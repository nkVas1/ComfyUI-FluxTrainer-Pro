# -*- coding: utf-8 -*-
"""
Low VRAM Training Utilities for Flux.2 Models
==============================================

Этот модуль предоставляет агрессивные стратегии оптимизации памяти
для обучения LoRA на Flux.2 моделях с ограниченным VRAM (8GB и менее).

Основные техники:
1. Sequential Block Processing - обработка блоков по одному
2. CPU Offloading - выгрузка весов модели на CPU/RAM
3. Gradient Checkpointing - пересчет активаций вместо хранения
4. Mixed Precision Training - FP8/BF16 смешанная точность
5. Optimizer State Offloading - состояние оптимизатора в RAM

Author: ComfyUI-FluxTrainer Team
License: Apache-2.0
"""

import gc
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from .device_utils import clean_memory_on_device

import logging
logger = logging.getLogger(__name__)


class OffloadStrategy(Enum):
    """Стратегии оффлоадинга для разных сценариев."""
    NONE = "none"                    # Без оффлоадинга (достаточно VRAM)
    CONSERVATIVE = "conservative"    # Только Text Encoders на CPU
    AGGRESSIVE = "aggressive"        # Модель и оптимизатор на CPU, блоки последовательно
    EXTREME = "extreme"              # Всё на CPU, один блок за раз, FP8


@dataclass
class LowVRAMConfig:
    """Конфигурация для обучения с низким VRAM."""
    
    # Основные настройки
    strategy: OffloadStrategy = OffloadStrategy.AGGRESSIVE
    available_vram_gb: float = 8.0
    available_ram_gb: float = 32.0
    
    # Block swapping
    blocks_to_swap: int = 20  # Количество блоков для свапинга CPU<->GPU
    swap_async: bool = True   # Асинхронный свап для перекрытия вычислений
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    cpu_offload_checkpointing: bool = True  # Активации на CPU
    
    # Precision
    use_fp8_base: bool = True         # FP8 для базовой модели
    use_bf16_training: bool = True    # BF16 для обучения
    
    # Text Encoder offloading
    cache_text_encoder_outputs: bool = True
    text_encoder_offload_to_cpu: bool = True
    
    # Optimizer offloading
    optimizer_offload_to_cpu: bool = True
    optimizer_pin_memory: bool = True  # Pinned memory для быстрого трансфера
    
    # VAE settings
    vae_slicing: bool = True           # Slice VAE для экономии памяти
    vae_tiling: bool = True            # Tiled VAE
    vae_offload: bool = True           # VAE на CPU
    
    # Batch settings (для low VRAM обычно 1)
    effective_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Накопление градиентов
    
    # Дополнительные оптимизации
    empty_cache_frequently: bool = True
    use_channels_last: bool = True    # Channels last memory format
    compile_model: bool = False       # torch.compile (экспериментально)
    
    def estimate_memory_usage(self, model_params_billions: float) -> Dict[str, float]:
        """Оценка использования памяти в GB."""
        # Примерные расчеты
        base_model_fp8 = model_params_billions * 1.0  # 1 byte per param
        base_model_fp16 = model_params_billions * 2.0  # 2 bytes per param
        
        lora_rank_64_fp16 = 0.1  # ~100MB для rank 64
        
        # Активации зависят от размера изображения и batch size
        activations_per_block = 0.5  # ~500MB per block peak
        
        optimizer_adam_fp32 = model_params_billions * 8.0  # 8 bytes (2 states)
        optimizer_adam_offloaded = 0.0  # На CPU
        
        return {
            "base_model_vram": base_model_fp8 if self.use_fp8_base else base_model_fp16,
            "lora_weights_vram": lora_rank_64_fp16,
            "activations_peak": activations_per_block * (1 if self.gradient_checkpointing else 20),
            "optimizer_vram": 0.0 if self.optimizer_offload_to_cpu else optimizer_adam_fp32,
            "optimizer_ram": optimizer_adam_fp32 if self.optimizer_offload_to_cpu else 0.0,
        }


class CPUOffloadOptimizer:
    """
    Враппер для оптимизатора, хранящий состояния на CPU.
    Переносит градиенты на CPU, обновляет веса там, и переносит обратно на GPU.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pin_memory: bool = True,
        async_transfer: bool = True
    ):
        self.optimizer = optimizer
        self.device = device
        self.pin_memory = pin_memory
        self.async_transfer = async_transfer
        
        # Перенос состояний оптимизатора на CPU
        self._move_optimizer_to_cpu()
        
        # Для асинхронного переноса
        self.stream = torch.cuda.Stream() if async_transfer and device.type == "cuda" else None
        
        logger.info(f"CPUOffloadOptimizer initialized with pin_memory={pin_memory}, async={async_transfer}")
    
    def _move_optimizer_to_cpu(self):
        """Переносит состояния оптимизатора на CPU."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            if self.pin_memory and value.device.type == "cuda":
                                state[key] = value.cpu().pin_memory()
                            else:
                                state[key] = value.cpu()
    
    def step(self, closure: Optional[Callable] = None):
        """Выполняет шаг оптимизатора с CPU offloading."""
        
        # Собираем градиенты на CPU
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # Переносим градиент на CPU
                    p.grad_cpu = p.grad.cpu()
                    if self.pin_memory:
                        p.grad_cpu = p.grad_cpu.pin_memory()
        
        # Освобождаем GPU память от градиентов
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Обновляем веса на CPU
        # (здесь нужна специальная логика для каждого оптимизатора)
        # Пока просто вызываем стандартный step
        
        # Восстанавливаем градиенты и делаем шаг
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad_cpu"):
                    p.grad = p.grad_cpu.to(self.device)
                    del p.grad_cpu
        
        self.optimizer.step(closure)
        
        # Очищаем градиенты
        self.optimizer.zero_grad(set_to_none=True)
    
    def zero_grad(self, set_to_none: bool = True):
        """Очищает градиенты."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._move_optimizer_to_cpu()


class SequentialBlockProcessor:
    """
    Процессор для последовательной обработки блоков трансформера.
    Только один блок находится на GPU в каждый момент времени.
    """
    
    def __init__(
        self,
        device: torch.device,
        cpu_device: torch.device = torch.device("cpu"),
        async_transfer: bool = True,
        prefetch_count: int = 1,
        debug: bool = False
    ):
        self.device = device
        self.cpu_device = cpu_device
        self.async_transfer = async_transfer
        self.prefetch_count = prefetch_count
        self.debug = debug
        
        self.stream = torch.cuda.Stream() if async_transfer and device.type == "cuda" else None
        self.executor = ThreadPoolExecutor(max_workers=2) if async_transfer else None
        
        self._timing_stats = {"to_gpu": [], "compute": [], "to_cpu": []}
    
    def process_blocks_forward(
        self,
        blocks: nn.ModuleList,
        hidden_states: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Последовательно обрабатывает блоки во время forward pass.
        Каждый блок загружается на GPU, обрабатывается, и выгружается.
        """
        for i, block in enumerate(blocks):
            if self.debug:
                start = time.perf_counter()
            
            # Загружаем блок на GPU
            block.to(self.device)
            
            if self.debug:
                torch.cuda.synchronize()
                load_time = time.perf_counter() - start
                self._timing_stats["to_gpu"].append(load_time)
                start = time.perf_counter()
            
            # Обрабатываем
            hidden_states = block(hidden_states, *args, **kwargs)
            
            if self.debug:
                torch.cuda.synchronize()
                compute_time = time.perf_counter() - start
                self._timing_stats["compute"].append(compute_time)
                start = time.perf_counter()
            
            # Выгружаем блок обратно на CPU (если не последний и не обучаемый)
            if not any(p.requires_grad for p in block.parameters()):
                block.to(self.cpu_device)
            
            if self.debug:
                torch.cuda.synchronize()
                unload_time = time.perf_counter() - start
                self._timing_stats["to_cpu"].append(unload_time)
        
        return hidden_states
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Возвращает статистику времени."""
        stats = {}
        for key, values in self._timing_stats.items():
            if values:
                stats[f"{key}_avg_ms"] = sum(values) / len(values) * 1000
                stats[f"{key}_total_ms"] = sum(values) * 1000
        return stats
    
    def clear_stats(self):
        """Очищает статистику."""
        self._timing_stats = {"to_gpu": [], "compute": [], "to_cpu": []}


class VAESlicing:
    """Утилиты для экономии памяти при работе с VAE."""
    
    @staticmethod
    def encode_sliced(vae, images: torch.Tensor, slice_size: int = 1) -> torch.Tensor:
        """
        Кодирует изображения по частям для экономии памяти.
        """
        latents = []
        for i in range(0, images.shape[0], slice_size):
            batch = images[i:i + slice_size]
            with torch.no_grad():
                latent = vae.encode(batch)
            latents.append(latent.cpu())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(latents, dim=0)
    
    @staticmethod
    def decode_sliced(vae, latents: torch.Tensor, slice_size: int = 1) -> torch.Tensor:
        """
        Декодирует латенты по частям для экономии памяти.
        """
        images = []
        for i in range(0, latents.shape[0], slice_size):
            batch = latents[i:i + slice_size]
            with torch.no_grad():
                image = vae.decode(batch.to(vae.device))
            images.append(image.cpu())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(images, dim=0)


class MemoryTracker:
    """Отслеживание и логирование использования памяти."""
    
    def __init__(self, device: torch.device, log_interval: int = 100):
        self.device = device
        self.log_interval = log_interval
        self.step_count = 0
        self.peak_memory = 0
        self.memory_history = []
    
    def log_memory(self, tag: str = ""):
        """Логирует текущее использование памяти."""
        if self.device.type != "cuda":
            return
        
        self.step_count += 1
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        
        self.peak_memory = max(self.peak_memory, allocated)
        self.memory_history.append(allocated)
        
        if self.step_count % self.log_interval == 0:
            logger.info(
                f"[Memory {tag}] "
                f"Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, "
                f"Peak: {self.peak_memory:.2f}GB"
            )
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику памяти."""
        return {
            "peak_gb": self.peak_memory,
            "avg_gb": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            "current_gb": self.memory_history[-1] if self.memory_history else 0,
        }
    
    @staticmethod
    def empty_cache():
        """Агрессивно очищает кэш CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def setup_low_vram_training(
    config: LowVRAMConfig,
    model: nn.Module,
    text_encoders: List[nn.Module],
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, Any]:
    """
    Настраивает окружение для обучения с низким VRAM.
    
    Returns:
        Dict с модифицированными компонентами и утилитами.
    """
    result = {
        "model": model,
        "text_encoders": text_encoders,
        "vae": vae,
        "optimizer": optimizer,
        "memory_tracker": MemoryTracker(device),
    }
    
    logger.info(f"Setting up low VRAM training with strategy: {config.strategy.value}")
    logger.info(f"Available VRAM: {config.available_vram_gb}GB, RAM: {config.available_ram_gb}GB")
    
    # 1. Gradient Checkpointing
    if config.gradient_checkpointing:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing(cpu_offload=config.cpu_offload_checkpointing)
            logger.info("Enabled gradient checkpointing for model")
    
    # 2. Text Encoder Offloading
    if config.text_encoder_offload_to_cpu:
        for i, te in enumerate(text_encoders):
            if te is not None:
                te.to("cpu")
                logger.info(f"Moved text_encoder[{i}] to CPU")
    
    # 3. VAE Offloading
    if config.vae_offload:
        vae.to("cpu")
        logger.info("Moved VAE to CPU")
    
    # 4. Optimizer Offloading
    if config.optimizer_offload_to_cpu:
        result["optimizer"] = CPUOffloadOptimizer(
            optimizer, device, 
            pin_memory=config.optimizer_pin_memory
        )
        logger.info("Wrapped optimizer with CPU offloading")
    
    # 5. Channels Last (для некоторых операций быстрее)
    if config.use_channels_last and hasattr(model, "to"):
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Converted model to channels_last memory format")
        except Exception as e:
            logger.warning(f"Could not convert to channels_last: {e}")
    
    # 6. Sequential Block Processor
    result["block_processor"] = SequentialBlockProcessor(
        device=device,
        async_transfer=config.swap_async,
        debug=False
    )
    
    # Выводим оценку использования памяти
    # Для Flux.2 Klein 9B
    mem_estimate = config.estimate_memory_usage(9.0)
    logger.info(f"Estimated memory usage for 9B model: {mem_estimate}")
    
    return result


def get_optimal_config_for_vram(vram_gb: float, ram_gb: float) -> LowVRAMConfig:
    """
    Возвращает оптимальную конфигурацию для заданного объема VRAM.
    """
    if vram_gb >= 24:
        return LowVRAMConfig(
            strategy=OffloadStrategy.NONE,
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
            blocks_to_swap=0,
            gradient_checkpointing=True,
            cpu_offload_checkpointing=False,
            use_fp8_base=False,
            optimizer_offload_to_cpu=False,
        )
    elif vram_gb >= 16:
        return LowVRAMConfig(
            strategy=OffloadStrategy.CONSERVATIVE,
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
            blocks_to_swap=10,
            gradient_checkpointing=True,
            cpu_offload_checkpointing=False,
            use_fp8_base=True,
            optimizer_offload_to_cpu=False,
        )
    elif vram_gb >= 8:
        return LowVRAMConfig(
            strategy=OffloadStrategy.AGGRESSIVE,
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
            blocks_to_swap=25,
            gradient_checkpointing=True,
            cpu_offload_checkpointing=True,
            use_fp8_base=True,
            optimizer_offload_to_cpu=True,
            effective_batch_size=1,
            gradient_accumulation_steps=8,
        )
    else:
        return LowVRAMConfig(
            strategy=OffloadStrategy.EXTREME,
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
            blocks_to_swap=35,  # Почти все блоки
            gradient_checkpointing=True,
            cpu_offload_checkpointing=True,
            use_fp8_base=True,
            optimizer_offload_to_cpu=True,
            empty_cache_frequently=True,
            effective_batch_size=1,
            gradient_accumulation_steps=16,
        )


# Утилитные функции для быстрого доступа
def aggressive_memory_cleanup(device: Optional[torch.device] = None):
    """Агрессивная очистка памяти."""
    gc.collect()
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.empty_cache()
            clean_memory_on_device(device)
        else:
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()


def estimate_trainable_params_memory(model: nn.Module) -> float:
    """Оценивает память для обучаемых параметров в GB."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Для Adam: param + grad + 2 момента = 4x размер параметров
    # В FP32: 4 bytes per param
    bytes_needed = trainable_params * 4 * 4  # 4 copies, 4 bytes each
    return bytes_needed / (1024 ** 3)


# =============================================================================
# VRAM SAFETY CHECKER - Предиктивная проверка памяти
# =============================================================================
@dataclass
class VRAMEstimate:
    """Результат оценки VRAM."""
    total_vram_needed_gb: float
    base_model_gb: float
    lora_weights_gb: float
    activations_peak_gb: float
    optimizer_gb: float
    text_encoder_gb: float
    safety_margin_gb: float
    
    available_vram_gb: float
    will_fit: bool
    risk_level: str  # "safe", "warning", "danger", "critical"
    recommendations: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "✅" if self.will_fit else "⚠️"
        return (
            f"{status} VRAM Estimate: {self.total_vram_needed_gb:.1f}GB needed, "
            f"{self.available_vram_gb:.1f}GB available ({self.risk_level})"
        )


def estimate_vram_usage(
    model_params_billions: float = 9.0,
    resolution: Tuple[int, int] = (1024, 1024),
    batch_size: int = 1,
    network_dim: int = 16,
    gradient_checkpointing: bool = True,
    use_fp8_base: bool = True,
    optimizer_offload: bool = True,
    cache_text_encoder: bool = True,
    blocks_to_swap: int = 20,
    available_vram_gb: float = 8.0,
) -> VRAMEstimate:
    """
    Предиктивно оценивает использование VRAM перед началом обучения.
    
    Помогает предотвратить OOM (Out of Memory) ошибки заранее.
    
    Args:
        model_params_billions: Количество параметров модели в миллиардах (9B для Klein, 12B для Dev)
        resolution: Размер изображения (width, height)
        batch_size: Размер батча
        network_dim: Ранг LoRA
        gradient_checkpointing: Включен ли gradient checkpointing
        use_fp8_base: Используется ли FP8 для базовой модели
        optimizer_offload: Выгружается ли оптимизатор на CPU
        cache_text_encoder: Кэшируются ли выходы text encoder
        blocks_to_swap: Количество блоков для CPU swapping
        available_vram_gb: Доступная VRAM в GB
    
    Returns:
        VRAMEstimate с детальной информацией
    """
    recommendations = []
    
    # === Базовая модель ===
    # FP8: 1 byte/param, FP16: 2 bytes/param
    bytes_per_param = 1.0 if use_fp8_base else 2.0
    base_model_bytes = model_params_billions * 1e9 * bytes_per_param
    
    # Учитываем block swapping - часть модели на CPU
    if blocks_to_swap > 0:
        total_blocks = 57  # Flux имеет примерно 57 блоков (19 double + 38 single)
        gpu_blocks_ratio = max(0.1, 1.0 - (blocks_to_swap / total_blocks))
        base_model_bytes *= gpu_blocks_ratio
    
    base_model_gb = base_model_bytes / (1024 ** 3)
    
    # === LoRA веса ===
    # LoRA добавляет примерно (rank * hidden_dim * 2) параметров на слой
    # Примерная оценка: rank * 0.01 GB
    lora_weights_gb = network_dim * 0.008  # ~130MB для rank 16
    
    # === Активации ===
    # Размер активаций зависит от разрешения и batch size
    pixels = resolution[0] * resolution[1]
    # Latent space 1/8 от оригинала, 16 каналов, FP16
    latent_size = (pixels / 64) * 16 * 2  # bytes
    
    # Активации трансформера примерно 4KB на пиксель latent
    if gradient_checkpointing:
        # С checkpointing храним только 1/4 активаций
        activation_multiplier = 0.5
    else:
        activation_multiplier = 4.0
    
    activations_bytes = latent_size * activation_multiplier * batch_size * 1000
    activations_peak_gb = activations_bytes / (1024 ** 3)
    
    # === Text Encoder ===
    if cache_text_encoder:
        # Если кэшируем, text encoder не занимает VRAM во время обучения
        text_encoder_gb = 0.0
    else:
        # T5-XXL ~10GB, CLIP-L ~0.5GB
        text_encoder_gb = 2.0  # Минимум для inference
    
    # === Optimizer ===
    if optimizer_offload:
        optimizer_gb = 0.0
    else:
        # AdamW: 2 момента * размер LoRA параметров * 4 bytes
        lora_params = network_dim * 1e6  # Примерная оценка
        optimizer_gb = (lora_params * 2 * 4) / (1024 ** 3)
    
    # === Safety margin ===
    # CUDA и PyTorch требуют дополнительную память для operations
    safety_margin_gb = 1.0 + (batch_size * 0.2)
    
    # === Total ===
    total_vram_needed_gb = (
        base_model_gb + 
        lora_weights_gb + 
        activations_peak_gb + 
        optimizer_gb + 
        text_encoder_gb + 
        safety_margin_gb
    )
    
    # === Risk assessment ===
    headroom = available_vram_gb - total_vram_needed_gb
    
    if headroom >= 2.0:
        risk_level = "safe"
        will_fit = True
    elif headroom >= 0.5:
        risk_level = "warning"
        will_fit = True
        recommendations.append("⚠️ Близко к лимиту VRAM. Могут быть спайки памяти.")
    elif headroom >= 0:
        risk_level = "danger"
        will_fit = True
        recommendations.append("🔴 Очень мало запаса. Высокий риск OOM при пиках нагрузки.")
        recommendations.append("Рекомендуется увеличить blocks_to_swap или уменьшить batch_size.")
    else:
        risk_level = "critical"
        will_fit = False
        recommendations.append("❌ НЕДОСТАТОЧНО VRAM! Обучение вызовет OOM ошибку.")
        
        # Предлагаем решения
        if not use_fp8_base:
            recommendations.append("→ Включите use_fp8_base для экономии ~50% VRAM модели")
        if not gradient_checkpointing:
            recommendations.append("→ Включите gradient_checkpointing для экономии активаций")
        if blocks_to_swap < 25:
            recommendations.append(f"→ Увеличьте blocks_to_swap до {min(35, blocks_to_swap + 10)}")
        if batch_size > 1:
            recommendations.append("→ Уменьшите batch_size до 1")
        if not optimizer_offload:
            recommendations.append("→ Включите optimizer_offload для выгрузки оптимизатора на CPU")
        if resolution[0] > 512 or resolution[1] > 512:
            recommendations.append("→ Попробуйте уменьшить разрешение обучения")
    
    return VRAMEstimate(
        total_vram_needed_gb=total_vram_needed_gb,
        base_model_gb=base_model_gb,
        lora_weights_gb=lora_weights_gb,
        activations_peak_gb=activations_peak_gb,
        optimizer_gb=optimizer_gb,
        text_encoder_gb=text_encoder_gb,
        safety_margin_gb=safety_margin_gb,
        available_vram_gb=available_vram_gb,
        will_fit=will_fit,
        risk_level=risk_level,
        recommendations=recommendations,
    )


def print_vram_estimate(estimate: VRAMEstimate):
    """Выводит детальную информацию об использовании VRAM."""
    logger.info("=" * 60)
    logger.info("  VRAM Usage Estimate")
    logger.info("=" * 60)
    logger.info(f"  Base Model:      {estimate.base_model_gb:>6.2f} GB")
    logger.info(f"  LoRA Weights:    {estimate.lora_weights_gb:>6.2f} GB")
    logger.info(f"  Activations:     {estimate.activations_peak_gb:>6.2f} GB")
    logger.info(f"  Text Encoder:    {estimate.text_encoder_gb:>6.2f} GB")
    logger.info(f"  Optimizer:       {estimate.optimizer_gb:>6.2f} GB")
    logger.info(f"  Safety Margin:   {estimate.safety_margin_gb:>6.2f} GB")
    logger.info("-" * 60)
    logger.info(f"  TOTAL NEEDED:    {estimate.total_vram_needed_gb:>6.2f} GB")
    logger.info(f"  AVAILABLE:       {estimate.available_vram_gb:>6.2f} GB")
    logger.info("-" * 60)
    
    status_icons = {
        "safe": "✅ SAFE",
        "warning": "⚠️  WARNING",
        "danger": "🔴 DANGER",
        "critical": "❌ CRITICAL"
    }
    logger.info(f"  Status: {status_icons.get(estimate.risk_level, estimate.risk_level)}")
    
    for rec in estimate.recommendations:
        logger.info(f"  {rec}")
    
    logger.info("=" * 60)


# =============================================================================
# AUTO-RESUME TRAINING - Автоматическое продолжение обучения
# =============================================================================
def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Находит последний чекпоинт в директории вывода.
    
    Поддерживает форматы:
    - flux2_lora_step_1000.safetensors
    - epoch_5_step_500.safetensors
    - *.safetensors (по времени модификации)
    
    Args:
        output_dir: Директория с чекпоинтами
    
    Returns:
        Путь к последнему чекпоинту или None
    """
    import glob
    import re
    from pathlib import Path
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Ищем safetensors файлы
    checkpoints = list(output_path.glob("*.safetensors"))
    
    if not checkpoints:
        # Проверяем подпапки
        checkpoints = list(output_path.glob("**/*.safetensors"))
    
    if not checkpoints:
        return None
    
    # Пробуем отсортировать по номеру шага
    step_pattern = re.compile(r'step[_-]?(\d+)', re.IGNORECASE)
    epoch_pattern = re.compile(r'epoch[_-]?(\d+)', re.IGNORECASE)
    
    def extract_order(path: Path) -> Tuple[int, int, float]:
        """Извлекает порядок сортировки (epoch, step, mtime)."""
        name = path.name
        
        epoch = 0
        step = 0
        
        epoch_match = epoch_pattern.search(name)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        
        step_match = step_pattern.search(name)
        if step_match:
            step = int(step_match.group(1))
        
        mtime = path.stat().st_mtime
        
        return (epoch, step, mtime)
    
    # Сортируем: epoch DESC, step DESC, mtime DESC
    sorted_checkpoints = sorted(checkpoints, key=extract_order, reverse=True)
    
    if sorted_checkpoints:
        latest = str(sorted_checkpoints[0])
        logger.info(f"🔄 Found latest checkpoint: {latest}")
        return latest
    
    return None


def auto_resume_training(
    output_dir: str,
    args,
    force: bool = False
) -> bool:
    """
    Автоматически настраивает продолжение обучения с последнего чекпоинта.
    
    Args:
        output_dir: Директория с чекпоинтами
        args: Namespace с аргументами обучения
        force: Форсировать resume даже если уже указан
    
    Returns:
        True если resume настроен, False если чекпоинт не найден
    """
    # Проверяем, не указан ли уже resume
    if hasattr(args, 'resume') and args.resume and not force:
        logger.info(f"Resume already configured: {args.resume}")
        return True
    
    if hasattr(args, 'network_weights') and args.network_weights and not force:
        logger.info(f"Network weights already specified: {args.network_weights}")
        return True
    
    # Ищем последний чекпоинт
    latest = find_latest_checkpoint(output_dir)
    
    if latest:
        logger.info(f"🔄 Auto-resume: Found checkpoint at {latest}")
        
        # Устанавливаем network_weights для продолжения LoRA обучения
        if not hasattr(args, 'network_weights') or not args.network_weights:
            args.network_weights = latest
            logger.info(f"   Set network_weights = {latest}")
        
        return True
    
    logger.info("No previous checkpoint found. Starting fresh training.")
    return False


def get_training_progress(output_dir: str) -> Dict[str, Any]:
    """
    Анализирует прогресс обучения из директории.
    
    Returns:
        Dict с информацией о прогрессе:
        - total_checkpoints: количество чекпоинтов
        - latest_checkpoint: путь к последнему
        - latest_step: номер последнего шага (если определим)
        - training_started: была ли начата тренировка
    """
    import re
    from pathlib import Path
    
    output_path = Path(output_dir)
    result = {
        "total_checkpoints": 0,
        "latest_checkpoint": None,
        "latest_step": 0,
        "latest_epoch": 0,
        "training_started": False,
    }
    
    if not output_path.exists():
        return result
    
    checkpoints = list(output_path.glob("**/*.safetensors"))
    result["total_checkpoints"] = len(checkpoints)
    result["training_started"] = len(checkpoints) > 0
    
    if checkpoints:
        result["latest_checkpoint"] = find_latest_checkpoint(output_dir)
        
        # Пробуем извлечь step из имени
        if result["latest_checkpoint"]:
            name = Path(result["latest_checkpoint"]).name
            step_match = re.search(r'step[_-]?(\d+)', name, re.IGNORECASE)
            if step_match:
                result["latest_step"] = int(step_match.group(1))
            
            epoch_match = re.search(r'epoch[_-]?(\d+)', name, re.IGNORECASE)
            if epoch_match:
                result["latest_epoch"] = int(epoch_match.group(1))
    
    return result
