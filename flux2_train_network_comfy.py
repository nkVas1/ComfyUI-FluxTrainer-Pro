# -*- coding: utf-8 -*-
"""
Flux.2 Network Trainer for ComfyUI
===================================

Специализированный тренер для обучения LoRA на моделях Flux.2 (Klein 9B и Dev)
с агрессивной оптимизацией памяти для работы на GPU с 8GB VRAM.

Основные возможности:
- Поддержка Flux.2 Klein 9B Base (9B параметров)
- Поддержка Flux.2 Dev (32B параметров)
- Агрессивный CPU offloading для блоков трансформера
- Gradient checkpointing с offload на CPU
- Optimizer state offloading
- Sequential block processing
- Совместимость с ComfyUI-MultiGPU подходом

Author: ComfyUI-FluxTrainer Team
License: Apache-2.0
"""

import torch
import copy
import math
import gc
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse

from .library import flux_models, flux_train_utils, flux_utils, sd3_train_utils, strategy_base, strategy_flux, train_util
from .library.device_utils import clean_memory_on_device
from .library.low_vram_utils import (
    LowVRAMConfig, 
    OffloadStrategy, 
    get_optimal_config_for_vram,
    setup_low_vram_training,
    aggressive_memory_cleanup,
    MemoryTracker,
    CPUOffloadOptimizer,
    SequentialBlockProcessor
)
from .train_network import NetworkTrainer, setup_parser
from .flux_train_network_comfy import FluxNetworkTrainer

from accelerate import Accelerator

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Flux2NetworkTrainer(FluxNetworkTrainer):
    """
    Тренер для Flux.2 моделей с расширенной поддержкой low VRAM.
    Наследует FluxNetworkTrainer и добавляет:
    - Агрессивный block swapping
    - CPU offloading для оптимизатора
    - Улучшенное управление памятью
    """
    
    def __init__(self):
        super().__init__()
        self.model_version = flux_utils.MODEL_VERSION_FLUX_V2
        self.low_vram_config: Optional[LowVRAMConfig] = None
        self.memory_tracker: Optional[MemoryTracker] = None
        self.block_processor: Optional[SequentialBlockProcessor] = None
        self.is_flux2_klein = False
        self.is_flux2_dev = False
        
    def assert_extra_args(self, args, train_dataset_group):
        """Расширенная проверка аргументов для Flux.2."""
        super().assert_extra_args(args, train_dataset_group)
        
        # Проверяем и устанавливаем low VRAM конфигурацию
        if hasattr(args, 'low_vram_mode') and args.low_vram_mode:
            vram_gb = getattr(args, 'available_vram_gb', 8.0)
            ram_gb = getattr(args, 'available_ram_gb', 32.0)
            self.low_vram_config = get_optimal_config_for_vram(vram_gb, ram_gb)
            logger.info(f"Low VRAM mode enabled: strategy={self.low_vram_config.strategy.value}")
            
            # Форсируем определенные настройки для low VRAM
            if self.low_vram_config.strategy in [OffloadStrategy.AGGRESSIVE, OffloadStrategy.EXTREME]:
                if args.blocks_to_swap is None or args.blocks_to_swap < self.low_vram_config.blocks_to_swap:
                    logger.info(f"Overriding blocks_to_swap: {args.blocks_to_swap} -> {self.low_vram_config.blocks_to_swap}")
                    args.blocks_to_swap = self.low_vram_config.blocks_to_swap
                
                if not args.gradient_checkpointing:
                    logger.info("Enabling gradient_checkpointing for low VRAM mode")
                    args.gradient_checkpointing = True
                
                if not args.cpu_offload_checkpointing:
                    logger.info("Enabling cpu_offload_checkpointing for low VRAM mode")
                    args.cpu_offload_checkpointing = True
        
        # Рекомендации для Flux.2 Dev (32B)
        if self.is_flux2_dev:
            logger.warning(
                "Flux.2 Dev (32B) detected. This model requires significant resources. "
                "Recommended: blocks_to_swap >= 40, batch_size=1, gradient_accumulation >= 8"
            )
    
    def load_target_model(self, args, weight_dtype, accelerator):
        """
        Загружает модель Flux.2 с определением версии и оптимизацией памяти.
        """
        # Определяем тип модели до загрузки
        loading_dtype = None if args.fp8_base else weight_dtype
        
        # Анализируем чекпоинт для определения версии
        is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), _ = flux_utils.analyze_checkpoint_state(
            args.pretrained_model_name_or_path
        )
        
        # Определяем версию Flux
        keys = []  # Будут заполнены при загрузке
        model_version, model_name = flux_utils.detect_flux_version(keys, num_double_blocks, num_single_blocks)
        
        self.is_flux2_klein = model_name == flux_utils.MODEL_NAME_FLUX2_KLEIN_9B
        self.is_flux2_dev = model_name == flux_utils.MODEL_NAME_FLUX2_DEV
        
        if self.is_flux2_klein or self.is_flux2_dev:
            logger.info(f"Detected Flux.2 model: {model_name}")
            self.model_version = flux_utils.MODEL_VERSION_FLUX_V2
        
        # Загрузка с оптимизацией для low VRAM
        if self.low_vram_config and self.low_vram_config.strategy != OffloadStrategy.NONE:
            logger.info("Loading model with low VRAM optimizations...")
            
            # Очищаем память перед загрузкой
            aggressive_memory_cleanup(accelerator.device)
            
            # Загружаем на CPU сначала
            self.is_schnell, model = flux_utils.load_flow_model(
                args.pretrained_model_name_or_path, 
                loading_dtype, 
                "cpu",  # Загружаем на CPU
                disable_mmap=args.disable_mmap_load_safetensors
            )
        else:
            # Стандартная загрузка
            self.is_schnell, model = flux_utils.load_flow_model(
                args.pretrained_model_name_or_path, 
                loading_dtype, 
                "cpu", 
                disable_mmap=args.disable_mmap_load_safetensors
            )
        
        if args.fp8_base:
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn or model.dtype == torch.float8_e5m2:
                logger.info(f"Loaded {model.dtype} FLUX.2 model")
        
        # Настройка block swapping
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"Enabling block swap for Flux.2: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        
        # Загрузка text encoders
        clip_l = flux_utils.load_clip_l(
            args.clip_l, weight_dtype, "cpu", 
            disable_mmap=args.disable_mmap_load_safetensors
        )
        clip_l.eval()
        
        # T5XXL - большой text encoder, важно правильно обработать
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None
        else:
            loading_dtype = weight_dtype
        
        logger.info("Loading T5XXL (this may take a while for large models)...")
        t5xxl = flux_utils.load_t5xxl(
            args.t5xxl, loading_dtype, "cpu", 
            disable_mmap=args.disable_mmap_load_safetensors
        )
        t5xxl.eval()
        
        if args.fp8_base and not args.fp8_base_unet:
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")
        
        # VAE
        ae = flux_utils.load_ae(
            args.ae, weight_dtype, "cpu", 
            disable_mmap=args.disable_mmap_load_safetensors
        )
        
        # Инициализация memory tracker
        self.memory_tracker = MemoryTracker(accelerator.device, log_interval=50)
        
        # Очистка после загрузки
        aggressive_memory_cleanup(accelerator.device)
        
        model_version_str = flux_utils.MODEL_VERSION_FLUX_V2 if (self.is_flux2_klein or self.is_flux2_dev) else flux_utils.MODEL_VERSION_FLUX_V1
        return model_version_str, [clip_l, t5xxl], ae, model
    
    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        """
        Кэширование выходов text encoder с оптимизацией памяти для Flux.2.
        """
        if self.low_vram_config and self.low_vram_config.cache_text_encoder_outputs:
            logger.info("Caching text encoder outputs for low VRAM mode...")
            
            # Выгружаем всё лишнее
            unet.to("cpu")
            vae.to("cpu")
            aggressive_memory_cleanup(accelerator.device)
        
        # Вызываем родительский метод
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)
        
        # После кэширования выгружаем text encoders на CPU
        if self.low_vram_config and self.low_vram_config.text_encoder_offload_to_cpu:
            for i, te in enumerate(text_encoders):
                if te is not None:
                    te.to("cpu")
                    logger.info(f"Offloaded text_encoder[{i}] to CPU after caching")
            aggressive_memory_cleanup(accelerator.device)
    
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
    ):
        """
        Вычисление предсказания шума с оптимизацией памяти для Flux.2.
        """
        # Логируем использование памяти
        if self.memory_tracker:
            self.memory_tracker.log_memory("before_forward")
        
        # Очищаем память перед forward pass для extreme mode
        if self.low_vram_config and self.low_vram_config.strategy == OffloadStrategy.EXTREME:
            aggressive_memory_cleanup(accelerator.device)
        
        # Вызываем родительский метод
        result = super().get_noise_pred_and_target(
            args, accelerator, noise_scheduler, latents, batch,
            text_encoder_conds, unet, network, weight_dtype, train_unet
        )
        
        if self.memory_tracker:
            self.memory_tracker.log_memory("after_forward")
        
        return result
    
    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Подготовка модели с учетом low VRAM оптимизаций.
        """
        if not self.is_swapping_blocks:
            return super(FluxNetworkTrainer, self).prepare_unet_with_accelerator(args, accelerator, unet)
        
        flux: flux_models.Flux = unet
        
        # Для low VRAM режима используем особую логику
        if self.low_vram_config and self.low_vram_config.strategy in [OffloadStrategy.AGGRESSIVE, OffloadStrategy.EXTREME]:
            logger.info("Preparing Flux.2 model with aggressive memory optimization")
            
            # Подготавливаем без автоматического размещения
            flux = accelerator.prepare(flux, device_placement=[False])
            
            # Вручную размещаем только необходимые части
            unwrapped = accelerator.unwrap_model(flux)
            unwrapped.move_to_device_except_swap_blocks(accelerator.device)
            unwrapped.prepare_block_swap_before_forward()
            
            # Принудительная очистка
            aggressive_memory_cleanup(accelerator.device)
        else:
            flux = accelerator.prepare(flux, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(flux).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(flux).prepare_block_swap_before_forward()
        
        return flux
    
    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        """Вызывается перед каждым шагом обучения."""
        super().on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype)
        
        # Для extreme mode очищаем память перед каждым шагом
        if self.low_vram_config and self.low_vram_config.empty_cache_frequently:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_sai_model_spec(self, args):
        """Возвращает спецификацию модели для метаданных."""
        if self.is_flux2_klein:
            return train_util.get_sai_model_spec(None, args, False, True, False, flux="flux2_klein")
        elif self.is_flux2_dev:
            return train_util.get_sai_model_spec(None, args, False, True, False, flux="flux2_dev")
        else:
            return super().get_sai_model_spec(args)
    
    def update_metadata(self, metadata, args):
        """Обновляет метаданные с информацией о Flux.2."""
        super().update_metadata(metadata, args)
        
        # Добавляем информацию о версии Flux
        metadata["ss_flux_version"] = self.model_version
        if self.is_flux2_klein:
            metadata["ss_flux_model"] = "flux2_klein_9b"
        elif self.is_flux2_dev:
            metadata["ss_flux_model"] = "flux2_dev"
        
        # Low VRAM информация
        if self.low_vram_config:
            metadata["ss_low_vram_strategy"] = self.low_vram_config.strategy.value
            metadata["ss_blocks_to_swap"] = self.low_vram_config.blocks_to_swap


def add_flux2_training_arguments(parser: argparse.ArgumentParser):
    """Добавляет аргументы специфичные для Flux.2 обучения."""
    
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        help="Enable aggressive memory optimizations for GPUs with 8GB or less VRAM"
    )
    parser.add_argument(
        "--available_vram_gb",
        type=float,
        default=8.0,
        help="Available VRAM in GB (used to auto-configure memory optimizations)"
    )
    parser.add_argument(
        "--available_ram_gb",
        type=float,
        default=32.0,
        help="Available system RAM in GB"
    )
    parser.add_argument(
        "--force_sequential_blocks",
        action="store_true",
        help="Force sequential processing of transformer blocks (slower but uses less memory)"
    )
    parser.add_argument(
        "--optimizer_cpu_offload",
        action="store_true",
        help="Offload optimizer states to CPU RAM"
    )
    parser.add_argument(
        "--flux2_model_type",
        type=str,
        choices=["auto", "klein_9b", "dev"],
        default="auto",
        help="Specify Flux.2 model type (auto-detected by default)"
    )
    
    return parser


def setup_flux2_parser() -> argparse.ArgumentParser:
    """Создает парсер аргументов для Flux.2 тренера."""
    parser = setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    add_flux2_training_arguments(parser)
    return parser
