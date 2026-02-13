# -*- coding: utf-8 -*-
"""
ComfyUI Nodes for Flux.2 Training
==================================

Ноды для обучения LoRA на моделях Flux.2 (Klein 9B и Dev) 
с поддержкой low VRAM GPU (8GB и менее).

Особенности:
- Автоматическое определение версии модели
- Настраиваемые стратегии экономии памяти
- Поддержка CPU offloading
- Полная интеграция с существующей системой FluxTrainer
- LAZY IMPORTS - ноды загружаются даже если зависимости недоступны

Author: ComfyUI-FluxTrainer-Pro Team
License: Apache-2.0
"""

import os
import sys
import json
import shlex
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable

import folder_paths
import comfy.model_management as mm
import comfy.utils

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# LAZY IMPORT SYSTEM - Ключевое решение для стабильной загрузки
# =============================================================================
# Кэш для загруженных модулей - загружаем один раз при первом использовании
_CACHED_MODULES: Dict[str, Any] = {}
_IMPORT_ERROR: Optional[str] = None


def _lazy_import_flux_utils():
    """Загружает только flux_utils для анализа моделей (легкая зависимость)."""
    if "flux_utils" in _CACHED_MODULES:
        return _CACHED_MODULES["flux_utils"]
    
    try:
        from .library import flux_utils
        _CACHED_MODULES["flux_utils"] = flux_utils
        return flux_utils
    except Exception as e:
        logger.warning(f"Could not import flux_utils: {e}")
        return None


def _lazy_import_training():
    """
    Загружает тяжелые модули обучения ТОЛЬКО когда они реально нужны.
    Это предотвращает падение ComfyUI при старте из-за ошибок в diffusers/triton/bitsandbytes.
    
    Ошибка появится ТОЛЬКО при нажатии Queue Prompt, а не при загрузке нод!
    
    NOTE: Патч triton выполняется в __init__.py (глобально) до загрузки любых нод.
    """
    global _IMPORT_ERROR
    
    if "FluxNetworkTrainer" in _CACHED_MODULES:
        return _CACHED_MODULES
    
    # =======================================================================
    # WINDOWS ENVIRONMENT SETUP - Переменные окружения для Triton/CUDA
    # Сам патч triton уже выполнен в __init__.py
    # =======================================================================
    if sys.platform == 'win32':
        python_home = os.path.dirname(sys.executable)
        
        # Добавляем путь к include для Triton
        include_path = os.path.join(python_home, 'include')
        if os.path.exists(include_path):
            os.environ.setdefault('INCLUDE', include_path)
        
        # Путь к ptxas.exe для CUDA компиляции (если есть)
        ptxas_candidates = [
            os.path.join(python_home, 'Library', 'bin', 'ptxas.exe'),
            os.path.join(os.environ.get('CUDA_PATH', ''), 'bin', 'ptxas.exe'),
        ]
        for ptxas_path in ptxas_candidates:
            if os.path.exists(ptxas_path):
                os.environ['TRITON_PTXAS_PATH'] = ptxas_path
                break
        
        # Отключаем JIT компиляцию Triton если нет компилятора
        if not os.path.exists(os.path.join(python_home, 'include', 'Python.h')):
            os.environ.setdefault('TRITON_DISABLE_LINE_INFO', '1')
            logger.debug("[Flux2] Windows Embedded Python detected")
    
    try:
        import toml
        import torch
        from .flux_train_network_comfy import FluxNetworkTrainer
        from .train_network import setup_parser as train_network_setup_parser
        from .library import flux_train_utils, flux_utils, train_util
        from .library.low_vram_utils import (
            LowVRAMConfig, 
            OffloadStrategy, 
            get_optimal_config_for_vram,
            aggressive_memory_cleanup,
            estimate_vram_usage,
            print_vram_estimate,
            auto_resume_training,
            get_training_progress,
            find_latest_checkpoint,
        )
        
        # IPEX (Intel GPU) - строго опционально
        clean_memory_on_device: Callable = lambda *args, **kwargs: None
        try:
            from .library.device_utils import init_ipex, clean_memory_on_device as _clean
            init_ipex()
            clean_memory_on_device = _clean
        except ImportError:
            pass
        except Exception as ipex_err:
            logger.debug(f"IPEX not available: {ipex_err}")
        
        _CACHED_MODULES.update({
            "toml": toml,
            "torch": torch,
            "FluxNetworkTrainer": FluxNetworkTrainer,
            "train_network_setup_parser": train_network_setup_parser,
            "flux_train_utils": flux_train_utils,
            "flux_utils": flux_utils,
            "train_util": train_util,
            "LowVRAMConfig": LowVRAMConfig,
            "OffloadStrategy": OffloadStrategy,
            "get_optimal_config_for_vram": get_optimal_config_for_vram,
            "aggressive_memory_cleanup": aggressive_memory_cleanup,
            "estimate_vram_usage": estimate_vram_usage,
            "print_vram_estimate": print_vram_estimate,
            "auto_resume_training": auto_resume_training,
            "get_training_progress": get_training_progress,
            "find_latest_checkpoint": find_latest_checkpoint,
            "clean_memory_on_device": clean_memory_on_device,
        })
        
        logger.info("[Flux2] Training modules loaded successfully")
        return _CACHED_MODULES
        
    except Exception as e:
        _IMPORT_ERROR = str(e)
        import traceback
        # sys already imported at module level (line 21)
        
        # Детальная диагностика ошибки
        error_lower = str(e).lower()
        traceback_str = traceback.format_exc()
        
        # Определяем тип ошибки и даём конкретные рекомендации
        if "python.h" in error_lower or "include file" in error_lower:
            problem = "[ERROR] COMPILATION ERROR: Python.h not found"
            diagnosis = [
                "Вы используете embedded/portable Python, который не поддерживает компиляцию C расширений.",
                "",
                "🔧 РЕШЕНИЕ:",
                "1. Запустите: python install.py",
                "   Это установит pre-built wheels для triton и bitsandbytes",
                "",
                "2. Или установите полный Python с python.org:",
                "   - Скачайте 'Windows installer (64-bit)' с https://python.org",
                "   - При установке выберите 'Add Python to PATH'",
                "   - Переустановите ComfyUI с полным Python",
            ]
        elif "triton" in error_lower:
            problem = "[ERROR] TRITON ERROR: Could not load triton"
            diagnosis = [
                "Triton требует специальной сборки для Windows.",
                "",
                "🔧 РЕШЕНИЕ:",
                "1. Запустите: python install.py",
                "   Это установит pre-built triton для Windows",
                "",
                "2. Или вручную: pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cpXXX-win_amd64.whl",
                "   (замените XXX на вашу версию Python: 310, 311, 312)",
            ]
        elif "bitsandbytes" in error_lower:
            problem = "[ERROR] BITSANDBYTES ERROR: Could not load bitsandbytes"
            diagnosis = [
                "bitsandbytes требует CUDA и специальной сборки для Windows.",
                "",
                "🔧 РЕШЕНИЕ:",
                "1. Запустите: python install.py",
                "   Это установит pre-built bitsandbytes для Windows",
                "",
                "2. Или вручную: pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui",
            ]
        elif "torch" in error_lower or "cuda" in error_lower:
            problem = "[ERROR] TORCH/CUDA ERROR: Problem with PyTorch or CUDA"
            diagnosis = [
                "PyTorch не настроен правильно или отсутствует CUDA.",
                "",
                "🔧 РЕШЕНИЕ:",
                "1. Проверьте установку CUDA: nvidia-smi",
                "2. Переустановите PyTorch с CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            ]
        else:
            problem = f"[ERROR] IMPORT ERROR: {type(e).__name__}"
            diagnosis = [
                f"Не удалось загрузить модули обучения: {e}",
                "",
                "🔧 РЕШЕНИЕ:",
                "1. Запустите: python install.py",
                "2. Проверьте requirements.txt: pip install -r requirements.txt",
                "3. Убедитесь, что CUDA установлена правильно",
            ]
        
        # Формируем красивое сообщение
        separator = "=" * 70
        error_lines = [
            "",
            separator,
            problem,
            separator,
            "",
        ] + diagnosis + [
            "",
            f"Python: {sys.version}",
            f"Executable: {sys.executable}",
            "",
            "Полный traceback:",
            traceback_str,
            separator,
        ]
        
        error_msg = "\n".join(error_lines)
        logger.error(error_msg)
        
        # Также выводим в консоль для видимости
        print(error_msg)
        
        raise RuntimeError(error_msg)


class Flux2TrainModelSelect:
    """
    Выбор моделей Flux.2 для обучения.
    Поддерживает Flux.2 Klein 9B Base и Flux.2 Dev.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": (folder_paths.get_filename_list("unet"), {
                    "tooltip": "Flux.2 transformer model (flux2_klein_9b or flux2_dev)"
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "VAE model (ae.safetensors)"
                }),
                "clip_l": (folder_paths.get_filename_list("clip"), {
                    "tooltip": "CLIP-L text encoder"
                }),
                "t5": (folder_paths.get_filename_list("clip"), {
                    "tooltip": "T5-XXL text encoder"
                }),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "tooltip": "Pre-trained LoRA path to continue training from (optional)"
                }),
            }
        }

    RETURN_TYPES = ("TRAIN_FLUX2_MODELS",)
    RETURN_NAMES = ("flux2_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer/Flux2"

    def loadmodel(self, transformer, vae, clip_l, t5, lora_path=""):
        # LAZY IMPORT - загружаем только при использовании
        flux_utils = _lazy_import_flux_utils()
        
        transformer_path = folder_paths.get_full_path("unet", transformer)
        vae_path = folder_paths.get_full_path("vae", vae)
        clip_path = folder_paths.get_full_path("clip", clip_l)
        t5_path = folder_paths.get_full_path("clip", t5)

        # Определяем тип модели
        model_type = "auto"
        if flux_utils:
            try:
                is_diffusers, is_schnell, (num_double, num_single), _ = flux_utils.analyze_checkpoint_state(transformer_path)
                if num_double > 24 or num_single > 50:
                    model_type = "flux2_dev"
                    logger.info(f"Detected Flux.2 Dev model (blocks: {num_double}/{num_single})")
                else:
                    model_type = "flux2_klein_9b"
                    logger.info(f"Detected Flux.2 Klein 9B model (blocks: {num_double}/{num_single})")
            except Exception as e:
                logger.warning(f"Could not auto-detect model type: {e}")

        flux2_models = {
            "transformer": transformer_path,
            "vae": vae_path,
            "clip_l": clip_path,
            "t5": t5_path,
            "lora_path": lora_path,
            "model_type": model_type
        }
        
        return (flux2_models,)


class Flux2TrainModelPaths:
    """
    Ручной ввод путей к моделям Flux.2.
    Используйте, если файлы не лежат в стандартных папках ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path or filename in models/unet"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path or filename in models/vae"
                }),
                "clip_l_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path or filename in models/clip"
                }),
                "t5_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path or filename in models/clip"
                }),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Pre-trained LoRA path to continue training from"
                }),
            }
        }

    RETURN_TYPES = ("TRAIN_FLUX2_MODELS",)
    RETURN_NAMES = ("flux2_models",)
    FUNCTION = "loadmodel_paths"
    CATEGORY = "FluxTrainer/Flux2"

    def _resolve(self, path_value: str, folder_key: str, required: bool = True) -> str:
        if not path_value or not path_value.strip():
            if required:
                friendly_names = {
                    "unet": "Transformer (UNet)",
                    "vae": "VAE",
                    "clip": "CLIP / T5",
                }
                name = friendly_names.get(folder_key, folder_key)
                raise ValueError(
                    f"Путь к модели '{name}' не указан. "
                    f"Укажите путь к файлу или выберите модель через виджет."
                )
            return ""
        path_value = path_value.strip()
        if os.path.isabs(path_value) and os.path.exists(path_value):
            return path_value
        # Try resolve relative to ComfyUI models folders
        resolved = folder_paths.get_full_path(folder_key, path_value)
        if resolved and os.path.exists(resolved):
            return resolved
        # Fallback: direct path check (relative)
        if os.path.exists(path_value):
            return os.path.abspath(path_value)
        raise FileNotFoundError(
            f"Файл не найден: '{path_value}'. "
            f"Проверьте путь и наличие файла."
        )

    def loadmodel_paths(self, transformer_path, vae_path, clip_l_path, t5_path, lora_path=""):
        # LAZY IMPORT
        flux_utils = _lazy_import_flux_utils()
        
        transformer_path = self._resolve(transformer_path, "unet")
        vae_path = self._resolve(vae_path, "vae")
        clip_path = self._resolve(clip_l_path, "clip")
        t5_path = self._resolve(t5_path, "clip")

        # Определяем тип модели
        model_type = "auto"
        if flux_utils:
            try:
                is_diffusers, is_schnell, (num_double, num_single), _ = flux_utils.analyze_checkpoint_state(transformer_path)
                if num_double > 24 or num_single > 50:
                    model_type = "flux2_dev"
                    logger.info(f"Detected Flux.2 Dev model (blocks: {num_double}/{num_single})")
                else:
                    model_type = "flux2_klein_9b"
                    logger.info(f"Detected Flux.2 Klein 9B model (blocks: {num_double}/{num_single})")
            except Exception as e:
                logger.warning(f"Could not auto-detect model type: {e}")

        flux2_models = {
            "transformer": transformer_path,
            "vae": vae_path,
            "clip_l": clip_path,
            "t5": t5_path,
            "lora_path": lora_path,
            "model_type": model_type,
        }

        return (flux2_models,)


class Flux2LowVRAMConfig:
    """
    Конфигурация для обучения с низким VRAM.
    Автоматически подбирает оптимальные настройки для вашей GPU.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strategy": (["auto", "none", "conservative", "aggressive", "extreme"], {
                    "default": "aggressive",
                    "tooltip": "Memory optimization strategy. 'auto' detects based on VRAM."
                }),
                "available_vram_gb": ("FLOAT", {
                    "default": 8.0, 
                    "min": 4.0, 
                    "max": 48.0, 
                    "step": 0.5,
                    "tooltip": "Your GPU VRAM in GB (e.g., 8.0 for RTX 3060 Ti)"
                }),
                "available_ram_gb": ("FLOAT", {
                    "default": 32.0, 
                    "min": 8.0, 
                    "max": 256.0, 
                    "step": 1.0,
                    "tooltip": "Your system RAM in GB"
                }),
                "blocks_to_swap": ("INT", {
                    "default": 20, 
                    "min": 0, 
                    "max": 50, 
                    "step": 1,
                    "tooltip": "Number of transformer blocks to swap between GPU and CPU (higher = less VRAM, slower)"
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable gradient checkpointing (saves memory, slightly slower)"
                }),
                "cpu_offload_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload gradient checkpoints to CPU (saves more VRAM)"
                }),
                "cache_text_encoder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache text encoder outputs (recommended for low VRAM)"
                }),
                "optimizer_cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep optimizer states in RAM instead of VRAM"
                }),
            },
            "optional": {
                "use_fp8_base": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load base model in FP8 precision (saves ~50% VRAM)"
                }),
                "empty_cache_frequently": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Aggressively clear CUDA cache (slower but more stable)"
                }),
            }
        }

    RETURN_TYPES = ("FLUX2_LOW_VRAM_CONFIG",)
    RETURN_NAMES = ("low_vram_config",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer/Flux2"

    def create_config(
        self, 
        strategy, 
        available_vram_gb, 
        available_ram_gb,
        blocks_to_swap,
        gradient_checkpointing,
        cpu_offload_checkpointing,
        cache_text_encoder,
        optimizer_cpu_offload,
        use_fp8_base=True,
        empty_cache_frequently=True
    ):
        # LAZY IMPORT - загружаем модули обучения только при использовании
        modules = _lazy_import_training()
        LowVRAMConfig = modules["LowVRAMConfig"]
        OffloadStrategy = modules["OffloadStrategy"]
        get_optimal_config_for_vram = modules["get_optimal_config_for_vram"]
        
        # Определяем стратегию
        if strategy == "auto":
            config = get_optimal_config_for_vram(available_vram_gb, available_ram_gb)
        else:
            strategy_enum = {
                "none": OffloadStrategy.NONE,
                "conservative": OffloadStrategy.CONSERVATIVE,
                "aggressive": OffloadStrategy.AGGRESSIVE,
                "extreme": OffloadStrategy.EXTREME
            }.get(strategy, OffloadStrategy.AGGRESSIVE)
            
            config = LowVRAMConfig(
                strategy=strategy_enum,
                available_vram_gb=available_vram_gb,
                available_ram_gb=available_ram_gb,
                blocks_to_swap=blocks_to_swap,
                gradient_checkpointing=gradient_checkpointing,
                cpu_offload_checkpointing=cpu_offload_checkpointing,
                cache_text_encoder_outputs=cache_text_encoder,
                optimizer_offload_to_cpu=optimizer_cpu_offload,
                use_fp8_base=use_fp8_base,
                empty_cache_frequently=empty_cache_frequently,
            )
        
        # Выводим рекомендации
        mem_estimate = config.estimate_memory_usage(9.0)  # Для Klein 9B
        logger.info(f"Low VRAM Config: strategy={config.strategy.value}")
        logger.info(f"  Blocks to swap: {config.blocks_to_swap}")
        logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")
        logger.info(f"  Estimated VRAM usage: ~{sum(v for k,v in mem_estimate.items() if 'vram' in k):.1f}GB")
        
        return (config,)


# =============================================================================
# NODE: Flux2InitTraining - Главный нод для инициализации обучения
# =============================================================================
class Flux2InitTraining:
    """
    Инициализация обучения Flux.2 LoRA.
    Основной узел для настройки тренировочной сессии.
    Оптимизирован для работы с ограниченным VRAM.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flux2_models": ("TRAIN_FLUX2_MODELS",),
                "dataset": ("JSON",),
                "optimizer_settings": ("ARGS",),
                "output_name": ("STRING", {"default": "flux2_lora", "multiline": False}),
                "output_dir": ("STRING", {"default": "flux2_trainer_output", "multiline": False, 
                    "tooltip": "Output directory path (relative to ComfyUI folder)"}),
                
                # Network type - LoRA or DoRA
                "network_type": (["lora", "dora"], {
                    "default": "lora",
                    "tooltip": "LoRA - classic Low-Rank Adaptation. DoRA - Weight-Decomposed LoRA (better quality, slightly more VRAM)"
                }),
                
                # LoRA settings
                "network_dim": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "LoRA rank (dim). Lower = less VRAM. Recommended: 8-32 for low VRAM"}),
                "network_alpha": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 128.0, "step": 0.1}),
                
                # Training settings
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6,
                    "tooltip": "Learning rate. Recommended: 1e-4 to 5e-4"}),
                "max_train_steps": ("INT", {"default": 1000, "min": 1, "max": 100000, "step": 1}),
                
                # Data settings
                "cache_latents": (["disk", "memory", "disabled"], {"default": "disk",
                    "tooltip": "Cache VAE latents. 'disk' recommended for low VRAM"}),
                "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"default": "disk",
                    "tooltip": "Cache text encoder outputs. 'disk' recommended for low VRAM"}),
                
                # Precision
                "gradient_dtype": (["bf16", "fp16"], {"default": "bf16"}),
                "save_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                
                # Memory optimization
                "optimizer_fusing": (["fused_backward_pass", "blockwise_fused_optimizers"], {
                    "default": "fused_backward_pass",
                    "tooltip": "Memory optimization for optimizer. Both significantly reduce VRAM"}),
                
                # Sample prompts
                "sample_prompts": ("STRING", {"multiline": True, 
                    "default": "a photo of sks person | a painting of sks person in anime style",
                    "tooltip": "Sample prompts for validation. Separate multiple prompts with |"}),
            },
            "optional": {
                "low_vram_config": ("FLUX2_LOW_VRAM_CONFIG", {
                    "tooltip": "Low VRAM configuration from Flux2LowVRAMConfig node"
                }),
                "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"], {
                    "default": "logit_normal",
                    "tooltip": "Timestep weighting scheme. logit_normal recommended for Flux"
                }),
                "timestep_sampling": (["sigmoid", "uniform", "shift"], {
                    "default": "sigmoid",
                    "tooltip": "Timestep sampling method. sigmoid recommended for Flux"
                }),
                "auto_resume": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically resume from latest checkpoint if found in output_dir"
                }),
                "check_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check VRAM availability before training and warn if insufficient"
                }),
                "additional_args": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Additional training arguments"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "STRING", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "output_path", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/Flux2"

    def init_training(
        self, 
        flux2_models, 
        dataset, 
        optimizer_settings, 
        output_name,
        output_dir,
        network_type,
        network_dim, 
        network_alpha,
        learning_rate,
        max_train_steps,
        cache_latents,
        cache_text_encoder_outputs,
        gradient_dtype,
        save_dtype,
        optimizer_fusing,
        sample_prompts,
        low_vram_config=None,
        weighting_scheme="logit_normal",
        timestep_sampling="sigmoid",
        auto_resume=True,
        check_vram=True,
        additional_args=None,
        seed=42,
        prompt=None, 
        extra_pnginfo=None,
    ):
        # LAZY IMPORT - ключевое изменение для стабильной загрузки!
        # Ошибка появится ТОЛЬКО здесь при Queue Prompt, а не при загрузке нод
        modules = _lazy_import_training()
        toml = modules["toml"]
        torch = modules["torch"]
        FluxNetworkTrainer = modules["FluxNetworkTrainer"]
        train_network_setup_parser = modules["train_network_setup_parser"]
        flux_train_utils = modules["flux_train_utils"]
        get_optimal_config_for_vram = modules["get_optimal_config_for_vram"]
        estimate_vram_usage = modules["estimate_vram_usage"]
        print_vram_estimate = modules["print_vram_estimate"]
        auto_resume_fn = modules["auto_resume_training"]
        get_training_progress = modules["get_training_progress"]
        find_latest_checkpoint = modules["find_latest_checkpoint"]
        
        mm.soft_empty_cache()
        
        # ===================================================================
        # VALIDATION - Проверяем параметры LoRA для предотвращения "garbage LoRA"
        # ===================================================================
        # Правило: network_alpha должен быть <= network_dim
        # Если alpha > dim, веса "взрываются" и LoRA получается битой
        if network_alpha > network_dim:
            logger.warning(
                f"[WARN] network_alpha ({network_alpha}) > network_dim ({network_dim})! "
                f"Это может привести к нестабильному обучению. "
                f"Автоматически устанавливаю network_alpha = {network_dim}"
            )
            network_alpha = float(network_dim)
        
        # Проверка network_type
        is_dora = network_type.lower() == "dora"
        
        # ===================================================================
        # SAFE MODE - Fallback на Adafactor если bitsandbytes недоступен
        # ===================================================================
        bnb_available = False
        try:
            import bitsandbytes
            bnb_available = True
        except ImportError:
            logger.warning("[WARN] bitsandbytes not available. 8-bit optimizers disabled.")
        
        # Получаем тип оптимизатора из настроек
        current_optimizer = optimizer_settings.get("optimizer_type", "adafactor")
        
        # Список оптимизаторов, требующих bitsandbytes
        bnb_optimizers = ["adamw8bit", "lion8bit", "ademamix8bit", "pagedademamix8bit"]
        
        if current_optimizer.lower() in [o.lower() for o in bnb_optimizers] and not bnb_available:
            logger.warning(
                f"[WARN] Optimizer '{current_optimizer}' requires bitsandbytes which is not available. "
                f"Automatically switching to Adafactor (works without bitsandbytes)."
            )
            optimizer_settings["optimizer_type"] = "adafactor"
            optimizer_settings["optimizer_args"] = [
                "scale_parameter=False",
                "relative_step=False",
                "warmup_init=False"
            ]
        
        # Создаём конфиг по умолчанию если не передан
        if low_vram_config is None:
            low_vram_config = get_optimal_config_for_vram(8.0, 32.0)
        
        # Проверяем директорию
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        total, used, free = shutil.disk_usage(output_dir)
        required_free_space = 2 * (2**30)  # 2 GB минимум
        if free <= required_free_space:
            raise ValueError(f"Insufficient disk space. Required: {required_free_space/2**30:.1f}GB. Available: {free/2**30:.1f}GB")
        
        # ===================================================================
        # VRAM SAFETY CHECK - Проверяем доступную память перед обучением
        # ===================================================================
        if check_vram:
            try:
                # Определяем размер модели из model_type
                model_type_str = flux2_models.get("model_type", "auto")
                if model_type_str in ("flux2_dev", "flux_12b"):
                    model_params_b = 12.0
                else:
                    model_params_b = 9.0  # Klein 9B или auto
                
                import torch
                gpu_vram_gb = 8.0
                if torch.cuda.is_available():
                    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                
                vram_estimate = estimate_vram_usage(
                    model_params_billions=model_params_b,
                    network_dim=network_dim,
                    batch_size=1,
                    use_fp8_base=low_vram_config.use_fp8_base,
                    gradient_checkpointing=low_vram_config.gradient_checkpointing,
                    cache_text_encoder=cache_text_encoder_outputs != "disabled",
                    blocks_to_swap=low_vram_config.blocks_to_swap,
                    available_vram_gb=gpu_vram_gb,
                )
                print_vram_estimate(vram_estimate)
                
                if vram_estimate.risk_level == "critical":
                    raise ValueError(
                        f"VRAM CRITICAL: Требуется ~{vram_estimate.total_estimated_gb:.1f}GB, "
                        f"доступно {vram_estimate.available_vram_gb:.1f}GB. "
                        "Уменьшите network_dim или включите FP8 base."
                    )
                elif vram_estimate.risk_level == "danger":
                    logger.warning(
                        f"[WARN] VRAM WARNING: Required ~{vram_estimate.total_estimated_gb:.1f}GB, "
                        f"доступно {vram_estimate.available_vram_gb:.1f}GB. Возможны проблемы с памятью."
                    )
            except Exception as e:
                logger.warning(f"Could not estimate VRAM: {e}")
        
        # ===================================================================
        # AUTO-RESUME - Автоматическое продолжение с последнего чекпоинта
        # ===================================================================
        resume_checkpoint = None
        if auto_resume:
            try:
                latest_ckpt = find_latest_checkpoint(output_dir)
                if latest_ckpt:
                    resume_checkpoint = latest_ckpt
                    progress = get_training_progress(output_dir)
                    logger.info("=" * 60)
                    logger.info("[AUTO-RESUME] Found checkpoint to continue!")
                    logger.info(f"   Файл: {os.path.basename(resume_checkpoint)}")
                    if progress:
                        logger.info(f"   Прогресс: шаг {progress.get('last_step', '?')}, "
                                  f"эпоха {progress.get('last_epoch', '?')}")
                    logger.info("=" * 60)
                else:
                    logger.info("[AUTO-RESUME] No previous checkpoint found. Starting fresh.")
            except Exception as e:
                logger.warning(f"Auto-resume check failed: {e}")
        
        # Парсим датасет
        dataset_config = dataset["datasets"]
        dataset_json = json.loads(dataset_config)
        
        # Получаем размеры из датасета  
        width = dataset.get("width", 1024)
        height = dataset.get("height", 1024)
        
        # Настраиваем кэширование
        cache_latents_to_disk = cache_latents == "disk"
        cache_latents_enabled = cache_latents != "disabled"
        cache_te_to_disk = cache_text_encoder_outputs == "disk"
        cache_te_enabled = cache_text_encoder_outputs != "disabled"
        
        # ========================================================
        # AUTO-FIX: При включённом кэшировании Text Encoder
        # автоматически отключаем несовместимые параметры датасета
        # (shuffle_caption, caption_dropout_rate, token_warmup_step,
        #  caption_tag_dropout_rate)
        # ========================================================
        if cache_te_enabled and "general" in dataset_json:
            general = dataset_json["general"]
            incompatible_keys = {
                "shuffle_caption": False,
                "caption_dropout_rate": 0.0,
                "token_warmup_step": 0,
                "caption_tag_dropout_rate": 0.0,
            }
            fixed_keys = []
            for key, safe_value in incompatible_keys.items():
                if key in general:
                    current = general[key]
                    # Проверяем, установлено ли несовместимое значение
                    if isinstance(current, bool) and current:
                        general[key] = safe_value
                        fixed_keys.append(f"{key}: {current} → {safe_value}")
                    elif isinstance(current, (int, float)) and current > 0:
                        general[key] = safe_value
                        fixed_keys.append(f"{key}: {current} → {safe_value}")
            if fixed_keys:
                logger.warning(
                    "[AUTO-FIX] Кэширование Text Encoder включено — "
                    "автоматически отключены несовместимые параметры датасета:\n  "
                    + "\n  ".join(fixed_keys)
                )
        
        dataset_toml = toml.dumps(dataset_json)
        
        # Создаём парсер и аргументы
        parser = train_network_setup_parser()
        flux_train_utils.add_flux_train_arguments(parser)
        
        if additional_args:
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()
        
        # Парсим sample prompts
        if '|' in sample_prompts:
            prompts_list = [p.strip() for p in sample_prompts.split('|')]
        else:
            prompts_list = [sample_prompts.strip()]
        
        # Формируем конфигурацию
        network_suffix = "dora" if is_dora else "lora"
        
        # Network args для LoRA/DoRA и Flux-specific настройки
        network_args_dict = {}
        if is_dora:
            # DoRA: Weight-Decomposed Low-Rank Adaptation
            # Добавляет decomposed weight magnitude для лучшего качества
            network_args_dict["dora_wd"] = True
        
        # Flux-specific: train_on_input улучшает качество на некоторых моделях
        # Опционально можно добавить через additional_args
        
        config_dict = {
            # Модели
            "pretrained_model_name_or_path": flux2_models["transformer"],
            "clip_l": flux2_models["clip_l"],
            "t5xxl": flux2_models["t5"],
            "ae": flux2_models["vae"],
            
            # LoRA/DoRA
            "network_module": ".networks.lora_flux",
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": network_args_dict if network_args_dict else None,
            
            # Training
            "learning_rate": learning_rate,
            "max_train_steps": max_train_steps,
            "seed": seed,
            
            # Output - включаем тип сети в имя
            "output_dir": output_dir,
            "output_name": f"{output_name}_{network_suffix}_rank{network_dim}_{save_dtype}",
            "save_model_as": "safetensors",
            "save_precision": save_dtype,
            
            # Dataset
            "dataset_config": dataset_toml,
            "width": int(width),
            "height": int(height),
            
            # Caching
            "cache_latents": cache_latents_enabled,
            "cache_latents_to_disk": cache_latents_to_disk,
            "cache_text_encoder_outputs": cache_te_enabled,
            "cache_text_encoder_outputs_to_disk": cache_te_to_disk,
            
            # Precision
            "mixed_precision": gradient_dtype,
            "full_bf16": gradient_dtype == "bf16",
            "full_fp16": gradient_dtype == "fp16",
            
            # Memory optimizations from low_vram_config
            "gradient_checkpointing": low_vram_config.gradient_checkpointing,
            "cpu_offload_checkpointing": low_vram_config.cpu_offload_checkpointing,
            "blocks_to_swap": low_vram_config.blocks_to_swap,
            "fp8_base": low_vram_config.use_fp8_base,
            "fp8_base_unet": low_vram_config.use_fp8_base,
            
            # Optimizer fusing
            "fused_backward_pass": optimizer_fusing == "fused_backward_pass",
            "blockwise_fused_optimizers": optimizer_fusing == "blockwise_fused_optimizers",
            
            # Misc
            "sample_prompts": prompts_list,
            "network_train_unet_only": True,
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 2,
            "num_cpu_threads_per_process": 1,
            "disable_mmap_load_safetensors": False,
            "mem_eff_attn": True,
            "xformers": False,
            "sdpa": True,
            
            # Flux-specific - используем параметры из INPUT
            "t5xxl_max_token_length": 512,
            "apply_t5_attn_mask": True,
            "weighting_scheme": weighting_scheme,
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "mode_scale": 1.29,
            "guidance_scale": 1.0,
            "discrete_flow_shift": 1.0,
            "loss_type": "l2",
            "timestep_sampling": timestep_sampling,
            "sigmoid_scale": 1.0,
            "model_prediction_type": "raw",
            "alpha_mask": dataset.get("alpha_mask", False),
        }
        
        # Добавляем lora_path если есть (для fine-tuning существующей LoRA)
        if flux2_models.get("lora_path"):
            config_dict["network_weights"] = flux2_models["lora_path"]
        
        # Добавляем resume checkpoint если найден
        if resume_checkpoint:
            config_dict["network_weights"] = resume_checkpoint
            logger.info(f"[RESUME] Resuming from: {os.path.basename(resume_checkpoint)}")
        
        # Обновляем из optimizer_settings
        config_dict.update(optimizer_settings)
        
        # Применяем к args
        for key, value in config_dict.items():
            setattr(args, key, value)
        
        # Сохраняем конфигурацию
        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
        
        # Сохраняем workflow
        if extra_pnginfo is not None:
            saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_workflow.json")
            with open(saved_workflow_file_path, 'w', encoding='utf-8') as f:
                json.dump(extra_pnginfo.get("workflow", {}), f, indent=4, ensure_ascii=False)
        
        # Инициализируем тренер
        logger.info("=" * 60)
        logger.info(f"Initializing Flux.2 {'DoRA' if is_dora else 'LoRA'} Training")
        logger.info(f"  Model type: {flux2_models.get('model_type', 'unknown')}")
        logger.info(f"  Network type: {network_type.upper()}")
        logger.info(f"  Output: {output_dir}/{output_name}")
        logger.info(f"  Network dim: {network_dim}, alpha: {network_alpha}")
        logger.info(f"  Blocks to swap: {low_vram_config.blocks_to_swap}")
        logger.info(f"  FP8 base: {low_vram_config.use_fp8_base}")
        if resume_checkpoint:
            logger.info(f"  Resuming from: {os.path.basename(resume_checkpoint)}")
        logger.info("=" * 60)
        
        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)
        
        final_output_path = os.path.join(output_dir, f"{output_name}_rank{network_dim}_{save_dtype}")
        epochs_count = network_trainer.num_train_epochs
        
        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        
        return (trainer, epochs_count, final_output_path, args)


# =============================================================================
# NODE: Flux2TrainLoop - Цикл обучения
# =============================================================================
class Flux2TrainLoop:
    """
    Цикл обучения Flux.2 LoRA.
    Выполняет указанное количество шагов обучения.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Number of training steps to run"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "current_step",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer/Flux2"

    def train(self, network_trainer, steps):
        # LAZY IMPORT
        modules = _lazy_import_training()
        torch = modules["torch"]
        
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            trainer = network_trainer["network_trainer"]
            
            target_global_step = trainer.global_step + steps
            comfy_pbar = comfy.utils.ProgressBar(steps)
            trainer.comfy_pbar = comfy_pbar
            
            trainer.optimizer_train_fn()
            
            while trainer.global_step < target_global_step:
                steps_done = training_loop(
                    break_at_steps=target_global_step,
                    epoch=trainer.current_epoch.value,
                )
                
                # Прерываем если достигли максимума
                if trainer.global_step >= trainer.args.max_train_steps:
                    break
            
            result = {
                "network_trainer": trainer,
                "training_loop": training_loop,
            }
        
        return (result, trainer.global_step)


# =============================================================================
# NODE: Flux2TrainAndValidateLoop - Обучение с валидацией
# =============================================================================
class Flux2TrainAndValidateLoop:
    """
    Цикл обучения с периодической валидацией и сохранением.
    Рекомендуется для длительного обучения.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "validate_at_steps": ("INT", {"default": 250, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Generate validation samples every N steps"}),
                "save_at_steps": ("INT", {"default": 500, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Save checkpoint every N steps"}),
            },
            "optional": {
                "validation_settings": ("VALSETTINGS",),
            }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "final_step",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer/Flux2"

    def train(self, network_trainer, validate_at_steps, save_at_steps, validation_settings=None):
        # LAZY IMPORT
        modules = _lazy_import_training()
        torch = modules["torch"]
        
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            trainer = network_trainer["network_trainer"]
            
            target_global_step = trainer.args.max_train_steps
            comfy_pbar = comfy.utils.ProgressBar(target_global_step)
            trainer.comfy_pbar = comfy_pbar
            
            trainer.optimizer_train_fn()
            
            while trainer.global_step < target_global_step:
                next_validate_step = ((trainer.global_step // validate_at_steps) + 1) * validate_at_steps
                next_save_step = ((trainer.global_step // save_at_steps) + 1) * save_at_steps
                
                steps_done = training_loop(
                    break_at_steps=min(next_validate_step, next_save_step),
                    epoch=trainer.current_epoch.value,
                )
                
                # Валидация
                if trainer.global_step % validate_at_steps == 0:
                    self._validate(trainer, validation_settings)
                
                # Сохранение
                if trainer.global_step % save_at_steps == 0:
                    self._save(trainer)
                
                # Прерываем если достигли максимума
                if trainer.global_step >= trainer.args.max_train_steps:
                    break
            
            result = {
                "network_trainer": trainer,
                "training_loop": training_loop,
            }
        
        return (result, trainer.global_step)
    
    def _validate(self, trainer, validation_settings=None):
        params = (
            trainer.current_epoch.value,
            trainer.global_step,
            validation_settings
        )
        trainer.optimizer_eval_fn()
        image_tensors = trainer.sample_images(*params)
        trainer.optimizer_train_fn()
        logger.info(f"Validation at step: {trainer.global_step}")
    
    def _save(self, trainer):
        # LAZY IMPORT
        modules = _lazy_import_training()
        train_util = modules["train_util"]
        
        ckpt_name = train_util.get_step_ckpt_name(
            trainer.args, 
            "." + trainer.args.save_model_as, 
            trainer.global_step
        )
        trainer.optimizer_eval_fn()
        trainer.save_model(
            ckpt_name, 
            trainer.accelerator.unwrap_model(trainer.network), 
            trainer.global_step, 
            trainer.current_epoch.value + 1
        )
        trainer.optimizer_train_fn()
        logger.info(f"Saved checkpoint at step: {trainer.global_step}")


# =============================================================================
# NODE: Flux2TrainSave - Сохранение LoRA
# =============================================================================
class Flux2TrainSave:
    """
    Сохранение обученной LoRA модели.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "save_state": ("BOOLEAN", {"default": False,
                    "tooltip": "Also save the full training state (for resume)"}),
                "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": True,
                    "tooltip": "Copy LoRA to ComfyUI loras folder"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer", "lora_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer/Flux2"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        # LAZY IMPORT
        modules = _lazy_import_training()
        torch = modules["torch"]
        train_util = modules["train_util"]
        
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step
            
            ckpt_name = train_util.get_step_ckpt_name(
                trainer.args, 
                "." + trainer.args.save_model_as, 
                global_step
            )
            trainer.save_model(
                ckpt_name, 
                trainer.accelerator.unwrap_model(trainer.network), 
                global_step, 
                trainer.current_epoch.value + 1
            )
            
            # Удаляем старые чекпоинты
            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(
                    trainer.args, 
                    "." + trainer.args.save_model_as, 
                    remove_step_no
                )
                trainer.remove_model(remove_ckpt_name)
            
            # Сохраняем состояние если нужно
            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)
            
            lora_path = os.path.join(trainer.args.output_dir, ckpt_name)
            
            # Копируем в папку loras
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "flux2_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lora_path, os.path.join(destination_dir, ckpt_name))
                logger.info(f"Copied LoRA to: {destination_dir}/{ckpt_name}")
        
        return (network_trainer, lora_path, global_step)


# =============================================================================
# NODE: Flux2TrainEnd - Завершение обучения
# =============================================================================
class Flux2TrainEnd:
    """
    Завершение обучения и очистка ресурсов.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_lora_path",)
    FUNCTION = "end_training"
    CATEGORY = "FluxTrainer/Flux2"

    def end_training(self, network_trainer):
        # LAZY IMPORT
        modules = _lazy_import_training()
        train_util = modules["train_util"]
        clean_memory_on_device = modules["clean_memory_on_device"]
        
        trainer = network_trainer["network_trainer"]
        
        # Финальное сохранение
        final_ckpt_name = train_util.get_last_ckpt_name(
            trainer.args, 
            "." + trainer.args.save_model_as
        )
        trainer.save_model(
            final_ckpt_name,
            trainer.accelerator.unwrap_model(trainer.network),
            trainer.global_step,
            trainer.current_epoch.value + 1
        )
        
        final_path = os.path.join(trainer.args.output_dir, final_ckpt_name)
        
        # Очистка
        trainer.accelerator.end_training()
        clean_memory_on_device(trainer.accelerator.device)
        mm.soft_empty_cache()
        
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"  Final LoRA saved to: {final_path}")
        logger.info(f"  Total steps: {trainer.global_step}")
        logger.info("=" * 60)
        
        # === FluxTrainer Pro Dashboard: mark training finished ===
        try:
            from .training_state import TrainingState
            TrainingState.instance().finish_training(
                success=True,
                message=f"LoRA saved: {final_path}, steps: {trainer.global_step}"
            )
        except Exception:
            pass

        return (final_path,)


# =============================================================================
# NODE: Flux2TrainAdvancedSettings - Расширенные настройки
# =============================================================================
class Flux2TrainAdvancedSettings:
    """
    Расширенные настройки обучения Flux.2.
    Для опытных пользователей.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Precision settings
                "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
                "full_bf16": ("BOOLEAN", {"default": False}),
                "fp8_base": ("BOOLEAN", {"default": True}),
                
                # Training dynamics
                "timestep_sampling": (["sigmoid", "uniform", "logit_normal", "sigma", "shift", "flux_shift"], 
                    {"default": "sigmoid"}),
                "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"default": "raw"}),
                "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                
                # Weighting
                "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"], 
                    {"default": "logit_normal"}),
                "logit_mean": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "logit_std": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
                
                # Regularization
                "network_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "scale_weight_norms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                
                # Memory
                "max_data_loader_n_workers": ("INT", {"default": 2, "min": 0, "max": 16}),
                "persistent_data_loader_workers": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "t5xxl_max_token_length": ("INT", {"default": 512, "min": 77, "max": 1024}),
                "apply_t5_attn_mask": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLUX2_ADVANCED_SETTINGS",)
    RETURN_NAMES = ("advanced_settings",)
    FUNCTION = "create_settings"
    CATEGORY = "FluxTrainer/Flux2"

    def create_settings(self, **kwargs):
        return (kwargs,)


# =============================================================================
# NODE: Flux2MemoryEstimator - Оценка использования памяти
# =============================================================================
class Flux2MemoryEstimator:
    """
    Оценивает использование памяти для выбранной конфигурации.
    Помогает определить оптимальные настройки до начала обучения.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["flux2_klein_9b", "flux2_dev"], {"default": "flux2_klein_9b"}),
                "network_dim": ("INT", {"default": 16, "min": 1, "max": 256}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            },
            "optional": {
                "low_vram_config": ("FLUX2_LOW_VRAM_CONFIG",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("memory_report",)
    FUNCTION = "estimate"
    CATEGORY = "FluxTrainer/Flux2"
    OUTPUT_NODE = True

    def estimate(self, model_type, network_dim, batch_size, resolution, low_vram_config=None):
        # LAZY IMPORT - загрузка только при Queue Prompt
        modules = _lazy_import_training()
        
        # Параметры моделей
        model_params = {
            "flux2_klein_9b": 9.0,
            "flux2_dev": 32.0
        }
        
        params_b = model_params.get(model_type, 9.0)
        
        # Если low_vram_config не предоставлен, создаём стандартный
        if low_vram_config is None:
            from dataclasses import dataclass
            from enum import Enum
            
            class VRAMStrategy(Enum):
                AGGRESSIVE = "aggressive_offload"
            
            @dataclass
            class DefaultConfig:
                strategy: VRAMStrategy = VRAMStrategy.AGGRESSIVE
                blocks_to_swap: int = 18
                
                def estimate_memory_usage(self, params_b: float):
                    return {
                        'base_model_vram': params_b * 0.3,
                        'lora_weights_vram': 0.1,
                        'activations_peak': 2.0,
                        'optimizer_vram': 0.5,
                        'optimizer_ram': 4.0
                    }
            
            low_vram_config = DefaultConfig()
        
        mem = low_vram_config.estimate_memory_usage(params_b)
        total_vram = sum(v for k, v in mem.items() if 'vram' in k)
        
        # Статус
        if total_vram <= 8:
            status = "[OK] Should fit in 8GB VRAM"
        elif total_vram <= 12:
            status = "[WARN] May need 12GB VRAM"
        else:
            status = "[ERROR] Requires more than 12GB VRAM"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           FLUX.2 MEMORY ESTIMATION REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
║ Model: {model_type:<54} ║
║ Parameters: {params_b:.1f}B                                              ║
║ Resolution: {resolution}x{resolution:<47} ║
║ Batch Size: {batch_size:<52} ║
║ LoRA Dim: {network_dim:<54} ║
╠══════════════════════════════════════════════════════════════╣
║ VRAM USAGE ESTIMATE:                                         ║
║   Base Model (FP8):     {mem['base_model_vram']:.1f} GB                              ║
║   LoRA Weights:         {mem['lora_weights_vram']:.2f} GB                              ║
║   Activations Peak:     {mem['activations_peak']:.1f} GB                              ║
║   Optimizer (GPU):      {mem['optimizer_vram']:.1f} GB                              ║
║   ───────────────────────────────────────────────────────    ║
║   TOTAL VRAM:           ~{total_vram:.1f} GB                             ║
╠══════════════════════════════════════════════════════════════╣
║ RAM USAGE ESTIMATE:                                          ║
║   Optimizer (CPU):      {mem['optimizer_ram']:.1f} GB                              ║
║   Cached TE Outputs:    ~2.0 GB                              ║
╠══════════════════════════════════════════════════════════════╣
║ STRATEGY: {low_vram_config.strategy.value:<52} ║
║ Blocks to Swap: {low_vram_config.blocks_to_swap:<47} ║
╠══════════════════════════════════════════════════════════════╣
║ {status:<60} ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        return (report,)


# =============================================================================
# NODE MAPPINGS
# =============================================================================
# КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Ноды ВСЕГДА регистрируются!
# Ошибки зависимостей появятся только при Queue Prompt, а не при загрузке.
# Это позволяет ComfyUI корректно отображать все ноды в UI.

NODE_CLASS_MAPPINGS = {
    "Flux2TrainModelSelect": Flux2TrainModelSelect,
    "Flux2TrainModelPaths": Flux2TrainModelPaths,
    "Flux2LowVRAMConfig": Flux2LowVRAMConfig,
    "Flux2InitTraining": Flux2InitTraining,
    "Flux2TrainLoop": Flux2TrainLoop,
    "Flux2TrainAndValidateLoop": Flux2TrainAndValidateLoop,
    "Flux2TrainSave": Flux2TrainSave,
    "Flux2TrainEnd": Flux2TrainEnd,
    "Flux2TrainAdvancedSettings": Flux2TrainAdvancedSettings,
    "Flux2MemoryEstimator": Flux2MemoryEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2TrainModelSelect": "🔷 Flux.2 Model Select",
    "Flux2TrainModelPaths": "📁 Flux.2 Model Paths",
    "Flux2LowVRAMConfig": "💾 Flux.2 Low VRAM Config",
    "Flux2InitTraining": "🚀 Flux.2 Init Training",
    "Flux2TrainLoop": "🔄 Flux.2 Train Loop",
    "Flux2TrainAndValidateLoop": "🔄✅ Flux.2 Train & Validate",
    "Flux2TrainSave": "💾 Flux.2 Save LoRA",
    "Flux2TrainEnd": "🏁 Flux.2 End Training",
    "Flux2TrainAdvancedSettings": "⚙️ Flux.2 Advanced Settings",
    "Flux2MemoryEstimator": "📊 Flux.2 Memory Estimator",
}

# Log registration
logger.info(f"[ComfyUI-FluxTrainer-Pro] Registered {len(NODE_CLASS_MAPPINGS)} Flux.2 nodes (lazy imports enabled)")
