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

Author: ComfyUI-FluxTrainer Team
License: Apache-2.0
"""

import os
import sys
import json
import toml
import shlex
import shutil
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import folder_paths
import comfy.model_management as mm
import comfy.utils

# --- Safe Imports ---
IMPORTS_OK = True
IMPORT_ERROR_MSG = ""

try:
    from .flux_train_network_comfy import FluxNetworkTrainer
    from .train_network import setup_parser as train_network_setup_parser
    from .library import flux_train_utils, flux_utils, train_util
    from .library.low_vram_utils import (
        LowVRAMConfig, 
        OffloadStrategy, 
        get_optimal_config_for_vram,
        aggressive_memory_cleanup
    )
    from .library.device_utils import init_ipex, clean_memory_on_device
    init_ipex()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR_MSG = str(e)
    print(f"\n[ComfyUI-FluxTrainer-Pro] ❌ Critical Import Error: {e}")
    print("[ComfyUI-FluxTrainer-Pro] Check requirements.txt and installed packages.\n")
# --------------------

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))


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
                    "forceInput": True, 
                    "default": "", 
                    "tooltip": "Pre-trained LoRA path to continue training from"
                }),
            }
        }

    RETURN_TYPES = ("TRAIN_FLUX2_MODELS",)
    RETURN_NAMES = ("flux2_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer/Flux2"

    def loadmodel(self, transformer, vae, clip_l, t5, lora_path=""):
        transformer_path = folder_paths.get_full_path("unet", transformer)
        vae_path = folder_paths.get_full_path("vae", vae)
        clip_path = folder_paths.get_full_path("clip", clip_l)
        t5_path = folder_paths.get_full_path("clip", t5)

        # Определяем тип модели
        model_type = "auto"
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
        additional_args=None,
        seed=42,
        prompt=None, 
        extra_pnginfo=None,
    ):
        mm.soft_empty_cache()
        
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
        
        # Парсим датасет
        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))
        
        # Получаем размеры из датасета  
        width = dataset.get("width", 1024)
        height = dataset.get("height", 1024)
        
        # Создаём парсер и аргументы
        parser = train_network_setup_parser()
        flux_train_utils.add_flux_train_arguments(parser)
        
        if additional_args:
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()
        
        # Настраиваем кэширование
        cache_latents_to_disk = cache_latents == "disk"
        cache_latents_enabled = cache_latents != "disabled"
        cache_te_to_disk = cache_text_encoder_outputs == "disk"
        cache_te_enabled = cache_text_encoder_outputs != "disabled"
        
        # Парсим sample prompts
        if '|' in sample_prompts:
            prompts_list = [p.strip() for p in sample_prompts.split('|')]
        else:
            prompts_list = [sample_prompts.strip()]
        
        # Формируем конфигурацию
        config_dict = {
            # Модели
            "pretrained_model_name_or_path": flux2_models["transformer"],
            "clip_l": flux2_models["clip_l"],
            "t5xxl": flux2_models["t5"],
            "ae": flux2_models["vae"],
            
            # LoRA
            "network_module": ".networks.lora_flux",
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            
            # Training
            "learning_rate": learning_rate,
            "max_train_steps": max_train_steps,
            "seed": seed,
            
            # Output
            "output_dir": output_dir,
            "output_name": f"{output_name}_rank{network_dim}_{save_dtype}",
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
            
            # Flux-specific
            "t5xxl_max_token_length": 512,
            "apply_t5_attn_mask": True,
            "weighting_scheme": "logit_normal",
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "mode_scale": 1.29,
            "guidance_scale": 1.0,
            "discrete_flow_shift": 1.0,
            "loss_type": "l2",
            "timestep_sampling": "sigmoid",
            "sigmoid_scale": 1.0,
            "model_prediction_type": "raw",
            "alpha_mask": dataset.get("alpha_mask", False),
        }
        
        # Добавляем lora_path если есть
        if flux2_models.get("lora_path"):
            config_dict["network_weights"] = flux2_models["lora_path"]
        
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
        logger.info("Initializing Flux.2 LoRA Training")
        logger.info(f"  Model type: {flux2_models.get('model_type', 'unknown')}")
        logger.info(f"  Output: {output_dir}/{output_name}")
        logger.info(f"  Network dim: {network_dim}, alpha: {network_alpha}")
        logger.info(f"  Blocks to swap: {low_vram_config.blocks_to_swap}")
        logger.info(f"  FP8 base: {low_vram_config.use_fp8_base}")
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
        # Параметры моделей
        model_params = {
            "flux2_klein_9b": 9.0,
            "flux2_dev": 32.0
        }
        
        params_b = model_params.get(model_type, 9.0)
        
        if low_vram_config is None:
            low_vram_config = LowVRAMConfig()
        
        mem = low_vram_config.estimate_memory_usage(params_b)
        total_vram = sum(v for k, v in mem.items() if 'vram' in k)
        
        # Статус
        if total_vram <= 8:
            status = "✅ Should fit in 8GB VRAM"
        elif total_vram <= 12:
            status = "⚠️ May need 12GB VRAM"
        else:
            status = "❌ Requires more than 12GB VRAM"
        
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
if IMPORTS_OK:
    NODE_CLASS_MAPPINGS = {
        "Flux2TrainModelSelect": Flux2TrainModelSelect,
        "Flux2LowVRAMConfig": Flux2LowVRAMConfig,
        "Flux2InitTraining": Flux2InitTraining,
        "Flux2TrainLoop": Flux2TrainLoop,
        "Flux2TrainAndValidateLoop": Flux2TrainAndValidateLoop,
        "Flux2TrainSave": Flux2TrainSave,
        "Flux2TrainEnd": Flux2TrainEnd,
        "Flux2TrainAdvancedSettings": Flux2TrainAdvancedSettings,
        "Flux2MemoryEstimator": Flux2MemoryEstimator,
    }
else:
    class DependencyErrorNode:
        @classmethod
        def INPUT_TYPES(s): return {"required": {}}
        RETURN_TYPES = ()
        FUNCTION = "error"
        CATEGORY = "FluxTrainer/Flux2"
        def error(self): raise ImportError(f"Missing dependencies: {IMPORT_ERROR_MSG}")
    
    NODE_CLASS_MAPPINGS = {k: DependencyErrorNode for k in [
        "Flux2TrainModelSelect", "Flux2LowVRAMConfig", "Flux2InitTraining", 
        "Flux2TrainLoop", "Flux2TrainAndValidateLoop", "Flux2TrainSave", 
        "Flux2TrainEnd", "Flux2TrainAdvancedSettings", "Flux2MemoryEstimator"
    ]}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2TrainModelSelect": "Flux.2 Model Select" if IMPORTS_OK else "⚠️ Flux.2 Model Select (Error)",
    "Flux2LowVRAMConfig": "Flux.2 Low VRAM Config" if IMPORTS_OK else "⚠️ Flux.2 Low VRAM Config (Error)",
    "Flux2InitTraining": "Flux.2 Init Training" if IMPORTS_OK else "⚠️ Flux.2 Init Training (Error)",
    "Flux2TrainLoop": "Flux.2 Train Loop" if IMPORTS_OK else "⚠️ Flux.2 Train Loop (Error)",
    "Flux2TrainAndValidateLoop": "Flux.2 Train & Validate" if IMPORTS_OK else "⚠️ Flux.2 Train & Validate (Error)",
    "Flux2TrainSave": "Flux.2 Save LoRA" if IMPORTS_OK else "⚠️ Flux.2 Save LoRA (Error)",
    "Flux2TrainEnd": "Flux.2 End Training" if IMPORTS_OK else "⚠️ Flux.2 End Training (Error)",
    "Flux2TrainAdvancedSettings": "Flux.2 Advanced Settings" if IMPORTS_OK else "⚠️ Flux.2 Advanced Settings (Error)",
    "Flux2MemoryEstimator": "Flux.2 Memory Estimator" if IMPORTS_OK else "⚠️ Flux.2 Memory Estimator (Error)",
}
