# -*- coding: utf-8 -*-
"""
ComfyUI Extended Utility Nodes for FluxTrainer-Pro
===================================================

Расширенные утилиты для профессионального обучения:
- Предпросмотр датасета
- Визуализация прогресса обучения  
- Сравнение моделей
- Слияние LoRA
- Управление чекпоинтами
- Мониторинг памяти

Author: ComfyUI-FluxTrainer-Pro Team
License: Apache-2.0
"""

import os
import sys
import json
import time
import math
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from collections import deque

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import folder_paths
import comfy.model_management as mm
import comfy.utils

# --- Safe Imports ---
IMPORTS_OK = True
IMPORT_ERROR_MSG = ""
try:
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR_MSG = str(e)
    print(f"\n[ComfyUI-FluxTrainer-Pro] ❌ Critical Import Error (Extended Nodes): {e}")
# --------------------

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# DATASET UTILITIES
# =============================================================================

class DatasetPreviewGrid:
    """
    Создает превью-сетку из датасета для визуальной проверки перед обучением.
    Показывает случайные изображения с их подписями.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Path to the training dataset folder"
                }),
                "grid_cols": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of columns in the preview grid"
                }),
                "grid_rows": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of rows in the preview grid"
                }),
                "image_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of each preview image"
                }),
                "show_captions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show caption text on images"
                }),
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Extension of caption files"
                }),
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random image selection (0 = random)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT",)
    RETURN_NAMES = ("preview_grid", "dataset_info", "total_images",)
    FUNCTION = "create_preview"
    CATEGORY = "FluxTrainer/Utilities"

    def create_preview(self, dataset_path, grid_cols, grid_rows, image_size, 
                       show_captions, caption_extension, random_seed):
        from PIL import ImageDraw, ImageFont
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Найти все изображения
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(dataset_path, f'*{ext}')))
            all_images.extend(glob.glob(os.path.join(dataset_path, f'*{ext.upper()}')))
        
        if not all_images:
            raise ValueError(f"No images found in: {dataset_path}")
        
        total_images = len(all_images)
        num_preview = min(grid_cols * grid_rows, total_images)
        
        # Случайный выбор
        if random_seed == 0:
            random_seed = int(time.time()) % 10000
        np.random.seed(random_seed)
        selected_indices = np.random.choice(len(all_images), size=num_preview, replace=False)
        selected_images = [all_images[i] for i in selected_indices]
        
        # Создать сетку
        grid_width = grid_cols * image_size
        grid_height = grid_rows * image_size
        grid = Image.new('RGB', (grid_width, grid_height), color=(40, 40, 40))
        
        # Попытаться загрузить шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Статистика
        captions_found = 0
        resolutions = []
        
        for idx, img_path in enumerate(selected_images):
            row = idx // grid_cols
            col = idx % grid_cols
            
            try:
                img = Image.open(img_path).convert('RGB')
                resolutions.append(img.size)
                
                # Масштабирование с сохранением пропорций
                img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Центрирование на ячейке
                x_offset = col * image_size + (image_size - img.width) // 2
                y_offset = row * image_size + (image_size - img.height) // 2
                
                grid.paste(img, (x_offset, y_offset))
                
                # Подпись
                if show_captions:
                    caption_path = os.path.splitext(img_path)[0] + caption_extension
                    if os.path.exists(caption_path):
                        captions_found += 1
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()[:50] + '...' if len(f.read()) > 50 else f.read().strip()
                        
                        draw = ImageDraw.Draw(grid)
                        text_x = col * image_size + 5
                        text_y = row * image_size + image_size - 20
                        
                        # Тень текста
                        draw.text((text_x+1, text_y+1), caption[:30], fill=(0, 0, 0), font=font)
                        draw.text((text_x, text_y), caption[:30], fill=(255, 255, 255), font=font)
                        
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")
        
        # Статистика датасета
        if resolutions:
            avg_width = sum(r[0] for r in resolutions) // len(resolutions)
            avg_height = sum(r[1] for r in resolutions) // len(resolutions)
        else:
            avg_width = avg_height = 0
            
        dataset_info = f"""Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Path: {dataset_path}
📸 Total Images: {total_images}
📝 Captions Found: {captions_found}/{num_preview} previewed
📐 Average Resolution: {avg_width}x{avg_height}
🎲 Random Seed: {random_seed}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        # Конвертация в тензор
        grid_tensor = transforms.ToTensor()(grid)
        grid_tensor = grid_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        
        return (grid_tensor, dataset_info, total_images)


class DatasetValidator:
    """
    Проверяет датасет на проблемы перед обучением.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Path to the training dataset folder"
                }),
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Extension of caption files"
                }),
                "min_resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Minimum acceptable resolution"
                }),
                "check_duplicates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check for duplicate images"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT",)
    RETURN_NAMES = ("validation_report", "is_valid", "issue_count",)
    FUNCTION = "validate"
    CATEGORY = "FluxTrainer/Utilities"

    def validate(self, dataset_path, caption_extension, min_resolution, check_duplicates):
        import hashlib
        
        issues = []
        warnings = []
        
        if not os.path.exists(dataset_path):
            return (f"❌ Dataset path does not exist: {dataset_path}", False, 1)
        
        # Найти изображения
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(dataset_path, f'*{ext}')))
            all_images.extend(glob.glob(os.path.join(dataset_path, f'*{ext.upper()}')))
        
        if not all_images:
            return (f"❌ No images found in: {dataset_path}", False, 1)
        
        # Проверки
        missing_captions = []
        low_resolution = []
        corrupt_images = []
        file_hashes = {}
        duplicates = []
        
        for img_path in all_images:
            # Проверка подписи
            caption_path = os.path.splitext(img_path)[0] + caption_extension
            if not os.path.exists(caption_path):
                missing_captions.append(os.path.basename(img_path))
            
            try:
                img = Image.open(img_path)
                w, h = img.size
                
                # Проверка разрешения
                if min(w, h) < min_resolution:
                    low_resolution.append(f"{os.path.basename(img_path)} ({w}x{h})")
                
                # Проверка дубликатов
                if check_duplicates:
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in file_hashes:
                        duplicates.append((os.path.basename(img_path), 
                                         os.path.basename(file_hashes[file_hash])))
                    else:
                        file_hashes[file_hash] = img_path
                        
            except Exception as e:
                corrupt_images.append(os.path.basename(img_path))
        
        # Формирование отчета
        report_lines = [
            "📊 Dataset Validation Report",
            "═" * 50,
            f"📁 Path: {dataset_path}",
            f"📸 Total Images: {len(all_images)}",
            "",
        ]
        
        issue_count = 0
        
        if corrupt_images:
            issue_count += len(corrupt_images)
            report_lines.append(f"❌ Corrupt Images ({len(corrupt_images)}):")
            for img in corrupt_images[:5]:
                report_lines.append(f"   • {img}")
            if len(corrupt_images) > 5:
                report_lines.append(f"   ... and {len(corrupt_images) - 5} more")
            report_lines.append("")
        
        if missing_captions:
            issue_count += len(missing_captions)
            report_lines.append(f"⚠️ Missing Captions ({len(missing_captions)}):")
            for img in missing_captions[:5]:
                report_lines.append(f"   • {img}")
            if len(missing_captions) > 5:
                report_lines.append(f"   ... and {len(missing_captions) - 5} more")
            report_lines.append("")
        
        if low_resolution:
            issue_count += len(low_resolution)
            report_lines.append(f"⚠️ Low Resolution ({len(low_resolution)}):")
            for img in low_resolution[:5]:
                report_lines.append(f"   • {img}")
            if len(low_resolution) > 5:
                report_lines.append(f"   ... and {len(low_resolution) - 5} more")
            report_lines.append("")
        
        if duplicates:
            issue_count += len(duplicates)
            report_lines.append(f"⚠️ Duplicate Images ({len(duplicates)}):")
            for dup in duplicates[:5]:
                report_lines.append(f"   • {dup[0]} == {dup[1]}")
            if len(duplicates) > 5:
                report_lines.append(f"   ... and {len(duplicates) - 5} more")
            report_lines.append("")
        
        if issue_count == 0:
            report_lines.append("✅ Dataset is valid! No issues found.")
        else:
            report_lines.append(f"⚠️ Found {issue_count} issues. Review before training.")
        
        report = "\n".join(report_lines)
        is_valid = len(corrupt_images) == 0
        
        return (report, is_valid, issue_count)


# =============================================================================
# TRAINING PROGRESS & MONITORING
# =============================================================================

class TrainingProgressDisplay:
    """
    Отображает прогресс обучения с расширенной статистикой.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "show_eta": ("BOOLEAN", {"default": True, "tooltip": "Show estimated time remaining"}),
                "show_loss_stats": ("BOOLEAN", {"default": True, "tooltip": "Show loss statistics"}),
                "show_lr": ("BOOLEAN", {"default": True, "tooltip": "Show learning rate"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "INT",)
    RETURN_NAMES = ("progress_report", "current_loss", "avg_loss", "current_step",)
    FUNCTION = "display"
    CATEGORY = "FluxTrainer/Utilities"

    def display(self, network_trainer, show_eta, show_loss_stats, show_lr):
        trainer = network_trainer["network_trainer"]
        
        # Получение данных
        current_step = trainer.global_step if hasattr(trainer, 'global_step') else 0
        total_steps = trainer.args.max_train_steps if hasattr(trainer, 'args') else 0
        
        loss_list = trainer.loss_recorder.global_loss_list if hasattr(trainer, 'loss_recorder') else []
        current_loss = loss_list[-1] if loss_list else 0.0
        avg_loss = sum(loss_list[-100:]) / len(loss_list[-100:]) if loss_list else 0.0
        
        # Расчет прогресса
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        
        # ETA расчет
        if show_eta and hasattr(trainer, 'training_start_time'):
            elapsed = time.time() - trainer.training_start_time
            if current_step > 0:
                time_per_step = elapsed / current_step
                remaining_steps = total_steps - current_step
                eta_seconds = time_per_step * remaining_steps
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "Calculating..."
        else:
            eta = "N/A"
        
        # Learning rate
        if show_lr and hasattr(trainer, 'optimizer'):
            current_lr = trainer.optimizer.param_groups[0]['lr']
        else:
            current_lr = 0.0
        
        # Формирование отчета
        report_lines = [
            "🎯 Training Progress",
            "═" * 40,
            f"",
            f"📊 Step: {current_step:,} / {total_steps:,}",
            f"📈 Progress: {progress_pct:.1f}%",
            f"",
        ]
        
        # Прогресс-бар
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        report_lines.append(f"[{bar}]")
        report_lines.append("")
        
        if show_loss_stats:
            report_lines.extend([
                f"📉 Current Loss: {current_loss:.6f}",
                f"📉 Average Loss (100): {avg_loss:.6f}",
                f"",
            ])
        
        if show_lr:
            report_lines.append(f"🎚️ Learning Rate: {current_lr:.2e}")
        
        if show_eta:
            report_lines.extend([
                f"",
                f"⏱️ ETA: {eta}",
            ])
        
        report_lines.append("═" * 40)
        
        report = "\n".join(report_lines)
        
        return (report, current_loss, avg_loss, current_step)


class MemoryMonitorDisplay:
    """
    Мониторинг использования памяти в реальном времени.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "update_trigger": ("*", {"tooltip": "Any input to trigger update"}),
            },
            "optional": {
                "network_trainer": ("NETWORKTRAINER",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("memory_chart", "memory_info",)
    FUNCTION = "monitor"
    CATEGORY = "FluxTrainer/Utilities"

    def monitor(self, update_trigger, network_trainer=None):
        try:
            import psutil
        except ImportError:
            psutil = None
        
        # GPU память
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_free = gpu_total - gpu_allocated
        else:
            gpu_allocated = gpu_reserved = gpu_total = gpu_free = 0
        
        # RAM
        if psutil:
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024**3
            ram_total = ram.total / 1024**3
            ram_available = ram.available / 1024**3
        else:
            ram_used = ram_total = ram_available = 0
        
        # Создание графика
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # GPU график
        ax1 = axes[0]
        gpu_data = [gpu_allocated, gpu_reserved - gpu_allocated, gpu_free]
        gpu_labels = ['Allocated', 'Reserved', 'Free']
        gpu_colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']
        ax1.pie(gpu_data, labels=gpu_labels, colors=gpu_colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'GPU Memory ({gpu_total:.1f} GB)')
        
        # RAM график
        ax2 = axes[1]
        ram_data = [ram_used, ram_available]
        ram_labels = ['Used', 'Available']
        ram_colors = ['#FF6B6B', '#4ECDC4']
        ax2.pie(ram_data, labels=ram_labels, colors=ram_colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'System RAM ({ram_total:.1f} GB)')
        
        plt.tight_layout()
        
        # Конвертация в изображение
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        image = Image.open(buf).convert('RGB')
        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        
        # Текстовый отчет
        memory_info = f"""💾 Memory Status
═══════════════════════════════
🎮 GPU (CUDA):
   • Allocated: {gpu_allocated:.2f} GB
   • Reserved:  {gpu_reserved:.2f} GB  
   • Free:      {gpu_free:.2f} GB
   • Total:     {gpu_total:.2f} GB
   • Usage:     {(gpu_allocated/gpu_total*100):.1f}%

💻 System RAM:
   • Used:      {ram_used:.2f} GB
   • Available: {ram_available:.2f} GB
   • Total:     {ram_total:.2f} GB
   • Usage:     {(ram_used/ram_total*100):.1f}%
═══════════════════════════════"""
        
        return (image_tensor, memory_info)


class LossGraphAdvanced:
    """
    Расширенная визуализация loss с несколькими метриками.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "plot_style": (plt.style.available, {"default": 'seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default'}),
                "show_moving_avg": ("BOOLEAN", {"default": True}),
                "moving_avg_window": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "show_min_max": ("BOOLEAN", {"default": True}),
                "show_trend": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "height": ("INT", {"default": 600, "min": 300, "max": 1200}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("loss_graph", "min_loss", "max_loss", "final_loss",)
    FUNCTION = "plot"
    CATEGORY = "FluxTrainer/Utilities"

    def plot(self, network_trainer, plot_style, show_moving_avg, moving_avg_window,
             show_min_max, show_trend, width, height):
        
        loss_list = network_trainer["network_trainer"].loss_recorder.global_loss_list
        
        if not loss_list:
            # Пустой график
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center', fontsize=20)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            image_tensor = transforms.ToTensor()(image)
            image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            return (image_tensor, 0.0, 0.0, 0.0)
        
        losses = np.array(loss_list)
        steps = np.arange(len(losses))
        
        plt.style.use(plot_style)
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # Основной график loss
        ax.plot(steps, losses, alpha=0.3, color='blue', label='Loss (raw)')
        
        # Moving average
        if show_moving_avg and len(losses) > moving_avg_window:
            ma = np.convolve(losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            ma_steps = steps[moving_avg_window-1:]
            ax.plot(ma_steps, ma, color='red', linewidth=2, 
                   label=f'Moving Avg ({moving_avg_window})')
        
        # Min/Max точки
        if show_min_max:
            min_idx = np.argmin(losses)
            max_idx = np.argmax(losses)
            ax.scatter([min_idx], [losses[min_idx]], color='green', s=100, 
                      zorder=5, label=f'Min: {losses[min_idx]:.4f}')
            ax.scatter([max_idx], [losses[max_idx]], color='red', s=100, 
                      zorder=5, label=f'Max: {losses[max_idx]:.4f}')
        
        # Trend line
        if show_trend and len(losses) > 10:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), '--', color='orange', alpha=0.8, 
                   label=f'Trend (slope: {z[0]:.2e})')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Добавить статистику
        stats_text = f'Final: {losses[-1]:.4f} | Min: {losses.min():.4f} | Avg: {losses.mean():.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        image = Image.open(buf).convert('RGB')
        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        
        return (image_tensor, float(losses.min()), float(losses.max()), float(losses[-1]))


# =============================================================================
# MODEL UTILITIES
# =============================================================================

class LoRAMerger:
    """
    Слияние нескольких LoRA моделей с настраиваемыми весами.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_1": (folder_paths.get_filename_list("loras"),),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_name": ("STRING", {"default": "merged_lora"}),
                "save_dtype": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "lora_2": (["None"] + folder_paths.get_filename_list("loras"),),
                "weight_2": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3": (["None"] + folder_paths.get_filename_list("loras"),),
                "weight_3": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_4": (["None"] + folder_paths.get_filename_list("loras"),),
                "weight_4": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_lora_path",)
    FUNCTION = "merge"
    CATEGORY = "FluxTrainer/Utilities"

    def merge(self, lora_1, weight_1, output_name, save_dtype,
              lora_2="None", weight_2=1.0, lora_3="None", weight_3=1.0,
              lora_4="None", weight_4=1.0):
        from safetensors.torch import load_file, save_file
        
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        target_dtype = dtype_map[save_dtype]
        
        # Загрузка первой LoRA
        lora_path_1 = folder_paths.get_full_path("loras", lora_1)
        merged = load_file(lora_path_1)
        
        # Применить вес
        for key in merged:
            merged[key] = merged[key].float() * weight_1
        
        # Слияние остальных
        loras_to_merge = [
            (lora_2, weight_2),
            (lora_3, weight_3),
            (lora_4, weight_4),
        ]
        
        for lora_name, weight in loras_to_merge:
            if lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_sd = load_file(lora_path)
                
                for key in lora_sd:
                    if key in merged:
                        merged[key] = merged[key] + lora_sd[key].float() * weight
                    else:
                        merged[key] = lora_sd[key].float() * weight
        
        # Конвертация в целевой dtype
        for key in merged:
            merged[key] = merged[key].to(target_dtype)
        
        # Сохранение
        output_dir = os.path.join(folder_paths.models_dir, "loras", "merged")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_name}.safetensors")
        
        save_file(merged, output_path)
        logger.info(f"Merged LoRA saved to: {output_path}")
        
        return (output_path,)


class CheckpointManager:
    """
    Управление чекпоинтами обучения.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoints_folder": ("STRING", {"default": "", "tooltip": "Folder containing training checkpoints"}),
                "action": (["list", "cleanup_old", "get_best"],),
                "keep_count": ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "Number of checkpoints to keep"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("result", "best_checkpoint",)
    FUNCTION = "manage"
    CATEGORY = "FluxTrainer/Utilities"

    def manage(self, checkpoints_folder, action, keep_count):
        if not os.path.exists(checkpoints_folder):
            return (f"Folder not found: {checkpoints_folder}", "")
        
        # Найти все чекпоинты
        checkpoints = []
        for f in os.listdir(checkpoints_folder):
            if f.endswith('.safetensors'):
                path = os.path.join(checkpoints_folder, f)
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path) / 1024 / 1024  # MB
                checkpoints.append({
                    'name': f,
                    'path': path,
                    'mtime': mtime,
                    'size': size
                })
        
        checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
        
        if action == "list":
            lines = ["📁 Checkpoints:"]
            for i, ckpt in enumerate(checkpoints):
                date = datetime.fromtimestamp(ckpt['mtime']).strftime('%Y-%m-%d %H:%M')
                lines.append(f"{i+1}. {ckpt['name']} ({ckpt['size']:.1f} MB) - {date}")
            result = "\n".join(lines)
            best = checkpoints[0]['path'] if checkpoints else ""
            
        elif action == "cleanup_old":
            to_delete = checkpoints[keep_count:]
            deleted = []
            for ckpt in to_delete:
                try:
                    os.remove(ckpt['path'])
                    deleted.append(ckpt['name'])
                except Exception as e:
                    logger.error(f"Could not delete {ckpt['name']}: {e}")
            
            result = f"Deleted {len(deleted)} old checkpoints:\n" + "\n".join(deleted)
            best = checkpoints[0]['path'] if checkpoints else ""
            
        elif action == "get_best":
            if checkpoints:
                result = f"Best (latest) checkpoint: {checkpoints[0]['name']}"
                best = checkpoints[0]['path']
            else:
                result = "No checkpoints found"
                best = ""
        
        return (result, best)


class PresetManager:
    """
    Сохранение и загрузка пресетов настроек обучения.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        presets_dir = os.path.join(script_directory, "presets")
        if os.path.exists(presets_dir):
            presets = [f.replace('.json', '') for f in os.listdir(presets_dir) if f.endswith('.json')]
        else:
            presets = []
        
        return {
            "required": {
                "action": (["save", "load", "list"],),
                "preset_name": ("STRING", {"default": "my_preset"}),
            },
            "optional": {
                "existing_preset": (["None"] + presets,),
                "optimizer_config": ("ARGS",),
                "network_config": ("ARGS",),
            }
        }

    RETURN_TYPES = ("STRING", "ARGS", "ARGS",)
    RETURN_NAMES = ("status", "optimizer_config", "network_config",)
    FUNCTION = "manage"
    CATEGORY = "FluxTrainer/Utilities"

    def manage(self, action, preset_name, existing_preset="None", 
               optimizer_config=None, network_config=None):
        
        presets_dir = os.path.join(script_directory, "presets")
        os.makedirs(presets_dir, exist_ok=True)
        
        if action == "save":
            if optimizer_config is None and network_config is None:
                return ("Nothing to save - connect optimizer_config or network_config", None, None)
            
            preset_data = {
                "optimizer": optimizer_config,
                "network": network_config,
                "created": datetime.now().isoformat()
            }
            
            preset_path = os.path.join(presets_dir, f"{preset_name}.json")
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2, default=str)
            
            return (f"Saved preset: {preset_name}", optimizer_config, network_config)
        
        elif action == "load":
            load_name = existing_preset if existing_preset != "None" else preset_name
            preset_path = os.path.join(presets_dir, f"{load_name}.json")
            
            if not os.path.exists(preset_path):
                return (f"Preset not found: {load_name}", None, None)
            
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
            
            return (f"Loaded preset: {load_name}", 
                   preset_data.get("optimizer"), 
                   preset_data.get("network"))
        
        elif action == "list":
            presets = [f.replace('.json', '') for f in os.listdir(presets_dir) if f.endswith('.json')]
            if presets:
                status = "Available presets:\n" + "\n".join([f"• {p}" for p in presets])
            else:
                status = "No presets found"
            return (status, None, None)


# =============================================================================
# NODE MAPPINGS
# =============================================================================
if IMPORTS_OK:
    NODE_CLASS_MAPPINGS = {
        # Dataset utilities
        "DatasetPreviewGrid": DatasetPreviewGrid,
        "DatasetValidator": DatasetValidator,
        
        # Training progress
        "TrainingProgressDisplay": TrainingProgressDisplay,
        "MemoryMonitorDisplay": MemoryMonitorDisplay,
        "LossGraphAdvanced": LossGraphAdvanced,
        
        # Model utilities
        "LoRAMerger": LoRAMerger,
        "CheckpointManager": CheckpointManager,
        "PresetManager": PresetManager,
    }
else:
    class DependencyErrorNodeExtended:
        @classmethod
        def INPUT_TYPES(s): return {"required": {}}
        RETURN_TYPES = ()
        FUNCTION = "error"
        CATEGORY = "FluxTrainer/Utilities"
        def error(self): raise ImportError(f"Missing dependencies: {IMPORT_ERROR_MSG}")

    NODE_CLASS_MAPPINGS = {k: DependencyErrorNodeExtended for k in [
         "DatasetPreviewGrid", "DatasetValidator", "TrainingProgressDisplay", 
         "MemoryMonitorDisplay", "LossGraphAdvanced", "LoRAMerger", 
         "CheckpointManager", "PresetManager"
    ]}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Dataset utilities
    "DatasetPreviewGrid": "📸 Dataset Preview Grid" if IMPORTS_OK else "⚠️ Dataset Preview (Error)",
    "DatasetValidator": "✅ Dataset Validator" if IMPORTS_OK else "⚠️ Dataset Validator (Error)",
    
    # Training progress  
    "TrainingProgressDisplay": "📊 Training Progress" if IMPORTS_OK else "⚠️ Training Progress (Error)",
    "MemoryMonitorDisplay": "💾 Memory Monitor" if IMPORTS_OK else "⚠️ Memory Monitor (Error)",
    "LossGraphAdvanced": "📈 Advanced Loss Graph" if IMPORTS_OK else "⚠️ Loss Graph (Error)",
    
    # Model utilities
    "LoRAMerger": "🔀 LoRA Merger" if IMPORTS_OK else "⚠️ LoRA Merger (Error)",
    "CheckpointManager": "📁 Checkpoint Manager" if IMPORTS_OK else "⚠️ Checkpoint Manager (Error)",
    "PresetManager": "⚙️ Preset Manager" if IMPORTS_OK else "⚠️ Preset Manager (Error)",
}
