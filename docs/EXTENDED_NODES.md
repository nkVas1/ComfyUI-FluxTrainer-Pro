# Extended Nodes Documentation
## Документация расширенных нод

This document describes the additional utility nodes included in ComfyUI-FluxTrainer-Pro.

Этот документ описывает дополнительные утилитарные ноды, включенные в ComfyUI-FluxTrainer-Pro.

---

## 📁 Dataset Utilities / Утилиты датасета

### DatasetPreviewGrid / Предпросмотр датасета

Creates a visual grid preview of your training dataset before starting training.

Создаёт визуальную сетку предпросмотра вашего тренировочного датасета перед началом обучения.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_path | STRING | "" | Path to the training dataset folder |
| grid_cols | INT | 4 | Number of columns in preview |
| grid_rows | INT | 4 | Number of rows in preview |
| image_size | INT | 256 | Size of each preview image |
| show_captions | BOOLEAN | True | Display caption text on images |
| caption_extension | STRING | ".txt" | Extension of caption files |
| random_seed | INT | 0 | Seed for random selection (0=random) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| preview_grid | IMAGE | Grid image for preview |
| dataset_info | STRING | Statistics about the dataset |
| total_images | INT | Total number of images found |

**Usage Tips:**
- Use this node to verify your dataset before training
- Check if captions are properly associated with images
- Ensure images have sufficient quality

---

### DatasetValidator / Валидатор датасета

Validates your dataset for potential issues before training.

Проверяет ваш датасет на потенциальные проблемы перед обучением.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_path | STRING | "" | Path to dataset folder |
| caption_extension | STRING | ".txt" | Extension of caption files |
| min_resolution | INT | 512 | Minimum acceptable resolution |
| check_duplicates | BOOLEAN | True | Check for duplicate images |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| validation_report | STRING | Detailed validation report |
| is_valid | BOOLEAN | True if no critical issues found |
| issue_count | INT | Number of issues detected |

**Checks Performed:**
- ✅ Corrupt images
- ✅ Missing caption files
- ✅ Low resolution images
- ✅ Duplicate images (by MD5 hash)

---

## 📊 Training Progress / Прогресс обучения

### TrainingProgressDisplay / Отображение прогресса

Shows real-time training progress with statistics.

Показывает прогресс обучения в реальном времени со статистикой.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| network_trainer | NETWORKTRAINER | - | The active trainer object |
| show_eta | BOOLEAN | True | Show estimated time remaining |
| show_loss_stats | BOOLEAN | True | Show loss statistics |
| show_lr | BOOLEAN | True | Show current learning rate |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| progress_report | STRING | Formatted progress report |
| current_loss | FLOAT | Current step loss value |
| avg_loss | FLOAT | Average loss (last 100 steps) |
| current_step | INT | Current training step |

---

### LossGraphAdvanced / Расширенный график потерь

Advanced loss visualization with moving average and trend analysis.

Расширенная визуализация потерь с скользящим средним и анализом тренда.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| network_trainer | NETWORKTRAINER | - | The active trainer object |
| plot_style | CHOICE | "default" | Matplotlib plot style |
| show_moving_avg | BOOLEAN | True | Display moving average line |
| moving_avg_window | INT | 100 | Window size for moving average |
| show_min_max | BOOLEAN | True | Highlight min/max points |
| show_trend | BOOLEAN | True | Show trend line |
| width/height | INT | 1024/600 | Graph dimensions in pixels |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| loss_graph | IMAGE | Loss visualization graph |
| min_loss | FLOAT | Minimum loss achieved |
| max_loss | FLOAT | Maximum loss observed |
| final_loss | FLOAT | Final/current loss value |

---

### MemoryMonitorDisplay / Мониторинг памяти

Real-time GPU and RAM usage monitoring.

Мониторинг использования GPU и RAM в реальном времени.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| update_trigger | * | - | Any input to trigger update |
| network_trainer | NETWORKTRAINER | - | Optional trainer for context |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| memory_chart | IMAGE | Pie charts showing memory usage |
| memory_info | STRING | Detailed memory report |

---

## 🔧 Model Utilities / Утилиты моделей

### LoRAMerger / Слияние LoRA

Merge multiple LoRA models with configurable weights.

Слияние нескольких LoRA моделей с настраиваемыми весами.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| lora_1 | LORA | - | First LoRA (required) |
| weight_1 | FLOAT | 1.0 | Weight for first LoRA (-2 to 2) |
| lora_2/3/4 | LORA | None | Additional LoRAs (optional) |
| weight_2/3/4 | FLOAT | 1.0 | Weights for additional LoRAs |
| output_name | STRING | "merged_lora" | Name for output file |
| save_dtype | CHOICE | "bf16" | Output precision (fp16/bf16/fp32) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| merged_lora_path | STRING | Path to merged LoRA file |

**Usage Examples:**
- **Average merge**: weight_1=0.5, weight_2=0.5
- **Dominant merge**: weight_1=0.8, weight_2=0.2
- **Additive**: weight_1=1.0, weight_2=0.5
- **Negative merge**: weight_1=1.0, weight_2=-0.3

---

### CheckpointManager / Менеджер чекпоинтов

Manage training checkpoints.

Управление чекпоинтами обучения.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| checkpoints_folder | STRING | "" | Folder containing checkpoints |
| action | CHOICE | "list" | list/cleanup_old/get_best |
| keep_count | INT | 5 | Checkpoints to keep (for cleanup) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| result | STRING | Action result/report |
| best_checkpoint | STRING | Path to best checkpoint |

**Actions:**
- **list**: List all checkpoints with dates and sizes
- **cleanup_old**: Remove old checkpoints, keeping only `keep_count` latest
- **get_best**: Get path to the most recent (best) checkpoint

---

### PresetManager / Менеджер пресетов

Save and load training configuration presets.

Сохранение и загрузка пресетов настроек обучения.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| action | CHOICE | "save" | save/load/list |
| preset_name | STRING | "my_preset" | Name for the preset |
| existing_preset | CHOICE | None | Existing preset to load |
| optimizer_config | ARGS | - | Optimizer configuration |
| network_config | ARGS | - | Network configuration |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| status | STRING | Action status message |
| optimizer_config | ARGS | Loaded optimizer config |
| network_config | ARGS | Loaded network config |

**Preset Storage:**
Presets are saved in `[extension]/presets/` as JSON files.

---

## 💡 Best Practices / Лучшие практики

### Workflow Recommendations

1. **Always validate your dataset first**
   - Use `DatasetValidator` to check for issues
   - Use `DatasetPreviewGrid` for visual verification

2. **Monitor memory during training**
   - Use `MemoryMonitorDisplay` to track VRAM usage
   - Adjust settings if usage exceeds 90%

3. **Save presets for successful configurations**
   - Use `PresetManager` to save working settings
   - Share presets with others

4. **Analyze training with advanced loss graphs**
   - Watch for convergence in the loss graph
   - If loss plateaus, consider adjusting learning rate

---

## 🔗 Integration Examples

### Complete Training Workflow

```
DatasetValidator → DatasetPreviewGrid → Flux2InitTraining → 
Flux2TrainAndValidateLoop → LossGraphAdvanced + TrainingProgressDisplay → 
Flux2TrainSave → Flux2TrainEnd
```

### Dataset Preparation Only

```
DatasetValidator → DatasetPreviewGrid → PreviewImage
```

### Post-Training Analysis

```
CheckpointManager (list) → LoRAMerger → Output
```

---

## 📝 Notes

- All extended nodes use the category prefix `FluxTrainer/Utilities`
- Nodes are designed to work with both Flux.1 and Flux.2 training
- Memory-intensive operations include automatic cleanup

---

*This documentation is part of ComfyUI-FluxTrainer-Pro v2.0.0*
