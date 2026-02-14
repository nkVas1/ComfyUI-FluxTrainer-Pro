# 🚀 Быстрый старт ComfyUI-FluxTrainer-Pro

## Требования

- **ComfyUI** (свежая версия)
- **Python 3.10+** 
- **NVIDIA GPU** с 8+ GB VRAM (рекомендуется 12+ GB)
- **CUDA 12.x**

## Установка

### Шаг 1: Клонирование

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro.git
```

### Шаг 2: Установка зависимостей

```bash
cd ComfyUI-FluxTrainer-Pro
python install.py
```

**⚠️ Windows Embedded Python?** Скрипт автоматически установит pre-built wheels для triton и bitsandbytes!

### Шаг 3: Перезапуск ComfyUI

После установки обязательно перезапустите ComfyUI.

---

## Первое обучение LoRA

### 1. Подготовьте изображения

Создайте папку с изображениями (минимум 5-10 картинок):
```
training_data/
├── image1.png
├── image2.jpg
└── image3.png
```

### 2. Выберите модели

Добавьте ноду `Flux2TrainModelSelect` и укажите:
- **Transformer**: flux2_klein_9b или flux2_dev
- **VAE**: ae.safetensors
- **Text Encoder**: для Klein 9B используйте `qwen_3_8b.safetensors`

### 3. Создайте датасет

Используйте ноду `FluxTrainDatasetAdd` для настройки датасета.

### 4. Настройте оптимизатор

Рекомендуемые настройки для начала:
- **Optimizer**: `adafactor` (не требует bitsandbytes!)
- **Learning Rate**: `1e-4`
- **LR Scheduler**: `constant_with_warmup`
- **Warmup Steps**: `100`

### 5. Инициализируйте обучение

Нода `Flux2InitTraining`:
- **network_type**: `lora` (или `dora` для лучшего качества)
- **network_dim**: `16` (8-32 для Low VRAM)
- **network_alpha**: `16` (должен быть ≤ network_dim!)
- **max_train_steps**: `1000`

### 6. Запустите!

Подключите `Flux2TrainLoop` и нажмите **Queue Prompt**.

---

## Советы для Low VRAM (8-12 GB)

1. **Используйте Adafactor** - не требует bitsandbytes и работает стабильно
2. **network_dim: 8-16** - меньше = меньше VRAM
3. **cache_latents: disk** - кэширование на диск экономит VRAM
4. **gradient_dtype: bf16** - bf16 стабильнее fp16
5. **Включите FP8 base** через `Flux2LowVRAMConfig`
6. Для 8 ГБ VRAM используйте `blocks_to_swap=25` и не включайте одновременно `cpu_offload_checkpointing`

---

## Решение проблем

### ❌ "Python.h not found"

```bash
cd custom_nodes/ComfyUI-FluxTrainer-Pro
python install.py
```

### ❌ "bitsandbytes error"

Используйте Adafactor вместо adamw8bit - он не требует bitsandbytes!

### ❌ "CUDA out of memory"

1. Уменьшите `network_dim` до 8
2. Включите `cache_latents: disk`
3. Используйте `Flux2LowVRAMConfig` с FP8

### ❌ "Garbage LoRA" (битый результат)

Проверьте:
- `network_alpha` ≤ `network_dim`
- `save_dtype` совпадает с `gradient_dtype`
- Используете `bf16` а не `fp16`

---

## Пример workflow

Импортируйте один из готовых примеров из папки `example_workflows/`:

- `flux2_lora_low_vram_example.json` - для 8-12 GB VRAM
- `flux2_complete_training.json` - полный пример

---

## Полезные ссылки

- [README.md](../README.md) - Полная документация
- [CHANGELOG.md](../CHANGELOG.md) - История изменений
- [GitHub Issues](https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro/issues) - Сообщить о проблеме
