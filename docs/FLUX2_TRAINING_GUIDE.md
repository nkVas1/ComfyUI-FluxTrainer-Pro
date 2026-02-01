# Flux.2 Training Guide / Руководство по обучению Flux.2

## English

### Overview

This extension adds support for training LoRA on **Flux.2** models:
- **Flux.2 Klein 9B Base** - 9 billion parameters, optimized for consumer hardware
- **Flux.2 Dev** - 32 billion parameters, full capacity model

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| VRAM | 8GB | 16GB+ |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB+ |

### Low VRAM Strategies

The extension provides automatic memory optimization strategies:

| Strategy | VRAM Required | Description |
|----------|---------------|-------------|
| `none` | 24GB+ | No offloading, fastest training |
| `conservative` | 16-24GB | Text encoders on CPU |
| `aggressive` | 8-16GB | Model blocks swapped, optimizer on CPU |
| `extreme` | <8GB | Everything offloaded, slowest but works |

### Nodes

#### 1. Flux.2 Model Select
Select your Flux.2 model files:
- **transformer**: The main transformer model (.safetensors)
- **vae**: VAE model (ae.safetensors)
- **clip_l**: CLIP-L text encoder
- **t5**: T5-XXL text encoder (fp8 recommended for low VRAM)

#### 2. Flux.2 Low VRAM Config
Configure memory optimization:
- **strategy**: Memory strategy (auto/none/conservative/aggressive/extreme)
- **blocks_to_swap**: Number of blocks to swap between GPU and CPU (0-57)
- **gradient_checkpointing**: Enable gradient checkpointing
- **cpu_offload_checkpointing**: Offload checkpoints to CPU (~2GB saved)

#### 3. Flux.2 Init Training ⭐ (Main Node)
Initialize training session:
- **network_dim/alpha**: LoRA rank and alpha
- **learning_rate**: Training learning rate
- **max_train_steps**: Total training steps
- **cache_latents/cache_text_encoder_outputs**: Caching options (use "disk" for low VRAM)
- **optimizer_fusing**: Memory optimization (fused_backward_pass recommended)
- **sample_prompts**: Validation prompts separated by `|`

#### 4. Flux.2 Train Loop
Execute training steps:
- **steps**: Number of steps to train

#### 5. Flux.2 Train & Validate
Training with periodic validation and saving:
- **validate_at_steps**: Generate samples every N steps
- **save_at_steps**: Save checkpoint every N steps

#### 6. Flux.2 Save LoRA
Save trained LoRA:
- **save_state**: Also save full training state (for resume)
- **copy_to_comfy_lora_folder**: Copy LoRA to ComfyUI loras folder

#### 7. Flux.2 End Training
Finalize training, save final LoRA and cleanup resources.

#### 8. Flux.2 Memory Estimator
Estimate memory usage before training to verify your settings will work.

#### 9. Flux.2 Advanced Settings
Expert settings for fine-tuning training dynamics:
- Timestep sampling, weighting scheme, regularization, etc.

### Training Tips for 8GB GPU

1. **Use Flux.2 Klein 9B** - it's designed for consumer hardware
2. **Set strategy to `aggressive`** - this is optimized for 8GB
3. **Use low LoRA rank**: network_dim=8 or 16
4. **Enable all offloading options**
5. **Set batch_size=1 with gradient_accumulation=4-8**
6. **Cache text encoder outputs to disk** - saves ~4GB VRAM
7. **Use FP8 base model** - halves model memory
8. **Use blocks_to_swap=20-30** - balance between speed and VRAM

### Recommended Settings for RTX 3060 Ti (8GB)

```
Model: Flux.2 Klein 9B
Strategy: aggressive
Blocks to swap: 25
Gradient checkpointing: True
CPU offload checkpointing: True
Cache latents: disk
Cache text encoder outputs: disk
FP8 base: True
Network dim: 16
Network alpha: 16
Batch size: 1
Gradient accumulation: 4
Learning rate: 1e-4
Optimizer fusing: fused_backward_pass
```

### Workflow Example

See `example_workflows/flux2_lora_low_vram_example.json` for a complete workflow.

---

## Русский

### Обзор

Это расширение добавляет поддержку обучения LoRA на моделях **Flux.2**:
- **Flux.2 Klein 9B Base** — 9 миллиардов параметров, оптимизирована для потребительского оборудования
- **Flux.2 Dev** — 32 миллиарда параметров, полноценная модель

### Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| VRAM | 8 ГБ | 16+ ГБ |
| RAM | 32 ГБ | 64 ГБ |
| Накопитель | 50 ГБ | 100+ ГБ |

### Стратегии для низкого VRAM

Расширение предоставляет автоматическую оптимизацию памяти:

| Стратегия | Требуется VRAM | Описание |
|-----------|----------------|----------|
| `none` | 24+ ГБ | Без выгрузки, самое быстрое обучение |
| `conservative` | 16-24 ГБ | Text encoders на CPU |
| `aggressive` | 8-16 ГБ | Блоки модели свапятся, оптимизатор на CPU |
| `extreme` | <8 ГБ | Всё выгружается, медленно но работает |

### Ноды

#### 1. Flux.2 Model Select
Выбор файлов модели Flux.2:
- **transformer**: Основная модель трансформера (.safetensors)
- **vae**: VAE модель (ae.safetensors)
- **clip_l**: CLIP-L text encoder
- **t5**: T5-XXL text encoder (рекомендуется fp8 для низкого VRAM)

#### 2. Flux.2 Low VRAM Config
Настройка оптимизации памяти:
- **strategy**: Стратегия памяти (auto/none/conservative/aggressive/extreme)
- **blocks_to_swap**: Количество блоков для свапа между GPU и CPU (0-57)
- **gradient_checkpointing**: Включить gradient checkpointing
- **cpu_offload_checkpointing**: Выгрузка checkpoints на CPU (~2 ГБ экономии)

#### 3. Flux.2 Init Training ⭐ (Главный нод)
Инициализация сессии обучения:
- **network_dim/alpha**: Ранг и альфа LoRA
- **learning_rate**: Скорость обучения
- **max_train_steps**: Общее количество шагов обучения
- **cache_latents/cache_text_encoder_outputs**: Опции кэширования (используйте "disk" для низкого VRAM)
- **optimizer_fusing**: Оптимизация памяти (рекомендуется fused_backward_pass)
- **sample_prompts**: Промпты для валидации, разделённые `|`

#### 4. Flux.2 Train Loop
Выполнение шагов обучения:
- **steps**: Количество шагов для обучения

#### 5. Flux.2 Train & Validate
Обучение с периодической валидацией и сохранением:
- **validate_at_steps**: Генерировать примеры каждые N шагов
- **save_at_steps**: Сохранять чекпоинт каждые N шагов

#### 6. Flux.2 Save LoRA
Сохранение обученной LoRA:
- **save_state**: Также сохранить полное состояние обучения (для продолжения)
- **copy_to_comfy_lora_folder**: Копировать LoRA в папку loras ComfyUI

#### 7. Flux.2 End Training
Завершение обучения, сохранение финальной LoRA и очистка ресурсов.

#### 8. Flux.2 Memory Estimator
Оценка использования памяти перед обучением.

#### 9. Flux.2 Advanced Settings
Экспертные настройки для тонкой настройки динамики обучения:
- Сэмплирование timesteps, схема весов, регуляризация и т.д.

### Советы по обучению для 8 ГБ GPU

1. **Используйте Flux.2 Klein 9B** — она разработана для потребительского оборудования
2. **Установите strategy=`aggressive`** — оптимизировано для 8 ГБ
3. **Используйте низкий ранг LoRA**: network_dim=8 или 16
4. **Включите все опции выгрузки**
5. **Установите batch_size=1 с gradient_accumulation=4-8**
6. **Кэшируйте выходы text encoder на диск** — экономит ~4 ГБ VRAM
7. **Используйте FP8 базовую модель** — уменьшает память модели вдвое
8. **Используйте blocks_to_swap=20-30** — баланс между скоростью и VRAM

### Рекомендуемые настройки для RTX 3060 Ti (8 ГБ)

```
Модель: Flux.2 Klein 9B
Стратегия: aggressive
Блоков для свапа: 25
Gradient checkpointing: True
CPU offload checkpointing: True
Кэширование latents: disk
Кэширование text encoder: disk
FP8 база: True
Network dim: 16
Network alpha: 16
Batch size: 1
Gradient accumulation: 4
Learning rate: 1e-4
Optimizer fusing: fused_backward_pass
```

### Пример Workflow

Смотрите `example_workflows/flux2_lora_low_vram_example.json` для полного примера workflow.

### Ожидаемая производительность

| GPU | Модель | Время на шаг | Примечание |
|-----|--------|--------------|------------|
| RTX 3060 Ti (8GB) | Klein 9B | ~30-60 сек | С aggressive strategy |
| RTX 4090 (24GB) | Klein 9B | ~5-10 сек | Без offloading |
| RTX 4090 (24GB) | Dev 32B | ~15-30 сек | С conservative strategy |

### Устранение неполадок

#### CUDA Out of Memory
1. Увеличьте `blocks_to_swap`
2. Включите `optimizer_cpu_offload`
3. Уменьшите `network_dim` (LoRA rank)
4. Уменьшите разрешение изображений в датасете

#### Медленное обучение
1. Уменьшите `blocks_to_swap` если возможно
2. Убедитесь что `persistent_data_loader_workers=True`
3. Используйте SSD для кэширования

#### NaN loss
1. Уменьшите learning rate
2. Попробуйте `mixed_precision=bf16` вместо fp16
3. Включите gradient clipping

---

## Architecture Notes

### How Low VRAM Training Works

1. **Block Swapping**: Transformer blocks are loaded to GPU one at a time, processed, then moved back to CPU. This allows running models much larger than VRAM.

2. **Gradient Checkpointing**: Instead of storing all activations for backward pass, we recompute them. Trades compute for memory.

3. **CPU Offload Checkpointing**: Activations are stored on CPU RAM instead of GPU VRAM during checkpointing.

4. **Optimizer Offloading**: Adam optimizer states (2x model size) are kept in CPU RAM, with gradients transferred for updates.

5. **Text Encoder Caching**: T5 and CLIP outputs are pre-computed and cached, so these large models don't need to stay in VRAM during training.

### Memory Breakdown (Flux.2 Klein 9B, FP8)

| Component | VRAM (standard) | VRAM (optimized) |
|-----------|-----------------|------------------|
| Base Model | ~9 GB | ~9 GB (on CPU) |
| LoRA Weights (rank 16) | ~0.1 GB | ~0.1 GB |
| Activations | ~4 GB | ~0.5 GB (checkpointed) |
| Optimizer States | ~0.4 GB | 0 GB (on CPU) |
| **Total** | **~13.5 GB** | **~0.6 GB** |

With aggressive offloading, training is possible on 8GB cards, but significantly slower.
