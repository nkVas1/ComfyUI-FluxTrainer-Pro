# 🚀 Быстрый старт ComfyUI-FluxTrainer-Pro

## Установка

1. **Клонируйте репозиторий** в папку `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro.git
   ```

2. **Установите зависимости** (в виртуальном окружении ComfyUI!):
   ```bash
   cd ComfyUI-FluxTrainer-Pro
   pip install -r requirements.txt
   ```

3. **Перезапустите ComfyUI**

## Первый workflow за 5 минут

### Flux LoRA Training (Legacy)

1. Создайте новый workflow
2. Добавьте ноды:
   - `FluxTrain ModelSelect` - выберите модели transformer, vae, clip_l, t5
   - `TrainDatasetGeneralConfig` - базовые настройки датасета  
   - `TrainDatasetAdd` - укажите путь к папке с изображениями
   - `Optimizer Config` - настройки оптимизатора
   - `Init Flux LoRA Training` - инициализация
   - `Flux Train Loop` - цикл обучения
   - `Flux Train Save LoRA` - сохранение результата

3. Соедините ноды и запустите

### Flux.2 LoRA Training (Low VRAM)

Для GPU с 8GB VRAM используйте ноды `Flux.2`:

1. `Flux.2 Model Select` или `Flux.2 Model Paths`
2. `Flux.2 Low VRAM Config` - настройки для экономии памяти
3. `Flux.2 Init Training`
4. `Flux.2 Train Loop`
5. `Flux.2 Save LoRA`

## Решение проблем

### Ноды показывают "UNKNOWN"

- Убедитесь, что модели лежат в правильных папках ComfyUI
- Используйте `Flux.2 Model Paths` для ручного ввода путей
- Проверьте консоль ComfyUI на ошибки импорта

### Ошибки зависимостей

```bash
# Активируйте виртуальное окружение ComfyUI!
pip install -r requirements.txt
```

### Недостаточно VRAM

- Уменьшите `batch_size` до 1
- Включите `gradient_checkpointing`
- Используйте `cpu_offloading`
- Уменьшите разрешение изображений

## Ссылки

- [Полная документация](docs/FLUX2_TRAINING_GUIDE.md)
- [Примеры workflows](example_workflows/)
- [Changelog](CHANGELOG.md)
- [Оригинальный репозиторий](https://github.com/kijai/ComfyUI-FluxTrainer)
