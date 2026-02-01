# 🚀 ComfyUI-FluxTrainer-Pro

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2](https://img.shields.io/badge/Flux.2-Supported-purple.svg)](https://blackforestlabs.ai)

**Professional Flux & Flux.2 LoRA Training for ComfyUI**

*Fork of [kijai/ComfyUI-FluxTrainer](https://github.com/kijai/ComfyUI-FluxTrainer) with extended Flux.2 support and low VRAM optimization*

[English](#english) | [Русский](#русский)

</div>

---

<a name="english"></a>
## 🇬🇧 English

### ✨ Features

#### 🆕 Flux.2 Support
- **Flux.2 Klein 9B Base** — 9 billion parameters, consumer GPU friendly
- **Flux.2 Dev** — Full 32 billion parameter model
- Auto-detection of model type from checkpoint

#### 💾 Low VRAM Optimization (8GB+)
- **Block Swapping** — Dynamic GPU↔CPU offloading (up to 35 blocks)
- **Gradient Checkpointing** — With optional CPU offload
- **Optimizer Offloading** — Keep optimizer states in RAM
- **FP8 Loading** — 50% VRAM reduction for base model
- **Auto Strategy** — Automatic optimization based on available VRAM

#### 🎛️ Extended Nodes
| Category | Nodes |
|----------|-------|
| **Model Selection** | FluxTrainModelSelect, Flux2TrainModelSelect |
| **Dataset** | TrainDatasetGeneralConfig, TrainDatasetAdd, TrainDatasetRegularization |
| **Optimizer** | OptimizerConfig, OptimizerConfigAdafactor, OptimizerConfigProdigy |
| **Training** | InitFluxLoRATraining, FluxTrainLoop, FluxTrainAndValidateLoop |
| **Save/Load** | FluxTrainSave, FluxTrainSaveModel, FluxTrainResume |
| **Validation** | FluxTrainValidate, FluxTrainValidationSettings |
| **Utilities** | VisualizeLoss, ExtractFluxLoRA, UploadToHuggingFace |
| **Flux.2 Specific** | Flux2LowVRAMConfig, Flux2OptimizerConfig, Flux2LoRAConfig |
| **Memory** | Flux2MemoryMonitor |

### 📦 Installation

#### Method 1: ComfyUI Manager (Recommended)
Search for "FluxTrainer-Pro" in ComfyUI Manager.

#### Method 2: Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro.git
cd ComfyUI-FluxTrainer-Pro
pip install -r requirements.txt
# OR run the provided helper:
python install.py
```

#### Method 3: Portable Windows
```bash
cd ComfyUI_windows_portable
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-FluxTrainer-Pro\requirements.txt
```

### 🚀 Quick Start

#### For Standard Flux Training
1. Use **FluxTrain ModelSelect** node
2. Add **TrainDatasetGeneralConfig** → **TrainDatasetAdd**
3. Choose optimizer with **OptimizerConfig**
4. Initialize with **Init Flux LoRA Training**
5. Connect to **Flux Train Loop** → **Flux Train Save**

#### For Flux.2 on 8GB GPU
1. Use **Flux2 Model Select** node
2. Add **Flux2 Low VRAM Config** with:
   - `strategy`: aggressive
   - `blocks_to_swap`: 25
   - Enable all offloading options
3. Use **Flux2 Optimizer Config** with:
   - `optimizer_type`: adamw8bit
   - `cpu_offload_optimizer`: true
4. Set batch_size=1, gradient_accumulation=8

### 📊 VRAM Requirements

| Model | Min VRAM | Recommended | Config |
|-------|----------|-------------|--------|
| Flux.1 | 12GB | 16GB+ | Standard |
| Flux.2 Klein 9B | 8GB | 12GB+ | aggressive + 25 blocks |
| Flux.2 Dev | 12GB | 24GB+ | conservative |

### 📚 Documentation

- [FLUX2_TRAINING_GUIDE.md](docs/FLUX2_TRAINING_GUIDE.md) — Complete Flux.2 training guide
- [CHANGELOG.md](CHANGELOG.md) — Version history
- [CREDITS.md](CREDITS.md) — Attribution and credits

---

<a name="русский"></a>
## 🇷🇺 Русский

### ✨ Возможности

#### 🆕 Поддержка Flux.2
- **Flux.2 Klein 9B Base** — 9 миллиардов параметров, для потребительских GPU
- **Flux.2 Dev** — Полная модель с 32 миллиардами параметров
- Автоопределение типа модели из чекпоинта

#### 💾 Оптимизация для низкого VRAM (8GB+)
- **Block Swapping** — Динамическая выгрузка GPU↔CPU (до 35 блоков)
- **Gradient Checkpointing** — С опциональной выгрузкой на CPU
- **Optimizer Offloading** — Хранение состояния оптимизатора в RAM
- **FP8 Loading** — 50% экономия VRAM для базовой модели
- **Auto Strategy** — Автоматическая оптимизация по доступному VRAM

### 📦 Установка

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro.git
cd ComfyUI-FluxTrainer-Pro
pip install -r requirements.txt
# ИЛИ запустите помощника установки:
python install.py
```

### 🚀 Быстрый старт для 8GB GPU

1. Используйте ноду **Flux2 Model Select**
2. Добавьте **Flux2 Low VRAM Config**:
   - `strategy`: aggressive
   - `blocks_to_swap`: 25
   - Включите все опции offloading
3. Используйте **Flux2 Optimizer Config**:
   - `optimizer_type`: adamw8bit
   - `cpu_offload_optimizer`: true
4. Установите batch_size=1, gradient_accumulation=8

### 📊 Требования к VRAM

| Модель | Мин. VRAM | Рекомендуемый | Конфиг |
|--------|-----------|---------------|--------|
| Flux.1 | 12GB | 16GB+ | Стандартный |
| Flux.2 Klein 9B | 8GB | 12GB+ | aggressive + 25 блоков |
| Flux.2 Dev | 12GB | 24GB+ | conservative |

---

## 🙏 Credits / Благодарности

This project is a **fork** of [kijai/ComfyUI-FluxTrainer](https://github.com/kijai/ComfyUI-FluxTrainer).

Based on:
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — Core training scripts
- [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) — LyCORIS networks
- [LoganBooker/prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free) — Optimizer

See [CREDITS.md](CREDITS.md) for full attribution.

## 📄 License

Apache-2.0 — Same as original project. See [LICENSE.md](LICENSE.md).

---

<div align="center">

**Made with ❤️ for the ComfyUI Community**

*If you find this useful, please ⭐ the repository!*

</div>

