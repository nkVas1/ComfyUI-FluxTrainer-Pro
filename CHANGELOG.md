# Changelog / История изменений
All notable changes to ComfyUI-FluxTrainer-Pro will be documented in this file.

Все значимые изменения в ComfyUI-FluxTrainer-Pro документируются в этом файле.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-01-XX

### 🎉 Fork from kijai/ComfyUI-FluxTrainer
This version marks the creation of ComfyUI-FluxTrainer-Pro as a derivative work.
Эта версия отмечает создание ComfyUI-FluxTrainer-Pro как производной работы.

### Added / Добавлено

#### Flux.2 Support
- **Flux2TrainModelSelect**: Model selection with auto-detection of Klein 9B / Dev
- **Flux2LowVRAMConfig**: Memory optimization configuration node
- **Flux2OptimizerConfig**: Extended optimizer settings for low VRAM
- **Flux2LoRAConfig**: LoRA network configuration with memory presets
- **InitFlux2LoRATraining**: Training initialization with advanced settings
- **Flux2TrainLoop**: Training loop with real-time memory monitoring
- **Flux2TrainSave**: Enhanced save with multiple formats and HuggingFace upload
- **Flux2TrainValidate**: Validation with sample generation
- **Flux2MemoryMonitor**: Real-time VRAM/RAM monitoring display

#### Low VRAM Optimizations
- Block swapping between GPU and CPU (up to 35 blocks)
- Gradient checkpointing with CPU offload
- Optimizer state offloading to RAM
- Automatic strategy selection based on available VRAM
- FP8 base model loading (50% VRAM savings)
- Aggressive memory cleanup routines

#### Extended Nodes (Coming Soon)
- **DatasetPreviewGrid**: Visual preview of training dataset
- **TrainingProgressDisplay**: Real-time training statistics
- **ModelComparisonGrid**: A/B comparison of trained models
- **LoRAMerger**: Merge multiple LoRAs with weights
- **AutoBatchCalculator**: Automatic batch size optimization
- **CheckpointManager**: Manage training checkpoints
- **PresetManager**: Save/load training configurations

#### Documentation
- Bilingual documentation (English/Russian)
- Detailed FLUX2_TRAINING_GUIDE.md
- Example workflows for different VRAM configurations
- CREDITS.md with proper attribution

### Changed / Изменено
- Project renamed to ComfyUI-FluxTrainer-Pro
- Updated pyproject.toml with new metadata
- Enhanced README with Flux.2 instructions
- Improved error handling and logging

### Fixed / Исправлено
- Memory leaks in training loop
- CUDA cache management for low VRAM
- Import errors for Flux.2 utilities

---

## [1.0.2] - Previous Version (kijai/ComfyUI-FluxTrainer)

See [kijai/ComfyUI-FluxTrainer](https://github.com/kijai/ComfyUI-FluxTrainer) for original changelog.

---

## Versioning / Версионирование

- **Major** (X.0.0): Breaking changes, new model support
- **Minor** (0.X.0): New features, nodes, significant improvements
- **Patch** (0.0.X): Bug fixes, small improvements

---

## Upgrade Guide / Руководство по обновлению

### From kijai/ComfyUI-FluxTrainer 1.0.2 to Pro 2.0.0

1. Backup your existing workflows
2. Remove old ComfyUI-FluxTrainer folder
3. Clone ComfyUI-FluxTrainer-Pro
4. Install new requirements: `pip install -r requirements.txt`
5. Update node names in workflows if needed (old nodes still work)

### Совместимость
All original nodes are preserved. New Flux.2 nodes are additional.
Все оригинальные ноды сохранены. Новые Flux.2 ноды являются дополнительными.
