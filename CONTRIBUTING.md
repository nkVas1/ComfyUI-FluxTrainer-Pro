# Contributing to ComfyUI-FluxTrainer-Pro
## Участие в разработке

Thank you for your interest in contributing! / Спасибо за ваш интерес к проекту!

---

## 🌐 Language / Язык

This project accepts contributions in both **English** and **Russian**.
Этот проект принимает вклады на **английском** и **русском** языках.

---

## 📋 How to Contribute / Как внести вклад

### 1. Bug Reports / Отчёты об ошибках

Before reporting a bug, please:
- Check if the issue already exists
- Include your system information (OS, GPU, VRAM, Python version)
- Include the full error message and traceback
- Describe steps to reproduce the issue

Прежде чем сообщать об ошибке:
- Проверьте, не существует ли уже такая проблема
- Укажите информацию о системе (ОС, GPU, VRAM, версия Python)
- Включите полное сообщение об ошибке
- Опишите шаги для воспроизведения

### 2. Feature Requests / Запросы функций

We welcome suggestions for new features! Please:
- Describe the feature clearly
- Explain the use case
- Consider if it fits the project scope

Мы приветствуем предложения! Пожалуйста:
- Опишите функцию чётко
- Объясните случай использования
- Подумайте, вписывается ли она в рамки проекта

### 3. Code Contributions / Код

#### Setup / Настройка

```bash
# Clone the repository / Клонируйте репозиторий
git clone https://github.com/nkVas1/ComfyUI-FluxTrainer-Pro.git
cd ComfyUI-FluxTrainer-Pro

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

#### Code Style / Стиль кода

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions/classes
- Use meaningful variable names

```python
# Good / Хорошо
def calculate_memory_usage(model_params_b: float, use_fp8: bool = True) -> Dict[str, float]:
    """
    Calculate estimated VRAM usage for training.
    
    Args:
        model_params_b: Model parameters in billions
        use_fp8: Whether to use FP8 precision
        
    Returns:
        Dictionary with memory estimates in GB
    """
    ...

# Bad / Плохо
def calc_mem(p, f):
    ...
```

#### Node Development / Разработка нод

When creating new nodes, follow these guidelines:

```python
class MyNewNode:
    """
    Brief description of what this node does.
    
    Краткое описание того, что делает эта нода.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "param1": ("TYPE", {
                    "default": "value",
                    "tooltip": "Description of this parameter"  # Always add tooltips!
                }),
            },
            "optional": {
                # Optional parameters
            }
        }
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    RETURN_NAMES = ("output_name",)
    FUNCTION = "process"
    CATEGORY = "FluxTrainer/YourCategory"  # Use consistent categories
    
    def process(self, param1):
        # Implementation
        pass
```

#### Testing / Тестирование

Please test your changes before submitting:

1. **Unit tests** (if applicable)
2. **Integration test with ComfyUI**
3. **Test on low VRAM GPU** (if adding memory-related features)

#### Commit Messages / Сообщения коммитов

Use clear, descriptive commit messages:

```
feat: Add dataset validation node
fix: Fix memory leak in training loop
docs: Update README with new node examples
refactor: Improve optimizer offloading performance
```

---

## 🔀 Pull Request Process / Процесс Pull Request

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes
4. **Push** to your fork
5. Open a **Pull Request**

### PR Checklist / Чеклист PR

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] No new dependencies without discussion

---

## 📁 Project Structure / Структура проекта

```
ComfyUI-FluxTrainer-Pro/
├── __init__.py           # Entry point, node registration
├── nodes.py              # Main Flux.1 nodes
├── nodes_flux2.py        # Flux.2 specific nodes
├── nodes_extended.py     # Utility nodes
├── nodes_sd3.py          # SD3 support
├── nodes_sdxl.py         # SDXL support
├── library/              # Core utilities
│   ├── flux_utils.py     # Flux model utilities
│   ├── low_vram_utils.py # Memory optimization
│   ├── train_util.py     # Training utilities
│   └── ...
├── networks/             # LoRA implementations
├── lycoris/              # LyCORIS support
├── docs/                 # Documentation
├── example_workflows/    # Example workflows
└── presets/              # Training presets
```

---

## 🎯 Areas for Contribution / Области для вклада

### High Priority / Высокий приоритет
- [ ] More low VRAM optimizations
- [ ] Better progress visualization
- [ ] Automatic hyperparameter tuning
- [ ] Multi-GPU support improvements

### Medium Priority / Средний приоритет
- [ ] Additional optimizer implementations
- [ ] Dataset augmentation nodes
- [ ] Training schedulers
- [ ] Checkpoint comparison tools

### Documentation / Документация
- [ ] Video tutorials
- [ ] More example workflows
- [ ] Translation to other languages

---

## 📜 Code of Conduct / Кодекс поведения

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Focus on the work, not the person

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.

---

## 🙏 Thank You!

Your contributions help make this project better for everyone!
Ваш вклад помогает улучшить этот проект для всех!
