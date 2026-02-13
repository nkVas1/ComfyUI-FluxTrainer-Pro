"""
ComfyUI-FluxTrainer-Pro
=======================

Professional Flux/Flux.2 LoRA training for ComfyUI with low VRAM optimization.
Fork of kijai/ComfyUI-FluxTrainer with Flux.2 Klein 9B and Dev support.

Author: nkVas1 (fork), kijai (original)
License: Apache-2.0
"""

import sys
import os
import traceback
import logging

__version__ = "2.5.0"

# =============================================================================
# WINDOWS TRITON GLOBAL PATCH - v2.5.0
# =============================================================================
# ВАЖНО: Этот патч ДОЛЖЕН быть в начале файла, ПЕРЕД любыми другими импортами!
# 
# Проблема: На Windows Embedded Python (ComfyUI portable) triton пытается 
# компилировать JIT-ядра через cl.exe, но Python.h отсутствует.
# Другие ноды ComfyUI (KSampler, hy3, и др.) могут загрузить triton ПЕРЕД
# нашими нодами, поэтому простой mock недостаточен - нужно патчить 
# УЖЕ ЗАГРУЖЕННЫЙ triton module.
#
# Решение: Агрессивный патч, который работает в обоих случаях:
# 1. Если triton НЕ загружен - создаём полный mock-модуль
# 2. Если triton УЖЕ загружен - патчим его декораторы на месте
# =============================================================================

def _patch_triton_for_windows():
    """
    Универсальный патч для triton на Windows.
    Предотвращает краш при JIT-компиляции triton ядер.
    """
    # Только Windows и только если нет Python.h
    if sys.platform != 'win32':
        return False
    
    python_home = os.path.dirname(sys.executable)
    python_h_path = os.path.join(python_home, 'include', 'Python.h')
    
    # Если Python.h есть - полноценная установка, патч не нужен
    if os.path.exists(python_h_path):
        return False
    
    from types import ModuleType
    from unittest.mock import MagicMock
    
    # --- No-op декораторы для замены triton.autotune/jit/heuristics ---
    
    def _add_jit_attrs(func):
        """
        Добавляет атрибуты triton.JITFunction к обычной Python-функции.
        Критично: bitsandbytes и другие обращаются к fn.arg_names после @triton.jit.
        Без этого @triton.heuristics и @triton.autotune крашатся.
        """
        if not hasattr(func, 'arg_names'):
            try:
                code = func.__code__
                func.arg_names = list(code.co_varnames[:code.co_argcount])
            except (AttributeError, TypeError):
                func.arg_names = []
        if not hasattr(func, 'params'):
            func.params = [{'name': n, 'is_constexpr': False} for n in getattr(func, 'arg_names', [])]
        if not hasattr(func, 'src'):
            func.src = ''
        if not hasattr(func, 'kernel'):
            func.kernel = None
        if not hasattr(func, 'best_config'):
            func.best_config = None
        return func
    
    def _noop_autotune(*args, **kwargs):
        """No-op autotune decorator - возвращает функцию без изменений"""
        def decorator(func):
            return _add_jit_attrs(func)
        # Поддержка @triton.autotune без скобок (маловероятно, но на всякий случай)
        if args and callable(args[0]):
            return _add_jit_attrs(args[0])
        return decorator
    
    def _noop_jit(*args, **kwargs):
        """
        No-op jit decorator - возвращает функцию с добавленными JIT-атрибутами.
        Критично: добавляет arg_names, params и др. атрибуты, которые ожидаются
        другими triton декораторами (heuristics, autotune).
        """
        def decorator(func):
            return _add_jit_attrs(func)
        # Поддержка @triton.jit без скобок
        if args and callable(args[0]):
            return _add_jit_attrs(args[0])
        return decorator
    
    def _noop_heuristics(values=None, *args, **kwargs):
        """
        No-op heuristics decorator.
        В реальном triton: heuristics({key: fn}) создаёт Heuristics обёртку,
        которая обращается к fn.arg_names. Наш no-op просто пропускает.
        """
        def decorator(func):
            return _add_jit_attrs(func)
        return decorator
    
    class _TritonConfig:
        """
        Полноценная заглушка для triton.Config.
        Принимает любые аргументы как в оригинальном triton.Config.
        
        Использование в bitsandbytes:
            triton.Config({}, num_stages=1, num_warps=8)
        """
        def __init__(self, kwargs=None, num_stages=1, num_warps=4, num_ctas=1, 
                     maxnreg=None, pre_hook=None, post_hook=None):
            self.kwargs = kwargs or {}
            self.num_stages = num_stages
            self.num_warps = num_warps
            self.num_ctas = num_ctas
            self.maxnreg = maxnreg
            self.pre_hook = pre_hook
            self.post_hook = post_hook
        
        def __repr__(self):
            return f"TritonConfig(num_stages={self.num_stages}, num_warps={self.num_warps})"
        
        def all_kwargs(self):
            """Возвращает все kwargs для совместимости"""
            return self.kwargs
    
    # --- Общая функция для создания mock-субмодулей ---
    def _create_mock_submodule(name):
        """Создаёт MagicMock и регистрирует в sys.modules"""
        mock = MagicMock()
        mock.__name__ = name
        mock.__package__ = name.rsplit('.', 1)[0] if '.' in name else name
        sys.modules[name] = mock
        return mock
    
    # --- CASE 1: Triton уже загружен - патчим существующий модуль ---
    triton_module = sys.modules.get('triton')
    
    if triton_module is not None:
        # Проверяем, не наш ли это уже mock
        if getattr(triton_module, '_patched_by_fluxtrainer', False):
            return True  # Уже патчили
        
        # Это реальный triton - патчим его декораторы
        triton_module.autotune = _noop_autotune
        triton_module.jit = _noop_jit
        triton_module.heuristics = _noop_heuristics
        triton_module.Config = _TritonConfig
        triton_module.cdiv = lambda x, y: (x + y - 1) // y
        triton_module.next_power_of_2 = lambda x: 1 << (x - 1).bit_length() if x > 0 else 1
        triton_module._patched_by_fluxtrainer = True
        
        # Патчим уже загруженные субмодули с реальным кодом
        # Критично: triton.runtime.autotuner содержит РЕАЛЬНЫЙ код heuristics/autotune,
        # который крашится при обращении к fn.arg_names на обычных функциях.
        _autotuner = sys.modules.get('triton.runtime.autotuner')
        if _autotuner is not None:
            _autotuner.autotune = _noop_autotune
            _autotuner.heuristics = _noop_heuristics
            # Заменяем Heuristics/Autotuner классы на безопасные обёртки
            class _SafeHeuristics:
                def __init__(self, fn, *a, **kw):
                    self.fn = fn
                    self.arg_names = getattr(fn, 'arg_names', [])
                def __call__(self, *a, **kw):
                    return self.fn(*a, **kw) if callable(self.fn) else None
            _autotuner.Heuristics = _SafeHeuristics
        
        _runtime_jit = sys.modules.get('triton.runtime.jit')
        if _runtime_jit is not None:
            _runtime_jit.jit = _noop_jit
        
        # Патчим/создаём проблемные субмодули
        _problematic_submodules = [
            'triton.common',
            'triton.common.libdevice', 
            'triton.compiler',
            'triton.compiler.compiler',
            'triton.runtime',
            'triton.runtime.driver',
            'triton.backends',
            'triton.backends.nvidia',
            'triton.backends.nvidia.compiler',
            'triton.language',
        ]
        
        for submod_name in _problematic_submodules:
            if submod_name not in sys.modules:
                _create_mock_submodule(submod_name)
        
        print(f"[ComfyUI-FluxTrainer-Pro] Triton decorators patched (existing module)")
        return True
    
    # --- CASE 2: Triton не загружен - создаём полный mock ---
    triton_mock = ModuleType('triton')
    triton_mock.__version__ = '0.0.0-fluxtrainer-mock'
    triton_mock.__path__ = []
    triton_mock.__package__ = 'triton'
    triton_mock._patched_by_fluxtrainer = True
    
    triton_mock.autotune = _noop_autotune
    triton_mock.jit = _noop_jit
    triton_mock.heuristics = _noop_heuristics
    triton_mock.Config = _TritonConfig
    triton_mock.cdiv = lambda x, y: (x + y - 1) // y
    triton_mock.next_power_of_2 = lambda x: 1 << (x - 1).bit_length() if x > 0 else 1
    triton_mock.language = MagicMock()
    
    # Регистрируем mock-модули
    sys.modules['triton'] = triton_mock
    
    # Создаём все нужные субмодули как MagicMock
    _submodules = [
        'triton.common',
        'triton.common.libdevice',
        'triton.compiler', 
        'triton.compiler.compiler',
        'triton.runtime',
        'triton.runtime.driver',
        'triton.runtime.jit',
        'triton.runtime.autotuner',
        'triton.backends',
        'triton.backends.nvidia',
        'triton.backends.nvidia.compiler',
        'triton.language',
        'triton.language.math',
        'triton.language.core',
    ]
    
    for submod_name in _submodules:
        _create_mock_submodule(submod_name)
    
    print(f"[ComfyUI-FluxTrainer-Pro] Triton mock installed (new module)")
    return True


# Применяем патч СРАЗУ при загрузке модуля (до любых других импортов)
_triton_patched = _patch_triton_for_windows()

# Настройка логгера
logger = logging.getLogger("ComfyUI-FluxTrainer-Pro")

# --- Version Check ---
_MIN_PYTHON = (3, 10)
if sys.version_info < _MIN_PYTHON:
    print(f"[ComfyUI-FluxTrainer-Pro] [WARN] Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ required, got {sys.version_info.major}.{sys.version_info.minor}")

print(f"[ComfyUI-FluxTrainer-Pro] v{__version__} initializing...")

# --- Dependency check with detailed diagnostics ---
_critical_deps_ok = True
_missing_deps = []
_broken_deps = {}

# Проверяем критические зависимости
for _dep in ["torch", "toml", "safetensors", "accelerate"]:
    try:
        __import__(_dep)
    except ImportError as e:
        _missing_deps.append(_dep)
        _critical_deps_ok = False
    except Exception as e:
        _broken_deps[_dep] = str(e)
        _critical_deps_ok = False

# Проверяем опциональные зависимости (triton, bitsandbytes) - НЕ критично для загрузки нод
_optional_deps_status = {}
for _opt_dep in ["triton", "bitsandbytes", "diffusers"]:
    try:
        __import__(_opt_dep)
        _optional_deps_status[_opt_dep] = "[OK]"
    except ImportError:
        _optional_deps_status[_opt_dep] = "[WARN] Not installed"
    except Exception as e:
        _optional_deps_status[_opt_dep] = f"[ERROR] Broken: {e}"

if _missing_deps:
    print(f"[ComfyUI-FluxTrainer-Pro] [WARN] Missing core dependencies: {', '.join(_missing_deps)}")
    print("[ComfyUI-FluxTrainer-Pro]    Run: pip install -r requirements.txt")

if _broken_deps:
    print(f"[ComfyUI-FluxTrainer-Pro] [ERROR] Broken dependencies:")
    for dep, err in _broken_deps.items():
        print(f"    {dep}: {err}")

# Показываем статус опциональных зависимостей
print("[ComfyUI-FluxTrainer-Pro] Optional dependencies:")
for dep, status in _optional_deps_status.items():
    print(f"    {dep}: {status}")

# Initialize empty mappings as fallback
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# =============================================================================
# FALLBACK ERROR NODE - показывается если критические зависимости сломаны
# =============================================================================
class FluxTrainerDependencyError:
    """
    Нода-заглушка, которая показывается при ошибках зависимостей.
    Позволяет пользователю видеть проблему прямо в интерфейсе ComfyUI.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        error_msg = "DEPENDENCY ERROR\\n\\n"
        error_msg += "Missing: " + ", ".join(_missing_deps) if _missing_deps else ""
        error_msg += "\\n\\nBroken: " + str(_broken_deps) if _broken_deps else ""
        error_msg += "\n\nSOLUTION:\n"
        error_msg += "1. Run: python install.py\\n"
        error_msg += "2. Or: pip install -r requirements.txt\\n"
        error_msg += "3. Restart ComfyUI"
        
        return {
            "required": {
                "error_info": ("STRING", {
                    "multiline": True, 
                    "default": error_msg
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "show_error"
    CATEGORY = "FluxTrainer"
    OUTPUT_NODE = True
    
    def show_error(self, error_info):
        print(f"[FluxTrainer] [ERROR] {error_info}")
        return ()


# Если критические зависимости сломаны, загружаем только ноду-заглушку
if not _critical_deps_ok:
    NODE_CLASS_MAPPINGS = {"FluxTrainerDependencyError": FluxTrainerDependencyError}
    NODE_DISPLAY_NAME_MAPPINGS = {"FluxTrainerDependencyError": "[!] Flux Trainer - Install Error"}
    print(f"[ComfyUI-FluxTrainer-Pro] [ERROR] Critical dependencies missing. Only error node loaded.")
else:
    # --- Load main Flux nodes ---
    try:
        from .nodes import NODE_CLASS_MAPPINGS as _NCM_FLUX, NODE_DISPLAY_NAME_MAPPINGS as _NDM_FLUX
        NODE_CLASS_MAPPINGS.update(_NCM_FLUX)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_FLUX)
        print("[ComfyUI-FluxTrainer-Pro] [OK] Loaded Flux nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] [ERROR] Failed to load Flux nodes: {e}")

    # --- Load SD3 nodes ---
    try:
        from .nodes_sd3 import NODE_CLASS_MAPPINGS as _NCM_SD3, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SD3
        NODE_CLASS_MAPPINGS.update(_NCM_SD3)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_SD3)
        print("[ComfyUI-FluxTrainer-Pro] [OK] Loaded SD3 nodes")
    except Exception as e:
        print(f"[ComfyUI-FluxTrainer-Pro] [WARN] SD3 nodes not loaded: {e}")

    # --- Load SDXL nodes ---
    try:
        from .nodes_sdxl import NODE_CLASS_MAPPINGS as _NCM_SDXL, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SDXL
        NODE_CLASS_MAPPINGS.update(_NCM_SDXL)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_SDXL)
        print("[ComfyUI-FluxTrainer-Pro] [OK] Loaded SDXL nodes")
    except Exception as e:
        print(f"[ComfyUI-FluxTrainer-Pro] [WARN] SDXL nodes not loaded: {e}")

    # --- Load Flux.2 nodes ---
    try:
        from .nodes_flux2 import NODE_CLASS_MAPPINGS as _NCM_FLUX2, NODE_DISPLAY_NAME_MAPPINGS as _NDM_FLUX2
        NODE_CLASS_MAPPINGS.update(_NCM_FLUX2)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_FLUX2)
        print("[ComfyUI-FluxTrainer-Pro] [OK] Loaded Flux.2 nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] [ERROR] Failed to load Flux.2 nodes: {e}")

    # --- Load Extended utility nodes ---
    try:
        from .nodes_extended import NODE_CLASS_MAPPINGS as _NCM_EXT, NODE_DISPLAY_NAME_MAPPINGS as _NDM_EXT
        NODE_CLASS_MAPPINGS.update(_NCM_EXT)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_EXT)
        print("[ComfyUI-FluxTrainer-Pro] [OK] Loaded Extended nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] [ERROR] Failed to load Extended nodes: {e}")

# --- Web extensions directory ---
WEB_DIRECTORY = "./web"

# Summary
print(f"[ComfyUI-FluxTrainer-Pro] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]