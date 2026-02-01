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

__version__ = "2.3.0"

# Настройка логгера
logger = logging.getLogger("ComfyUI-FluxTrainer-Pro")

# --- Version Check ---
_MIN_PYTHON = (3, 10)
if sys.version_info < _MIN_PYTHON:
    print(f"[ComfyUI-FluxTrainer-Pro] ⚠️ Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ required, got {sys.version_info.major}.{sys.version_info.minor}")

print(f"[ComfyUI-FluxTrainer-Pro] 🚀 v{__version__} initializing...")

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
        _optional_deps_status[_opt_dep] = "✅ OK"
    except ImportError:
        _optional_deps_status[_opt_dep] = "⚠️ Not installed"
    except Exception as e:
        _optional_deps_status[_opt_dep] = f"❌ Broken: {e}"

if _missing_deps:
    print(f"[ComfyUI-FluxTrainer-Pro] ⚠️ Missing core dependencies: {', '.join(_missing_deps)}")
    print("[ComfyUI-FluxTrainer-Pro]    Run: pip install -r requirements.txt")

if _broken_deps:
    print(f"[ComfyUI-FluxTrainer-Pro] ❌ Broken dependencies:")
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
        error_msg += "\\n\\n🔧 SOLUTION:\\n"
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
        print(f"[FluxTrainer] ❌ {error_info}")
        return ()


# Если критические зависимости сломаны, загружаем только ноду-заглушку
if not _critical_deps_ok:
    NODE_CLASS_MAPPINGS = {"FluxTrainerDependencyError": FluxTrainerDependencyError}
    NODE_DISPLAY_NAME_MAPPINGS = {"FluxTrainerDependencyError": "⚠️ Flux Trainer - Install Error"}
    print(f"[ComfyUI-FluxTrainer-Pro] ❌ Critical dependencies missing. Only error node loaded.")
else:
    # --- Load main Flux nodes ---
    try:
        from .nodes import NODE_CLASS_MAPPINGS as _NCM_FLUX, NODE_DISPLAY_NAME_MAPPINGS as _NDM_FLUX
        NODE_CLASS_MAPPINGS.update(_NCM_FLUX)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_FLUX)
        print("[ComfyUI-FluxTrainer-Pro] ✅ Loaded Flux nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] ❌ Failed to load Flux nodes: {e}")

    # --- Load SD3 nodes ---
    try:
        from .nodes_sd3 import NODE_CLASS_MAPPINGS as _NCM_SD3, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SD3
        NODE_CLASS_MAPPINGS.update(_NCM_SD3)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_SD3)
        print("[ComfyUI-FluxTrainer-Pro] ✅ Loaded SD3 nodes")
    except Exception as e:
        print(f"[ComfyUI-FluxTrainer-Pro] ⚠️ SD3 nodes not loaded: {e}")

    # --- Load SDXL nodes ---
    try:
        from .nodes_sdxl import NODE_CLASS_MAPPINGS as _NCM_SDXL, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SDXL
        NODE_CLASS_MAPPINGS.update(_NCM_SDXL)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_SDXL)
        print("[ComfyUI-FluxTrainer-Pro] ✅ Loaded SDXL nodes")
    except Exception as e:
        print(f"[ComfyUI-FluxTrainer-Pro] ⚠️ SDXL nodes not loaded: {e}")

    # --- Load Flux.2 nodes ---
    try:
        from .nodes_flux2 import NODE_CLASS_MAPPINGS as _NCM_FLUX2, NODE_DISPLAY_NAME_MAPPINGS as _NDM_FLUX2
        NODE_CLASS_MAPPINGS.update(_NCM_FLUX2)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_FLUX2)
        print("[ComfyUI-FluxTrainer-Pro] ✅ Loaded Flux.2 nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] ❌ Failed to load Flux.2 nodes: {e}")

    # --- Load Extended utility nodes ---
    try:
        from .nodes_extended import NODE_CLASS_MAPPINGS as _NCM_EXT, NODE_DISPLAY_NAME_MAPPINGS as _NDM_EXT
        NODE_CLASS_MAPPINGS.update(_NCM_EXT)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDM_EXT)
        print("[ComfyUI-FluxTrainer-Pro] ✅ Loaded Extended nodes")
    except Exception as e:
        traceback.print_exc()
        print(f"[ComfyUI-FluxTrainer-Pro] ❌ Failed to load Extended nodes: {e}")

# --- Web extensions directory ---
WEB_DIRECTORY = "./web"

# Summary
print(f"[ComfyUI-FluxTrainer-Pro] 📦 Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]