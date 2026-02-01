"""
ComfyUI-FluxTrainer-Pro
=======================

Professional Flux/Flux.2 LoRA training for ComfyUI with low VRAM optimization.
Fork of kijai/ComfyUI-FluxTrainer with Flux.2 Klein 9B and Dev support.

Author: nkVas1 (fork), kijai (original)
License: Apache-2.0
"""

import traceback

# Initialize empty mappings as fallback
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

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

# Summary
print(f"[ComfyUI-FluxTrainer-Pro] 📦 Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]