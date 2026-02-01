# -*- coding: utf-8 -*-
"""
ComfyUI-FluxTrainer-Pro v2.1 - Advanced Dependency Installer
=============================================================

Этот скрипт автоматически устанавливает все необходимые зависимости,
включая предварительно скомпилированные wheels для Windows Embedded Python.

РЕШАЕТ ПРОБЛЕМУ:
  "error: include file 'Python.h' not found"
  
Скрипт определяет версию Python, платформу и скачивает готовые бинарники
для triton и bitsandbytes, чтобы ничего не нужно было компилировать.

Author: ComfyUI-FluxTrainer-Pro Team
License: Apache-2.0
"""

import subprocess
import sys
import os
import platform
import importlib.util
import urllib.request
import tempfile
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================
SCRIPT_DIR = Path(__file__).parent

# Репозитории с pre-built wheels для Windows
TRITON_WHEELS = {
    # Python 3.10
    (3, 10): "https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp310-cp310-win_amd64.whl",
    # Python 3.11
    (3, 11): "https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp311-cp311-win_amd64.whl",
    # Python 3.12
    (3, 12): "https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp312-cp312-win_amd64.whl",
}

# Bitsandbytes для Windows - используем официальный пакет с pre-built binaries
BNB_WINDOWS_INDEX = "https://jllllll.github.io/bitsandbytes-windows-webui"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def is_installed(package_name: str) -> bool:
    """Проверяет, установлен ли пакет."""
    return importlib.util.find_spec(package_name) is not None


def get_package_version(package_name: str) -> str:
    """Получает версию установленного пакета."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def run_pip(*args):
    """Запускает pip с переданными аргументами."""
    cmd = [sys.executable, "-m", "pip"] + list(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def is_embedded_python() -> bool:
    """Проверяет, является ли это embedded/portable Python."""
    exe_path = Path(sys.executable)
    # Типичные признаки embedded Python
    patterns = ["python_embeded", "python_embedded", "portable"]
    return any(p in str(exe_path).lower() for p in patterns)


def has_python_headers() -> bool:
    """Проверяет наличие Python.h для компиляции."""
    import sysconfig
    include_dir = sysconfig.get_path("include")
    if include_dir:
        python_h = Path(include_dir) / "Python.h"
        return python_h.exists()
    return False


def print_header():
    """Выводит заголовок скрипта."""
    print()
    print("=" * 70)
    print("  ComfyUI-FluxTrainer-Pro v2.1 - Dependency Installer")
    print("=" * 70)
    print()
    print(f"  Python:    {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  Platform:   {platform.system()} {platform.machine()}")
    print(f"  Embedded:   {is_embedded_python()}")
    print(f"  Has Python.h: {has_python_headers()}")
    print()
    print("-" * 70)
    print()


def print_success(msg: str):
    print(f"✅ {msg}")


def print_warning(msg: str):
    print(f"⚠️  {msg}")


def print_error(msg: str):
    print(f"❌ {msg}")


def print_info(msg: str):
    print(f"ℹ️  {msg}")


# =============================================================================
# INSTALLATION LOGIC
# =============================================================================
def install_basic_requirements():
    """Устанавливает базовые зависимости из requirements.txt."""
    print_info("Installing basic requirements...")
    
    requirements_file = SCRIPT_DIR / "requirements.txt"
    
    if requirements_file.exists():
        try:
            run_pip("install", "-r", str(requirements_file), "--prefer-binary")
            print_success("Basic requirements installed")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Some requirements failed to install: {e}")
            # Продолжаем, чтобы попробовать установить критические пакеты вручную
    else:
        print_warning("requirements.txt not found, installing core packages manually")
    
    # Fallback - устанавливаем критические пакеты
    core_packages = [
        "accelerate>=0.33.0",
        "transformers>=4.44.0",
        "diffusers>=0.30.0",
        "safetensors>=0.4.4",
        "huggingface-hub>=0.24.5",
        "toml>=0.10.2",
        "matplotlib",
        "sentencepiece>=0.2.0",
        "protobuf",
    ]
    
    for pkg in core_packages:
        pkg_name = pkg.split(">")[0].split("=")[0].replace("-", "_")
        if not is_installed(pkg_name):
            try:
                run_pip("install", pkg, "--prefer-binary")
                print_success(f"Installed {pkg}")
            except Exception as e:
                print_warning(f"Failed to install {pkg}: {e}")
    
    return True


def install_triton_windows():
    """Устанавливает pre-built Triton для Windows."""
    print_info("Checking Triton for Windows...")
    
    py_ver = (sys.version_info.major, sys.version_info.minor)
    
    if is_installed("triton"):
        version = get_package_version("triton")
        print_success(f"Triton already installed (v{version})")
        return True
    
    if py_ver not in TRITON_WHEELS:
        print_warning(f"No pre-built Triton wheel for Python {py_ver[0]}.{py_ver[1]}")
        print_info("Triton may not be available. Some optimizers might not work.")
        return False
    
    wheel_url = TRITON_WHEELS[py_ver]
    print_info(f"Installing Triton from pre-built wheel...")
    print_info(f"URL: {wheel_url}")
    
    try:
        run_pip("install", wheel_url)
        print_success("Triton installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Triton: {e}")
        print_warning("Some advanced optimizers (8-bit Adam, etc.) may not work.")
        return False


def install_bitsandbytes_windows():
    """Устанавливает bitsandbytes с pre-built binaries для Windows."""
    print_info("Checking bitsandbytes for Windows...")
    
    if is_installed("bitsandbytes"):
        version = get_package_version("bitsandbytes")
        print_success(f"bitsandbytes already installed (v{version})")
        return True
    
    print_info("Installing bitsandbytes with Windows pre-built binaries...")
    
    try:
        # Сначала пробуем официальный пакет (версии >= 0.43.0 имеют Windows поддержку)
        run_pip("install", "bitsandbytes>=0.43.0", "--prefer-binary")
        print_success("bitsandbytes installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print_warning("Official package failed, trying Windows-specific repository...")
    
    try:
        # Fallback на специальный индекс с Windows binaries
        run_pip("install", "bitsandbytes>=0.43.0", 
                "--prefer-binary",
                "--extra-index-url", BNB_WINDOWS_INDEX)
        print_success("bitsandbytes installed from Windows repository!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install bitsandbytes: {e}")
        print_warning("8-bit optimizers will not be available.")
        return False


def install_optional_optimizers():
    """Устанавливает опциональные оптимизаторы."""
    print_info("Installing optional optimizers...")
    
    optimizers = [
        ("prodigyopt", "prodigyopt>=1.0"),
        ("lion_pytorch", "lion-pytorch>=0.0.6"),
        ("schedulefree", "schedulefree>=1.2.7"),
    ]
    
    for pkg_name, pkg_spec in optimizers:
        if not is_installed(pkg_name):
            try:
                run_pip("install", pkg_spec, "--prefer-binary")
                print_success(f"Installed {pkg_name}")
            except Exception as e:
                print_warning(f"Optional: {pkg_name} failed to install: {e}")
        else:
            print_success(f"{pkg_name} already installed")


def verify_installation():
    """Проверяет успешность установки."""
    print()
    print("-" * 70)
    print("  Installation Verification")
    print("-" * 70)
    print()
    
    critical = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("safetensors", "Safetensors"),
    ]
    
    optional = [
        ("triton", "Triton (advanced optimizers)"),
        ("bitsandbytes", "Bitsandbytes (8-bit training)"),
        ("prodigyopt", "Prodigy optimizer"),
        ("lion_pytorch", "Lion optimizer"),
    ]
    
    all_critical_ok = True
    
    for pkg, name in critical:
        if is_installed(pkg):
            ver = get_package_version(pkg)
            print_success(f"{name}: v{ver}")
        else:
            print_error(f"{name}: NOT INSTALLED")
            all_critical_ok = False
    
    print()
    
    for pkg, name in optional:
        if is_installed(pkg):
            ver = get_package_version(pkg)
            print_success(f"{name}: v{ver}")
        else:
            print_warning(f"{name}: not available")
    
    print()
    return all_critical_ok


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Главная функция установки."""
    print_header()
    
    is_windows = platform.system() == "Windows"
    is_embedded = is_embedded_python()
    
    # Предупреждение для embedded Python
    if is_embedded and not has_python_headers():
        print_warning("Embedded Python detected WITHOUT Python.h headers!")
        print_info("This script will install pre-built wheels to avoid compilation.")
        print()
    
    # Interactive confirmation
    if sys.stdin.isatty():
        print("This will install packages to the Python environment shown above.")
        response = input("Continue? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Installation cancelled.")
            return
        print()
    
    # Step 1: Basic requirements
    install_basic_requirements()
    print()
    
    # Step 2: Windows-specific packages (triton, bitsandbytes)
    if is_windows:
        print("-" * 70)
        print("  Windows-Specific Packages")
        print("-" * 70)
        print()
        
        install_triton_windows()
        install_bitsandbytes_windows()
        print()
    
    # Step 3: Optional optimizers
    install_optional_optimizers()
    
    # Step 4: Verification
    all_ok = verify_installation()
    
    print("-" * 70)
    if all_ok:
        print_success("Installation complete! Please restart ComfyUI.")
    else:
        print_error("Some critical packages failed to install.")
        print_info("Check the errors above and try manual installation.")
    print("-" * 70)
    print()


if __name__ == "__main__":
    main()
