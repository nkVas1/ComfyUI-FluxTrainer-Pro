import subprocess
import sys
import os

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install():
    print("--- ComfyUI-FluxTrainer-Pro Dependency Installer ---")
    print(f"Python Executable: {sys.executable}")
    print("NOTE: This script installs packages to the Python environment listed above.")
    print("If you are using a portable ComfyUI, make sure you run this script with that ComfyUI's python.")
    print("Example: G:\\ComfyUI\\python_embeded\\python.exe install.py")
    print("-" * 50)
    
    # Simple interactive check if run in visible terminal
    if sys.stdin.isatty():
        response = input("Do you want to proceed? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_file):
        print(f"Found requirements.txt at {requirements_file}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Successfully installed requirements from file.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
    else:
        print("requirements.txt not found. Installing core packages manually...")
        packages = [
            "accelerate>=0.33.0",
            "transformers>=4.44.0",
            "diffusers>=0.30.0",
            "matplotlib",
            "toml",
            "pandas",
            "scipy"
        ]
        for p in packages:
            try:
                install(p)
            except Exception as e:
                print(f"Failed to install {p}: {e}")

    print("\n--- Installation Complete ---")
    print("Please restart ComfyUI.")

if __name__ == "__main__":
    check_and_install()
