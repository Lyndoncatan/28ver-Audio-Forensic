
import sys
import importlib

packages = ["demucs", "mediapipe", "scipy", "numpy", "tensorflow"]
missing = []

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

for package in packages:
    try:
        importlib.import_module(package)
        print(f"[OK] {package}")
    except ImportError:
        print(f"[MISSING] {package}")
        missing.append(package)

if missing:
    sys.exit(1)
sys.exit(0)
