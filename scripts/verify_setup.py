"""
Environment & CUDA Verification Script
=======================================
Run this after setting up the environment to verify everything is configured correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
import importlib
import platform


def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def check_python():
    print_header("Python Environment")
    print(f"  Python Version : {sys.version}")
    print(f"  Platform       : {platform.platform()}")
    print(f"  Architecture   : {platform.machine()}")
    print(f"  Executable     : {sys.executable}")


def check_cuda():
    print_header("CUDA / GPU Configuration")
    try:
        import torch

        print(f"  PyTorch Version       : {torch.__version__}")
        print(f"  CUDA Available        : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA Version          : {torch.version.cuda}")
            print(f"  cuDNN Version         : {torch.backends.cudnn.version()}")
            print(f"  cuDNN Enabled         : {torch.backends.cudnn.enabled}")
            print(f"  Number of GPUs        : {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024 ** 3)
                print(f"\n  --- GPU {i}: {props.name} ---")
                print(f"      Compute Capability : {props.major}.{props.minor}")
                print(f"      Total Memory       : {total_mem:.2f} GB")
                print(f"      Multi-Processors   : {props.multi_processor_count}")

            # Quick tensor test on GPU
            print(f"\n  [TEST] Creating tensor on GPU...")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            print(f"  [PASS] GPU matrix multiplication successful (shape: {z.shape})")

            # Memory after test
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            print(f"  [INFO] GPU Memory Allocated : {allocated:.1f} MB")
            print(f"  [INFO] GPU Memory Reserved  : {reserved:.1f} MB")

            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
        else:
            print("  [WARN] CUDA is NOT available. Training will fall back to CPU.")
            print("         Make sure you have:")
            print("           1. An NVIDIA GPU with CUDA support")
            print("           2. NVIDIA drivers installed")
            print("           3. PyTorch installed with CUDA (pip install torch --index-url ...)")

    except ImportError:
        print("  [FAIL] PyTorch is not installed!")


def check_libraries():
    print_header("Library Versions")

    libraries = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("tokenizers", "Tokenizers"),
        ("accelerate", "Accelerate"),
        ("safetensors", "Safetensors"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
        ("tensorboard", "TensorBoard"),
    ]

    all_ok = True
    for module_name, display_name in libraries:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed (version N/A)")
            print(f"  [OK]   {display_name:<16} : {version}")
        except ImportError:
            print(f"  [MISS] {display_name:<16} : NOT INSTALLED")
            all_ok = False

    # Optional libraries
    print("\n  --- Optional ---")
    optional = [("wandb", "Weights & Biases")]
    for module_name, display_name in optional:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK]   {display_name:<16} : {version}")
        except ImportError:
            print(f"  [SKIP] {display_name:<16} : not installed (optional)")

    return all_ok


def check_project_structure():
    print_header("Project Structure")
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/tokenized",
        "models/checkpoints",
        "models/trained",
        "models/tokenizer",
        "scripts",
        "configs",
        "logs",
        "notebooks",
    ]

    all_ok = True
    for d in expected_dirs:
        full_path = project_root / d
        exists = full_path.is_dir()
        status = "[OK]  " if exists else "[MISS]"
        print(f"  {status} {d}/")
        if not exists:
            all_ok = False

    return all_ok


def main():
    print("\n" + "+" * 60)
    print("  LLM Training Environment — Verification Report")
    print("+" * 60)

    check_python()
    check_cuda()
    libs_ok = check_libraries()
    struct_ok = check_project_structure()

    print_header("Summary")
    if libs_ok and struct_ok:
        print("  [SUCCESS] All checks passed! Your environment is ready for training.")
    else:
        print("  [WARNING] Some checks failed. Review the output above and fix issues.")

    print()


if __name__ == "__main__":
    main()
