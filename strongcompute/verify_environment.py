#!/usr/bin/env python3
"""
StrongCompute Environment Verification Script

Run this inside the StrongCompute container to verify all dependencies
are correctly installed for the CaML linear probe experiments.

Usage:
    python verify_environment.py
    python verify_environment.py --full  # Include model loading test
"""

import sys
import argparse


def check_import(module_name: str, package_name: str = None) -> bool:
    """Try to import a module and report status."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_cuda() -> dict:
    """Check CUDA availability and version."""
    result = {"available": False, "version": None, "device_name": None, "memory_gb": None}

    try:
        import torch
        result["available"] = torch.cuda.is_available()
        if result["available"]:
            result["version"] = torch.version.cuda
            result["device_name"] = torch.cuda.get_device_name(0)
            result["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as e:
        result["error"] = str(e)

    return result


def check_torch() -> dict:
    """Check PyTorch version and capabilities."""
    result = {"installed": False, "version": None, "cuda_available": False}

    try:
        import torch
        result["installed"] = True
        result["version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    return result


def test_model_loading(model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> dict:
    """Test loading the target model."""
    result = {"success": False, "model_name": model_name, "error": None}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"  Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Quick inference test
        print(f"  Testing inference...")
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        result["success"] = True
        result["num_layers"] = model.config.num_hidden_layers
        result["hidden_size"] = model.config.hidden_size

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify CaML environment")
    parser.add_argument("--full", action="store_true", help="Include model loading test")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to test loading")
    args = parser.parse_args()

    print("=" * 60)
    print("CaML Research Environment Verification")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Python version
    print("1. Python Version")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 10
    status = "OK" if py_ok else "WARN"
    print(f"   [{status}] Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("       Expected Python 3.10+")
    print()

    # 2. CUDA
    print("2. CUDA / GPU")
    cuda = check_cuda()
    if cuda["available"]:
        print(f"   [OK] CUDA {cuda['version']}")
        print(f"   [OK] {cuda['device_name']} ({cuda['memory_gb']} GB)")
    else:
        print("   [FAIL] CUDA not available")
        all_passed = False
    print()

    # 3. PyTorch
    print("3. PyTorch")
    torch_info = check_torch()
    if torch_info["installed"]:
        print(f"   [OK] PyTorch {torch_info['version']}")
        cuda_status = "OK" if torch_info["cuda_available"] else "FAIL"
        print(f"   [{cuda_status}] CUDA support")
    else:
        print("   [FAIL] PyTorch not installed")
        all_passed = False
    print()

    # 4. Core dependencies
    print("4. Core Dependencies")
    dependencies = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
    ]

    for module, package in dependencies:
        ok = check_import(module)
        status = "OK" if ok else "FAIL"
        print(f"   [{status}] {package}")
        if not ok:
            all_passed = False
    print()

    # 5. Interpretability tools (optional)
    print("5. Interpretability Tools (optional)")
    interp_deps = [
        ("transformer_lens", "transformer-lens"),
        ("einops", "einops"),
    ]

    for module, package in interp_deps:
        ok = check_import(module)
        status = "OK" if ok else "WARN"
        print(f"   [{status}] {package}")
    print()

    # 6. Model loading test (if requested)
    if args.full:
        print("6. Model Loading Test")
        print(f"   Testing: {args.model}")
        model_result = test_model_loading(args.model)
        if model_result["success"]:
            print(f"   [OK] Model loaded successfully")
            print(f"       Layers: {model_result['num_layers']}")
            print(f"       Hidden size: {model_result['hidden_size']}")
        else:
            print(f"   [FAIL] Model loading failed")
            print(f"       Error: {model_result['error']}")
            all_passed = False
        print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("All checks passed! Environment is ready.")
    else:
        print("Some checks failed. Review the output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
