"""Verify that all dependencies and tools are correctly installed."""

import sys


def check_python():
    """Check Python version."""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"{'✓' if ok else '✗'} Python {v.major}.{v.minor}.{v.micro} (need ≥3.10)")
    return ok


def check_torch():
    """Check PyTorch installation."""
    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"✓ PyTorch {torch.__version__} (CUDA: {gpu})")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_transformers():
    """Check Transformers installation."""
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        return True
    except ImportError:
        print("✗ Transformers not installed")
        return False


def check_peft():
    """Check PEFT (LoRA) installation."""
    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
        return True
    except ImportError:
        print("✗ PEFT not installed")
        return False


def check_sentence_transformers():
    """Check Sentence Transformers installation."""
    try:
        import sentence_transformers
        print(f"✓ Sentence Transformers {sentence_transformers.__version__}")
        return True
    except ImportError:
        print("✗ Sentence Transformers not installed")
        return False


def check_faiss():
    """Check FAISS installation."""
    try:
        import faiss
        print(f"✓ FAISS (indexed: {faiss.IndexFlatL2})")
        return True
    except ImportError:
        print("✗ FAISS not installed")
        return False


def check_java():
    """Check Java installation (for HeidelTime)."""
    import subprocess
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        version = result.stderr.split("\n")[0]
        print(f"✓ Java: {version}")
        return True
    except FileNotFoundError:
        print("✗ Java not installed (needed for HeidelTime)")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Temporal News Reasoning — Installation Check")
    print("=" * 50)

    checks = [
        check_python(),
        check_torch(),
        check_transformers(),
        check_peft(),
        check_sentence_transformers(),
        check_faiss(),
        check_java(),
    ]

    print("=" * 50)
    passed = sum(checks)
    total = len(checks)
    print(f"Result: {passed}/{total} checks passed")

    if passed < total:
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("All checks passed!")
