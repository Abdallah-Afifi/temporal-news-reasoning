"""Verify that all dependencies and tools are correctly installed."""

import re
import subprocess
import sys


def _major_minor(version: str) -> tuple[int, int] | tuple[None, None]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return (None, None)
    return int(match.group(1)), int(match.group(2))


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


def check_torch_family_compatibility():
    """Check torch/torchvision/torchaudio major.minor compatibility."""
    try:
        import torch
        import torchvision
        import torchaudio

        torch_mm = _major_minor(torch.__version__)
        vision_mm = _major_minor(torchvision.__version__)
        audio_mm = _major_minor(torchaudio.__version__)

        expected_vision_by_torch = {
            (2, 4): (0, 19),
            (2, 5): (0, 20),
            (2, 6): (0, 21),
            (2, 7): (0, 22),
            (2, 8): (0, 23),
            (2, 9): (0, 24),
        }
        expected_vision = expected_vision_by_torch.get(torch_mm)

        audio_ok = audio_mm == torch_mm
        vision_ok = expected_vision is None or vision_mm == expected_vision
        family_ok = audio_ok and vision_ok

        extra = ""
        if expected_vision is not None:
            extra = f", expected torchvision={expected_vision[0]}.{expected_vision[1]}.x"
        print(
            f"{'✓' if family_ok else '✗'} "
            f"Torch family: torch={torch.__version__}, torchvision={torchvision.__version__}, "
            f"torchaudio={torchaudio.__version__}{extra}"
        )
        return family_ok
    except ImportError as exc:
        print(f"✗ Torch family incomplete: {exc}")
        return False


def check_flash_attention_compatibility():
    """Check flash-attn import and broad compatibility with torch major.minor."""
    try:
        import torch
        import flash_attn

        torch_mm = _major_minor(torch.__version__)
        fa_version = getattr(flash_attn, "__version__", "unknown")

        compatible = True
        detail = ""
        if torch_mm == (2, 4) and not fa_version.startswith("2.8"):
            compatible = False
            detail = " (expected flash-attn 2.8.x with torch 2.4.x)"

        print(f"{'✓' if compatible else '✗'} FlashAttention {fa_version} with torch {torch.__version__}{detail}")
        return compatible
    except ImportError:
        print("✓ FlashAttention not installed (optional)")
        return True
    except Exception as exc:
        print(f"✗ FlashAttention check failed: {exc}")
        return False


def check_dependency_integrity():
    """Run pip check to detect broken requirements metadata."""
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    ok = result.returncode == 0
    if ok:
        print("✓ pip check: no broken requirements")
    else:
        print("✗ pip check found broken requirements")
        output = (result.stdout + "\n" + result.stderr).strip()
        if output:
            print(output)
    return ok


def check_cuda_runtime_health():
    """Detect broken CUDA runtime state that often indicates driver/runtime mismatch."""
    try:
        import torch

        built_with_cuda = torch.backends.cuda.is_built()
        if not built_with_cuda:
            print("✓ CUDA runtime health: CPU-only torch build")
            return True

        device_count = torch.cuda.device_count()
        cuda_available = torch.cuda.is_available()

        suspicious_state = device_count > 0 and not cuda_available
        if suspicious_state:
            print(
                "✗ CUDA runtime health: GPU detected but CUDA unavailable "
                f"(device_count={device_count}, is_available={cuda_available})"
            )
            return False

        print(
            "✓ CUDA runtime health: "
            f"device_count={device_count}, is_available={cuda_available}"
        )
        return True
    except Exception as exc:
        print(f"✗ CUDA runtime health check failed: {exc}")
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
        check_torch_family_compatibility(),
        check_flash_attention_compatibility(),
        check_dependency_integrity(),
        check_cuda_runtime_health(),
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
