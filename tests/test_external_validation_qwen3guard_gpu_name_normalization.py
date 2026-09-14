import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_external_validation_qwen3guard_kaggle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_name_normalizes_kaggle_torch_aliases():
    m = load_module()
    assert m.canonical_gpu_name("Tesla T4") == "Tesla T4"
    assert m.canonical_gpu_name("NVIDIA Tesla T4") == "Tesla T4"
    assert m.canonical_gpu_name("  NVIDIA   Tesla   T4  ") == "Tesla T4"


def test_expected_gpu_identity_is_canonical_t4():
    m = load_module()
    assert m.EXPECTED_GPU_CANONICAL_NAME == "Tesla T4"


def test_preflight_still_requires_two_t4s_and_capability_75():
    m = load_module()
    assert m.EXPECTED_GPU_COUNT == 2
    assert m.EXPECTED_CAPABILITY == (7, 5)
