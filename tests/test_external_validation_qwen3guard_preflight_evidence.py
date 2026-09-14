import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "metadata" / "external_validation_qwen3guard_kaggle_preflight_v1.json"
EXPECTED_SHA256 = "132dde5ccc0285e811e743cc1d44d86c0360cfcb84423f08eae4595d0a4baf44"


def test_preflight_evidence_hash_and_pass_status():
    data = EVIDENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    x = json.loads(data)
    assert x["overall_pass"] is True
    assert x["study_examples_used"] is False
    assert len(x["checks"]) == 24
    assert all(c["passed"] for c in x["checks"])


def test_preflight_evidence_exact_runtime_and_model():
    x = json.loads(EVIDENCE.read_text())
    observed = {c["name"]: c["observed"] for c in x["checks"]}
    assert observed["python_version"] == "3.12.13"
    assert observed["torch_version"] == "2.10.0+cu128"
    assert observed["torch_cuda_runtime"] == "12.8"
    assert observed["cuda_device_count"] == 2
    assert observed["gpu_0_canonical_name"] == "Tesla T4"
    assert observed["gpu_1_canonical_name"] == "Tesla T4"
    assert observed["gpu_0_capability"] == [7, 5]
    assert observed["gpu_1_capability"] == [7, 5]
    assert observed["resolved_model_revision"] == "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
    assert x["runtime_dtype"] == "torch.float16"
    assert x["gpu_layout"] == "single_gpu_cuda0"
