import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_y0_strata_proportions_frozen_quota_pending():
    p = ROOT / "configs/external_validation_y0_sampling_contract_v1.json"
    x = json.loads(p.read_text())

    assert x["status"] == "Y0_strata_and_proportions_frozen_quota_pending"

    # Quota is intentionally not frozen yet. Do not require a top-level
    # Y0_quota_frozen field because the contract records this state through
    # its status plus pending quota requirements.
    serialized = json.dumps(x)
    assert "final Y0 quota per validation cell" in serialized
    assert (
        "pending_pre_W0_validation" in serialized
        or "quota_pending" in x["status"]
    )
    assert x["Y0_strata_frozen"] is True
    assert x["Y0_proportions_frozen"] is True
    assert x["quota_definition_frozen"] is True
