from monitor_fusion.external_validation.confirmatory_joint_power import (
    load_joint_power_config,
    estimate_joint_power
)

def test_joint_power_contract():
    cfg = load_joint_power_config()
    assert cfg["independence_unit"] == "provenance_block"
    assert cfg["claims"]["relative_FNR_NI"]["margin"] == 0.03
    assert cfg["claims"]["absolute_FNR_ceiling"]["threshold"] == 0.10
    assert cfg["claims"]["joint_success_required"] is True

def test_power_output():
    result = estimate_joint_power(100, 20)
    assert "joint_power" in result
