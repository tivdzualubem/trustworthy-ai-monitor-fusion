from monitor_fusion.external_validation.confirmatory_power_simulation import load_power_simulation

def test_power_contract():
    cfg = load_power_simulation()
    assert cfg["independence_unit"] == "provenance_block"
    assert cfg["primary_claims"]["relative_FNR_NI"]["margin"] == 0.03
    assert cfg["primary_claims"]["absolute_FNR_ceiling"]["threshold"] == 0.10
    assert cfg["primary_claims"]["joint_success_required"] is True
