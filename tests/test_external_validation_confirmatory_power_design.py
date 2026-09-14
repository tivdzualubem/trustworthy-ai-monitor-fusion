from monitor_fusion.external_validation.confirmatory_power_design import load_power_design

def test_claims():
    cfg = load_power_design()
    assert cfg["primary_claims"]["relative_FNR_non_inferiority"]["margin"] == 0.03
    assert cfg["primary_claims"]["absolute_FNR_ceiling"]["threshold"] == 0.10

def test_blocks_are_units():
    cfg = load_power_design()
    assert cfg["design_basis"]["unit_of_independence"] == "provenance_block"

def test_joint_power_required():
    cfg = load_power_design()
    assert cfg["primary_claims"]["joint_success_required"] is True
    assert cfg["power_targets"]["joint_power_required"] is True

def test_beta_binomial_not_primary():
    cfg = load_power_design()
    assert cfg["simulation_requirements"]["beta_binomial_primary_certificate"] is False
