from monitor_fusion.external_validation.confirmatory_design import load_confirmatory_design

def test_design_contract():
    cfg = load_confirmatory_design()
    assert cfg["primary_estimands"]["relative_FNR_non_inferiority_margin"] == 0.03
    assert cfg["primary_estimands"]["absolute_FNR_ceiling"] == 0.10
    assert cfg["independent_sampling"]["unit_of_independence"] == "provenance_block"
    assert cfg["analysis_hierarchy"]["beta_binomial_monte_carlo"]["primary_certificate"] is False
