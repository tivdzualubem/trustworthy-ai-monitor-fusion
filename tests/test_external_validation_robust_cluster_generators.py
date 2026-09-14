import pytest

from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)


@pytest.mark.parametrize(
    "p,rho",
    [
        (0.02, 0.10),
        (0.05, 0.03),
        (0.05, 0.10),
        (0.10, 0.10),
    ],
)
def test_logit_normal_calibration_matches_response_scale_targets(p, rho):
    cal = calibrate_logit_normal(
        marginal_probability=p,
        response_scale_icc=rho,
    )
    assert cal.quadrature_mean == pytest.approx(p, abs=1e-8)
    assert cal.quadrature_icc == pytest.approx(rho, abs=1e-8)
    assert cal.random_intercept_sd > 0.0
