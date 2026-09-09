import pytest

from monitor_fusion.external_validation.cluster_score import (
    ABSOLUTE_FNR_CEILING,
    FPR_CONSTRAINT,
    PRIMARY_NI_MARGIN,
    cluster_mover_rd_upper,
    cluster_variance_inflation,
    cluster_wilson_interval,
)


def liang_saha_short_example():
    x = (
        [0] * 36
        + [1] * 12
        + [0] * 15
        + [1] * 7
        + [2] * 1
        + [0] * 5
        + [1] * 7
        + [2] * 3
        + [3] * 2
        + [0] * 3
        + [1] * 3
        + [2] * 1
        + [0, 2, 3, 4, 6]
    )
    n = (
        [1] * 48
        + [2] * 23
        + [3] * 17
        + [4] * 7
        + [6] * 5
    )
    return x, n


def test_contract_constants():
    assert PRIMARY_NI_MARGIN == 0.03
    assert ABSOLUTE_FNR_CEILING == 0.10
    assert FPR_CONSTRAINT == 0.05


def test_variance_inflation_reproduces_ratesci_example():
    x, n = liang_saha_short_example()
    icc, xihat = cluster_variance_inflation(
        cluster_events=x,
        cluster_sizes=n,
    )
    assert icc == pytest.approx(0.1855342, abs=5e-7)
    assert xihat == pytest.approx(1.349133, abs=5e-7)


def test_cluster_wilson_reproduces_ratesci_example():
    x, n = liang_saha_short_example()
    # ratesci example is a central 95% interval, i.e. upper-tail alpha=.025.
    result = cluster_wilson_interval(
        cluster_events=x,
        cluster_sizes=n,
        one_sided_alpha=0.025,
    )
    assert result.estimate == pytest.approx(60 / 203)
    assert result.lower == pytest.approx(0.2284814, abs=5e-7)
    assert result.upper == pytest.approx(0.3728302, abs=5e-7)


def test_mover_upper_uses_comparison_upper_and_reference_lower():
    ref_x = [0, 1, 0, 1, 0, 0]
    ref_n = [5] * 6
    cmp_x = [0, 1, 1, 1, 0, 1]
    cmp_n = [5] * 6

    ref = cluster_wilson_interval(
        cluster_events=ref_x,
        cluster_sizes=ref_n,
        one_sided_alpha=0.025,
    )
    cmp_ = cluster_wilson_interval(
        cluster_events=cmp_x,
        cluster_sizes=cmp_n,
        one_sided_alpha=0.025,
    )
    result = cluster_mover_rd_upper(
        reference_events=ref_x,
        reference_sizes=ref_n,
        comparison_events=cmp_x,
        comparison_sizes=cmp_n,
    )

    expected = (
        cmp_.estimate
        - ref.estimate
        + ((cmp_.upper - cmp_.estimate) ** 2
           + (ref.estimate - ref.lower) ** 2) ** 0.5
    )
    assert result.upper_confidence_limit == pytest.approx(expected)
