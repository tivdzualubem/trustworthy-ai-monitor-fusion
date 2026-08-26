from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_states_real_historical_v2_status():
    required = [
        "summed measured component latencies",
        "1,687 effective groups",
        "not a demonstrated mechanically enforced runtime bound",
        "not externally justified",
        "not fully regenerable",
        "fresh calibration",
        "multi-rater labels",
        "historically evaluated and included in the label audit",
        "Pareto comparison",
    ]

    for phrase in required:
        assert phrase in README


def test_readme_does_not_present_router_as_selected():
    assert "no router was selected" in README.lower()

def test_readme_records_legacy_split_freshness_correctly():
    lower = README.lower()
    assert "not used in the development-only pilot" in lower
    assert "neither split is eligible as fresh confirmatory data" in lower
    assert "remain sealed" not in lower
    assert "remain unopened" not in lower


def test_readme_does_not_treat_19_75_or_14_75_as_isolated_effects():
    assert "Grouping full-protocol contrast (confounded)" in README
    assert "Label full-protocol contrast (confounded)" in README
    assert "full-protocol contrasts, not isolated grouping or label effects" in README
