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
        "protected legacy `final_test`",
        "Pareto comparison",
    ]

    for phrase in required:
        assert phrase in README


def test_readme_does_not_present_router_as_selected():
    assert "no router was selected" in README.lower()
