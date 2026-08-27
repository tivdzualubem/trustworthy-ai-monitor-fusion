from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
TEX = (
    ROOT
    / "paper/evaluation_measurement_current/main.tex"
).read_text(encoding="utf-8")


def test_paper_has_corrected_provenance_boundary():
    required = [
        "were historically evaluated and were included in the label audit",
        "not fresh confirmatory data",
        "No result in this report is presented as fresh confirmation",
    ]
    for phrase in required:
        assert phrase in TEX


def test_grouping_and_label_claims_are_decomposed():
    for phrase in [
        "Fixed-policy measurement & 0/75 & 1/75",
        "Retraining/reselection, fixed outer holdout & 6/75 & 4/75",
        "Full protocol & 19/75 & 14/75",
        "cannot be interpreted as isolated measurement effects",
    ]:
        assert phrase in TEX

    stale = [
        "grouping changes 19 of 75",
        "label choice changes 14 of 75",
        "Replacing singleton grouping with the frozen near-duplicate grouping changes 19 of 75",
    ]
    for phrase in stale:
        assert phrase not in TEX


def test_numerical_result_is_boundary_diagnostic_not_general_failure():
    required = [
        "boundary/finite-precision diagnostic",
        "0/5 route and 0/5 prediction flips",
        "6.12\\% route-ambiguity fraction must not be interpreted",
        "planned threshold-tie diagnostic",
    ]
    for phrase in required:
        assert phrase in TEX


def test_security_direction_is_not_overclaimed():
    for phrase in [
        "does not support a standalone security-paper direction",
        "generic capacity/admission-control condition",
        "does not establish a novel security theorem",
    ]:
        assert phrase in TEX


def test_next_study_boundary_matches_ordered_plan():
    required = [
        "After that diagnostic, the existing data will no longer be used for discovery",
        "genuinely fresh source- and time-separated examples",
        "independent multi-rater labels",
        "genuinely different monitor families",
        "sample-size/power calculation for a 5\\% FPR certificate",
        "no router retuning on confirmation data",
        "No transport result is claimed",
    ]
    for phrase in required:
        assert phrase in TEX


def test_readme_matches_current_direction():
    readme_lower = README.lower()
    required = [
        "development-only evaluation/measurement pilot",
        "risk-certificate transport",
        "threshold-tie diagnostic",
        "stop using the existing data for discovery",
        "preregistered fresh-data protocol",
        "do not run",
        "independent multi-rater labels",
        "genuinely different monitor families",
        "sample-size/power calculation for a 5% fpr certificate",
    ]
    for phrase in required:
        assert phrase in readme_lower

    current_work = README.split("## Current work", 1)[1].lower()
    assert "threshold-tie diagnostic" in current_work
    assert "stop using the existing data for discovery" in current_work
    assert "preregistered fresh-data protocol" in current_work
    assert "do not run" in current_work
    assert "router retuning" in current_work
    assert "confirmation data" in current_work
    assert (
        "no router retuning on confirmation data" in current_work
        or (
            "confirmation data must not be used" in current_work
            and "router retuning" in current_work
        )
    )
    assert "no fresh data should be collected or evaluated" in current_work
