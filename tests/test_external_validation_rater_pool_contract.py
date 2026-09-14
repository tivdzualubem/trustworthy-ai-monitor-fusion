import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_primary_and_review_pools_are_separate():
    c = load("external_validation_rater_pool_contract_v1.json")
    assert c["status"] == "role_separation_frozen_recruitment_ethics_pending"
    assert c["pools"]["review_rater_pool"]["must_be_disjoint_from_primary_rater_pool"] is True
    assert c["per_example_independence"]["three_primary_raters_must_be_distinct"] is True
    assert c["per_example_independence"]["fourth_rater_must_be_distinct_from_all_three_primary_raters"] is True


def test_primary_label_is_not_posthoc_replaced():
    c = load("external_validation_rater_pool_contract_v1.json")
    k = c["label_consequences"]
    assert k["primary_reference_label"] == "three_rater_majority_vote"
    assert k["two_to_one_adjudication_changes_primary_label"] is False
    assert k["unanimous_audit_changes_primary_label"] is False


def test_human_participation_remains_blocked_pending_ethics_and_consent():
    c = load("external_validation_rater_pool_contract_v1.json")
    gate = c["human_participation_gate"]
    assert gate["recruitment_started"] is False
    assert gate["unpaid_volunteer_use_automatically_authorized"] is False
    assert gate["W0_authorized_by_this_contract"] is False
    assert "confirm institutional ethics or review requirements" in gate["before_recruitment_required"]
    assert "freeze consent language" in gate["before_recruitment_required"]


def test_annotation_contract_points_to_rater_pool_contract():
    a = load("external_validation_annotation_contract_v1.json")
    assert a["rater_pool_contract"]["path"] == "configs/external_validation_rater_pool_contract_v1.json"
    assert a["rater_pool_contract"]["primary_and_review_pools_disjoint"] is True
    assert a["disagreement"]["primary_label_changes_after_adjudication"] is False
    assert a["unanimous_audit"]["primary_label_changes_after_audit"] is False
