from copy import deepcopy

import pytest

from monitor_fusion.external_validation.attack_transforms import select_transform
from monitor_fusion.external_validation.dataset_schema import (
    CELL_SPECS,
    DatasetSchemaError,
    expected_provenance_cluster_id,
    validate_candidate_record,
    validate_dataset,
    validate_labeled_record,
    validate_primary_pre_scoring_record,
)


def human_direct(example_id="e1", cell_id="A_opt"):
    window, source, attack = CELL_SPECS[cell_id]
    return {
        "example_id": example_id,
        "cell_id": cell_id,
        "window_id": window,
        "source": source,
        "attack_family": attack,
        "common_category_id": "C1",
        "base_intent_id": f"intent-{example_id}",
        "template_id": None,
        "author_id": f"author-{example_id}",
        "generator_id": None,
        "generator_batch_id": None,
        "attack_transform_id": None,
        "dependency_text": "canonical direct request",
        "prompt": "canonical direct request",
        "response": "target response",
    }


def model_obfuscated(example_id="m1", cell_id="SF"):
    window, source, attack = CELL_SPECS[cell_id]
    base_intent_id = f"intent-{example_id}"
    return {
        "example_id": example_id,
        "cell_id": cell_id,
        "window_id": window,
        "source": source,
        "attack_family": attack,
        "common_category_id": "C4",
        "base_intent_id": base_intent_id,
        "template_id": None,
        "author_id": None,
        "generator_id": "mistral-source",
        "generator_batch_id": "batch-01",
        "attack_transform_id": select_transform(base_intent_id),
        "dependency_text": "canonical pre-obfuscation request",
        "prompt": "transformed request",
        "response": "target response",
    }


def add_unanimous_labels(record):
    row = deepcopy(record)
    row.update(
        {
            "rater_1_id": "r1",
            "rater_2_id": "r2",
            "rater_3_id": "r3",
            "rater_1_y": 1,
            "rater_2_y": 1,
            "rater_3_y": 1,
            "y": 1,
            "adjudicator_id": None,
            "adjudicated_y": None,
            "unanimous_audit_selected": False,
            "auditor_id": None,
            "audit_y": None,
        }
    )
    return row


def add_primary_fields(record, group="g1", rep=True):
    row = deepcopy(record)
    row.update(
        {
            "dependency_group_id": group,
            "is_dependency_representative": rep,
            "provenance_cluster_id": expected_provenance_cluster_id(row),
            "historical_overlap_checked": True,
            "historical_overlap_found": False,
        }
    )
    return row


def test_valid_human_direct_candidate():
    validate_candidate_record(human_direct())


def test_valid_model_obfuscated_candidate():
    validate_candidate_record(model_obfuscated())


def test_cell_source_attack_mismatch_is_rejected():
    row = human_direct()
    row["source"] = "model_generated"
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)


def test_human_requires_author_and_forbids_generator_provenance():
    row = human_direct()
    row["author_id"] = None
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)

    row = human_direct()
    row["generator_id"] = "unexpected"
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)


def test_model_requires_generator_batch_and_forbids_author():
    row = model_obfuscated()
    row["generator_batch_id"] = None
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)

    row = model_obfuscated()
    row["author_id"] = "unexpected"
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)


def test_frozen_obfuscation_assignment_is_enforced():
    row = model_obfuscated()
    row["attack_transform_id"] = "O999"
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)


def test_monitor_outputs_are_forbidden_before_scoring():
    row = human_direct()
    row["shieldgemma_score"] = 0.7
    with pytest.raises(DatasetSchemaError):
        validate_candidate_record(row)


def test_primary_y_must_equal_three_rater_majority():
    row = add_unanimous_labels(human_direct())
    row["y"] = 0
    with pytest.raises(DatasetSchemaError):
        validate_labeled_record(row)


def test_2_1_disagreement_requires_independent_adjudication():
    row = add_unanimous_labels(human_direct())
    row.update(
        {
            "rater_3_y": 0,
            "adjudicator_id": None,
            "adjudicated_y": None,
        }
    )
    with pytest.raises(DatasetSchemaError):
        validate_labeled_record(row)

    row["adjudicator_id"] = "r4"
    row["adjudicated_y"] = 1
    validate_labeled_record(row)


def test_human_author_cannot_self_rate():
    row = add_unanimous_labels(human_direct())
    row["rater_1_id"] = row["author_id"]
    with pytest.raises(DatasetSchemaError):
        validate_labeled_record(row)


def test_selected_unanimous_audit_requires_independent_auditor():
    row = add_unanimous_labels(human_direct())
    row["unanimous_audit_selected"] = True
    with pytest.raises(DatasetSchemaError):
        validate_labeled_record(row)

    row["auditor_id"] = "r5"
    row["audit_y"] = 1
    validate_labeled_record(row)


def test_primary_record_requires_clean_historical_overlap_screen():
    row = add_primary_fields(add_unanimous_labels(human_direct()))
    validate_primary_pre_scoring_record(row)

    bad = deepcopy(row)
    bad["historical_overlap_found"] = True
    with pytest.raises(DatasetSchemaError):
        validate_primary_pre_scoring_record(bad)


def test_provenance_cluster_rule_is_enforced():
    row = add_primary_fields(add_unanimous_labels(model_obfuscated()))
    row["provenance_cluster_id"] = "model:wrong-batch"
    with pytest.raises(DatasetSchemaError):
        validate_primary_pre_scoring_record(row)


def test_dataset_requires_unique_example_ids():
    row = add_primary_fields(add_unanimous_labels(human_direct()))
    with pytest.raises(DatasetSchemaError):
        validate_dataset([row, deepcopy(row)], stage="primary_pre_scoring")


def test_dependency_group_cannot_cross_cells():
    first = add_primary_fields(
        add_unanimous_labels(human_direct("e1", "A_opt")),
        group="shared",
        rep=True,
    )
    second = add_primary_fields(
        add_unanimous_labels(human_direct("e2", "A_val")),
        group="shared",
        rep=False,
    )
    with pytest.raises(DatasetSchemaError):
        validate_dataset(
            [first, second],
            stage="primary_pre_scoring",
        )


def test_exactly_one_representative_per_dependency_group():
    first = add_primary_fields(
        add_unanimous_labels(human_direct("e1")),
        group="same",
        rep=True,
    )
    second = add_primary_fields(
        add_unanimous_labels(human_direct("e2")),
        group="same",
        rep=True,
    )
    with pytest.raises(DatasetSchemaError):
        validate_dataset(
            [first, second],
            stage="primary_pre_scoring",
        )


def test_valid_primary_dataset_summary():
    first = add_primary_fields(
        add_unanimous_labels(human_direct("e1")),
        group="g1",
        rep=True,
    )
    second = add_primary_fields(
        add_unanimous_labels(human_direct("e2")),
        group="g2",
        rep=True,
    )
    summary = validate_dataset(
        [first, second],
        stage="primary_pre_scoring",
    )
    assert summary.row_count == 2
    assert summary.unique_example_count == 2
    assert summary.dependency_group_count == 2
    assert summary.dependency_representative_count == 2
