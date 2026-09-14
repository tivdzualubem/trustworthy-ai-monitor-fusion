from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from monitor_fusion.external_validation.attack_transforms import select_transform


PROTOCOL_ID = "safety_monitor_external_validation_preregistration_v1"

CELL_SPECS = {
    "A_opt": ("W0", "human", "direct"),
    "A_val": ("W0", "human", "direct"),
    "T": ("W1", "human", "direct"),
    "S": ("W1", "model_generated", "direct"),
    "F": ("W1", "human", "obfuscated"),
    "SF": ("W1", "model_generated", "obfuscated"),
}

COMMON_CATEGORY_IDS = frozenset(
    {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}
)

CANDIDATE_FIELDS = (
    "example_id",
    "cell_id",
    "window_id",
    "source",
    "attack_family",
    "common_category_id",
    "base_intent_id",
    "template_id",
    "author_id",
    "generator_id",
    "generator_batch_id",
    "attack_transform_id",
    "dependency_text",
    "prompt",
    "response",
)

ANNOTATION_FIELDS = (
    "rater_1_id",
    "rater_2_id",
    "rater_3_id",
    "rater_1_y",
    "rater_2_y",
    "rater_3_y",
    "y",
    "adjudicator_id",
    "adjudicated_y",
    "unanimous_audit_selected",
    "auditor_id",
    "audit_y",
)

PRIMARY_FIELDS = (
    "dependency_group_id",
    "is_dependency_representative",
    "provenance_cluster_id",
    "historical_overlap_checked",
    "historical_overlap_found",
)

KNOWN_MONITOR_FIELD_FRAGMENTS = (
    "llama_guard",
    "llamaguard",
    "shieldgemma",
    "shield_gemma",
    "granite_guardian",
    "graniteguardian",
    "monitor_score",
    "monitor_decision",
    "monitor_output",
)


class DatasetSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetValidationSummary:
    stage: str
    row_count: int
    unique_example_count: int
    dependency_group_count: int | None
    dependency_representative_count: int | None


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_string_is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _require_fields(record: Mapping[str, Any], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise DatasetSchemaError(
            f"missing required fields: {sorted(missing)}"
        )


def _require_binary(value: Any, field: str) -> int:
    if isinstance(value, bool):
        value = int(value)
    if value not in (0, 1):
        raise DatasetSchemaError(f"{field} must be 0 or 1")
    return int(value)


def _assert_no_monitor_outputs(record: Mapping[str, Any]) -> None:
    for key in record:
        normalized = str(key).strip().lower()
        if any(fragment in normalized for fragment in KNOWN_MONITOR_FIELD_FRAGMENTS):
            raise DatasetSchemaError(
                f"study monitor output field is forbidden before scoring: {key}"
            )


def validate_candidate_record(record: Mapping[str, Any]) -> None:
    _require_fields(record, CANDIDATE_FIELDS)
    _assert_no_monitor_outputs(record)

    for field in (
        "example_id",
        "cell_id",
        "window_id",
        "source",
        "attack_family",
        "common_category_id",
        "base_intent_id",
        "dependency_text",
        "prompt",
        "response",
    ):
        if not _is_nonempty_string(record[field]):
            raise DatasetSchemaError(f"{field} must be a non-empty string")

    cell_id = record["cell_id"]
    if cell_id not in CELL_SPECS:
        raise DatasetSchemaError(f"unknown study cell: {cell_id}")

    expected_window, expected_source, expected_attack = CELL_SPECS[cell_id]

    if record["window_id"] != expected_window:
        raise DatasetSchemaError(
            f"cell {cell_id} requires window {expected_window}"
        )
    if record["source"] != expected_source:
        raise DatasetSchemaError(
            f"cell {cell_id} requires source {expected_source}"
        )
    if record["attack_family"] != expected_attack:
        raise DatasetSchemaError(
            f"cell {cell_id} requires attack_family {expected_attack}"
        )

    if record["common_category_id"] not in COMMON_CATEGORY_IDS:
        raise DatasetSchemaError(
            f"unknown common_category_id: {record['common_category_id']}"
        )

    source = record["source"]
    if source == "human":
        if not _is_nonempty_string(record["author_id"]):
            raise DatasetSchemaError("human source requires author_id")
        if not _optional_string_is_empty(record["generator_id"]):
            raise DatasetSchemaError(
                "human source must not carry generator_id"
            )
        if not _optional_string_is_empty(record["generator_batch_id"]):
            raise DatasetSchemaError(
                "human source must not carry generator_batch_id"
            )
    else:
        if not _is_nonempty_string(record["generator_id"]):
            raise DatasetSchemaError(
                "model_generated source requires generator_id"
            )
        if not _is_nonempty_string(record["generator_batch_id"]):
            raise DatasetSchemaError(
                "model_generated source requires generator_batch_id"
            )
        if not _optional_string_is_empty(record["author_id"]):
            raise DatasetSchemaError(
                "model_generated source must not carry author_id"
            )

    attack_family = record["attack_family"]
    transform_id = record["attack_transform_id"]

    if attack_family == "direct":
        if not _optional_string_is_empty(transform_id):
            raise DatasetSchemaError(
                "direct rows must not carry an obfuscation transform"
            )
    else:
        expected_transform = select_transform(record["base_intent_id"])
        if transform_id != expected_transform:
            raise DatasetSchemaError(
                "obfuscated row transform does not match frozen "
                "base_intent_id assignment"
            )


def validate_labeled_record(record: Mapping[str, Any]) -> None:
    validate_candidate_record(record)
    _require_fields(record, ANNOTATION_FIELDS)

    rater_ids = [
        record["rater_1_id"],
        record["rater_2_id"],
        record["rater_3_id"],
    ]
    if not all(_is_nonempty_string(value) for value in rater_ids):
        raise DatasetSchemaError("all three rater IDs must be non-empty")
    if len(set(rater_ids)) != 3:
        raise DatasetSchemaError("the three primary raters must be distinct")

    if record["source"] == "human" and record["author_id"] in rater_ids:
        raise DatasetSchemaError("human author cannot rate their own example")

    labels = [
        _require_binary(record["rater_1_y"], "rater_1_y"),
        _require_binary(record["rater_2_y"], "rater_2_y"),
        _require_binary(record["rater_3_y"], "rater_3_y"),
    ]
    majority = int(sum(labels) >= 2)
    observed_y = _require_binary(record["y"], "y")
    if observed_y != majority:
        raise DatasetSchemaError("y must equal the three-rater majority")

    unanimous = len(set(labels)) == 1

    adjudicator_id = record["adjudicator_id"]
    adjudicated_y = record["adjudicated_y"]

    if unanimous:
        if not _optional_string_is_empty(adjudicator_id):
            raise DatasetSchemaError(
                "unanimous primary rating must not have adjudicator_id"
            )
        if adjudicated_y is not None:
            raise DatasetSchemaError(
                "unanimous primary rating must not have adjudicated_y"
            )
    else:
        if not _is_nonempty_string(adjudicator_id):
            raise DatasetSchemaError(
                "every 2-1 disagreement requires adjudicator_id"
            )
        if adjudicator_id in rater_ids:
            raise DatasetSchemaError(
                "adjudicator must be independent of the three primary raters"
            )
        if record["source"] == "human" and adjudicator_id == record["author_id"]:
            raise DatasetSchemaError(
                "human author cannot adjudicate their own example"
            )
        _require_binary(adjudicated_y, "adjudicated_y")

    selected = record["unanimous_audit_selected"]
    if not isinstance(selected, bool):
        raise DatasetSchemaError(
            "unanimous_audit_selected must be boolean"
        )

    auditor_id = record["auditor_id"]
    audit_y = record["audit_y"]

    if selected:
        if not unanimous:
            raise DatasetSchemaError(
                "only unanimous 3-0 cases may be selected for unanimous audit"
            )
        if not _is_nonempty_string(auditor_id):
            raise DatasetSchemaError(
                "selected unanimous audit requires auditor_id"
            )
        if auditor_id in rater_ids:
            raise DatasetSchemaError(
                "auditor must be independent of the three primary raters"
            )
        if record["source"] == "human" and auditor_id == record["author_id"]:
            raise DatasetSchemaError(
                "human author cannot audit their own example"
            )
        _require_binary(audit_y, "audit_y")
    else:
        if not _optional_string_is_empty(auditor_id):
            raise DatasetSchemaError(
                "non-selected unanimous audit must not have auditor_id"
            )
        if audit_y is not None:
            raise DatasetSchemaError(
                "non-selected unanimous audit must not have audit_y"
            )


def expected_provenance_cluster_id(record: Mapping[str, Any]) -> str:
    if record["source"] == "human":
        return f"human:{record['author_id']}"
    return f"model:{record['generator_batch_id']}"


def validate_primary_pre_scoring_record(record: Mapping[str, Any]) -> None:
    validate_labeled_record(record)
    _require_fields(record, PRIMARY_FIELDS)

    if not _is_nonempty_string(record["dependency_group_id"]):
        raise DatasetSchemaError(
            "dependency_group_id must be a non-empty string"
        )

    if not isinstance(record["is_dependency_representative"], bool):
        raise DatasetSchemaError(
            "is_dependency_representative must be boolean"
        )

    expected_cluster = expected_provenance_cluster_id(record)
    if record["provenance_cluster_id"] != expected_cluster:
        raise DatasetSchemaError(
            "provenance_cluster_id does not match frozen source rule"
        )

    if record["historical_overlap_checked"] is not True:
        raise DatasetSchemaError(
            "historical overlap screening must be completed"
        )

    if record["historical_overlap_found"] is not False:
        raise DatasetSchemaError(
            "historical-overlap examples are ineligible"
        )


def validate_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str,
) -> DatasetValidationSummary:
    rows = list(records)
    if not rows:
        raise DatasetSchemaError("dataset must contain at least one row")

    validators = {
        "candidate": validate_candidate_record,
        "labeled": validate_labeled_record,
        "primary_pre_scoring": validate_primary_pre_scoring_record,
    }
    if stage not in validators:
        raise DatasetSchemaError(f"unknown dataset stage: {stage}")

    validator = validators[stage]
    for record in rows:
        validator(record)

    example_ids = [record["example_id"] for record in rows]
    if len(set(example_ids)) != len(example_ids):
        raise DatasetSchemaError("example_id values must be unique")

    if stage != "primary_pre_scoring":
        return DatasetValidationSummary(
            stage=stage,
            row_count=len(rows),
            unique_example_count=len(set(example_ids)),
            dependency_group_count=None,
            dependency_representative_count=None,
        )

    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for record in rows:
        by_group.setdefault(record["dependency_group_id"], []).append(record)

    representative_count = 0
    for group_id, group_rows in by_group.items():
        cells = {record["cell_id"] for record in group_rows}
        if len(cells) != 1:
            raise DatasetSchemaError(
                f"dependency group crosses study cells: {group_id}"
            )

        reps = [
            record for record in group_rows
            if record["is_dependency_representative"]
        ]
        if len(reps) != 1:
            raise DatasetSchemaError(
                f"dependency group must have exactly one representative: "
                f"{group_id}"
            )
        representative_count += 1

    return DatasetValidationSummary(
        stage=stage,
        row_count=len(rows),
        unique_example_count=len(set(example_ids)),
        dependency_group_count=len(by_group),
        dependency_representative_count=representative_count,
    )
