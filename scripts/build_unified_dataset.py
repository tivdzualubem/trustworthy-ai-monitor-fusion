from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]

XSTEST_ROOT = ROOT / "data/raw/sources/xstest"
SOURCE_AUDIT = ROOT / "data/metadata/source_audit.json"

OUTPUT_PARQUET = ROOT / "data/processed/unified_dataset.parquet"
OUTPUT_CSV = ROOT / "data/processed/unified_dataset.csv"
MANIFEST_PATH = ROOT / "data/metadata/unified_dataset_manifest.json"

JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "judge_comparison"

WILDGUARD_REPO = "allenai/wildguardmix"
WILDGUARD_CONFIG = "wildguardtest"

PROCESSING_VERSION = "0.1.0"
HELD_OUT_ATTACK_FAMILY = "GCG"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_dataframe(loaded: Dataset | DatasetDict) -> pd.DataFrame:
    if isinstance(loaded, Dataset):
        return loaded.to_pandas()

    if isinstance(loaded, DatasetDict):
        datasets = list(loaded.values())

        if not datasets:
            raise ValueError("DatasetDict contains no splits.")

        return concatenate_datasets(datasets).to_pandas()

    raise TypeError(
        f"Unsupported dataset object: {type(loaded).__name__}"
    )


def map_binary_label(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not pd.isna(value):
        integer = int(value)

        if integer in {0, 1}:
            return integer

    normalized = str(value).strip().lower()

    false_values = {
        "0",
        "false",
        "safe",
        "unharmful",
        "benign",
        "no",
    }
    true_values = {
        "1",
        "true",
        "harmful",
        "unsafe",
        "yes",
    }

    if normalized in false_values:
        return 0

    if normalized in true_values:
        return 1

    raise ValueError(
        f"Cannot map {field_name} value {value!r} to 0 or 1."
    )


def jbb_attack_family(index: int) -> str:
    if 0 <= index <= 99:
        return "PAIR"

    if 100 <= index <= 149:
        return "GCG"

    if 150 <= index <= 199:
        return "random_search"

    raise ValueError(
        f"JailbreakBench adversarial index outside 0--199: {index}"
    )


def jbb_target_model(index: int) -> str:
    if 0 <= index <= 149:
        return "vicuna-13b-v1.5"

    if 150 <= index <= 199:
        return "mixed_random_search_target_models"

    raise ValueError(
        f"JailbreakBench adversarial index outside 0--199: {index}"
    )


def build_jailbreakbench() -> tuple[pd.DataFrame, str]:
    audit = json.loads(
        SOURCE_AUDIT.read_text(encoding="utf-8")
    )

    revision = audit[
        "sources"
    ]["jailbreakbench"]["revision"]

    loaded = load_dataset(
        JBB_REPO,
        JBB_CONFIG,
        revision=revision,
    )
    frame = as_dataframe(loaded)

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    required = {
        "index",
        "goal",
        "prompt",
        "target_response",
        "human_majority",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "JailbreakBench is missing columns: "
            f"{sorted(missing)}"
        )

    if len(frame) != 300:
        raise ValueError(
            "Expected 300 JailbreakBench rows, "
            f"found {len(frame)}."
        )

    frame["index"] = frame["index"].astype(int)

    frame = frame.loc[
        frame["index"].between(0, 199)
    ].copy()

    if len(frame) != 200:
        raise ValueError(
            "Expected 200 adversarial JBB rows, "
            f"found {len(frame)}."
        )

    frame["attack_family"] = frame["index"].map(
        jbb_attack_family
    )

    expected_family_counts = {
        "PAIR": 100,
        "GCG": 50,
        "random_search": 50,
    }

    actual_family_counts = (
        frame["attack_family"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if actual_family_counts != expected_family_counts:
        raise ValueError(
            "Unexpected JBB attack-family counts: "
            f"{actual_family_counts}"
        )

    records: list[dict[str, Any]] = []
    retrieved_at = utc_now()

    for row in frame.itertuples(index=False):
        y = map_binary_label(
            row.human_majority,
            "human_majority",
        )

        records.append(
            {
                "example_id": (
                    f"jbb-{int(row.index):03d}"
                ),
                "prompt": str(row.prompt),
                "response": str(row.target_response),
                "y": y,
                "source_dataset": (
                    "jailbreakbench_judge_comparison"
                ),
                "source_record_id": str(
                    int(row.index)
                ),
                "prompt_harmful": 1,
                "response_refusal": pd.NA,
                "jailbreak_success": y,
                "over_refusal": pd.NA,
                "harm_category": pd.NA,
                "attack_family": str(
                    row.attack_family
                ),
                "toxicity_category": pd.NA,
                "original_label": str(
                    row.human_majority
                ),
                "annotation_source": (
                    "JailbreakBench human-majority "
                    "response evaluation"
                ),
                "annotation_notes": (
                    "For this adversarial subset only, "
                    "the human-majority success label is "
                    "used as a source-specific proxy for "
                    "whether the response materially "
                    "satisfies the harmful goal. "
                    "Jailbreak success remains stored as "
                    "a separate auxiliary variable."
                ),
                "goal": str(row.goal),
                "target_model": jbb_target_model(
                    int(row.index)
                ),
                "target_model_revision": pd.NA,
                "generation_template": pd.NA,
                "temperature": pd.NA,
                "top_p": pd.NA,
                "max_new_tokens": pd.NA,
                "seed": pd.NA,
                "source_url": JBB_REPO,
                "source_license": "MIT",
                "retrieved_at": retrieved_at,
                "processing_version": (
                    PROCESSING_VERSION
                ),
            }
        )

    return pd.DataFrame(records), revision




def build_xstest() -> tuple[pd.DataFrame, str]:
    prompts_path = XSTEST_ROOT / "xstest_prompts.csv"
    completions_path = (
        XSTEST_ROOT
        / "model_completions"
        / "xstest_v2_completions_gpt4.csv"
    )

    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    if not completions_path.exists():
        raise FileNotFoundError(completions_path)

    prompts = pd.read_csv(prompts_path)
    completions = pd.read_csv(completions_path)

    # XSTest prompt IDs are integers such as 1, while the completion
    # files use strings such as "v2-1". Create a normalized merge key
    # without altering either source's original ID.
    prompts["merge_id"] = (
        prompts["id"]
        .astype("string")
        .str.strip()
    )
    completions["merge_id"] = (
        completions["id"]
        .astype("string")
        .str.strip()
        .str.replace(r"^v2-", "", regex=True)
    )

    if prompts["merge_id"].isna().any():
        raise ValueError("XSTest prompts contain a missing merge ID.")

    if completions["merge_id"].isna().any():
        raise ValueError("XSTest completions contain a missing merge ID.")

    if prompts["merge_id"].duplicated().any():
        raise ValueError("XSTest prompt merge IDs are not unique.")

    if completions["merge_id"].duplicated().any():
        raise ValueError("XSTest completion merge IDs are not unique.")

    prompt_ids = set(prompts["merge_id"])
    completion_ids = set(completions["merge_id"])

    if prompt_ids != completion_ids:
        raise ValueError(
            "XSTest prompt and completion ID sets differ. "
            f"Only in prompts: {sorted(prompt_ids - completion_ids)[:10]}; "
            f"only in completions: "
            f"{sorted(completion_ids - prompt_ids)[:10]}"
        )

    completion_columns = [
        "merge_id",
        "completion",
        "annotation_1",
        "annotation_2",
        "agreement",
        "final_label",
    ]

    missing = set(completion_columns).difference(
        completions.columns
    )
    if missing:
        raise ValueError(
            f"XSTest completion file is missing: {sorted(missing)}"
        )

    merged = prompts.merge(
        completions[completion_columns],
        on="merge_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 450:
        raise ValueError(
            f"Expected 450 merged XSTest rows, found {len(merged)}."
        )

    safe = merged.loc[
        merged["label"].astype(str).str.lower() == "safe"
    ].copy()

    if len(safe) != 250:
        raise ValueError(
            f"Expected 250 safe XSTest rows, found {len(safe)}."
        )

    refusal_labels = {
        "2_full_refusal",
        "3_partial_refusal",
    }

    records: list[dict[str, Any]] = []
    retrieved_at = utc_now()

    for row in safe.itertuples(index=False):
        final_label = str(row.final_label)
        response_refusal = int(
            final_label in refusal_labels
        )

        records.append(
            {
                "example_id": f"xstest-{int(row.id):03d}",
                "prompt": str(row.prompt),
                "response": str(row.completion),
                "y": 0,
                "source_dataset": "xstest_safe_gpt4",
                "source_record_id": str(int(row.id)),
                "prompt_harmful": 0,
                "response_refusal": response_refusal,
                "jailbreak_success": pd.NA,
                "over_refusal": response_refusal,
                "harm_category": (
                    str(row.focus)
                    if not pd.isna(row.focus)
                    else pd.NA
                ),
                "attack_family": pd.NA,
                "toxicity_category": pd.NA,
                "original_label": final_label,
                "annotation_source": (
                    "XSTest safe-prompt label and manual "
                    "response refusal annotation"
                ),
                "annotation_notes": (
                    "Used only as benign/over-refusal data. "
                    "A refusal does not change Y from 0."
                ),
                "goal": pd.NA,
                "target_model": "gpt-4",
                "target_model_revision": (
                    "repository-provided completion"
                ),
                "generation_template": pd.NA,
                "temperature": pd.NA,
                "top_p": pd.NA,
                "max_new_tokens": pd.NA,
                "seed": pd.NA,
                "source_url": (
                    "https://github.com/paul-rottger/xstest"
                ),
                "source_license": "CC-BY-4.0",
                "retrieved_at": retrieved_at,
                "processing_version": PROCESSING_VERSION,
            }
        )

    audit = json.loads(
        SOURCE_AUDIT.read_text(encoding="utf-8")
    )
    revision = audit["sources"]["xstest"]["revision"]

    return pd.DataFrame(records), revision


def build_wildguard() -> tuple[pd.DataFrame, str]:
    audit = json.loads(
        SOURCE_AUDIT.read_text(encoding="utf-8")
    )
    revision = audit["sources"]["wildguardmix"]["revision"]

    dataset = load_dataset(
        WILDGUARD_REPO,
        WILDGUARD_CONFIG,
        split="test",
        revision=revision,
    )
    frame = dataset.to_pandas()

    required_columns = {
        "prompt",
        "response",
        "adversarial",
        "prompt_harm_label",
        "response_refusal_label",
        "response_harm_label",
        "subcategory",
    }
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "WildGuardTest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame.loc[
        frame["prompt"].notna()
        & frame["response"].notna()
        & frame["response_harm_label"].isin(
            ["harmful", "unharmful"]
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No labelled WildGuardTest prompt-response pairs remain."
        )

    records: list[dict[str, Any]] = []
    retrieved_at = utc_now()

    for source_index, row in frame.iterrows():
        response_harm_label = str(
            row["response_harm_label"]
        ).strip().lower()

        y = int(response_harm_label == "harmful")

        prompt_harm_label = row["prompt_harm_label"]

        if prompt_harm_label == "harmful":
            prompt_harmful: int | Any = 1
        elif prompt_harm_label == "unharmful":
            prompt_harmful = 0
        else:
            prompt_harmful = pd.NA

        refusal_label = row["response_refusal_label"]

        if refusal_label == "refusal":
            response_refusal: int | Any = 1
        elif refusal_label == "compliance":
            response_refusal = 0
        else:
            response_refusal = pd.NA

        adversarial = bool(row["adversarial"])

        jailbreak_success: int | Any = pd.NA

        if (
            adversarial
            and not pd.isna(prompt_harmful)
            and int(prompt_harmful) == 1
        ):
            jailbreak_success = y

        over_refusal: int | Any = pd.NA

        if (
            not pd.isna(prompt_harmful)
            and int(prompt_harmful) == 0
            and not pd.isna(response_refusal)
        ):
            over_refusal = int(
                int(response_refusal) == 1
            )

        subcategory = row["subcategory"]

        if pd.isna(subcategory):
            harm_category: str | Any = pd.NA
        else:
            harm_category = str(subcategory)

        toxicity_category: str | Any = pd.NA

        if harm_category == "toxic_language_hate_speech":
            toxicity_category = (
                "toxic_language_hate_speech"
            )

        records.append(
            {
                "example_id": (
                    f"wildguard-{int(source_index):04d}"
                ),
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "y": y,
                "source_dataset": "wildguardtest",
                "source_record_id": str(
                    int(source_index)
                ),
                "prompt_harmful": prompt_harmful,
                "response_refusal": response_refusal,
                "jailbreak_success": jailbreak_success,
                "over_refusal": over_refusal,
                "harm_category": harm_category,
                "attack_family": pd.NA,
                "toxicity_category": toxicity_category,
                "original_label": response_harm_label,
                "annotation_source": (
                    "WildGuardTest human "
                    "response-harm label"
                ),
                "annotation_notes": (
                    "Prompt harmfulness and response "
                    "refusal remain separate from Y."
                ),
                "goal": pd.NA,
                "target_model": "mixed_or_unspecified",
                "target_model_revision": pd.NA,
                "generation_template": pd.NA,
                "temperature": pd.NA,
                "top_p": pd.NA,
                "max_new_tokens": pd.NA,
                "seed": pd.NA,
                "source_url": WILDGUARD_REPO,
                "source_license": "ODC-BY",
                "retrieved_at": retrieved_at,
                "processing_version": (
                    PROCESSING_VERSION
                ),
            }
        )

    result = pd.DataFrame(records)

    if not set(result["y"].astype(int)) <= {0, 1}:
        raise ValueError(
            "WildGuard primary labels contain "
            "values outside 0 and 1."
        )

    return result, revision




def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    result["normalized_prompt"] = result["prompt"].map(
        normalize_text
    )
    result["normalized_response"] = result[
        "response"
    ].map(normalize_text)

    if (result["normalized_prompt"] == "").any():
        raise ValueError(
            "Dataset contains an empty prompt."
        )

    if (result["normalized_response"] == "").any():
        raise ValueError(
            "Dataset contains an empty response."
        )

    result["group_id"] = result[
        "normalized_prompt"
    ].map(hash_text)

    result["pair_id"] = (
        result["normalized_prompt"]
        + "\n"
        + result["normalized_response"]
    ).map(hash_text)

    conflicting_pairs = (
        result.groupby("pair_id")["y"]
        .nunique()
        .loc[lambda values: values > 1]
    )

    if not conflicting_pairs.empty:
        raise ValueError(
            "Identical prompt-response pairs have "
            "conflicting Y labels: "
            f"{len(conflicting_pairs)} pairs."
        )

    source_priority = {
        "jailbreakbench_judge_comparison": 0,
        "xstest_safe_gpt4": 1,
        "wildguardtest": 2,
    }

    result["_source_priority"] = result[
        "source_dataset"
    ].map(source_priority)

    if result["_source_priority"].isna().any():
        unknown_sources = sorted(
            result.loc[
                result["_source_priority"].isna(),
                "source_dataset",
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(
            f"Unknown source datasets: {unknown_sources}"
        )

    before_deduplication = len(result)

    result = (
        result.sort_values(
            ["_source_priority", "example_id"]
        )
        .drop_duplicates("pair_id", keep="first")
        .drop(columns=["_source_priority"])
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_deduplication - len(result)
    )

    held_out_mask = (
        result["attack_family"]
        .fillna("")
        .astype(str)
        .eq(HELD_OUT_ATTACK_FAMILY)
    )

    held_out_groups = set(
        result.loc[
            held_out_mask,
            "group_id",
        ].astype(str)
    )

    if not held_out_groups:
        raise ValueError(
            "No examples were found for held-out "
            f"attack family {HELD_OUT_ATTACK_FAMILY!r}."
        )

    result["split"] = pd.NA

    result.loc[
        result["group_id"]
        .astype(str)
        .isin(held_out_groups),
        "split",
    ] = "held_out_shift"

    remaining = result.loc[
        result["split"].isna()
    ].copy()

    group_rows: list[dict[str, str]] = []

    for group_id, group in remaining.groupby(
        "group_id",
        sort=True,
    ):
        sources = ",".join(
            sorted(
                group["source_dataset"]
                .astype(str)
                .unique()
            )
        )

        labels = ",".join(
            str(value)
            for value in sorted(
                group["y"].astype(int).unique()
            )
        )

        group_rows.append(
            {
                "group_id": str(group_id),
                "stratum": (
                    f"{sources}|y={labels}"
                ),
                "order_key": hash_text(
                    f"split-seed-2026|{group_id}"
                ),
            }
        )

    if not group_rows:
        raise ValueError(
            "No in-distribution groups remain "
            "after reserving the shift split."
        )

    groups = pd.DataFrame(group_rows)

    split_order = [
        "policy_train",
        "policy_selection",
        "calibration",
        "final_test",
    ]
    proportions = [0.40, 0.20, 0.20, 0.20]

    group_to_split: dict[str, str] = {}

    for _, stratum_groups in groups.groupby(
        "stratum",
        sort=True,
    ):
        ordered = (
            stratum_groups
            .sort_values("order_key")
            .reset_index(drop=True)
        )

        number_of_groups = len(ordered)

        boundaries = [
            round(
                number_of_groups
                * proportions[0]
            ),
            round(
                number_of_groups
                * sum(proportions[:2])
            ),
            round(
                number_of_groups
                * sum(proportions[:3])
            ),
            number_of_groups,
        ]

        start = 0

        for split_name, end in zip(
            split_order,
            boundaries,
            strict=True,
        ):
            selected_group_ids = ordered.iloc[
                start:end
            ]["group_id"]

            for selected_group_id in selected_group_ids:
                group_to_split[
                    str(selected_group_id)
                ] = split_name

            start = end

    remaining_mask = result["split"].isna()

    result.loc[
        remaining_mask,
        "split",
    ] = (
        result.loc[
            remaining_mask,
            "group_id",
        ]
        .astype(str)
        .map(group_to_split)
    )

    if result["split"].isna().any():
        unresolved = result.loc[
            result["split"].isna(),
            ["example_id", "group_id"],
        ]
        raise ValueError(
            "Some examples were not assigned to a "
            f"split: {unresolved.head().to_dict('records')}"
        )

    allowed_splits = {
        "policy_train",
        "policy_selection",
        "calibration",
        "final_test",
        "held_out_shift",
    }

    observed_splits = set(
        result["split"].astype(str)
    )

    if observed_splits != allowed_splits:
        raise ValueError(
            "Unexpected split set. "
            f"Expected {sorted(allowed_splits)}, "
            f"observed {sorted(observed_splits)}."
        )

    group_split_counts = (
        result.groupby("group_id")["split"]
        .nunique()
    )

    if (group_split_counts > 1).any():
        raise ValueError(
            "At least one normalized prompt group "
            "appears in multiple splits."
        )

    gcg_outside_shift = (
        result.loc[
            result["split"] != "held_out_shift",
            "attack_family",
        ]
        .fillna("")
        .astype(str)
        .eq(HELD_OUT_ATTACK_FAMILY)
        .any()
    )

    if gcg_outside_shift:
        raise ValueError(
            f"{HELD_OUT_ATTACK_FAMILY} leaked into "
            "an in-distribution split."
        )

    result = result.drop(
        columns=[
            "normalized_prompt",
            "normalized_response",
        ]
    )

    result.attrs["duplicates_removed"] = (
        duplicates_removed
    )

    return result




def build_manifest(
    frame: pd.DataFrame,
    revisions: dict[str, str],
) -> dict[str, Any]:
    split_counts = (
        frame["split"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    source_counts = (
        frame["source_dataset"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    label_counts = (
        frame["y"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    split_label_counts = (
        frame.groupby(["split", "y"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    split_label_dict = {
        str(split): {
            str(label): int(count)
            for label, count in row.items()
        }
        for split, row in split_label_counts.iterrows()
    }

    attack_counts = (
        frame.loc[
            frame["attack_family"].notna(),
            "attack_family",
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "created_at": utc_now(),
        "processing_version": PROCESSING_VERSION,
        "prediction_unit": "prompt_response_pair",
        "primary_target": (
            "response_should_be_intercepted"
        ),
        "selected_datasets": [
            "JailbreakBench judge_comparison adversarial subset",
            "XSTest safe subset with GPT-4 completions",
            "WildGuardTest",
        ],
        "source_revisions": revisions,
        "held_out_attack_family": (
            HELD_OUT_ATTACK_FAMILY
        ),
        "total_rows": int(len(frame)),
        "duplicates_removed": int(
            frame.attrs.get("duplicates_removed", 0)
        ),
        "source_counts": {
            str(key): int(value)
            for key, value in source_counts.items()
        },
        "label_counts": {
            str(key): int(value)
            for key, value in label_counts.items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in split_counts.items()
        },
        "split_label_counts": split_label_dict,
        "attack_family_counts": {
            str(key): int(value)
            for key, value in attack_counts.items()
        },
        "outputs": {
            "parquet": str(
                OUTPUT_PARQUET.relative_to(ROOT)
            ),
            "csv": str(
                OUTPUT_CSV.relative_to(ROOT)
            ),
        },
    }


def main() -> None:
    jbb, jbb_revision = build_jailbreakbench()
    xstest, xstest_revision = build_xstest()
    wildguard, wildguard_revision = build_wildguard()

    unified = pd.concat(
        [jbb, xstest, wildguard],
        ignore_index=True,
        sort=False,
    )

    unified = assign_splits(unified)

    ordered_columns = [
        "example_id",
        "prompt",
        "response",
        "y",
        "source_dataset",
        "source_record_id",
        "split",
        "group_id",
        "pair_id",
        "prompt_harmful",
        "response_refusal",
        "jailbreak_success",
        "over_refusal",
        "harm_category",
        "attack_family",
        "toxicity_category",
        "original_label",
        "annotation_source",
        "annotation_notes",
        "goal",
        "target_model",
        "target_model_revision",
        "generation_template",
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
        "source_url",
        "source_license",
        "retrieved_at",
        "processing_version",
    ]

    unified = unified[ordered_columns]

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )
    unified.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    manifest = build_manifest(
        unified,
        revisions={
            "jailbreakbench": jbb_revision,
            "xstest": xstest_revision,
            "wildguardmix": wildguard_revision,
        },
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=== UNIFIED DATASET COMPLETE ===")
    print("Rows:", len(unified))
    print()
    print("Source counts:")
    print(
        unified["source_dataset"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("Primary Y counts:")
    print(
        unified["y"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("Split counts:")
    print(
        unified["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("Split by Y:")
    print(
        pd.crosstab(
            unified["split"],
            unified["y"],
        ).to_string()
    )
    print()
    print("Attack family by split:")
    print(
        pd.crosstab(
            unified["attack_family"],
            unified["split"],
            dropna=False,
        ).to_string()
    )
    print()
    print("Parquet:", OUTPUT_PARQUET)
    print("CSV:", OUTPUT_CSV)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
