from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HARMBENCH = ROOT / "data/raw/sources/harmbench"
XSTEST = ROOT / "data/raw/sources/xstest"
OUTPUT = ROOT / "data/metadata/attack_asset_inventory.json"

MAX_JSON_SIZE = 100 * 1024 * 1024


def describe_json(path: Path, repository: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(repository)),
        "size_bytes": path.stat().st_size,
    }

    if path.stat().st_size > MAX_JSON_SIZE:
        record["status"] = "skipped_too_large"
        return record

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        record["top_level_type"] = type(data).__name__

        if isinstance(data, list):
            record["num_items"] = len(data)
            if data:
                first = data[0]
                record["first_item_type"] = type(first).__name__
                if isinstance(first, dict):
                    record["first_item_keys"] = sorted(first.keys())

        elif isinstance(data, dict):
            record["num_top_level_keys"] = len(data)
            record["top_level_keys_preview"] = sorted(
                str(key) for key in list(data.keys())[:20]
            )

            if data:
                first_value = next(iter(data.values()))
                record["first_value_type"] = type(first_value).__name__

                if isinstance(first_value, dict):
                    record["first_value_keys"] = sorted(first_value.keys())
                elif isinstance(first_value, list):
                    record["first_value_length"] = len(first_value)
                    if first_value and isinstance(first_value[0], dict):
                        record["first_nested_item_keys"] = sorted(
                            first_value[0].keys()
                        )

        record["status"] = "inspected"

    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def relevant_directories(repository: Path) -> list[str]:
    keywords = {
        "test_case",
        "test_cases",
        "completion",
        "completions",
        "result",
        "results",
        "attack",
        "attacks",
    }

    matches = []
    for path in repository.rglob("*"):
        if not path.is_dir() or ".git" in path.parts:
            continue

        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts.intersection(keywords):
            matches.append(str(path.relative_to(repository)))

    return sorted(set(matches))


def inspect_xstest() -> dict[str, Any]:
    prompt_path = XSTEST / "xstest_prompts.csv"
    prompts = pd.read_csv(prompt_path)

    completion_files = sorted(
        (XSTEST / "model_completions").glob("*.csv")
    )

    completion_summaries = []
    for path in completion_files:
        frame = pd.read_csv(path)
        summary: dict[str, Any] = {
            "path": str(path.relative_to(XSTEST)),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

        for field in ["type", "final_label", "agreement"]:
            if field in frame.columns:
                counts = (
                    frame[field]
                    .fillna("<NULL>")
                    .astype(str)
                    .value_counts(dropna=False)
                    .sort_index()
                    .to_dict()
                )
                summary[f"{field}_counts"] = {
                    str(key): int(value)
                    for key, value in counts.items()
                }

        completion_summaries.append(summary)

    prompt_counts = {}
    for field in ["label", "type", "focus"]:
        if field in prompts.columns:
            counts = (
                prompts[field]
                .fillna("<NULL>")
                .astype(str)
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )
            prompt_counts[field] = {
                str(key): int(value)
                for key, value in counts.items()
            }

    return {
        "prompt_file": str(prompt_path.relative_to(XSTEST)),
        "prompt_rows": int(len(prompts)),
        "prompt_columns": list(prompts.columns),
        "prompt_distributions": prompt_counts,
        "completion_files": completion_summaries,
    }


def main() -> None:
    json_files = sorted(
        path
        for path in HARMBENCH.rglob("*.json")
        if ".git" not in path.parts
    )

    path_category_counts = Counter()
    for path in json_files:
        lowered = str(path.relative_to(HARMBENCH)).lower()
        if "test_case" in lowered:
            path_category_counts["test_cases"] += 1
        if "completion" in lowered:
            path_category_counts["completions"] += 1
        if "result" in lowered:
            path_category_counts["results"] += 1
        if "config" in lowered:
            path_category_counts["configs"] += 1

    inventory = {
        "harmbench": {
            "json_file_count": len(json_files),
            "path_category_counts": dict(path_category_counts),
            "relevant_directories": relevant_directories(HARMBENCH),
            "json_files": [
                describe_json(path, HARMBENCH)
                for path in json_files
            ],
        },
        "xstest": inspect_xstest(),
    }

    OUTPUT.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== HARMBENCH JSON SUMMARY ===")
    print("JSON files:", inventory["harmbench"]["json_file_count"])
    print(
        "Path categories:",
        inventory["harmbench"]["path_category_counts"],
    )

    print()
    print("=== RELEVANT HARMBENCH DIRECTORIES ===")
    for path in inventory["harmbench"]["relevant_directories"]:
        print(path)

    print()
    print("=== HARMBENCH TEST-CASE / COMPLETION JSON FILES ===")
    for item in inventory["harmbench"]["json_files"]:
        lowered = item["path"].lower()
        if "test_case" in lowered or "completion" in lowered:
            print(
                item["path"],
                "| type=",
                item.get("top_level_type"),
                "| items=",
                item.get("num_items", item.get("num_top_level_keys")),
                "| keys=",
                item.get(
                    "first_item_keys",
                    item.get("first_value_keys"),
                ),
            )

    print()
    print("=== XSTEST PROMPT DISTRIBUTIONS ===")
    for field, counts in inventory["xstest"][
        "prompt_distributions"
    ].items():
        print(f"{field}: {counts}")

    print()
    print("=== XSTEST COMPLETION FILES ===")
    for item in inventory["xstest"]["completion_files"]:
        print(
            item["path"],
            "| rows=",
            item["rows"],
            "| final_label=",
            item.get("final_label_counts"),
        )

    print()
    print("Inventory written to:", OUTPUT)


if __name__ == "__main__":
    main()
