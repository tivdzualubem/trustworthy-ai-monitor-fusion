from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data/metadata/source_audit.json"
SOURCE_ROOT = ROOT / "data/raw/sources"
OUTPUT_PATH = ROOT / "data/metadata/dataset_inventory.json"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clone_at_revision(
    name: str,
    url: str,
    revision: str,
) -> Path:
    destination = SOURCE_ROOT / name

    if not destination.exists():
        print(f"Cloning {name}...")
        subprocess.run(
            ["git", "clone", "--quiet", url, str(destination)],
            check=True,
        )

    print(f"Checking out pinned {name} revision: {revision}")
    subprocess.run(
        ["git", "fetch", "--quiet", "origin"],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", revision],
        cwd=destination,
        check=True,
    )

    actual_revision = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual_revision != revision:
        raise RuntimeError(
            f"{name}: expected {revision}, obtained {actual_revision}"
        )

    return destination


def inspect_tabular_file(path: Path, repository: Path) -> dict[str, Any]:
    relative_path = str(path.relative_to(repository))
    record: dict[str, Any] = {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            frame = pd.read_csv(path)
            record["rows"] = int(len(frame))
            record["columns"] = list(frame.columns)

        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
            record["rows"] = int(len(frame))
            record["columns"] = list(frame.columns)

        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
            record["rows"] = int(len(frame))
            record["columns"] = list(frame.columns)

    except Exception as exc:
        record["inspection_error"] = f"{type(exc).__name__}: {exc}"

    return record


def inspect_repository(repository: Path) -> list[dict[str, Any]]:
    allowed_suffixes = {".csv", ".jsonl", ".ndjson", ".parquet"}
    maximum_size = 100 * 1024 * 1024

    files = []
    for path in sorted(repository.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if path.stat().st_size > maximum_size:
            continue

        files.append(inspect_tabular_file(path, repository))

    return files


def clean_counter(values: list[Any]) -> dict[str, int]:
    normalized = [
        "<NULL>" if value is None else str(value)
        for value in values
    ]
    return dict(sorted(Counter(normalized).items()))


def inspect_wildguard(
    revision: str,
) -> tuple[dict[str, Any], Any]:
    print("Loading pinned WildGuardTest...")
    dataset = load_dataset(
        "allenai/wildguardmix",
        "wildguardtest",
        split="test",
        revision=revision,
    )

    fields_to_count = [
        "adversarial",
        "prompt_harm_label",
        "response_refusal_label",
        "response_harm_label",
        "subcategory",
    ]

    label_counts = {
        field: clean_counter(dataset[field])
        for field in fields_to_count
    }

    summary = {
        "revision": revision,
        "configuration": "wildguardtest",
        "split": "test",
        "num_rows": dataset.num_rows,
        "columns": dataset.column_names,
        "features": {
            key: str(value)
            for key, value in dataset.features.items()
        },
        "label_counts": label_counts,
    }

    return summary, dataset


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    sources = audit["sources"]

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

    harmbench = clone_at_revision(
        "harmbench",
        sources["harmbench"]["url"],
        sources["harmbench"]["revision"],
    )
    xstest = clone_at_revision(
        "xstest",
        sources["xstest"]["url"],
        sources["xstest"]["revision"],
    )

    wildguard_summary, _ = inspect_wildguard(
        sources["wildguardmix"]["revision"]
    )

    inventory = {
        "harmbench": {
            "revision": sources["harmbench"]["revision"],
            "candidate_tabular_files": inspect_repository(harmbench),
        },
        "xstest": {
            "revision": sources["xstest"]["revision"],
            "candidate_tabular_files": inspect_repository(xstest),
        },
        "wildguardtest": wildguard_summary,
    }

    OUTPUT_PATH.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== HARMBENCH TABULAR FILES ===")
    for item in inventory["harmbench"]["candidate_tabular_files"]:
        print(
            f"{item['path']} | rows={item.get('rows')} | "
            f"columns={item.get('columns')}"
        )

    print()
    print("=== XSTEST TABULAR FILES ===")
    for item in inventory["xstest"]["candidate_tabular_files"]:
        print(
            f"{item['path']} | rows={item.get('rows')} | "
            f"columns={item.get('columns')}"
        )

    print()
    print("=== WILDGUARDTEST ===")
    print("Rows:", wildguard_summary["num_rows"])
    print("Columns:", wildguard_summary["columns"])

    for field, counts in wildguard_summary["label_counts"].items():
        print(f"{field}: {counts}")

    print()
    print("Inventory written to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
