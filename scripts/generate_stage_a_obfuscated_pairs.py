"""Generate frozen Stage-A human/model obfuscated formulations.

This performs deterministic representation transforms only.
It performs no semantic review, severity review, guard scoring, or q estimation.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from monitor_fusion.external_validation.attack_transforms import (
    apply_transform,
    select_transform,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = ROOT / "configs/stage_a_obfuscation_generation_contract_v1.json"
OUT = ROOT / "results/stage_a_obfuscation_generation_v1"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen_pre_generation"

    # Fail closed rather than overwrite a prior generation artifact.
    if OUT.exists():
        raise RuntimeError(
            f"{OUT} already exists; refusing to regenerate or overwrite."
        )

    sources = contract["source_inputs"]

    pair_path = ROOT / sources["approved_pair_worksheet"]["path"]
    human_path = ROOT / sources["human_direct_worksheet"]["path"]
    map_path = ROOT / sources["human_direct_map"]["path"]
    model_path = ROOT / sources["model_direct_generations"]["path"]

    for spec, path in [
        (sources["approved_pair_worksheet"], pair_path),
        (sources["human_direct_worksheet"], human_path),
        (sources["human_direct_map"], map_path),
        (sources["model_direct_generations"], model_path),
    ]:
        assert path.exists(), path
        assert sha_file(path) == spec["sha256"], (
            f"Frozen source fingerprint mismatch: {path}"
        )

    registry = ROOT / contract["transform_registry"]["path"]
    assert sha_file(registry) == contract["transform_registry"]["sha256"]

    pairs = read_csv(pair_path)
    humans = read_csv(human_path)
    mapping = read_csv(map_path)
    models = read_csv(model_path)

    assert len(pairs) == len(humans) == len(mapping) == len(models) == 104

    pair_by_id = {r["base_intent_id"]: r for r in pairs}
    model_by_id = {r["base_intent_id"]: r for r in models}
    human_by_authoring_id = {r["authoring_item_id"]: r for r in humans}

    assert len(pair_by_id) == 104
    assert len(model_by_id) == 104
    assert len(human_by_authoring_id) == 104

    human_by_id = {}

    for m in mapping:
        aid = m["authoring_item_id"]
        bid = m["base_intent_id"]

        assert aid in human_by_authoring_id
        h = human_by_authoring_id[aid]

        assert h["human_direct_lock_status"] == "locked"
        assert h["human_direct_review_decision"] == "include"
        assert h["human_formulation_author"] == "project_author"
        assert h["human_direct_text"].strip()
        assert sha_text(h["human_direct_text"]) == h["human_direct_text_hash"]

        human_by_id[bid] = h

    assert len(human_by_id) == 104
    assert set(human_by_id) == set(pair_by_id) == set(model_by_id)

    for bid, m in model_by_id.items():
        assert m["generation_status"] == "generated_pending_eligibility_review"
        assert m["model_direct_text"].strip()
        assert sha_text(m["model_direct_text"]) == m["model_direct_text_sha256"]

    counts = Counter(r["obfuscation_transform_id"] for r in pairs)

    assert counts == Counter(
        contract["assignment"]["expected_counts"]
    )

    output_rows = []

    for p in pairs:
        bid = p["base_intent_id"]
        transform_id = p["obfuscation_transform_id"]

        # Verify the old assignment itself still matches the frozen rule.
        assert transform_id == select_transform(bid)

        human_direct = human_by_id[bid]["human_direct_text"]
        model_direct = model_by_id[bid]["model_direct_text"]

        human_obfuscated = apply_transform(human_direct, transform_id)
        model_obfuscated = apply_transform(model_direct, transform_id)

        assert human_obfuscated
        assert model_obfuscated

        output_rows.append({
            "base_intent_id": bid,
            "reviewed_category": p["reviewed_category"],
            "obfuscation_transform_id": transform_id,

            "human_direct_text_sha256":
                sha_text(human_direct),
            "human_obfuscated_text":
                human_obfuscated,
            "human_obfuscated_text_sha256":
                sha_text(human_obfuscated),

            "model_direct_text_sha256":
                sha_text(model_direct),
            "model_obfuscated_text":
                model_obfuscated,
            "model_obfuscated_text_sha256":
                sha_text(model_obfuscated),

            "obfuscation_status":
                "generated_pending_semantic_and_severity_review",
        })

    assert len(output_rows) == 104
    assert len({r["base_intent_id"] for r in output_rows}) == 104

    OUT.mkdir(parents=True, exist_ok=False)

    csv_path = OUT / "obfuscated_pairs.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "artifact_id": "stage_a_obfuscation_generation_v1",
        "status": "generated_pending_semantic_and_severity_review",
        "scope": "Stage-A public/development paired-formulation feasibility study only",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "rows": 104,
        "human_obfuscated_rows": 104,
        "model_obfuscated_rows": 104,
        "transform_counts": dict(sorted(counts.items())),
        "same_transform_for_human_and_model": True,
        "manual_post_transform_editing": False,
        "adaptive_transform_selection": False,
        "source_hashes": {
            "approved_pair_worksheet": sha_file(pair_path),
            "human_direct_worksheet": sha_file(human_path),
            "human_direct_map": sha_file(map_path),
            "model_direct_generations": sha_file(model_path),
            "transform_registry": sha_file(registry),
            "generation_contract": sha_file(CONTRACT),
        },
        "output_csv": str(csv_path.relative_to(ROOT)),
        "output_csv_sha256": sha_file(csv_path),
        "semantic_equivalence_review_performed": False,
        "severity_review_performed": False,
        "guard_scoring_performed": False,
        "q_estimation_performed": False,
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("STAGE-A OBFUSCATION GENERATION: PASS")
    print("rows:", len(output_rows))
    print("transform counts:", dict(sorted(counts.items())))
    print("human obfuscated:", 104)
    print("model obfuscated:", 104)
    print("semantic review performed: False")
    print("severity review performed: False")
    print("guard scoring performed: False")
    print("q estimation performed: False")
    print("output sha256:", sha_file(csv_path))


if __name__ == "__main__":
    main()
