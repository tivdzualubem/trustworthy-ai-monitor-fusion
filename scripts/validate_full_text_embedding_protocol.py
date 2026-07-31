#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)


def validate() -> dict[str, object]:
    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )
    embedding = protocol[
        "predictor_families"
    ]["frozen_embedding"]

    assert embedding["full_text_coverage_required"] is True
    assert embedding["truncation_allowed"] is False
    assert embedding["long_text_strategy"] == (
        "deterministic_non_overlapping_token_chunks"
    )
    assert embedding["chunk_size_rule"] == (
        "model_max_seq_length minus tokenizer special-token count"
    )
    assert embedding["chunk_overlap_tokens"] == 0
    assert embedding["chunk_aggregation"] == (
        "content_token_count_weighted_mean_then_l2_normalize"
    )
    assert embedding["minimum_required_token_coverage"] == 1.0
    assert embedding["record_per_example_chunk_count"] is True
    assert embedding["record_per_example_token_coverage"] is True
    assert (
        embedding["record_per_example_content_token_count"]
        is True
    )

    amendments = protocol.get("protocol_amendments", [])
    matching = [
        item
        for item in amendments
        if item.get("amendment_id")
        == "full_text_embedding_coverage_v1"
    ]
    assert len(matching) == 1

    amendment = matching[0]
    assert amendment["stage"] == (
        "before_value_estimator_training"
    )
    assert amendment["affected_committed_results"] == "none"
    assert amendment["value_estimator_training_started"] is False
    assert amendment["final_test_or_shift_used"] is False

    return {
        "status": "PASS",
        "full_text_coverage_required": True,
        "truncation_allowed": False,
        "minimum_required_token_coverage": 1.0,
        "chunk_overlap_tokens": 0,
        "amendment_id": amendment["amendment_id"],
    }


def main() -> None:
    result = validate()
    print(
        "complete-text embedding protocol validation passed"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
