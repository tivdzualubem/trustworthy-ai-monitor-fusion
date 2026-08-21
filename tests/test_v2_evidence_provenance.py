from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "scripts/check_v2_evidence_provenance.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "check_v2_evidence_provenance",
        SCRIPT,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_historical_v2_provenance_fails_closed():
    module = load_module()

    status = module.build_status()

    assert status["historical_v2_status"] == (
        "published_snapshot_not_fully_regenerable_"
        "from_committed_raw_artifacts"
    )

    gap = status["known_provenance_gap"]

    assert gap[
        "raw_downstream_timing_not_preserved"
    ] is True

    assert (
        gap[
            "historical_latency_summary_exact_regeneration"
        ]
        is False
    )

    assert len(
        gap[
            "missing_candidate_bundle_timing_columns"
        ]
    ) == 12


def test_historical_v2_claim_boundaries():
    module = load_module()

    boundary = (
        module.build_status()[
            "scientific_boundary"
        ]
    )

    assert (
        boundary[
            "historical_component_cost_results_are_direct_e2e"
        ]
        is False
    )

    assert (
        boundary[
            "historical_component_cost_results_are_final_exact_cost"
        ]
        is False
    )

    assert (
        boundary[
            "historical_grouping_protects_near_duplicate_dependence"
        ]
        is False
    )

    assert (
        boundary[
            "historical_35s_bound_is_proven_runtime_enforced"
        ]
        is False
    )

    assert (
        boundary[
            "historical_1pct_equivalence_margin_externally_justified"
        ]
        is False
    )


def test_historical_v2_snapshot_is_not_rewritten():
    module = load_module()

    status = module.build_status()

    assert (
        status[
            "historical_tables_are_modified"
        ]
        is False
    )

    assert (
        status[
            "router_rescue_attempted"
        ]
        is False
    )

    assert (
        status[
            "protected_legacy_data_opened"
        ]
        is False
    )

    assert (
        status[
            "fresh_data_used"
        ]
        is False
    )
