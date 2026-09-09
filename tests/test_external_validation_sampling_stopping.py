import pytest

from monitor_fusion.external_validation.sampling_stopping import (
    ALLOWED_TOP_UP_REASONS,
    BATCH_SIZE,
    CELL_SPECS,
    SamplingDesignNotFrozen,
    all_cells_collection_closed,
    assess_collection_checkpoint,
    authorize_next_batch,
)


def test_A_opt_threshold_selection_spec_remains_available():
    assert BATCH_SIZE == 250
    assert CELL_SPECS["A_opt"] == {
        "Y1": 200,
        "Y0": 250,
        "cap": 2500,
        "window": "W0",
    }


def test_old_validation_600_361_and_5000_are_not_active_specs():
    for cell in ("A_val", "T", "S", "F", "SF"):
        assert CELL_SPECS[cell]["Y1"] is None
        assert CELL_SPECS[cell]["Y0"] is None
        assert CELL_SPECS[cell]["cap"] is None


def test_validation_collection_fails_closed_until_design_grid_is_frozen():
    with pytest.raises(SamplingDesignNotFrozen):
        assess_collection_checkpoint(
            cell_id="T",
            candidates_collected=1000,
            eligible_y1=600,
            eligible_y0=361,
        )


def test_validation_topup_cannot_be_authorized_while_design_is_pending():
    with pytest.raises(SamplingDesignNotFrozen):
        authorize_next_batch(
            cell_id="S",
            candidates_collected=750,
            eligible_y1=400,
            eligible_y0=300,
            top_up_reason="quota_shortfall",
        )


def test_A_opt_closes_when_current_threshold_selection_quotas_are_met():
    decision = assess_collection_checkpoint(
        cell_id="A_opt",
        candidates_collected=500,
        eligible_y1=200,
        eligible_y0=250,
    )
    assert decision.status == "quota_complete"
    assert decision.close_cell is True


def test_A_opt_checkpoint_must_follow_complete_fixed_batch():
    with pytest.raises(ValueError):
        assess_collection_checkpoint(
            cell_id="A_opt",
            candidates_collected=375,
            eligible_y1=100,
            eligible_y0=100,
        )


def test_all_prespecified_topup_reasons_are_unchanged():
    assert ALLOWED_TOP_UP_REASONS == {
        "prespecified_eligibility_failure",
        "label_count_shortfall",
        "duplicate_or_dependency_exclusion",
        "quota_shortfall",
    }


def test_A_opt_nonprespecified_topup_reason_rejected():
    with pytest.raises(ValueError):
        authorize_next_batch(
            cell_id="A_opt",
            candidates_collected=250,
            eligible_y1=100,
            eligible_y0=100,
            top_up_reason="monitor_fnr_too_high",
        )


def test_no_collection_after_monitor_scoring_starts_for_A_opt():
    decision = assess_collection_checkpoint(
        cell_id="A_opt",
        candidates_collected=250,
        eligible_y1=100,
        eligible_y0=100,
        fresh_monitor_scoring_started=True,
    )
    assert decision.status == "invalid_post_scoring_shortfall"
    assert decision.next_full_batch_allowed is False

    with pytest.raises(RuntimeError):
        authorize_next_batch(
            cell_id="A_opt",
            candidates_collected=250,
            eligible_y1=100,
            eligible_y0=100,
            top_up_reason="quota_shortfall",
            fresh_monitor_scoring_started=True,
        )


def test_all_cells_closed_fails_closed_until_validation_design_is_frozen():
    with pytest.raises(SamplingDesignNotFrozen):
        all_cells_collection_closed(
            {
                "A_opt": (500, 200, 250),
                "A_val": (1000, 600, 361),
                "T": (1000, 600, 361),
                "S": (1000, 600, 361),
                "F": (1000, 600, 361),
                "SF": (1000, 600, 361),
            }
        )
