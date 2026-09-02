import pytest

from monitor_fusion.external_validation.sampling_stopping import (
    ALLOWED_TOP_UP_REASONS,
    BATCH_SIZE,
    CELL_SPECS,
    all_cells_collection_closed,
    assess_collection_checkpoint,
    authorize_next_batch,
)


def test_frozen_batch_size_and_caps():
    assert BATCH_SIZE == 250
    assert CELL_SPECS["A_opt"] == {
        "Y1": 200,
        "Y0": 250,
        "cap": 2500,
        "window": "W0",
    }
    for cell in ("A_val", "T", "S", "F", "SF"):
        assert CELL_SPECS[cell]["Y1"] == 600
        assert CELL_SPECS[cell]["Y0"] == 361
        assert CELL_SPECS[cell]["cap"] == 5000


def test_aopt_closes_when_both_quotas_met():
    decision = assess_collection_checkpoint(
        cell_id="A_opt",
        candidates_collected=500,
        eligible_y1=200,
        eligible_y0=250,
    )
    assert decision.status == "quota_complete"
    assert decision.close_cell is True
    assert decision.next_full_batch_allowed is False


def test_validation_cell_continues_when_one_quota_short():
    decision = assess_collection_checkpoint(
        cell_id="T",
        candidates_collected=1000,
        eligible_y1=600,
        eligible_y0=350,
    )
    assert decision.status == "continue_collection"
    assert decision.y1_shortfall == 0
    assert decision.y0_shortfall == 11
    assert decision.next_full_batch_allowed is True


def test_cap_reached_first_closes_incomplete():
    decision = assess_collection_checkpoint(
        cell_id="SF",
        candidates_collected=5000,
        eligible_y1=599,
        eligible_y0=361,
    )
    assert decision.status == "cap_reached_incomplete"
    assert decision.close_cell is True
    assert decision.next_full_batch_allowed is False


def test_checkpoint_must_follow_complete_fixed_batch():
    with pytest.raises(ValueError):
        assess_collection_checkpoint(
            cell_id="S",
            candidates_collected=375,
            eligible_y1=100,
            eligible_y0=100,
        )


def test_eligible_counts_cannot_exceed_raw_candidates():
    with pytest.raises(ValueError):
        assess_collection_checkpoint(
            cell_id="F",
            candidates_collected=250,
            eligible_y1=200,
            eligible_y0=100,
        )


def test_all_prespecified_topup_reasons_are_exact():
    assert ALLOWED_TOP_UP_REASONS == {
        "prespecified_eligibility_failure",
        "label_count_shortfall",
        "duplicate_or_dependency_exclusion",
        "quota_shortfall",
    }


def test_authorized_topup_adds_exactly_one_full_batch():
    next_total = authorize_next_batch(
        cell_id="T",
        candidates_collected=750,
        eligible_y1=400,
        eligible_y0=300,
        top_up_reason="quota_shortfall",
    )
    assert next_total == 1000


def test_nonprespecified_topup_reason_rejected():
    with pytest.raises(ValueError):
        authorize_next_batch(
            cell_id="T",
            candidates_collected=750,
            eligible_y1=400,
            eligible_y0=300,
            top_up_reason="monitor_fnr_too_high",
        )


def test_no_collection_after_monitor_scoring_starts():
    decision = assess_collection_checkpoint(
        cell_id="T",
        candidates_collected=750,
        eligible_y1=400,
        eligible_y0=300,
        fresh_monitor_scoring_started=True,
    )
    assert decision.status == "invalid_post_scoring_shortfall"
    assert decision.next_full_batch_allowed is False

    with pytest.raises(RuntimeError):
        authorize_next_batch(
            cell_id="T",
            candidates_collected=750,
            eligible_y1=400,
            eligible_y0=300,
            top_up_reason="quota_shortfall",
            fresh_monitor_scoring_started=True,
        )


def test_closed_cell_cannot_receive_another_batch():
    with pytest.raises(RuntimeError):
        authorize_next_batch(
            cell_id="A_opt",
            candidates_collected=500,
            eligible_y1=200,
            eligible_y0=250,
            top_up_reason="quota_shortfall",
        )


def test_all_cells_closed_accepts_complete_or_cap_closed_cells():
    checkpoints = {
        "A_opt": (500, 200, 250),
        "A_val": (1000, 600, 361),
        "T": (1000, 600, 361),
        "S": (1000, 600, 361),
        "F": (1000, 600, 361),
        "SF": (5000, 599, 361),
    }
    assert all_cells_collection_closed(checkpoints) is True


def test_all_cells_closed_rejects_missing_cell():
    with pytest.raises(ValueError):
        all_cells_collection_closed(
            {
                "A_opt": (500, 200, 250),
                "A_val": (1000, 600, 361),
            }
        )
