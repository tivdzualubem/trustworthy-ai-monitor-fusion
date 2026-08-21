from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "scripts/reproduce_historical_v2_evidence.sh"
)

MAKEFILE = ROOT / "Makefile"

PROVENANCE_CHECKER = (
    ROOT
    / "scripts/check_v2_evidence_provenance.py"
)


def test_canonical_historical_v2_entrypoint_exists():
    assert SCRIPT.is_file()

    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        "check_v2_evidence_provenance.py --write"
        in text
    )

    assert (
        "historical_v2_csv_regeneration=false"
        in text
    )

    assert (
        "corrected_measurement_requires_new_namespace=true"
        in text
    )


def test_makefile_exposes_single_named_v2_historical_target():
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "reproduce-v2-historical:" in text

    assert (
        "bash scripts/reproduce_historical_v2_evidence.sh"
        in text
    )


def test_provenance_checker_is_part_of_entrypoint():
    assert PROVENANCE_CHECKER.is_file()

    entrypoint = SCRIPT.read_text(encoding="utf-8")

    assert PROVENANCE_CHECKER.name in entrypoint
