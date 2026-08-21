from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dependency_grouping_regenerates_exactly():
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/rebuild_evaluation_measurement_dependency_groups.py"
            ),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
