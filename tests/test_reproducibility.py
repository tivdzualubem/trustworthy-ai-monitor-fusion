from pathlib import Path
import subprocess
import sys

import monitor_fusion


ROOT = Path(__file__).resolve().parents[1]


def test_package_import() -> None:
    assert monitor_fusion.__file__ is not None


def test_committed_reproducibility_snapshot() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_reproducibility.py"),
            "--strict-hashes",
        ],
        cwd=ROOT,
        check=True,
    )
