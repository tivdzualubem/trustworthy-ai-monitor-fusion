#!/usr/bin/env python3
"""Materialize the frozen legacy development-only parquet view."""

from __future__ import annotations

import json
from pathlib import Path

from monitor_fusion.evaluation.data_boundary import (
    DEVELOPMENT_VIEW_DIRECTORY,
    load_protocol,
)
from monitor_fusion.evaluation.development_view import (
    materialize_development_view,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs/exact_cost_risk_cascade_protocol_v2.json"
)


def main() -> None:
    protocol = load_protocol(PROTOCOL_PATH)

    source_paths = [
        ROOT / relative
        for relative in protocol["data_boundary"][
            "sealed_mixed_split_containers"
        ]
    ]

    if len(source_paths) != 2:
        raise SystemExit(
            "Frozen protocol must contain exactly two mixed containers"
        )

    result = materialize_development_view(
        source_paths[0],
        source_paths[1],
        ROOT / DEVELOPMENT_VIEW_DIRECTORY,
        protocol=protocol,
        protocol_sha256=sha256_file(
            PROTOCOL_PATH
        ),
        root=ROOT,
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_directory": str(
                    result.output_directory.relative_to(
                        ROOT
                    )
                ),
                "manifest": str(
                    result.manifest_path.relative_to(
                        ROOT
                    )
                ),
                "outputs": [
                    {
                        "path": artifact.relative_path,
                        "row_count": artifact.row_count,
                        "split_counts": (
                            artifact.split_counts
                        ),
                        "sha256": artifact.sha256,
                    }
                    for artifact in result.artifacts
                ],
                "source_file_hashes_recorded": False,
                "source_row_counts_recorded": False,
                "protected_rows_materialized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
