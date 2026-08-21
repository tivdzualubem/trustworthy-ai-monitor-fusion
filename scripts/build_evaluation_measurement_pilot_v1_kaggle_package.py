#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CPU_COMMIT = "20dd2ab38d09036a23db27406472122d7506abfa"
BRANCH = "evaluation-measurement-aug17-repair"

CPU_DIR = ROOT / "reports/evaluation_measurement_pilot_v1/cpu"
CPU_MANIFEST = CPU_DIR / "cpu_pilot_manifest.json"
TIMING_SAMPLE = CPU_DIR / "timing_sample.parquet"
POLICY_DEFINITIONS = CPU_DIR / "primary_policy_definitions.json"
PRIMARY_MATRIX = CPU_DIR / "primary_policy_prediction_matrix.parquet"

DEV_CACHE = (
    ROOT
    / "data/processed/v2_development_view/"
    / "monitor_score_cache_v3.development.parquet"
)
GROUPS = (
    ROOT
    / "data/metadata/evaluation_measurement_pilot_v1/"
    / "development_dependency_groups.csv"
)
PROTOCOL = ROOT / "configs/evaluation_measurement_pilot_v1.json"
RULE_FILTER = ROOT / "src/monitor_fusion/monitors/rule_filter.py"
BENCHMARK = (
    ROOT
    / "scripts/benchmark_evaluation_measurement_pilot_v1_t4.py"
)

EXPORT = (
    ROOT
    / "exports/evaluation_measurement_pilot_v1_t4_package.zip"
)
META = (
    ROOT
    / "data/metadata/"
    / "evaluation_measurement_pilot_v1_t4_package_manifest.json"
)
DOWNLOADS = Path("/mnt/c/Users/NOAH/Downloads")
DOWNLOAD_COPY = (
    DOWNLOADS
    / "evaluation_measurement_pilot_v1_t4_package.zip"
)

STACK_ORDER = [
    "rule_to_compact",
    "compact_to_qwen",
    "rule_compact_to_qwen",
]
SCORES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]


def progress(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stdout}"
        )
    return result.stdout.strip()


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def model_probability(
    model: dict,
    frame: pd.DataFrame,
) -> np.ndarray:
    X = frame[model["features"]].to_numpy(float)
    coef = np.asarray(model["coef"], dtype=float)
    intercept = float(model["intercept"])
    return sigmoid(intercept + X @ coef)


def deterministic_zip(
    source_directory: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source_directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_directory).as_posix()
            info = zipfile.ZipInfo(
                relative,
                date_time=(2026, 8, 21, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def notebook_document() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Evaluation Measurement Pilot v1 — T4 timing\n",
                "\n",
                "Kaggle settings: **GPU = T4**, **Internet = On**, "
                "and add a secret named `HF_TOKEN`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import zipfile, shutil\n",
                "\n",
                "matches = list(Path('/kaggle/input').rglob("
                "'evaluation_measurement_pilot_v1_t4_package.zip'))\n",
                "assert len(matches) == 1, matches\n",
                "zip_path = matches[0]\n",
                "root = Path('/kaggle/working/evaluation_measurement_pilot_v1_t4_package')\n",
                "if root.exists(): shutil.rmtree(root)\n",
                "root.mkdir(parents=True)\n",
                "with zipfile.ZipFile(zip_path) as z: z.extractall(root)\n",
                "print('package:', zip_path)\n",
                "print('root:', root)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q \"transformers==5.13.1\" "
                "\"huggingface-hub==0.36.2\" "
                "\"accelerate>=1.12\" \"safetensors>=0.6\"\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from kaggle_secrets import UserSecretsClient\n",
                "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n",
                "print('HF token loaded:', bool(os.environ.get('HF_TOKEN')))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "out = Path('/kaggle/working/evaluation_measurement_pilot_v1_t4_results')\n",
                "if out.exists(): shutil.rmtree(out)\n",
                "cmd = (\n",
                "    f\"python -u {root/'benchmark_evaluation_measurement_pilot_v1_t4.py'} \"\n",
                "    f\"--package-root {root} --output-dir {out}\"\n",
                ")\n",
                "print(cmd)\n",
                "rc = os.system(cmd)\n",
                "assert rc == 0, rc\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "result_zip = Path('/kaggle/working/evaluation_measurement_pilot_v1_t4_results.zip')\n",
                "if result_zip.exists(): result_zip.unlink()\n",
                "with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as z:\n",
                "    for p in sorted(out.rglob('*')):\n",
                "        if p.is_file(): z.write(p, p.relative_to(out))\n",
                "print('DOWNLOAD THIS:', result_zip)\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    progress("[1/8] Verifying CPU commit and frozen CPU artifacts...")
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit("Unexpected branch.")
    if git("rev-parse", "HEAD") != CPU_COMMIT:
        raise SystemExit(
            "HEAD must be the committed CPU pilot before packaging."
        )

    cpu_manifest = json.loads(
        CPU_MANIFEST.read_text(encoding="utf-8")
    )
    if cpu_manifest["freeze_commit"] != (
        "b9759bf1610f726396183bd7c79c11983bd8956b"
    ):
        raise SystemExit("CPU manifest freeze parent mismatch.")

    for name, expected in cpu_manifest["outputs"].items():
        path = CPU_DIR / name
        if sha256(path) != expected:
            raise SystemExit(
                f"CPU output hash mismatch: {name}"
            )

    progress("[2/8] Building label-blind reference-score file...")
    sample = pd.read_parquet(TIMING_SAMPLE)
    cache = pd.read_parquet(
        DEV_CACHE,
        columns=["example_id", *SCORES],
    )
    groups = pd.read_csv(GROUPS)

    forbidden = {
        "y",
        "y_original",
        "label",
        "audited_y",
        "old_y",
    }
    if forbidden.intersection(sample.columns):
        raise SystemExit(
            "Timing sample unexpectedly contains labels."
        )

    reference = sample[["example_id"]].merge(
        cache,
        on="example_id",
        how="left",
        validate="one_to_one",
    )
    if len(reference) != len(sample):
        raise SystemExit("Reference score merge changed row count.")
    if reference[SCORES].isna().any().any():
        raise SystemExit("Missing reference monitor scores.")
    if forbidden.intersection(reference.columns):
        raise SystemExit("Reference-score file contains labels.")

    if len(sample) != 363:
        raise SystemExit(
            f"Expected 363 group-closed timing rows, got {len(sample)}."
        )

    progress("[3/8] Verifying dependency-group closure...")
    all_groups = groups[
        ["example_id", "primary_dependency_group"]
    ].copy()
    selected_ids = set(
        sample["example_id"].astype(str)
    )
    selected_groups = set(
        sample["primary_dependency_group"].astype(str)
    )
    closure_ids = set(
        all_groups.loc[
            all_groups["primary_dependency_group"]
            .astype(str)
            .isin(selected_groups),
            "example_id",
        ].astype(str)
    )
    if closure_ids != selected_ids:
        raise SystemExit(
            "Timing sample is not closed over frozen dependency groups."
        )

    progress("[4/8] Verifying serialized logistic formulas against CPU predictions...")
    definitions = json.loads(
        POLICY_DEFINITIONS.read_text(encoding="utf-8")
    )
    matrix = pd.read_parquet(PRIMARY_MATRIX)
    dev_scores = pd.read_parquet(
        DEV_CACHE,
        columns=["example_id", *SCORES],
    )

    parity = {}
    for stack in STACK_ORDER:
        config = definitions["stacks"][stack]
        part = matrix[
            matrix["policy_id"].eq(
                f"{stack}::cheap_only"
            )
        ][
            [
                "example_id",
                "cheap_probability",
                "full_probability",
            ]
        ].merge(
            dev_scores,
            on="example_id",
            how="left",
            validate="one_to_one",
        )

        cheap_calc = model_probability(
            config["cheap_model"],
            part,
        )
        full_calc = model_probability(
            config["full_model"],
            part,
        )
        cheap_diff = float(
            np.max(
                np.abs(
                    cheap_calc
                    - part["cheap_probability"].to_numpy(float)
                )
            )
        )
        full_diff = float(
            np.max(
                np.abs(
                    full_calc
                    - part["full_probability"].to_numpy(float)
                )
            )
        )
        if cheap_diff > 1e-12 or full_diff > 1e-12:
            raise SystemExit(
                f"Serialized logistic parity failed for {stack}: "
                f"cheap={cheap_diff}, full={full_diff}"
            )
        parity[stack] = {
            "cheap_max_abs_difference": cheap_diff,
            "full_max_abs_difference": full_diff,
        }

    progress("[5/8] Assembling deterministic Kaggle package...")
    temp_root = Path(
        tempfile.mkdtemp(
            prefix="evaluation-measurement-pilot-v1-t4-package-"
        )
    )
    try:
        package = temp_root / "package"
        (package / "data").mkdir(parents=True)
        (package / "code").mkdir(parents=True)

        shutil.copy2(
            TIMING_SAMPLE,
            package / "data/timing_sample.parquet",
        )
        reference.to_parquet(
            package / "data/reference_scores.parquet",
            index=False,
            compression="zstd",
        )
        shutil.copy2(
            POLICY_DEFINITIONS,
            package / "data/primary_policy_definitions.json",
        )
        shutil.copy2(
            PROTOCOL,
            package / "evaluation_measurement_pilot_v1.json",
        )
        shutil.copy2(
            CPU_MANIFEST,
            package / "cpu_pilot_manifest.json",
        )
        shutil.copy2(
            RULE_FILTER,
            package / "code/rule_filter.py",
        )
        shutil.copy2(
            BENCHMARK,
            package / "benchmark_evaluation_measurement_pilot_v1_t4.py",
        )

        (package / "requirements-kaggle.txt").write_text(
            "\n".join(
                [
                    "transformers==5.13.1",
                    "huggingface-hub==0.36.2",
                    "accelerate>=1.12",
                    "safetensors>=0.6",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        (package / "run_on_kaggle.ipynb").write_text(
            json.dumps(
                notebook_document(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        readme = """# Evaluation Measurement Pilot v1 — T4 package

Use one NVIDIA T4 GPU, batch size 1, and Internet enabled.
Add a Kaggle secret named `HF_TOKEN`, then run `run_on_kaggle.ipynb`.

The benchmark measures all 15 frozen deployable policies on the 363-row
label-blind, dependency-group-closed timing sample. Primary cost is direct
wall-clock end-to-end policy latency. The comparison estimand is the sum of
monitor component latencies measured inside the same policy call.

There is no timeout, post-hoc clipping, or bounded mean-cost certificate.
Model loading is excluded. Twenty untimed maximal-stack warmup requests are
run after model loading. Policy order is balanced by deterministic cyclic
rotation across the 15 deployable policies for each example.

Download `evaluation_measurement_pilot_v1_t4_results.zip` after completion.
"""
        (package / "README.md").write_text(
            readme,
            encoding="utf-8",
        )

        protocol = json.loads(
            PROTOCOL.read_text(encoding="utf-8")
        )
        deployable_count = sum(
            1
            for stack in definitions["stacks"].values()
            for policy in stack["policies"]
            if bool(policy["deployable"])
        )
        if deployable_count != 15:
            raise SystemExit(
                f"Expected 15 deployable policies, got {deployable_count}."
            )

        files = {}
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
                files[path.relative_to(package).as_posix()] = sha256(path)

        package_manifest = {
            "artifact": "evaluation_measurement_pilot_v1_t4_package",
            "status": "frozen_gpu_measurement_package",
            "cpu_commit": CPU_COMMIT,
            "freeze_commit": (
                "b9759bf1610f726396183bd7c79c11983bd8956b"
            ),
            "protocol_sha256": sha256(PROTOCOL),
            "cpu_manifest_sha256": sha256(CPU_MANIFEST),
            "timing_sample_rows": int(len(sample)),
            "timing_sample_source_counts": {
                str(k): int(v)
                for k, v in sample.groupby(
                    "source_dataset"
                ).size().to_dict().items()
            },
            "labels_in_timing_package": False,
            "dependency_group_closed": True,
            "deployable_policies": deployable_count,
            "serialized_model_probability_parity": parity,
            "required_gpu": "NVIDIA T4",
            "batch_size": 1,
            "models": protocol["monitors"],
            "measurement_design": {
                "primary_estimand": (
                    "direct_wall_clock_end_to_end_policy_latency_ms"
                ),
                "comparison_estimand": (
                    "same_run_sum_of_monitor_component_latencies_ms"
                ),
                "warmup_requests": 20,
                "warmup_type": (
                    "untimed maximal rule+compact+qwen requests"
                ),
                "cuda_synchronization": True,
                "model_load_time_included": False,
                "tokenization_transfer_policy_logic_parse_included": True,
                "timeout_or_capping": False,
                "posthoc_clipping": False,
                "policy_measurement_order": (
                    "deterministic cyclic rotation across 15 deployable "
                    "policies by timing-example index"
                ),
                "cpu_cost_join_integrity_rule": (
                    "All selective routing decisions must match the cached-score "
                    "reference on the timing sample before direct-E2E cost is "
                    "joined to frozen CPU recall for Pareto analysis."
                ),
            },
            "files": files,
        }
        (package / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(
                package_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        progress("[6/8] Writing deterministic package ZIP...")
        deterministic_zip(package, EXPORT)
        shutil.copy2(EXPORT, DOWNLOAD_COPY)

        progress("[7/8] Writing repository package manifest...")
        metadata = {
            "artifact": "evaluation_measurement_pilot_v1_t4_package_build",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "cpu_commit": CPU_COMMIT,
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "benchmark": str(BENCHMARK.relative_to(ROOT)),
            "benchmark_sha256": sha256(BENCHMARK),
            "package_zip": str(EXPORT.relative_to(ROOT)),
            "package_zip_sha256": sha256(EXPORT),
            "download_copy": str(DOWNLOAD_COPY),
            "protocol_sha256": sha256(PROTOCOL),
            "cpu_manifest_sha256": sha256(CPU_MANIFEST),
            "timing_sample_sha256": sha256(TIMING_SAMPLE),
            "reference_scores_sha256": sha256(
                package / "data/reference_scores.parquet"
            ),
            "labels_in_timing_package": False,
            "dependency_group_closed": True,
            "deployable_policies": 15,
            "serialized_model_probability_parity": parity,
            "gpu_run_started": False,
        }
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    progress("[8/8] Complete.")
    print(
        "\nEVALUATION_MEASUREMENT_PILOT_V1_T4_PACKAGE=PASS",
        flush=True,
    )
    print(f"package_file={DOWNLOAD_COPY}", flush=True)
    print(f"package_sha256={sha256(EXPORT)}", flush=True)
    print("gpu_run_started=false", flush=True)


if __name__ == "__main__":
    main()
