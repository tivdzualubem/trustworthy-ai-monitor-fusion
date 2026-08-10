#!/usr/bin/env python3
"""Measure the frozen v2 development optional monitor on one T4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from monitor_fusion.evaluation.development_optional_monitor_timing import (  # noqa: E402
    EXPECTED_DEVELOPMENT_DATASET_SHA256,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_DEVELOPMENT_SPLIT_COUNTS,
    ORIGINAL_QWEN_SCORING_RUN_MANIFEST_SHA256,
    QWEN_BATCH_SIZE,
    QWEN_DTYPE,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    REQUIRED_INPUT_COLUMNS,
    TARGET_DEVICE,
    TARGET_GPU_SUBSTRING,
    cap_optional_monitor_latency,
    development_optional_monitor_contract,
    prepare_development_timing_frame,
)
from monitor_fusion.evaluation.latency_measurement import (  # noqa: E402
    measure_monotonic_stage,
    summarize_latency,
)


EXPECTED_PROTOCOL_SHA256 = (
    "96a0bfdf1a0954d9313ecd7a2ae1272a"
    "07f0df4ba6197eede2f0e0afd9f1c1c7"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def stream_sha256(
    example_ids: list[str],
) -> str:
    digest = hashlib.sha256()

    for example_id in example_ids:
        digest.update(
            example_id.encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def resolve_path(
    package_root: Path,
    value: Path,
) -> Path:
    if value.is_absolute():
        return value

    return package_root / value


def nvidia_smi_text() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "nvidia-smi unavailable\n"

    return completed.stdout + completed.stderr


def pip_freeze_text() -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return completed.stdout


def load_scoring_reference(
    path: Path,
) -> ModuleType:
    """Load the exact historical message builder and parser."""

    spec = importlib.util.spec_from_file_location(
        "_qwen_scoring_reference",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load Qwen scoring reference"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if getattr(
        module,
        "MODEL_ID",
        None,
    ) != QWEN_MODEL_ID:
        raise RuntimeError(
            "Historical Qwen scoring model identity changed"
        )

    for function_name in (
        "build_messages",
        "parse_guard_output",
    ):
        if not callable(
            getattr(
                module,
                function_name,
                None,
            )
        ):
            raise RuntimeError(
                "Historical scoring reference missing "
                f"{function_name}"
            )

    return module


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--package-root",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/v2_development_view/"
            "unified_dataset_label_audited_v1."
            "development.parquet"
        ),
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/"
            "exact_cost_risk_cascade_protocol_v2.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "v2_development_optional_monitor_timing_results"
        ),
    )

    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(
            "models/qwen3guard_v2_timing"
        ),
    )

    parser.add_argument(
        "--expected-gpu-substring",
        default=TARGET_GPU_SUBSTRING,
    )

    parser.add_argument(
        "--device",
        default=TARGET_DEVICE,
    )

    args = parser.parse_args()

    package_root = args.package_root.resolve()

    dataset_path = resolve_path(
        package_root,
        args.dataset,
    )

    protocol_path = resolve_path(
        package_root,
        args.protocol,
    )

    output_dir = resolve_path(
        package_root,
        args.output_dir,
    )

    model_cache_dir = resolve_path(
        Path.cwd(),
        args.model_cache_dir,
    )

    scoring_reference_path = (
        package_root
        / "scripts"
        / "score_qwen3guard_official_colab.py"
    )

    if args.device != TARGET_DEVICE:
        raise SystemExit(
            "Frozen v2 development timing requires cuda:0"
        )

    protocol_hash = sha256(
        protocol_path
    )

    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit(
            "Protocol SHA256 differs from frozen v2.1 protocol"
        )

    dataset_hash = sha256(
        dataset_path
    )

    if (
        dataset_hash
        != EXPECTED_DEVELOPMENT_DATASET_SHA256
    ):
        raise SystemExit(
            "Development dataset SHA256 differs from "
            "the materialized frozen v2.1 development view"
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    contract = development_optional_monitor_contract(
        protocol
    )

    frame = pd.read_parquet(
        dataset_path,
        columns=list(
            REQUIRED_INPUT_COLUMNS
        ),
    )

    stream = prepare_development_timing_frame(
        frame,
        protocol=protocol,
    )

    if len(stream) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(
            "Development timing population changed"
        )

    import huggingface_hub
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    scoring_reference = (
        load_scoring_reference(
            scoring_reference_path
        )
    )

    build_messages = (
        scoring_reference.build_messages
    )

    parse_guard_output = (
        scoring_reference.parse_guard_output
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for frozen development timing"
        )

    visible_devices = (
        torch.cuda.device_count()
    )

    if visible_devices != 1:
        raise SystemExit(
            "Exactly one GPU must be visible. "
            "Run with CUDA_VISIBLE_DEVICES=0."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    if (
        args.expected_gpu_substring.lower()
        not in gpu_name.lower()
    ):
        raise SystemExit(
            "Unexpected GPU for frozen timing target: "
            f"{gpu_name!r}"
        )

    model_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    token = (
        os.getenv("HF_TOKEN")
        or None
    )

    snapshot = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        revision=QWEN_MODEL_REVISION,
        token=token,
        local_dir=model_cache_dir,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.jinja",
            "*.safetensors",
        ],
        max_workers=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to(args.device)
        .eval()
    )

    if str(model.dtype) != QWEN_DTYPE:
        raise SystemExit(
            "Loaded Qwen dtype differs from frozen "
            f"development scoring provenance: {model.dtype}"
        )

    torch.set_grad_enabled(False)

    def synchronize() -> None:
        torch.cuda.synchronize(
            torch.device(args.device)
        )

    def invoke_qwen(
        prompt: str,
        response: str,
    ) -> dict[str, Any]:
        messages = build_messages(
            prompt,
            response,
            "prompt_response",
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(args.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        width = int(
            inputs["input_ids"].shape[-1]
        )

        output_ids = (
            generated[0][width:]
            .detach()
            .cpu()
            .tolist()
        )

        raw = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        # Exact parser from the scoring implementation is used
        # inside the measured optional-monitor stage.
        parsed = parse_guard_output(
            raw
        )

        return {
            "parse_ok":
                bool(parsed["parse_ok"]),
            "label":
                parsed["label"],
            "score":
                parsed["score"],
            "input_tokens":
                int(
                    inputs[
                        "attention_mask"
                    ][0].sum().item()
                ),
            "output_tokens":
                int(len(output_ids)),
        }

    def measure_row(
        row: Any,
        *,
        warmup: bool,
        cold_start: bool,
        phase: str,
    ) -> dict[str, Any]:
        result, observed_ms = (
            measure_monotonic_stage(
                lambda: invoke_qwen(
                    str(row.prompt),
                    str(row.response),
                ),
                accelerator_stage=True,
                synchronize=synchronize,
            )
        )

        recorded_ms, timed_out = (
            cap_optional_monitor_latency(
                observed_ms,
                protocol=protocol,
            )
        )

        return {
            "example_id":
                str(row.example_id),
            "effective_group":
                str(row.effective_group),
            "split":
                str(row.split),
            "stream_position":
                int(row.stream_position),
            "measurement_phase":
                phase,
            "optional_monitor_observed_latency_ms":
                float(observed_ms),
            "optional_monitor_stage_latency_ms":
                float(recorded_ms),
            "optional_monitor_timed_out":
                bool(timed_out),
            "cold_start":
                bool(cold_start),
            "warmup":
                bool(warmup),
            "qwen_parse_ok":
                bool(result["parse_ok"]),
            "qwen_label":
                result["label"],
            "qwen_score":
                result["score"],
            "qwen_input_tokens":
                int(result["input_tokens"]),
            "qwen_output_tokens":
                int(result["output_tokens"]),
        }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    nvidia_path = (
        output_dir
        / "nvidia_smi.txt"
    )

    nvidia_path.write_text(
        nvidia_smi_text(),
        encoding="utf-8",
    )

    pip_freeze_path = (
        output_dir
        / "pip_freeze.txt"
    )

    pip_freeze_path.write_text(
        pip_freeze_text(),
        encoding="utf-8",
    )

    raw_records: list[
        dict[str, Any]
    ] = []

    warmup_rows = stream.head(
        contract.warmup_requests
    )

    print(
        "=== WARMUP ===",
        flush=True,
    )

    for index, row in enumerate(
        warmup_rows.itertuples(
            index=False
        )
    ):
        record = measure_row(
            row,
            warmup=True,
            cold_start=(index == 0),
            phase="warmup",
        )

        raw_records.append(
            record
        )

        print(
            f"warmup {index + 1}/"
            f"{contract.warmup_requests} "
            f"{record['optional_monitor_stage_latency_ms']:.3f} ms",
            flush=True,
        )

    print(
        "=== STEADY STATE ===",
        flush=True,
    )

    steady_records: list[
        dict[str, Any]
    ] = []

    total = len(stream)

    for index, row in enumerate(
        stream.itertuples(
            index=False
        )
    ):
        record = measure_row(
            row,
            warmup=False,
            cold_start=False,
            phase="steady_state",
        )

        raw_records.append(
            record
        )

        steady_records.append(
            record
        )

        if (
            index == 0
            or (index + 1) % 25 == 0
            or index + 1 == total
        ):
            print(
                f"steady {index + 1}/{total} "
                f"{record['optional_monitor_stage_latency_ms']:.3f} ms",
                flush=True,
            )

    raw_frame = pd.DataFrame(
        raw_records
    )

    steady_frame = pd.DataFrame(
        steady_records
    )

    if (
        len(steady_frame)
        != EXPECTED_DEVELOPMENT_ROWS
    ):
        raise RuntimeError(
            "Steady-state development timing "
            "row count changed"
        )

    if (
        steady_frame[
            "example_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Steady-state example_id is not unique"
        )

    if steady_frame["warmup"].any():
        raise RuntimeError(
            "Warmup request leaked into "
            "cost-predictor targets"
        )

    observed_split_counts = {
        str(key): int(value)
        for key, value in (
            steady_frame["split"]
            .value_counts()
            .to_dict()
            .items()
        )
    }

    if (
        observed_split_counts
        != EXPECTED_DEVELOPMENT_SPLIT_COUNTS
    ):
        raise RuntimeError(
            "Steady-state split counts changed"
        )

    raw_path = (
        output_dir
        / "optional_monitor_raw_timing.parquet"
    )

    target_path = (
        output_dir
        / "optional_monitor_cost_targets.parquet"
    )

    raw_frame.to_parquet(
        raw_path,
        index=False,
    )

    target_columns = [
        "example_id",
        "effective_group",
        "split",
        "stream_position",
        "optional_monitor_stage_latency_ms",
        "optional_monitor_timed_out",
        "cold_start",
        "warmup",
    ]

    steady_frame[
        target_columns
    ].to_parquet(
        target_path,
        index=False,
    )

    summary = summarize_latency(
        steady_frame[
            "optional_monitor_stage_latency_ms"
        ].to_numpy(),
        steady_frame[
            "optional_monitor_timed_out"
        ].to_numpy(),
    )

    parse_ok_rate = float(
        steady_frame[
            "qwen_parse_ok"
        ].mean()
    )

    capability = (
        torch.cuda.get_device_capability(
            0
        )
    )

    software = {
        "python":
            platform.python_version(),
        "platform":
            platform.platform(),
        "pandas":
            pd.__version__,
        "torch":
            torch.__version__,
        "cuda_runtime":
            torch.version.cuda,
        "cudnn":
            torch.backends.cudnn.version(),
        "transformers":
            transformers.__version__,
        "huggingface_hub":
            huggingface_hub.__version__,
    }

    manifest = {
        "artifact":
            "v2_development_optional_monitor_timing",
        "status":
            "completed_fixed_hardware_development_measurement",
        "created_at":
            now(),
        "protocol_sha256":
            protocol_hash,
        "input_dataset": {
            "path":
                str(dataset_path),
            "sha256":
                dataset_hash,
            "rows":
                int(len(stream)),
            "split_counts":
                observed_split_counts,
            "labels_read":
                False,
            "stream_order":
                contract.stream_order,
            "ordered_example_ids_sha256":
                stream_sha256(
                    stream[
                        "example_id"
                    ].astype(str).tolist()
                ),
        },
        "invocation_provenance": {
            "original_qwen_scoring_run_manifest_sha256":
                ORIGINAL_QWEN_SCORING_RUN_MANIFEST_SHA256,
            "scoring_reference_script":
                str(
                    scoring_reference_path
                ),
            "scoring_reference_script_sha256":
                sha256(
                    scoring_reference_path
                ),
            "message_builder":
                "historical build_messages",
            "output_parser":
                "historical parse_guard_output",
        },
        "hardware": {
            "gpu_name":
                gpu_name,
            "device":
                args.device,
            "visible_cuda_devices":
                int(
                    visible_devices
                ),
            "total_memory_bytes":
                int(
                    torch.cuda.get_device_properties(
                        0
                    ).total_memory
                ),
            "compute_capability":
                [
                    int(
                        capability[0]
                    ),
                    int(
                        capability[1]
                    ),
                ],
            "cuda_visible_devices":
                os.getenv(
                    "CUDA_VISIBLE_DEVICES"
                ),
            "bf16_is_supported_reported_by_torch":
                bool(
                    torch.cuda.is_bf16_supported()
                ),
        },
        "model": {
            "model_id":
                QWEN_MODEL_ID,
            "revision":
                QWEN_MODEL_REVISION,
            "dtype":
                str(model.dtype),
            "batch_size":
                QWEN_BATCH_SIZE,
            "max_new_tokens":
                QWEN_MAX_NEW_TOKENS,
            "method":
                "official_chat_template_generate_parse",
            "do_sample":
                False,
            "explicit_qwen_max_length":
                None,
            "padding_side":
                tokenizer.padding_side,
        },
        "measurement": {
            "stage":
                contract.measurement_stage,
            "clock":
                "time.perf_counter_ns",
            "accelerator_synchronize_before_and_after":
                True,
            "model_load_time_included":
                False,
            "warmup_requests":
                contract.warmup_requests,
            "warmup_excluded_from_training_targets":
                True,
            "cold_start":
                "first warmup request after model load",
            "timeout_ms":
                contract.timeout_ms,
            "timeout_recording":
                "observed latency retained; "
                "cost target capped at protocol timeout",
            "steady_state_primary":
                True,
        },
        "steady_state_summary":
            summary.as_dict(),
        "qwen_parse_ok_rate":
            parse_ok_rate,
        "software":
            software,
        "files": {},
    }

    for path in (
        raw_path,
        target_path,
        nvidia_path,
        pip_freeze_path,
    ):
        manifest["files"][
            path.name
        ] = sha256(path)

    manifest_path = (
        output_dir
        / "run_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status":
                    "PASS",
                "output_dir":
                    str(output_dir),
                "steady_rows":
                    len(steady_frame),
                "warmup_rows":
                    len(warmup_rows),
                "split_counts":
                    observed_split_counts,
                "parse_ok_rate":
                    parse_ok_rate,
                "summary":
                    summary.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
