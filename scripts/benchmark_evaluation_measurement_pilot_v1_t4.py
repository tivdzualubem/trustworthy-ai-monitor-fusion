#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

COMPACT_ID = "KoalaAI/Text-Moderation"
COMPACT_REV = "838ade0edb66dcffc5532d08ff6ed5c899abfb5c"
QWEN_ID = "Qwen/Qwen3Guard-Gen-4B"
QWEN_REV = "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
UNSAFE = {"H", "H2", "HR", "S", "S3", "SH", "V", "V2"}
LABEL_SCORE = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.I)

STACK_ORDER = [
    "rule_to_compact",
    "compact_to_qwen",
    "rule_compact_to_qwen",
]
POLICY_KIND_ORDER = {
    "cheap_only": 0,
    "full_information": 1,
    "selective_deployable": 2,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expected-gpu-substring", default="T4")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def progress(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync() -> None:
    torch.cuda.synchronize()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logistic_probability(model: dict, features: dict[str, float]) -> float:
    value = float(model["intercept"])
    for coef, feature in zip(
        model["coef"],
        model["features"],
        strict=True,
    ):
        value += float(coef) * float(features[feature])
    return sigmoid(value)


def policy_id(stack: str, policy: dict) -> str:
    kind = str(policy["policy_kind"])
    target = policy.get("target_rate")
    if target is None:
        return f"{stack}::{kind}"
    return f"{stack}::{kind}::{float(target):.2f}"


def parse_guard(text: str) -> dict:
    match = SAFETY_RE.search(text or "")
    refusal = REFUSAL_RE.search(text or "")
    label = None
    if match:
        label = {
            "safe": "Safe",
            "unsafe": "Unsafe",
            "controversial": "Controversial",
        }[match.group(1).lower()]
    return {
        "label": label,
        "score": LABEL_SCORE.get(label),
        "refusal": (
            refusal.group(1).capitalize()
            if refusal
            else None
        ),
        "parse_ok": label is not None,
    }


def compact_text(prompt: str, response: str) -> str:
    return (
        "Classify whether the assistant response should be intercepted "
        "for safety.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Assistant response:\n{response}"
    )


def load_rule_function(package_root: Path):
    module_path = package_root / "code/rule_filter.py"
    module_name = "frozen_rule_filter"
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen rule filter.")
    module = importlib.util.module_from_spec(spec)

    # Python 3.12 dataclasses inspect sys.modules while decorators execute.
    # Register the dynamically loaded module before exec_module.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module.score_prompt_response


def verify_package(package_root: Path) -> dict:
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative, expected in manifest["files"].items():
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing package file: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Package hash mismatch for {relative}: "
                f"{expected} != {actual}"
            )

    return manifest


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean_ms": float(values.mean()),
        "std_ms": (
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
        ),
        "min_ms": float(values.min()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(values.max()),
    }


class Runtime:
    def __init__(
        self,
        package_root: Path,
        device: str,
        max_length: int,
        max_new_tokens: int,
        token: str,
    ) -> None:
        self.package_root = package_root
        self.device = device
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.rule_fn = load_rule_function(package_root)

        models_root = Path("/kaggle/working/evaluation_measurement_models")
        models_root.mkdir(parents=True, exist_ok=True)

        progress("    downloading/loading frozen compact monitor...")
        compact_path = snapshot_download(
            repo_id=COMPACT_ID,
            revision=COMPACT_REV,
            token=token,
            local_dir=models_root / "compact",
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "*.model",
            ],
            max_workers=1,
        )
        self.ctok = AutoTokenizer.from_pretrained(
            compact_path,
            local_files_only=True,
        )
        self.cmodel = (
            AutoModelForSequenceClassification.from_pretrained(
                compact_path,
                local_files_only=True,
                dtype=torch.float32,
            )
            .to(device)
            .eval()
        )
        self.id2label = {
            int(i): str(v)
            for i, v in self.cmodel.config.id2label.items()
        }

        progress("    downloading/loading frozen Qwen guard...")
        qwen_path = snapshot_download(
            repo_id=QWEN_ID,
            revision=QWEN_REV,
            token=token,
            local_dir=models_root / "qwen",
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "*.model",
                "*.jinja",
            ],
            max_workers=1,
        )
        self.qtok = AutoTokenizer.from_pretrained(
            qwen_path,
            local_files_only=True,
        )
        if self.qtok.pad_token_id is None:
            self.qtok.pad_token = self.qtok.eos_token
        self.qtok.padding_side = "left"
        self.qmodel = (
            AutoModelForCausalLM.from_pretrained(
                qwen_path,
                local_files_only=True,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .to(device)
            .eval()
        )

    def rule(
        self,
        prompt: str,
        response: str,
    ) -> tuple[float, float]:
        t0 = time.perf_counter_ns()
        score = float(
            self.rule_fn(prompt, response)["rule_score"]
        )
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        return score, elapsed

    def compact(
        self,
        prompt: str,
        response: str,
    ) -> tuple[float, float]:
        sync()
        t0 = time.perf_counter_ns()
        enc = self.ctok(
            [compact_text(prompt, response)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {
            key: value.to(self.device)
            for key, value in enc.items()
        }
        with torch.inference_mode():
            probability = torch.softmax(
                self.cmodel(**enc).logits.float(),
                dim=-1,
            )[0]
        sync()

        probs = {
            self.id2label[index]: float(value)
            for index, value in enumerate(
                probability.detach().cpu().tolist()
            )
        }
        score = sum(
            value
            for label, value in probs.items()
            if label.strip().upper() in UNSAFE
        )
        score = float(min(max(score, 0.0), 1.0))
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        return score, elapsed

    def qwen(
        self,
        prompt: str,
        response: str,
    ) -> dict:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        t0 = time.perf_counter_ns()
        text = self.qtok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.qtok(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        t1 = time.perf_counter_ns()

        sync()
        t2 = time.perf_counter_ns()
        with torch.inference_mode():
            generated = self.qmodel.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.qtok.eos_token_id,
            )
        sync()
        t3 = time.perf_counter_ns()

        width = int(inputs["input_ids"].shape[-1])
        output_ids = generated[0][width:].tolist()
        raw = self.qtok.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()
        parsed = parse_guard(raw)
        t4 = time.perf_counter_ns()

        if not parsed["parse_ok"]:
            raise RuntimeError(
                f"Qwen parse failure: {raw!r}"
            )

        return {
            **parsed,
            "raw_output": raw,
            "input_tokens": int(
                inputs["attention_mask"][0].sum().item()
            ),
            "output_tokens": int(len(output_ids)),
            "tokenize_transfer_latency_ms": (t1 - t0) / 1e6,
            "generate_latency_ms": (t3 - t2) / 1e6,
            "decode_parse_latency_ms": (t4 - t3) / 1e6,
            "total_latency_ms": (t4 - t0) / 1e6,
        }


def deployable_policy_records(definitions: dict) -> list[dict]:
    records: list[dict] = []

    for stack in STACK_ORDER:
        config = definitions["stacks"][stack]
        for policy in config["policies"]:
            if not bool(policy["deployable"]):
                continue
            kind = str(policy["policy_kind"])
            if kind not in POLICY_KIND_ORDER:
                continue
            records.append(
                {
                    "stack": stack,
                    "policy": policy,
                    "cheap_model": config["cheap_model"],
                    "full_model": config["full_model"],
                    "policy_id": policy_id(stack, policy),
                }
            )

    records.sort(
        key=lambda item: (
            STACK_ORDER.index(item["stack"]),
            POLICY_KIND_ORDER[
                item["policy"]["policy_kind"]
            ],
            -1.0
            if item["policy"].get("target_rate") is None
            else float(item["policy"]["target_rate"]),
        )
    )

    if len(records) != 15:
        raise RuntimeError(
            f"Expected 15 deployable policies; found {len(records)}"
        )
    return records


def reference_features(row) -> dict[str, float]:
    return {
        "rule_score": float(row.rule_score),
        "compact_unsafe_score": float(
            row.compact_unsafe_score
        ),
        "qwen_prompt_response_score": float(
            row.qwen_prompt_response_score
        ),
    }


def evaluate_reference_policy(
    record: dict,
    ref_features: dict[str, float],
) -> tuple[bool, float, int]:
    policy = record["policy"]
    kind = str(policy["policy_kind"])
    cheap = logistic_probability(
        record["cheap_model"],
        ref_features,
    )
    full = logistic_probability(
        record["full_model"],
        ref_features,
    )

    if kind == "cheap_only":
        acquired = False
        probability = cheap
    elif kind == "full_information":
        acquired = True
        probability = full
    elif kind == "selective_deployable":
        distance = abs(cheap - 0.5)
        acquired = (
            distance
            <= float(policy["acquisition_threshold"])
        )
        probability = full if acquired else cheap
    else:
        raise RuntimeError(f"Unexpected policy kind: {kind}")

    prediction = int(
        probability >= float(policy["decision_threshold"])
    )
    return acquired, probability, prediction


def run_policy(
    runtime: Runtime,
    record: dict,
    prompt: str,
    response: str,
    ref_features: dict[str, float],
) -> tuple[dict, list[dict]]:
    stack = record["stack"]
    policy = record["policy"]
    kind = str(policy["policy_kind"])

    rule_score = math.nan
    compact_score = math.nan
    qwen_score = math.nan

    rule_ms = 0.0
    compact_ms = 0.0
    qwen_ms = 0.0
    qwen_record: dict | None = None

    sync()
    t0 = time.perf_counter_ns()

    runtime_features: dict[str, float] = {}

    if stack in {
        "rule_to_compact",
        "rule_compact_to_qwen",
    }:
        rule_score, rule_ms = runtime.rule(
            prompt,
            response,
        )
        runtime_features["rule_score"] = rule_score

    if stack in {
        "compact_to_qwen",
        "rule_compact_to_qwen",
    }:
        compact_score, compact_ms = runtime.compact(
            prompt,
            response,
        )
        runtime_features[
            "compact_unsafe_score"
        ] = compact_score

    cheap = logistic_probability(
        record["cheap_model"],
        runtime_features,
    )

    if kind == "cheap_only":
        acquired = False
    elif kind == "full_information":
        acquired = True
    elif kind == "selective_deployable":
        acquired = (
            abs(cheap - 0.5)
            <= float(policy["acquisition_threshold"])
        )
    else:
        raise RuntimeError(
            f"Unexpected policy kind: {kind}"
        )

    if stack == "rule_to_compact" and acquired:
        compact_score, compact_ms = runtime.compact(
            prompt,
            response,
        )
        runtime_features[
            "compact_unsafe_score"
        ] = compact_score

    if (
        stack
        in {
            "compact_to_qwen",
            "rule_compact_to_qwen",
        }
        and acquired
    ):
        qwen_record = runtime.qwen(
            prompt,
            response,
        )
        qwen_score = float(qwen_record["score"])
        qwen_ms = float(
            qwen_record["total_latency_ms"]
        )
        runtime_features[
            "qwen_prompt_response_score"
        ] = qwen_score

    if acquired:
        full = logistic_probability(
            record["full_model"],
            runtime_features,
        )
        probability = full
    else:
        full = math.nan
        probability = cheap

    prediction = int(
        probability >= float(policy["decision_threshold"])
    )

    sync()
    direct_ms = (
        time.perf_counter_ns() - t0
    ) / 1e6

    component_ms = (
        float(rule_ms)
        + float(compact_ms)
        + float(qwen_ms)
    )

    ref_acquired, ref_probability, ref_prediction = (
        evaluate_reference_policy(
            record,
            ref_features,
        )
    )

    row = {
        "policy_id": record["policy_id"],
        "stack": stack,
        "policy_kind": kind,
        "target_rate": (
            math.nan
            if policy.get("target_rate") is None
            else float(policy["target_rate"])
        ),
        "acquisition_threshold": (
            math.nan
            if policy.get("acquisition_threshold") is None
            else float(policy["acquisition_threshold"])
        ),
        "decision_threshold": float(
            policy["decision_threshold"]
        ),
        "runtime_acquired": int(acquired),
        "reference_acquired": int(ref_acquired),
        "route_matches_reference": bool(
            acquired == ref_acquired
        ),
        "runtime_probability": float(probability),
        "reference_probability": float(ref_probability),
        "runtime_prediction": int(prediction),
        "reference_prediction": int(ref_prediction),
        "prediction_matches_reference": bool(
            prediction == ref_prediction
        ),
        "rule_score": rule_score,
        "reference_rule_score": float(
            ref_features["rule_score"]
        ),
        "compact_unsafe_score": compact_score,
        "reference_compact_unsafe_score": float(
            ref_features["compact_unsafe_score"]
        ),
        "qwen_prompt_response_score": qwen_score,
        "reference_qwen_prompt_response_score": float(
            ref_features["qwen_prompt_response_score"]
        ),
        "rule_latency_ms": float(rule_ms),
        "compact_latency_ms": float(compact_ms),
        "qwen_latency_ms": float(qwen_ms),
        "component_sum_latency_ms": float(component_ms),
        "direct_e2e_latency_ms": float(direct_ms),
        "unattributed_overhead_ms": float(
            direct_ms - component_ms
        ),
    }

    qwen_rows: list[dict] = []
    if qwen_record is not None:
        qwen_rows.append(
            {
                "policy_id": record["policy_id"],
                **qwen_record,
            }
        )

    return row, qwen_rows


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()

    progress("[1/10] Verifying package hashes and benchmark inputs...")
    package_manifest = verify_package(package_root)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_parquet(
        package_root / "data/timing_sample.parquet"
    )
    reference = pd.read_parquet(
        package_root / "data/reference_scores.parquet"
    )
    definitions = json.loads(
        (
            package_root
            / "data/primary_policy_definitions.json"
        ).read_text(encoding="utf-8")
    )

    forbidden = {
        "y",
        "y_original",
        "label",
        "audited_y",
        "old_y",
    }
    if forbidden.intersection(sample.columns):
        raise RuntimeError(
            "Timing sample contains forbidden label columns."
        )
    if forbidden.intersection(reference.columns):
        raise RuntimeError(
            "Reference-score file contains forbidden label columns."
        )

    if len(sample) != int(
        package_manifest["timing_sample_rows"]
    ):
        raise RuntimeError("Timing sample row count mismatch.")
    if sample["example_id"].astype(str).nunique() != len(sample):
        raise RuntimeError("Timing sample IDs are not unique.")

    merged = sample.merge(
        reference,
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(sample):
        raise RuntimeError(
            "Reference scores do not cover the timing sample."
        )

    policies = deployable_policy_records(definitions)

    progress("[2/10] Verifying NVIDIA T4 environment...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu_substring.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"Expected GPU containing "
            f"{args.expected_gpu_substring!r}; found {gpu_name!r}"
        )

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        (output_dir / "nvidia_smi.txt").write_text(
            smi.stdout + smi.stderr,
            encoding="utf-8",
        )
    except FileNotFoundError:
        (output_dir / "nvidia_smi.txt").write_text(
            "nvidia-smi unavailable\n",
            encoding="utf-8",
        )

    progress("[3/10] Loading pinned monitors; model load is excluded from timing...")
    runtime = Runtime(
        package_root,
        args.device,
        args.max_length,
        args.max_new_tokens,
        token,
    )

    progress("[4/10] Running 20 untimed maximal-stack warmup requests...")
    warmup_rows = int(
        package_manifest["measurement_design"][
            "warmup_requests"
        ]
    )
    if warmup_rows != 20:
        raise RuntimeError("Package warmup count is not 20.")

    for row in merged.head(warmup_rows).itertuples(index=False):
        runtime.rule(
            str(row.prompt),
            str(row.response),
        )
        runtime.compact(
            str(row.prompt),
            str(row.response),
        )
        runtime.qwen(
            str(row.prompt),
            str(row.response),
        )

    progress("[5/10] Measuring 15 deployable policies with balanced rotation...")
    policy_rows: list[dict] = []
    qwen_rows: list[dict] = []
    order_rows: list[dict] = []

    total_calls = len(merged) * len(policies)
    completed = 0

    for example_index, row in enumerate(
        merged.itertuples(index=False)
    ):
        rotation = example_index % len(policies)
        ordered = (
            policies[rotation:]
            + policies[:rotation]
        )

        ref = reference_features(row)

        for position, record in enumerate(ordered):
            completed += 1
            if completed == 1 or completed % 100 == 0:
                progress(
                    f"    measured {completed}/{total_calls} "
                    f"policy calls"
                )

            measurement, q_rows = run_policy(
                runtime,
                record,
                str(row.prompt),
                str(row.response),
                ref,
            )
            measurement.update(
                {
                    "example_index": int(example_index),
                    "measurement_order_position": int(position),
                    "example_id": str(row.example_id),
                    "source_dataset": str(row.source_dataset),
                    "primary_dependency_group": str(
                        row.primary_dependency_group
                    ),
                    "base_selected": bool(row.base_selected),
                }
            )
            policy_rows.append(measurement)

            for q_row in q_rows:
                q_row.update(
                    {
                        "example_index": int(example_index),
                        "measurement_order_position": int(position),
                        "example_id": str(row.example_id),
                        "source_dataset": str(row.source_dataset),
                    }
                )
                qwen_rows.append(q_row)

            order_rows.append(
                {
                    "example_index": int(example_index),
                    "example_id": str(row.example_id),
                    "measurement_order_position": int(position),
                    "policy_id": record["policy_id"],
                }
            )

    raw = pd.DataFrame(policy_rows)
    order_frame = pd.DataFrame(order_rows)

    progress("[6/10] Computing direct-E2E/component summaries and parity checks...")
    summary_rows: list[dict] = []
    source_rows: list[dict] = []

    for policy_name, part in raw.groupby(
        "policy_id",
        sort=True,
    ):
        direct = describe(
            part["direct_e2e_latency_ms"].to_numpy(float)
        )
        component = describe(
            part["component_sum_latency_ms"].to_numpy(float)
        )
        record = part.iloc[0]

        summary_rows.append(
            {
                "policy_id": policy_name,
                "stack": record["stack"],
                "policy_kind": record["policy_kind"],
                "target_rate": record["target_rate"],
                "runtime_acquisition_rate": float(
                    part["runtime_acquired"].mean()
                ),
                "reference_acquisition_rate": float(
                    part["reference_acquired"].mean()
                ),
                "route_match_rate": float(
                    part["route_matches_reference"].mean()
                ),
                "prediction_match_rate": float(
                    part[
                        "prediction_matches_reference"
                    ].mean()
                ),
                **{
                    f"direct_{key}": value
                    for key, value in direct.items()
                },
                **{
                    f"component_{key}": value
                    for key, value in component.items()
                },
                "mean_unattributed_overhead_ms": float(
                    part["unattributed_overhead_ms"].mean()
                ),
                "mean_direct_minus_component_ms": float(
                    (
                        part["direct_e2e_latency_ms"]
                        - part["component_sum_latency_ms"]
                    ).mean()
                ),
            }
        )

        for source, source_part in part.groupby(
            "source_dataset",
            sort=True,
        ):
            source_direct = describe(
                source_part[
                    "direct_e2e_latency_ms"
                ].to_numpy(float)
            )
            source_component = describe(
                source_part[
                    "component_sum_latency_ms"
                ].to_numpy(float)
            )
            source_rows.append(
                {
                    "policy_id": policy_name,
                    "source_dataset": source,
                    **{
                        f"direct_{key}": value
                        for key, value in source_direct.items()
                    },
                    **{
                        f"component_{key}": value
                        for key, value in source_component.items()
                    },
                }
            )

    summary = pd.DataFrame(summary_rows)
    source_summary = pd.DataFrame(source_rows)

    direct_means = summary.set_index(
        "policy_id"
    )["direct_mean_ms"]
    component_means = summary.set_index(
        "policy_id"
    )["component_mean_ms"]

    reversal_rows: list[dict] = []
    names = sorted(summary["policy_id"].tolist())
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            direct_diff = float(
                direct_means[left] - direct_means[right]
            )
            component_diff = float(
                component_means[left]
                - component_means[right]
            )
            direct_sign = int(np.sign(direct_diff))
            component_sign = int(np.sign(component_diff))
            reversal_rows.append(
                {
                    "policy_a": left,
                    "policy_b": right,
                    "direct_mean_difference_ms": direct_diff,
                    "component_mean_difference_ms": component_diff,
                    "ranking_reversal": bool(
                        direct_sign != 0
                        and component_sign != 0
                        and direct_sign != component_sign
                    ),
                }
            )
    reversals = pd.DataFrame(reversal_rows)

    route_mismatches = raw[
        ~raw["route_matches_reference"]
    ].copy()
    prediction_mismatches = raw[
        ~raw["prediction_matches_reference"]
    ].copy()

    progress("[7/10] Preserving raw Qwen generations and score differences...")
    with (
        output_dir / "qwen_raw_generations.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for record in qwen_rows:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

    score_validation = {
        "rule_max_abs_difference": float(
            np.nanmax(
                np.abs(
                    raw["rule_score"]
                    - raw["reference_rule_score"]
                )
            )
        ),
        "compact_max_abs_difference": float(
            np.nanmax(
                np.abs(
                    raw["compact_unsafe_score"]
                    - raw[
                        "reference_compact_unsafe_score"
                    ]
                )
            )
        ),
        "qwen_max_abs_difference": float(
            np.nanmax(
                np.abs(
                    raw["qwen_prompt_response_score"]
                    - raw[
                        "reference_qwen_prompt_response_score"
                    ]
                )
            )
        ),
        "route_mismatch_rows": int(
            len(route_mismatches)
        ),
        "route_match_rate": float(
            raw["route_matches_reference"].mean()
        ),
        "prediction_mismatch_rows": int(
            len(prediction_mismatches)
        ),
        "prediction_match_rate": float(
            raw["prediction_matches_reference"].mean()
        ),
        "cpu_cost_join_valid": bool(
            len(route_mismatches) == 0
        ),
        "cpu_cost_join_rule": (
            "Direct-E2E timing may be joined to frozen CPU recall "
            "for Pareto analysis only if all deployable selective "
            "routing decisions match the cached-score reference on "
            "the timing sample."
        ),
    }

    progress("[8/10] Writing raw outputs and environment record...")
    raw.to_parquet(
        output_dir / "policy_latency_raw.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_csv(
        output_dir / "policy_latency_summary.csv",
        index=False,
    )
    source_summary.to_csv(
        output_dir / "source_latency_summary.csv",
        index=False,
    )
    reversals.to_csv(
        output_dir / "cost_ranking_reversals.csv",
        index=False,
    )
    route_mismatches.to_csv(
        output_dir / "route_mismatches.csv",
        index=False,
    )
    prediction_mismatches.to_csv(
        output_dir / "prediction_mismatches.csv",
        index=False,
    )
    order_frame.to_csv(
        output_dir / "measurement_order.csv",
        index=False,
    )
    (
        output_dir / "score_and_route_validation.json"
    ).write_text(
        json.dumps(
            score_validation,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "device": args.device,
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(0).total_memory
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "batch_size": 1,
        "clock": "time.perf_counter_ns",
        "cuda_synchronize": True,
        "model_load_time_included": False,
        "warmup_requests": warmup_rows,
        "warmup_excluded": True,
        "timeout_or_capping": False,
        "posthoc_clipping": False,
        "policy_order": (
            "deterministic cyclic rotation across 15 deployable "
            "policies by timing-example index"
        ),
        "models": {
            "compact": {
                "model_id": COMPACT_ID,
                "revision": COMPACT_REV,
                "dtype": "float32",
            },
            "qwen": {
                "model_id": QWEN_ID,
                "revision": QWEN_REV,
                "dtype": "float16",
                "method": (
                    "official_chat_template_generate_structured_parse"
                ),
            },
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(
            environment,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    progress("[9/10] Writing result manifest with SHA-256 provenance...")
    output_files = [
        "nvidia_smi.txt",
        "policy_latency_raw.parquet",
        "policy_latency_summary.csv",
        "source_latency_summary.csv",
        "cost_ranking_reversals.csv",
        "route_mismatches.csv",
        "prediction_mismatches.csv",
        "measurement_order.csv",
        "score_and_route_validation.json",
        "qwen_raw_generations.jsonl",
        "environment.json",
    ]

    result_manifest = {
        "artifact": "evaluation_measurement_pilot_v1_t4_timing",
        "status": "completed_fixed_hardware_direct_e2e_measurement",
        "protocol_sha256": package_manifest[
            "protocol_sha256"
        ],
        "cpu_commit": package_manifest["cpu_commit"],
        "package_sha256": package_manifest.get(
            "package_zip_sha256_reference"
        ),
        "timing_rows": int(len(merged)),
        "deployable_policies": int(len(policies)),
        "raw_policy_calls": int(len(raw)),
        "primary_estimand": (
            "direct_wall_clock_end_to_end_policy_latency_ms"
        ),
        "comparison_estimand": (
            "same_run_sum_of_monitor_component_latencies_ms"
        ),
        "measurement_design": package_manifest[
            "measurement_design"
        ],
        "score_and_route_validation": score_validation,
        "cost_ranking_reversal_pairs": int(
            reversals["ranking_reversal"].sum()
        ),
        "bounded_mean_cost_certification": False,
        "bounded_mean_cost_reason": (
            "No mechanically enforced runtime support bound; raw "
            "uncapped latency is retained."
        ),
        "files": {
            name: sha256(output_dir / name)
            for name in output_files
        },
    }
    (
        output_dir / "t4_timing_manifest.json"
    ).write_text(
        json.dumps(
            result_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    progress("[10/10] Complete.")
    print("\nT4_TIMING_SUMMARY", flush=True)
    print(
        summary[
            [
                "policy_id",
                "runtime_acquisition_rate",
                "route_match_rate",
                "direct_mean_ms",
                "direct_p95_ms",
                "direct_p99_ms",
                "component_mean_ms",
                "mean_direct_minus_component_ms",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        "\nEVALUATION_MEASUREMENT_PILOT_V1_T4=PASS",
        flush=True,
    )
    print(
        f"results_dir={output_dir}",
        flush=True,
    )
    print(
        f"cpu_cost_join_valid="
        f"{str(score_validation['cpu_cost_join_valid']).lower()}",
        flush=True,
    )
    print(
        f"cost_ranking_reversal_pairs="
        f"{int(reversals['ranking_reversal'].sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
