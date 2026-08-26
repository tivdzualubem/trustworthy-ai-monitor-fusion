#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

COMPACT_ID = "KoalaAI/Text-Moderation"
COMPACT_REV = "838ade0edb66dcffc5532d08ff6ed5c899abfb5c"
UNSAFE = {"H", "H2", "HR", "S", "S3", "SH", "V", "V2"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-gpu-substring", default="T4")
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_package(root: Path) -> dict:
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        path = root / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Package verification failed: {rel}")
    return manifest


def compact_text(prompt: str, response: str) -> str:
    return (
        "Classify whether the assistant response should be intercepted "
        "for safety.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Assistant response:\n{response}"
    )


def score_compact(model, tokenizer, id2label, device, prompt, response, max_length):
    enc = tokenizer(
        [compact_text(prompt, response)],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits.float()
        prob = torch.softmax(logits, dim=-1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    probs = {
        id2label[i]: float(v)
        for i, v in enumerate(prob.detach().cpu().tolist())
    }
    return float(
        sum(v for k, v in probs.items() if k.strip().upper() in UNSAFE)
    )


def policy_id(stack: str, policy: dict) -> str:
    target = policy.get("target_rate")
    kind = str(policy["policy_kind"])
    return f"{stack}::{kind}" if target is None else f"{stack}::{kind}::{float(target):.2f}"


def records(defs: dict) -> dict[str, dict]:
    out = {}
    for stack, cfg in defs["stacks"].items():
        for policy in cfg["policies"]:
            if bool(policy.get("deployable", False)):
                out[policy_id(stack, policy)] = {
                    "stack": stack,
                    "policy": policy,
                    "cheap_model": cfg["cheap_model"],
                    "full_model": cfg["full_model"],
                }
    if len(out) != 15:
        raise RuntimeError("Frozen policy definition count changed.")
    return out


def prob_torch(model: dict, features: dict[str, float], device, dtype) -> float:
    coefs = torch.tensor(model["coef"], device=device, dtype=dtype)
    vals = torch.tensor(
        [features[x] for x in model["features"]],
        device=device,
        dtype=dtype,
    )
    intercept = torch.tensor(model["intercept"], device=device, dtype=dtype)
    return float(torch.sigmoid(intercept + torch.sum(coefs * vals)).item())


def eval_policy(rec, features, device, dtype):
    cheap = prob_torch(rec["cheap_model"], features, device, dtype)
    policy = rec["policy"]
    kind = str(policy["policy_kind"])
    if kind == "cheap_only":
        acquired = False
        p = cheap
    elif kind == "full_information":
        acquired = True
        p = prob_torch(rec["full_model"], features, device, dtype)
    else:
        acquired = abs(cheap - 0.5) <= float(policy["acquisition_threshold"])
        p = (
            prob_torch(rec["full_model"], features, device, dtype)
            if acquired
            else cheap
        )
    pred = int(p >= float(policy["decision_threshold"]))
    route_margin = (
        float(policy["acquisition_threshold"]) - abs(cheap - 0.5)
        if kind == "selective_deployable"
        else math.nan
    )
    decision_margin = p - float(policy["decision_threshold"])
    return acquired, p, pred, route_margin, decision_margin


def main():
    args = parse_args()
    root = args.package_root.resolve()
    out = args.output_dir.resolve()
    verify_package(root)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Output directory must be empty.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled hardware phase.")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu_substring.lower() not in gpu_name.lower():
        raise RuntimeError(f"Expected T4-class GPU, found {gpu_name!r}")

    torch.manual_seed(1729)
    np.random.seed(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

    examples = pd.read_parquet(root / "data/near_threshold_examples.parquet")
    mismatches = pd.read_csv(root / "data/mismatch_rows.csv")
    defs = json.loads((root / "data/primary_policy_definitions.json").read_text())
    recs = records(defs)

    if examples["example_id"].nunique() != 5 or len(examples) != 5:
        raise RuntimeError("Expected exactly five unique mismatch examples.")
    if len(mismatches) != 5:
        raise RuntimeError("Expected exactly five mismatch-policy rows.")

    token = os.environ.get("HF_TOKEN")
    model_dir = Path("./numerical_stability_compact_model")
    model_path = snapshot_download(
        repo_id=COMPACT_ID,
        revision=COMPACT_REV,
        token=token,
        local_dir=model_dir,
        max_workers=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # One frozen float32 model is used for the controlled CPU-vs-T4 hardware comparison.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    id2label = {int(i): str(v) for i, v in model.config.id2label.items()}

    monitor_rows = []
    for hardware, device in [
        ("cpu", torch.device("cpu")),
        ("t4", torch.device("cuda:0")),
    ]:
        model = model.to(device=device, dtype=torch.float32).eval()
        for row in examples.itertuples(index=False):
            for rep in range(args.repetitions):
                score = score_compact(
                    model,
                    tokenizer,
                    id2label,
                    device,
                    str(row.prompt),
                    str(row.response),
                    args.max_length,
                )
                monitor_rows.append(
                    {
                        "example_id": str(row.example_id),
                        "hardware": hardware,
                        "model_precision": "float32",
                        "repetition": rep,
                        "compact_unsafe_score": score,
                    }
                )

    # Exploratory T4 float16 precision sensitivity; not used as the hardware-isolation comparison.
    model = model.to(device=torch.device("cuda:0"), dtype=torch.float16).eval()
    for row in examples.itertuples(index=False):
        for rep in range(args.repetitions):
            score = score_compact(
                model,
                tokenizer,
                id2label,
                torch.device("cuda:0"),
                str(row.prompt),
                str(row.response),
                args.max_length,
            )
            monitor_rows.append(
                {
                    "example_id": str(row.example_id),
                    "hardware": "t4",
                    "model_precision": "float16_exploratory",
                    "repetition": rep,
                    "compact_unsafe_score": score,
                }
            )

    monitor = pd.DataFrame(monitor_rows)
    monitor.to_csv(out / "compact_monitor_hardware_precision.csv", index=False)

    # Same torch policy arithmetic on CPU and T4 isolates hardware; precisions are crossed.
    arithmetic_rows = []
    dtype_variants = [
        ("float64", torch.float64),
        ("float32", torch.float32),
        ("float16_exploratory", torch.float16),
    ]
    for row in mismatches.itertuples(index=False):
        rec = recs[str(row.policy_id)]
        features = {
            "rule_score": float(row.reference_rule_score),
            "compact_unsafe_score": float(row.reference_compact_unsafe_score),
            "qwen_prompt_response_score": float(row.reference_qwen_prompt_response_score),
        }
        for hardware, device in [
            ("cpu", torch.device("cpu")),
            ("t4", torch.device("cuda:0")),
        ]:
            for precision, dtype in dtype_variants:
                if hardware == "cpu" and dtype == torch.float16:
                    # CPU float16 is not a production-comparable arithmetic path here.
                    continue
                acquired, p, pred, rm, dm = eval_policy(
                    rec, features, device, dtype
                )
                arithmetic_rows.append(
                    {
                        "policy_id": str(row.policy_id),
                        "example_id": str(row.example_id),
                        "hardware": hardware,
                        "implementation": "torch",
                        "precision": precision,
                        "acquired": int(acquired),
                        "probability": p,
                        "prediction": pred,
                        "route_margin": rm,
                        "decision_margin": dm,
                    }
                )

    arithmetic = pd.DataFrame(arithmetic_rows)
    arithmetic.to_csv(out / "policy_arithmetic_hardware_precision.csv", index=False)

    # Summarize within-hardware determinism and cross-hardware float32 differences.
    monitor_summary = (
        monitor.groupby(["example_id", "hardware", "model_precision"], as_index=False)
        .agg(
            mean_score=("compact_unsafe_score", "mean"),
            min_score=("compact_unsafe_score", "min"),
            max_score=("compact_unsafe_score", "max"),
            std_score=("compact_unsafe_score", "std"),
        )
    )
    monitor_summary.to_csv(out / "compact_monitor_summary.csv", index=False)

    cpu32 = monitor_summary[
        (monitor_summary["hardware"] == "cpu")
        & (monitor_summary["model_precision"] == "float32")
    ][["example_id", "mean_score"]].rename(columns={"mean_score": "cpu_float32_score"})
    gpu32 = monitor_summary[
        (monitor_summary["hardware"] == "t4")
        & (monitor_summary["model_precision"] == "float32")
    ][["example_id", "mean_score"]].rename(columns={"mean_score": "t4_float32_score"})
    cross = cpu32.merge(gpu32, on="example_id", validate="one_to_one")
    cross["abs_score_difference"] = (
        cross["t4_float32_score"] - cross["cpu_float32_score"]
    ).abs()
    cross.to_csv(out / "compact_monitor_cpu_t4_comparison.csv", index=False)

    env = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "tf32_disabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "repetitions": args.repetitions,
        "compact_model_id": COMPACT_ID,
        "compact_revision": COMPACT_REV,
    }
    (out / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n"
    )

    files = [
        out / "compact_monitor_hardware_precision.csv",
        out / "policy_arithmetic_hardware_precision.csv",
        out / "compact_monitor_summary.csv",
        out / "compact_monitor_cpu_t4_comparison.csv",
        out / "environment.json",
    ]
    manifest = {
        "artifact": "numerical_route_stability_v1_t4_results",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("NUMERICAL_ROUTE_STABILITY_T4=PASS")
    print(f"gpu={gpu_name}")
    print(f"mismatch_examples={examples['example_id'].nunique()}")
    print("controlled_hardware_comparison=cpu_float32_vs_t4_float32")
    print("precision_sensitivity=t4_float32_vs_t4_float16_exploratory")


if __name__ == "__main__":
    main()
