#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import sklearn
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
SAFETY_RE = re.compile(
    r"Safety:\s*(Safe|Unsafe|Controversial)", re.I
)
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.I)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", type=Path, default=Path("."))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final_v3_policy_timing_results"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--warmup-rows", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--expected-gpu-substring", default="T4")
    return p.parse_args()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync():
    torch.cuda.synchronize()


def parse_guard(text: str):
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
        "refusal": refusal.group(1).capitalize() if refusal else None,
        "parse_ok": label is not None,
    }


def compact_text(prompt: str, response: str):
    return (
        "Classify whether the assistant response should be intercepted "
        "for safety.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Assistant response:\n{response}"
    )


def stats(component, name, values, call_rate=None):
    x = np.asarray(values, dtype=float)
    return {
        "component": component,
        "mode_or_policy": name,
        "n": len(x),
        "mean_ms": float(x.mean()),
        "std_ms": float(x.std(ddof=1)),
        "min_ms": float(x.min()),
        "p50_ms": float(np.quantile(x, 0.50)),
        "p95_ms": float(np.quantile(x, 0.95)),
        "p99_ms": float(np.quantile(x, 0.99)),
        "max_ms": float(x.max()),
        "expensive_call_rate": call_rate,
    }


class Runtime:
    def __init__(self, root, device, max_length, max_new_tokens, token):
        self.root = root
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        sys.path.insert(0, str(root / "src"))
        from monitor_fusion.monitors.rule_filter import (
            score_prompt_response,
        )
        self.rule_fn = score_prompt_response

        self.bundle = joblib.load(
            root / "artifacts/fusion_models_v3/fusion_bundle.joblib"
        )
        payload = json.loads(
            (
                root
                / "artifacts/fusion_models_v3/"
                "frozen_operating_points_v3.json"
            ).read_text()
        )
        self.op = payload["operating_points"]["0.050"]

        compact_dir = root / "models/compact"
        qwen_dir = root / "models/qwen"
        compact_path = snapshot_download(
            repo_id=COMPACT_ID,
            revision=COMPACT_REV,
            token=token,
            local_dir=compact_dir,
            allow_patterns=[
                "*.json", "*.txt", "*.safetensors", "*.model"
            ],
            max_workers=1,
        )
        self.ctok = AutoTokenizer.from_pretrained(
            compact_path, local_files_only=True
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

        qwen_path = snapshot_download(
            repo_id=QWEN_ID,
            revision=QWEN_REV,
            token=token,
            local_dir=qwen_dir,
            allow_patterns=[
                "*.json", "*.txt", "*.safetensors",
                "*.model", "*.jinja"
            ],
            max_workers=1,
        )
        self.qtok = AutoTokenizer.from_pretrained(
            qwen_path, local_files_only=True
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

    def rule(self, prompt, response):
        t0 = time.perf_counter_ns()
        score = float(
            self.rule_fn(prompt, response)["rule_score"]
        )
        return score, (time.perf_counter_ns() - t0) / 1e6

    def compact(self, prompt, response):
        sync()
        t0 = time.perf_counter_ns()
        enc = self.ctok(
            [compact_text(prompt, response)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            prob = torch.softmax(
                self.cmodel(**enc).logits.float(), dim=-1
            )[0]
        sync()
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        probs = {
            self.id2label[i]: float(value)
            for i, value in enumerate(prob.detach().cpu().tolist())
        }
        score = sum(
            value
            for label, value in probs.items()
            if label.strip().upper() in UNSAFE
        )
        return float(min(max(score, 0.0), 1.0)), elapsed

    def qwen(self, prompt, response):
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
            output_ids, skip_special_tokens=True
        ).strip()
        parsed = parse_guard(raw)
        t4 = time.perf_counter_ns()

        if not parsed["parse_ok"]:
            raise RuntimeError(f"Qwen parse failure: {raw!r}")

        return {
            **parsed,
            "raw_output": raw,
            "input_tokens": int(
                inputs["attention_mask"][0].sum().item()
            ),
            "output_tokens": len(output_ids),
            "tokenize_transfer_latency_ms": (t1 - t0) / 1e6,
            "generate_latency_ms": (t3 - t2) / 1e6,
            "decode_parse_latency_ms": (t4 - t3) / 1e6,
            "total_latency_ms": (t4 - t0) / 1e6,
        }

    def full(self, prompt, response):
        sync()
        t0 = time.perf_counter_ns()
        rule, rule_ms = self.rule(prompt, response)
        compact, compact_ms = self.compact(prompt, response)
        q = self.qwen(prompt, response)
        frame = pd.DataFrame([{
            "rule_score": rule,
            "compact_unsafe_score": compact,
            "qwen_prompt_response_score": float(q["score"]),
        }])
        probability = float(
            self.bundle["full_information_fusion"].predict_proba(
                frame[self.bundle["full_features"]]
            )[0, 1]
        )
        pred = int(
            probability
            >= float(self.op["full_information"]["threshold"])
        )
        sync()
        return {
            "prediction": pred,
            "probability": probability,
            "called_expensive": 1,
            "route": "full_information",
            "rule_score": rule,
            "compact_score": compact,
            "qwen_score": float(q["score"]),
            "cheap_probability": math.nan,
            "full_probability": probability,
            "rule_latency_ms": rule_ms,
            "compact_latency_ms": compact_ms,
            "qwen_total_latency_ms": q["total_latency_ms"],
            "total_latency_ms": (
                time.perf_counter_ns() - t0
            ) / 1e6,
        }, q

    def selective(self, prompt, response):
        sync()
        t0 = time.perf_counter_ns()
        rule, rule_ms = self.rule(prompt, response)
        compact, compact_ms = self.compact(prompt, response)
        cheap_frame = pd.DataFrame([{
            "rule_score": rule,
            "compact_unsafe_score": compact,
        }])
        cheap = float(
            self.bundle["cheap_router"].predict_proba(
                cheap_frame[self.bundle["cheap_features"]]
            )[0, 1]
        )
        op = self.op["selective_acquisition"]
        low = float(op["router_low_allow_threshold"])
        high = float(op["router_high_intercept_threshold"])

        q = None
        q_score = math.nan
        q_ms = 0.0
        full_prob = math.nan
        if cheap <= low:
            route, pred, called, probability = (
                "cheap_allow", 0, 0, cheap
            )
        elif cheap >= high:
            route, pred, called, probability = (
                "cheap_intercept", 1, 0, cheap
            )
        else:
            route, called = "expensive_fusion", 1
            q = self.qwen(prompt, response)
            q_score = float(q["score"])
            q_ms = q["total_latency_ms"]
            full_frame = pd.DataFrame([{
                "rule_score": rule,
                "compact_unsafe_score": compact,
                "qwen_prompt_response_score": q_score,
            }])
            full_prob = float(
                self.bundle[
                    "full_information_fusion"
                ].predict_proba(
                    full_frame[self.bundle["full_features"]]
                )[0, 1]
            )
            probability = full_prob
            pred = int(
                full_prob
                >= float(op["full_fusion_threshold"])
            )
        sync()
        return {
            "prediction": pred,
            "probability": probability,
            "called_expensive": called,
            "route": route,
            "rule_score": rule,
            "compact_score": compact,
            "qwen_score": q_score,
            "cheap_probability": cheap,
            "full_probability": full_prob,
            "rule_latency_ms": rule_ms,
            "compact_latency_ms": compact_ms,
            "qwen_total_latency_ms": q_ms,
            "total_latency_ms": (
                time.perf_counter_ns() - t0
            ) / 1e6,
        }, q


def main():
    a = args()
    root = a.package_root.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    gpu = torch.cuda.get_device_name(0)
    if a.expected_gpu_substring.lower() not in gpu.lower():
        raise RuntimeError(
            f"Expected {a.expected_gpu_substring!r}; found {gpu!r}"
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a Hugging Face read token")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    sample_path = root / "data/benchmark_sample_v3.parquet"
    bundle_path = (
        root / "artifacts/fusion_models_v3/fusion_bundle.joblib"
    )
    points_path = (
        root / "artifacts/fusion_models_v3/"
        "frozen_operating_points_v3.json"
    )
    sample = pd.read_parquet(sample_path)
    assert len(sample) == 128
    assert sample["example_id"].nunique() == 128
    assert sample["split"].eq("calibration").all()

    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        (out / "nvidia_smi.txt").write_text(
            smi.stdout + smi.stderr
        )
    except FileNotFoundError:
        (out / "nvidia_smi.txt").write_text(
            "nvidia-smi unavailable\n"
        )

    runtime = Runtime(
        root,
        a.device,
        a.max_length,
        a.max_new_tokens,
        token,
    )

    warm = sample.head(a.warmup_rows)
    for r in warm.itertuples(index=False):
        runtime.full(str(r.prompt), str(r.response))
    for r in warm.itertuples(index=False):
        runtime.selective(str(r.prompt), str(r.response))

    policy_rows, qwen_rows, raw_rows = [], [], []
    for index, r in enumerate(sample.itertuples(index=False)):
        ordered = (
            [
                ("full_information_always_on", runtime.full),
                ("selective_acquisition", runtime.selective),
            ]
            if index % 2 == 0
            else [
                ("selective_acquisition", runtime.selective),
                ("full_information_always_on", runtime.full),
            ]
        )
        for position, (name, fn) in enumerate(ordered, 1):
            result, q = fn(str(r.prompt), str(r.response))
            policy_rows.append({
                "example_id": str(r.example_id),
                "policy": name,
                "order_position": position,
                "batch_size": 1,
                **result,
            })
            if q is not None:
                qwen_rows.append({
                    "example_id": str(r.example_id),
                    "policy": name,
                    "mode": "prompt_response",
                    **{
                        k: v
                        for k, v in q.items()
                        if k != "raw_output"
                    },
                })
                raw_rows.append({
                    "created_at": now(),
                    "example_id": str(r.example_id),
                    "policy": name,
                    "raw_output": q["raw_output"],
                    "parsed": {
                        "label": q["label"],
                        "score": q["score"],
                        "refusal": q["refusal"],
                        "parse_ok": q["parse_ok"],
                    },
                })
        if (index + 1) % 16 == 0:
            print(f"completed {index + 1}/128", flush=True)

    policy = pd.DataFrame(policy_rows)
    qwen = pd.DataFrame(qwen_rows)
    expected = sample[[
        "example_id",
        "rule_score",
        "compact_unsafe_score",
        "qwen_prompt_response_score",
    ]].rename(columns={
        "rule_score": "expected_rule",
        "compact_unsafe_score": "expected_compact",
        "qwen_prompt_response_score": "expected_qwen",
    })
    check = policy.merge(
        expected, on="example_id", validate="many_to_one"
    )
    rule_diff = float(
        np.abs(check["rule_score"] - check["expected_rule"]).max()
    )
    compact_diff = float(
        np.abs(
            check["compact_score"] - check["expected_compact"]
        ).max()
    )
    called = check["called_expensive"].eq(1)
    qwen_diff = float(
        np.abs(
            check.loc[called, "qwen_score"]
            - check.loc[called, "expected_qwen"]
        ).max()
    )
    if rule_diff > 1e-12:
        raise RuntimeError(f"Rule mismatch: {rule_diff}")
    if compact_diff > 1e-4:
        raise RuntimeError(f"Compact mismatch: {compact_diff}")
    qwen_check = qwen.merge(
        expected[["example_id", "expected_qwen"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )
    qwen_check["abs_score_difference"] = np.abs(
        qwen_check["score"] - qwen_check["expected_qwen"]
    )
    mismatches = qwen_check[
        qwen_check["abs_score_difference"] > 0
    ].copy()
    mismatches.to_csv(
        out / "qwen_score_mismatches.csv",
        index=False,
    )
    if qwen_diff > 0:
        print(
            "WARNING: regenerated Qwen scores differ from cached v3 "
            "scores; "
            f"max_abs_difference={qwen_diff}, "
            f"mismatch_rows={len(mismatches)}, "
            f"unique_examples={mismatches['example_id'].nunique()}",
            flush=True,
        )
    if not qwen["parse_ok"].all():
        raise RuntimeError("Qwen parse failure")

    policy.to_parquet(out / "policy_latency.parquet", index=False)
    qwen.to_parquet(out / "qwen_stage_latency.parquet", index=False)
    with (out / "qwen_raw_generations.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    for name, group in policy.groupby("policy", sort=True):
        summary.append(stats(
            "end_to_end_policy",
            name,
            group["total_latency_ms"],
            float(group["called_expensive"].mean()),
        ))
    full_group = policy[
        policy["policy"].eq("full_information_always_on")
    ]
    summary.append(stats(
        "rule_filter_v1",
        "prompt_response",
        full_group["rule_latency_ms"],
    ))
    summary.append(stats(
        "koala_text_moderation",
        "prompt_response",
        full_group["compact_latency_ms"],
    ))
    qwen_full = qwen[
        qwen["policy"].eq("full_information_always_on")
    ]
    summary.append(stats(
        "qwen3guard_official",
        "prompt_response",
        qwen_full["total_latency_ms"],
    ))
    latency = pd.DataFrame(summary)
    latency.to_csv(out / "latency_summary.csv", index=False)

    stage_rows = []
    for name, group in qwen.groupby("policy", sort=True):
        for stage in [
            "tokenize_transfer_latency_ms",
            "generate_latency_ms",
            "decode_parse_latency_ms",
            "total_latency_ms",
        ]:
            row = stats(
                "qwen3guard_official_stage",
                f"{name}:{stage}",
                group[stage],
            )
            row["policy"] = name
            row["stage"] = stage
            stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(
        out / "qwen_stage_latency_summary.csv",
        index=False,
    )

    routes = (
        policy[policy["policy"].eq("selective_acquisition")]
        .groupby("route", sort=True)
        .agg(
            n=("example_id", "size"),
            mean_latency_ms=("total_latency_ms", "mean"),
            p50_latency_ms=(
                "total_latency_ms",
                lambda x: np.quantile(x, 0.50),
            ),
            p95_latency_ms=(
                "total_latency_ms",
                lambda x: np.quantile(x, 0.95),
            ),
        )
        .reset_index()
    )
    routes["rate"] = routes["n"] / 128.0
    routes.to_csv(
        out / "selective_route_summary.csv", index=False
    )

    full = full_group["total_latency_ms"].to_numpy(float)
    sel_group = policy[
        policy["policy"].eq("selective_acquisition")
    ]
    sel = sel_group["total_latency_ms"].to_numpy(float)
    comparison = {
        "full_mean_ms": float(full.mean()),
        "selective_mean_ms": float(sel.mean()),
        "mean_reduction_ms": float(full.mean() - sel.mean()),
        "mean_reduction_rate": float(
            1 - sel.mean() / full.mean()
        ),
        "full_p50_ms": float(np.quantile(full, 0.50)),
        "selective_p50_ms": float(np.quantile(sel, 0.50)),
        "full_p95_ms": float(np.quantile(full, 0.95)),
        "selective_p95_ms": float(np.quantile(sel, 0.95)),
        "full_p99_ms": float(np.quantile(full, 0.99)),
        "selective_p99_ms": float(np.quantile(sel, 0.99)),
        "selective_expensive_call_rate": float(
            sel_group["called_expensive"].mean()
        ),
    }

    manifest = {
        "artifact": "final_v3_policy_timing_gpu",
        "created_at": now(),
        "status": "completed_fixed_hardware_measurement",
        "benchmark_rows": 128,
        "benchmark_split": "calibration",
        "benchmark_sample_sha256": sha256(sample_path),
        "batch_size": 1,
        "warmup_rows_per_policy": a.warmup_rows,
        "warmup_excluded": True,
        "policy_order": "alternated_per_example",
        "clock": "time.perf_counter_ns",
        "cuda_synchronize": True,
        "model_load_time_included": False,
        "tokenization_transfer_included": True,
        "qwen_decode_parse_included": True,
        "policy_decision_logic_included": True,
        "model_refit_during_timing": False,
        "gpu": {
            "name": gpu,
            "device": a.device,
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "models": {
            "compact": {
                "model_id": COMPACT_ID,
                "revision": COMPACT_REV,
            },
            "qwen": {
                "model_id": QWEN_ID,
                "revision": QWEN_REV,
                "method": (
                    "official_chat_template_generate_structured_parse"
                ),
            },
        },
        "serialized_policy": {
            "bundle_sha256": sha256(bundle_path),
            "operating_points_sha256": sha256(points_path),
            "target_fpr_key": "0.050",
            "policy_version": "v3_author_reviewed_labels",
        },
        "score_validation": {
            "rule_max_abs_difference": rule_diff,
            "compact_max_abs_difference": compact_diff,
            "qwen_max_abs_difference": qwen_diff,
            "qwen_mismatch_rows": int(len(mismatches)),
            "qwen_mismatch_unique_examples": int(
                mismatches["example_id"].nunique()
            ),
            "qwen_exact_match_rate": float(
                1.0 - len(mismatches) / len(qwen_check)
            ),
            "qwen_exact_parity_required_for_timing": False,
            "interpretation": (
                "Latency uses regenerated official Qwen outputs. "
                "Differences from cached v3 classifications are recorded "
                "as measurement-reproducibility evidence."
            ),
        },
        "qwen_parse_rate": float(qwen["parse_ok"].mean()),
        "policy_comparison": comparison,
        "limitations": [
            "Fixed-run benchmark, not a real-time guarantee.",
            "Results are specific to the recorded GPU and software.",
            "The 128 audited-label calibration examples are reused "
            "from the controlled timing sample.",
        ],
    }
    (out / "README.md").write_text(
        "# Final v3 policy timing benchmark\n\n"
        f"GPU: `{gpu}`\n\n"
        f"Selective Qwen call rate: "
        f"`{comparison['selective_expensive_call_rate']:.6f}`\n\n"
        f"Mean latency reduction: "
        f"`{comparison['mean_reduction_rate']:.6f}`\n"
    )
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            manifest.setdefault("files", {})[path.name] = sha256(path)
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    archive = shutil.make_archive(str(out), "zip", root_dir=out)

    print("\n=== LATENCY SUMMARY ===")
    print(latency.to_string(index=False))
    print("\n=== SELECTIVE ROUTES ===")
    print(routes.to_string(index=False))
    print("\n=== POLICY COMPARISON ===")
    print(json.dumps(comparison, indent=2))
    print("\nmanifest:", out / "run_manifest.json")
    print("results archive:", archive)


if __name__ == "__main__":
    main()
