#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3Guard-Gen-4B"
MODES = ("prompt_only", "response_only", "prompt_response")

SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)

KNOWN_CATEGORIES = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
    "None",
]
CATEGORY_RE = re.compile("|".join(re.escape(c) for c in KNOWN_CATEGORIES), re.IGNORECASE)
LABEL_TO_SCORE = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_guard_output(text: str) -> dict[str, Any]:
    text = text or ""
    safety_match = SAFETY_RE.search(text)
    refusal_match = REFUSAL_RE.search(text)

    label = None
    if safety_match:
        label = safety_match.group(1).capitalize()
        if label.lower() == "controversial":
            label = "Controversial"

    refusal = None
    if refusal_match:
        refusal = refusal_match.group(1).capitalize()

    categories = []
    seen = set()
    for match in CATEGORY_RE.findall(text):
        for known in KNOWN_CATEGORIES:
            if match.lower() == known.lower() and known not in seen:
                categories.append(known)
                seen.add(known)

    return {
        "label": label,
        "score": LABEL_TO_SCORE.get(label),
        "categories": categories,
        "refusal": refusal,
        "parse_ok": label is not None,
    }


def build_messages(prompt: str, response: str, mode: str) -> list[dict[str, str]]:
    prompt = "" if pd.isna(prompt) else str(prompt)
    response = "" if pd.isna(response) else str(response)

    if mode == "prompt_only":
        return [{"role": "user", "content": prompt}]
    if mode == "response_only":
        return [{"role": "user", "content": response}]
    if mode == "prompt_response":
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    raise ValueError(f"Unknown mode: {mode}")


def append_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_parquet(path)
        out = pd.concat([old_df, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=["example_id", "mode"], keep="last")
    else:
        out = new_df
    out.to_parquet(path, index=False)


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["example_id"])["example_id"].astype(str))


def infer_batch(
    *,
    batch: pd.DataFrame,
    mode: str,
    tokenizer,
    model,
    max_new_tokens: int,
    batch_size: int,
    model_revision: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages_list = [build_messages(r.prompt, r.response, mode) for r in batch.itertuples(index=False)]

    t0 = time.perf_counter()
    templated = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    inputs = tokenizer(templated, return_tensors="pt", padding=True, truncation=True).to(model.device)
    t1 = time.perf_counter()

    sync_cuda()
    t2 = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_cuda()
    t3 = time.perf_counter()

    input_width = int(inputs["input_ids"].shape[-1])
    rows = []
    raw_rows = []

    t4 = time.perf_counter()
    for i, source_row in enumerate(batch.itertuples(index=False)):
        output_ids = generated[i][input_width:].tolist()
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed = parse_guard_output(raw_output)

        row = {
            "example_id": str(source_row.example_id),
            "mode": mode,
            "qwen_official_score": parsed["score"],
            "qwen_official_label": parsed["label"],
            "qwen_official_categories": json.dumps(parsed["categories"], ensure_ascii=False),
            "qwen_official_refusal": parsed["refusal"],
            "qwen_official_parse_ok": bool(parsed["parse_ok"]),
            "qwen_official_raw_output": raw_output,
            "qwen_official_input_tokens": int(inputs["attention_mask"][i].sum().item()),
            "qwen_official_output_tokens": int(len(output_ids)),
            "qwen_official_model_id": MODEL_ID,
            "qwen_official_model_revision": model_revision,
            "qwen_official_device": str(model.device),
            "qwen_official_precision": str(getattr(model, "dtype", "unknown")),
            "qwen_official_batch_size": int(batch_size),
            "qwen_official_tokenize_latency_ms_batch": (t1 - t0) * 1000.0,
            "qwen_official_generate_latency_ms_batch": (t3 - t2) * 1000.0,
            "qwen_official_per_example_generate_latency_ms": ((t3 - t2) * 1000.0) / len(batch),
        }
        rows.append(row)
        raw_rows.append({
            "created_at": utc_now(),
            "example_id": str(source_row.example_id),
            "mode": mode,
            "raw_output": raw_output,
            "parsed": parsed,
        })

    t5 = time.perf_counter()
    for row in rows:
        row["qwen_official_decode_parse_latency_ms_batch"] = (t5 - t4) * 1000.0
        row["qwen_official_total_latency_ms_batch"] = (t5 - t0) * 1000.0
        row["qwen_official_per_example_total_latency_ms"] = ((t5 - t0) * 1000.0) / len(batch)

    return rows, raw_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/unified_dataset.parquet")
    parser.add_argument("--output-dir", default="data/interim/qwen3guard_official")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--warmup-examples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.dataset)
    missing = {"example_id", "prompt", "response"} - set(dataset.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if args.limit is not None:
        dataset = dataset.head(args.limit).copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    model.eval()
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)

    manifest = {
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "dataset": args.dataset,
        "modes": args.modes,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "warmup_examples": args.warmup_examples,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "method": "official_chat_template_generate_parse",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    raw_path = output_dir / "raw_generations.jsonl"
    for mode in args.modes:
        print(f"=== MODE {mode} ===")
        mode_path = output_dir / f"{mode}_scores.parquet"
        done = load_done_ids(mode_path)
        pending = dataset[~dataset["example_id"].astype(str).isin(done)].copy()
        print("total", len(dataset), "done", len(done), "pending", len(pending))

        if pending.empty:
            continue

        warmup = pending.head(args.warmup_examples)
        if len(warmup):
            infer_batch(
                batch=warmup,
                mode=mode,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                model_revision=model_revision,
            )

        buffer = []
        with raw_path.open("a", encoding="utf-8") as raw_file:
            for start in range(0, len(pending), args.batch_size):
                batch = pending.iloc[start:start + args.batch_size]
                rows, raw_rows = infer_batch(
                    batch=batch,
                    mode=mode,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    model_revision=model_revision,
                )
                buffer.extend(rows)
                for rr in raw_rows:
                    raw_file.write(json.dumps(rr, ensure_ascii=False) + "\n")
                if len(buffer) >= args.checkpoint_every:
                    append_parquet(mode_path, buffer)
                    buffer.clear()
                    print(mode, "saved through", start + len(batch), "/", len(pending))
        append_parquet(mode_path, buffer)
        scored = pd.read_parquet(mode_path)
        print(scored["qwen_official_label"].value_counts(dropna=False).to_string())

    combined = []
    for mode in args.modes:
        p = output_dir / f"{mode}_scores.parquet"
        if p.exists():
            combined.append(pd.read_parquet(p))
    if combined:
        out = pd.concat(combined, ignore_index=True)
        out.to_parquet(output_dir / "qwen3guard_official_all_modes.parquet", index=False)
        out.to_csv(output_dir / "qwen3guard_official_all_modes.csv", index=False)
        print("combined rows", len(out))


if __name__ == "__main__":
    main()
