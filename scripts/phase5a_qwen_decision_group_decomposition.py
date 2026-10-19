#!/usr/bin/env python3
"""Phase 5A: decision-group decomposition of archived Phase 4D FNR benefit.

Exploratory development-only replay. Uses the exact saved Phase4D stratification,
acquisition, allocation and RNG functions. Does not run GPUs, open sealed splits,
alter prior outputs, or use unpurchased optional scores for decisions.
Outputs only aggregates and scalar per-repetition errors; no examples/labels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BRANCH = "stage-a-paired-transport-20260921"
PHASE4D = "scripts/phase4d_optional_score_falsification.py"
ORIGINAL_ERRORS = "results/phase4d_optional_score_falsification_v1/paired_scalar_errors.csv"
ORIGINAL_MANIFEST = "results/phase4d_optional_score_falsification_v1/manifest.json"
SOURCE = "reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet"
FOLDS = "reports/decision_value_real_data/development_outer_fold_assignments.csv"
PROTOCOL = "configs/decision_value_real_data_protocol_v1.json"
BOUNDARY = "configs/exact_cost_risk_cascade_protocol_v2.json"
OUT = "results/phase5a_qwen_decision_group_decomposition_v1"
ARTIFACT = "phase5a_qwen_decision_group_decomposition_v1"

# These are already frozen in the saved Phase 4D manifest. Fail closed.
EXPECTED_INPUT_SHA256 = {
    SOURCE: "0041158b10e7ee0dd1189a0611c5b07f18a44d88355e3c150ac044e4112fe6ec",
    FOLDS: "62a2d8df0ba55b047d7e4685488b4d4d0ef77d480bcdb26cc34e5586e935d2dd",
    PROTOCOL: "7c82846f9de712a56260d74fbc1901b084262f2a7a745007547a75eab3d5f914",
    BOUNDARY: "96a0bfdf1a0954d9313ecd7a2ae1272a07f0df4ba6197eede2f0e0afd9f1c1c7",
}
METRICS = ("allowed_harm_A", "blocked_harm_B", "FNR")
CONTROLS = ("cheap_only", "acquisition_indicator", "permuted_optional")
OUTPUT_NAMES = (
    "decision_group_metrics.csv", "paired_decision_group_contrasts.csv",
    "paired_scalar_components.csv", "interpretation.md", "manifest.json",
    "phase5a_qwen_decision_group_decomposition.py"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_ratio(a: float, b: float) -> float:
    return a / (a + b) if a + b > 0 else float("nan")


def exact_shapley(a_hat: float, b_hat: float, a_true: float, b_true: float):
    """Exact symmetric split of FNR estimation error, NOT a causal attribution."""
    ft = safe_ratio(a_true, b_true)
    fab = safe_ratio(a_hat, b_hat)
    fa = safe_ratio(a_hat, b_true)
    fb = safe_ratio(a_true, b_hat)
    if not all(math.isfinite(v) for v in (ft, fab, fa, fb)):
        return (float("nan"), float("nan"))
    c_a = 0.5 * ((fa - ft) + (fab - fb))
    c_b = 0.5 * ((fb - ft) + (fab - fa))
    assert abs(c_a + c_b - (fab - ft)) < 1e-12
    return c_a, c_b


def test_math():
    a, b = 228 / 1687, 63 / 1687
    for aa, bb in [(a, b), (a*1.1, b*.9), (0, b), (a*0.5, b*2)]:
        ca, cb = exact_shapley(aa, bb, a, b)
        assert math.isclose(ca+cb, safe_ratio(aa, bb)-safe_ratio(a, b), abs_tol=1e-12)
    assert math.isnan(safe_ratio(0, 0))
    assert math.isclose(safe_ratio(a, b), 228/291)
    print("PHASE5A_MATH_TEST_PASS")


def import_phase4d(root: Path):
    p = root / PHASE4D
    spec = importlib.util.spec_from_file_location("phase4d_frozen_reference", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the original Phase 4D sampler")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("acquire", "make_groups", "allocate", "SEED", "CALLS", "REVIEWS", "REPS", "ARMS"):
        if not hasattr(mod, name):
            raise ValueError("Phase4D reference missing: "+name)
    assert mod.SEED == 20261009 and mod.REPS == 400
    assert tuple(mod.CALLS) == (80, 320) and tuple(mod.REVIEWS) == (80, 160)
    assert tuple(mod.ARMS) == ("cheap_only", "acquisition_indicator",
                              "permuted_optional", "real_optional")
    return mod


def components_from_draw(y, d, groups, weights, budget, seed, old):
    """Replay the SAME sampling mechanism as the archived draw_estimate."""
    alloc = old.allocate(groups, budget, weights)
    rng = np.random.default_rng(seed)
    n = len(y)
    allow, block = 0.0, 0.0
    for ids, nh in zip(groups, alloc):
        ix = rng.choice(ids, size=int(nh), replace=False)
        yy, dd = y[ix], d[ix]
        allow += len(ids) * float(np.mean(yy*(1-dd)))
        block += len(ids) * float(np.mean(yy*dd))
    return allow/n, block/n


def paired_replay(df, old):
    records = []
    truths = {}
    for setup_idx, setup in enumerate(("compact_after_rule", "qwen_after_rule_compact")):
        g = df[df.setup_id == setup].sort_values("example_id")
        ids = g.example_id.astype(str).to_numpy()
        y = g.y.astype(int).to_numpy()
        d = g.base_prediction.astype(int).to_numpy()
        cheap = g.base_uncertainty.astype(float).to_numpy()
        opt = g.optional_monitor_score.astype(float).to_numpy()
        n = len(y)
        assert n == 1687 and len(set(ids)) == n and int(y.sum()) == 291
        assert np.isin(y, (0,1)).all() and np.isin(d, (0,1)).all()
        assert np.isfinite(cheap).all() and np.isfinite(opt).all()
        assert np.all((opt >= 0) & (opt <= 1))
        at = float(np.mean(y*(1-d)))
        bt = float(np.mean(y*d))
        ft = safe_ratio(at, bt)
        truths[setup] = {
            "development_examples": n,
            "harmful_responses": int(y.sum()),
            "allowed_harmful_count": int(np.sum(y*(1-d))),
            "blocked_harmful_count": int(np.sum(y*d)),
            "allowed_harm_A": at,
            "blocked_harm_B": bt,
            "FNR": ft,
        }
        cheap_groups, cheap_weights = old.make_groups(d, cheap, mode="cheap_only")
        for calls in old.CALLS:
            for rep in range(old.REPS):
                aq_rng = np.random.default_rng(np.random.SeedSequence([old.SEED, setup_idx, calls, rep, 0]))
                acquired = old.acquire(cheap, ids, calls, aq_rng)
                shuffled = np.full(n, np.nan)
                perm_rng = np.random.default_rng(np.random.SeedSequence([old.SEED, setup_idx, calls, rep, 1]))
                for dd in (0, 1):
                    ix = np.flatnonzero((d == dd) & acquired)
                    shuffled[ix] = perm_rng.permutation(opt[ix])
                by_arm = {
                    "cheap_only": (cheap_groups, cheap_weights),
                    "acquisition_indicator": old.make_groups(d, cheap, acquired, mode="acquisition_indicator"),
                    "permuted_optional": old.make_groups(d, cheap, acquired, shuffled, mode="permuted_optional"),
                    "real_optional": old.make_groups(d, cheap, acquired, opt, mode="real_optional"),
                }
                for reviews in old.REVIEWS:
                    seed = np.random.SeedSequence([old.SEED, setup_idx, calls, rep, reviews, 2])
                    for arm in old.ARMS:
                        groups, weights = by_arm[arm]
                        ah, bh = components_from_draw(y, d, groups, weights, reviews, seed, old)
                        fh = safe_ratio(ah, bh)
                        ca, cb = exact_shapley(ah, bh, at, bt)
                        records.append({
                            "setup_id": setup, "monitor_calls": calls,
                            "human_reviews": reviews, "repetition": rep,
                            "arm": arm, "allowed_harm_A_error": ah-at,
                            "blocked_harm_B_error": bh-bt,
                            "FNR_error": fh-ft if math.isfinite(fh) else float("nan"),
                            "FNR_allowed_shapley": ca,
                            "FNR_blocked_shapley": cb,
                            # ORACLE diagnostics below, not deployable estimators:
                            "FNR_error_oracle_blocked_true": safe_ratio(ah, bt)-ft,
                            "FNR_error_oracle_allowed_true": safe_ratio(at, bh)-ft,
                        })
        print("REPLAY_SETUP_OK", setup, truths[setup]["allowed_harmful_count"],
              truths[setup]["blocked_harmful_count"], flush=True)
    return pd.DataFrame(records), truths


def validate_replay(frame: pd.DataFrame, original: pd.DataFrame):
    keys = ["setup_id","monitor_calls","human_reviews","repetition","arm"]
    target_cols = keys + ["miss_error","FNR_error"]
    if original[keys].duplicated().any() or frame[keys].duplicated().any():
        raise ValueError("Duplicate original/replayed seed/design keys")
    joined = frame.merge(original[target_cols], on=keys, how="outer",
                         validate="one_to_one", indicator=True, suffixes=("_new", "_original"))
    if len(joined) != 12800 or not (joined["_merge"] == "both").all():
        raise ValueError("Replay does not cover the exact 12800 original records")
    errors = {}
    for new_col, original_col in [
        ("allowed_harm_A_error", "miss_error"),
        ("FNR_error_new","FNR_error_original"),
    ]:
        a = joined[new_col].to_numpy(dtype=float)
        b = joined[original_col].to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(a),np.isfinite(b)):
            raise ValueError("Replay undefined-estimate mismatch: "+new_col)
        difference = float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)])))
        if difference > 5e-10:
            raise ValueError("The original Phase4D result was not reproduced: "+
                             new_col+" max abs difference "+str(difference))
        errors[new_col] = difference
    return errors


def rmse(x):
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr*arr))) if np.isfinite(arr).all() and len(arr) else float("nan")


def metrics(frame):
    rows = []
    for keys,g in frame.groupby(["setup_id","monitor_calls","human_reviews","arm"],sort=True):
        row = dict(zip(["setup_id","monitor_calls","human_reviews","arm"],keys))
        row["repetitions"] = len(g)
        for metric in METRICS:
            row["rmse_"+metric] = rmse(g[metric+"_error"])
            row["bias_"+metric] = float(g[metric+"_error"].mean())
        ca = g["FNR_allowed_shapley"].to_numpy()
        cb = g["FNR_blocked_shapley"].to_numpy()
        row["fnr_allowed_component_rms"] = rmse(ca)
        row["fnr_blocked_component_rms"] = rmse(cb)
        row["fnr_cross_term_2_mean_ca_cb"] = float(2*np.mean(ca*cb))
        row["fnr_oracle_blocked_true_rmse"] = rmse(g["FNR_error_oracle_blocked_true"])
        row["fnr_oracle_allowed_true_rmse"] = rmse(g["FNR_error_oracle_allowed_true"])
        if np.isfinite(ca).all() and np.isfinite(cb).all():
            mse_sum = np.mean(ca*ca)+np.mean(cb*cb)+2*np.mean(ca*cb)
            if not math.isclose(float(mse_sum), float(row["rmse_FNR"]**2),
                                rel_tol=1e-8, abs_tol=1e-12):
                raise AssertionError("Exact FNR error decomposition failed")
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(frame, boot=500):
    rows = []
    group_cols = ["setup_id","monitor_calls","human_reviews"]
    for keys,g in frame.groupby(group_cols,sort=True):
        lookup={metric:g.pivot(index="repetition",columns="arm",values=metric+"_error")
                for metric in METRICS}
        for metric,piv in lookup.items():
            a = piv["real_optional"].to_numpy(float)
            for ctl in CONTROLS:
                b = piv[ctl].to_numpy(float)
                valid = np.isfinite(a)&np.isfinite(b)
                aa,bb = a[valid],b[valid]
                if len(aa) < 300:
                    raise ValueError("Too few finite paired estimates")
                delta = rmse(aa)-rmse(bb)
                seed_arr=[20261013,0 if keys[0]=="compact_after_rule" else 1,
                          keys[1],keys[2],METRICS.index(metric),CONTROLS.index(ctl)]
                rng=np.random.default_rng(np.random.SeedSequence(seed_arr))
                samples=rng.integers(len(aa),size=(boot,len(aa)))
                diffs=np.sqrt(np.mean(aa[samples]**2,axis=1))-np.sqrt(np.mean(bb[samples]**2,axis=1))
                rows.append(dict(zip(group_cols,keys)) | {
                    "metric":metric,"comparison":"real_optional_minus_"+ctl,
                    "n_paired":len(aa),
                    "delta_RMSE":float(delta),
                    "mc_bootstrap_lo95":float(np.quantile(diffs,.025)),
                    "mc_bootstrap_hi95":float(np.quantile(diffs,.975))})
    return pd.DataFrame(rows)


def windows_downloads():
    try:
        profile=subprocess.check_output(
            ["powershell.exe","-NoProfile","-Command",
             '[Environment]::GetFolderPath("UserProfile")'],
            text=True,timeout=8).strip().replace("\r","")
        candidate=Path(subprocess.check_output(
            ["wslpath","-u",profile],text=True,timeout=5).strip())/"Downloads"
        if candidate.is_dir():return candidate
    except (OSError,subprocess.SubprocessError,ValueError):
        pass
    candidate=Path("/mnt/c/Users/user/Downloads")
    return candidate if candidate.is_dir() else None


def main():
    test_math()
    if "--self-test" in sys.argv:
        return 0
    root = Path.cwd().resolve()
    branch=subprocess.check_output(
        ["git","branch","--show-current"],cwd=root,text=True,timeout=10).strip()
    if branch!=BRANCH:
        raise ValueError("Wrong branch; do not run in "+branch)
    out=root/OUT
    if out.exists():
        raise FileExistsError("Preserving existing output: "+str(out))
    files=list(EXPECTED_INPUT_SHA256)+[PHASE4D,ORIGINAL_MANIFEST,ORIGINAL_ERRORS]
    missing=[p for p in files if not (root/p).is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen repository files: "+", ".join(missing))
    hashes={p:sha256(root/p) for p in EXPECTED_INPUT_SHA256}
    if hashes!=EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen development input hash mismatch")
    old_manifest=json.loads((root/ORIGINAL_MANIFEST).read_text())
    if old_manifest.get("source_sha256")!=hashes:
        raise ValueError("Frozen Phase4D manifest input hashes do not match")
    boundary=json.loads((root/BOUNDARY).read_text())
    permitted=set(boundary.get("data_boundary",{}).get("permitted_existing_development_artifacts",[]))
    if SOURCE not in permitted or FOLDS not in permitted:
        raise PermissionError("Development data boundary does not allow these inputs")
    old=import_phase4d(root)
    data=pd.read_parquet(root/SOURCE,columns=[
        "example_id","setup_id","outer_fold","y","base_prediction",
        "base_uncertainty","optional_monitor_score"])
    folds=pd.read_csv(root/FOLDS,usecols=["example_id","outer_fold"],
                      keep_default_na=False)
    expected=1687
    if len(data)!=2*expected or len(folds)!=expected or folds.example_id.nunique()!=expected:
        raise ValueError("Unexpected source/fold population sizes")
    if set(data.setup_id)!={"compact_after_rule","qwen_after_rule_compact"}:
        raise ValueError("Missing expected setups")
    if data.groupby("example_id").y.nunique().max()!=1:
        raise ValueError("Labels differ across historical setups")
    for s,g in data.groupby("setup_id"):
        if len(g)!=expected or g.example_id.nunique()!=expected or g.y.sum()!=291:
            raise ValueError("Unexpected development rows/labels in "+s)
        check=g[["example_id","outer_fold"]].merge(
            folds,on="example_id",how="left",validate="one_to_one",
            suffixes=("_source","_frozen"))
        if len(check)!=expected or check.outer_fold_frozen.isna().any() or \
                not (check.outer_fold_source.astype(str)==
                     check.outer_fold_frozen.astype(str)).all():
            raise ValueError("Fold mismatch in "+s)
    print("PHASE5A_INPUT_CHECK_PASS",flush=True)
    replay,truth=paired_replay(data,old)
    original=pd.read_csv(root/ORIGINAL_ERRORS)
    max_diff=validate_replay(replay,original)
    summary=metrics(replay)
    pairs=contrasts(replay)
    out.mkdir(parents=True,exist_ok=False)
    replay.to_csv(out/"paired_scalar_components.csv",index=False,
                  float_format="%.12g",lineterminator="\n")
    summary.to_csv(out/"decision_group_metrics.csv",index=False,
                   float_format="%.12g",lineterminator="\n")
    pairs.to_csv(out/"paired_decision_group_contrasts.csv",index=False,
                 float_format="%.12g",lineterminator="\n")
    report=[
        "# Phase 5A: Decomposing the Qwen estimation signal by decision group", "",
        "## Design and scientific boundaries", "",
        "- Exploratory **historical development** replay of saved Phase 4D, not prospective confirmation.",
        "- A=P(Y=1,D=0), B=P(Y=1,D=1), FNR=A/(A+B).",
        "- D is **archived cross-fitted classifier**, not one frozen native deployed guard.",
        "- Optional Qwen/Koala scores are treated as bought **only on simulated acquired cases**.",
        "- This replay exactly matches the original seed/arm-level Phase 4D scalar errors.",
        "- Original scripts, experimental data, labels and 0/4 Stage-A screen remain unchanged.",
        "- This reports **estimation** variance, not an improvement in native monitor decision quality.",
        "- Primary future estimand is A; FNR is secondary, pending supervisor confirmation.", "",
        "## Source development population (not a real-world traffic guarantee)", "",
        "| Setup | N | Harmful Y=1 | Allowed harmful | Blocked harmful | A | B | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s,t in truth.items():
        report.append(f"| {s} | {t['development_examples']} | {t['harmful_responses']} "
            f"| {t['allowed_harmful_count']} | {t['blocked_harmful_count']} "
            f"| {t['allowed_harm_A']:.5f} | {t['blocked_harm_B']:.5f} | {t['FNR']:.5f} |")
    report+=["", "## How to read the decomposition", "",
        "- `rmse_allowed_harm_A` evaluates the direct missed-harm/exposure estimand.",
        "- `rmse_blocked_harm_B` evaluates harmful-response mass found among blocked cases.",
        "- `FNR_allowed_shapley` and `FNR_blocked_shapley` are an **exact symmetric algebraic decomposition** of each FNR error (not a causal partition).",
        "- Total FNR squared error = mean(CA²)+mean(CB²)+2 mean(CA CB).",
        "- Negative or positive covariance/cross terms can alter total FNR RMSE, so neither component alone should be claimed to explain all improvement.",
        "- The two `oracle_...` columns hypothetically insert the true value of one group and are **diagnostics only, not feasible estimators**.",
        "- Bootstrap intervals summarize Monte Carlo uncertainty in this fixed historical population, **not** confidence coverage for deployment.",
        "- Compare A, B, FNR jointly before deciding that any purchased score adds relevant practical value.",
        "- Existing-data baseline comparison, StratPPI, Active MultiPPI, independent response labels and full costs still remain before a paper claim.",
        "",
        "## Actual outputs", "",
        f"- Replay error rows: {len(replay)}; aggregate rows: {len(summary)}; paired contrasts: {len(pairs)}.",
        f"- Maximum original-versus-replay error discrepancies: {max_diff}.",
        "", "### Qwen result at 320 calls / 160 human reviews", "",
        "| Arm | A RMSE | B RMSE | FNR RMSE |",
        "|---|---:|---:|---:|",
    ]
    sub=summary[(summary.setup_id=="qwen_after_rule_compact")&
                (summary.monitor_calls==320)&(summary.human_reviews==160)]
    for _,item in sub.sort_values("arm").iterrows():
        report.append(f"| {item['arm']} | {item['rmse_allowed_harm_A']:.5f} "
                      f"| {item['rmse_blocked_harm_B']:.5f} "
                      f"| {item['rmse_FNR']:.5f} |")
    report+=["", "All findings need to be checked for actual utility, including full human/monitor costs and independently verified labels.", ""]
    (out/"interpretation.md").write_text("\n".join(report),encoding="utf-8")
    manifest={
        "artifact_id":ARTIFACT, "status":"exploratory_historical_development_only",
        "primary_future_estimand":"P(Y=1,D=0)",
        "secondary_future_estimand":"P(D=0|Y=1)",
        "decision_timing_proposal":"post-generation before delivery; NOT verified in legacy decisions",
        "historical_decision":"base_prediction, archived cross-fitted classifier",
        "source_branch":BRANCH, "source_sha256":hashes,
        "phase4d_script_sha256":sha256(root/PHASE4D),
        "phase5a_script_sha256":sha256(Path(__file__).resolve()),
        "phase4d_errors_sha256":sha256(root/ORIGINAL_ERRORS),
        "phase4d_manifest_sha256":sha256(root/ORIGINAL_MANIFEST),
        "full_replay_seed_level_errors_verified":True,
        "max_original_vs_replayed_errors":max_diff,
        "population_true_by_historical_setup":truth,
        "repetitions":old.REPS, "calls":list(old.CALLS),
        "human_review_budgets":list(old.REVIEWS),
        "all_source_data_development_only":True,
        "no_protected_rows_read":True, "no_raw_records_exported":True,
        "label_provenance":"historical labels, not independently re-annotated",
        "native_guard_validation":"not attempted",
        "cost_provenance":"unchanged historical optional call times; reviewer costs unmeasured",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,
                                        sort_keys=True)+"\n",encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), out/"phase5a_qwen_decision_group_decomposition.py")
    archive=out/(ARTIFACT+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for name in OUTPUT_NAMES:
            z.write(out/name,arcname=name)
    folder=windows_downloads()
    if folder is not None:
        dst=folder/archive.name
        if dst.exists():
            print("WINDOWS_DOWNLOADS_EXISTS; preserving:",dst)
        else:
            shutil.copy2(archive,dst)
            print("WINDOWS_DOWNLOADS:",dst)
    else:
        print("Results archive:",archive)
    print("PHASE5A_SUCCESS",len(replay),len(summary),len(pairs))
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("PHASE5A_STOP:",type(exc).__name__,str(exc))
        sys.exit(2)