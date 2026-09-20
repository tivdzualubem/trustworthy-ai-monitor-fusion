#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/trustworthy-ai-monitor-fusion"
OUTDIR="/mnt/c/Users/user/Downloads/external_validation_independent_block_feasibility_v1"
mkdir -p "$OUTDIR"
cd "$REPO"

PY=".venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found"; exit 1; }
[[ "$(git branch --show-current)" == "external-validation-precollection-freeze" ]] || {
  echo "ERROR: wrong branch: $(git branch --show-current)"; exit 1;
}

"$PY" - <<'PY' | tee "$OUTDIR/independent_block_feasibility_v1.txt"
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist, norm

ROOT = Path.cwd()
OUTDIR = Path('/mnt/c/Users/user/Downloads/external_validation_independent_block_feasibility_v1')
OUT_JSON = OUTDIR / 'independent_block_feasibility_v1.json'
OUT_CSV = OUTDIR / 'independent_block_power_grid_v1.csv'

# -----------------------------------------------------------------------------
# 0. Load the currently authoritative local contracts.
# -----------------------------------------------------------------------------
def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

redesign = load('configs/external_validation_confirmatory_redesign_v2.json')
pop = load('configs/external_validation_population_sampling_contract_v1.json')
prereg = load('configs/safety_monitor_external_validation_preregistration_v1.json')
source = load('configs/external_validation_source_generation_contract_v1.json')
truth = load('data/metadata/external_validation_preW0_truth_repair_v1.json')

margin = float(redesign['confirmed_primary_preservation_margin']['primary_margin'])
fnr_ceiling = float(redesign['confirmed_absolute_fnr_criterion']['maximum_FNR'])
alpha_safety = float(redesign['confirmatory_error_control']['primary_safety_one_sided_alpha'])
fpr_ceiling = float(redesign['confirmatory_error_control']['FPR_availability_family']['operating_constraint'])
alpha_fpr = float(redesign['confirmatory_error_control']['FPR_availability_family']['cellwise_one_sided_alpha'])

print('=== INDEPENDENT-BLOCK CONFIRMATORY REDESIGN FEASIBILITY v1 ===')
print('This is a READ-ONLY candidate-design study. It does not freeze the protocol, quotas, W0, or scoring.')
print()
print('LOCAL SCIENTIFIC STATE')
print(f"W0_authorized: {truth['W0_authorized']}")
print(f"population contract status: {pop['status']}")
print(f"primary FNR NI margin: {margin:.3f}")
print(f"absolute FNR ceiling: {fnr_ceiling:.3f}")
print(f"one-sided safety alpha: {alpha_safety:.3f}")
print(f"FPR ceiling: {fpr_ceiling:.3f} at alpha={alpha_fpr:.3f}")
print(f"model-source rows per generator batch: {source['independent_batch_mechanism']['rows_per_generator_batch']}")
print()

# The repo has already completed the abstract population/block redesign. The still-open
# scientific problem is a valid primary inference + power design. This candidate asks:
# what happens if every primary FNR/FPR analysis uses at most ONE retained response per
# independent provenance block? For model cells this is already the frozen generation
# mechanism. For human cells it means one retained Y=1 response per author for FNR and
# one retained Y=0 response per author for FPR, with no author reused across cells.
# This is evaluated only as a feasibility candidate, not silently frozen.

candidate = {
    'name': 'singleton_primary_analysis_block_candidate',
    'human_FNR_rule': 'at most one retained eligible Y=1 dependency representative per author within its assigned cell',
    'human_FPR_rule': 'at most one retained eligible Y=0 dependency representative per author within its assigned cell',
    'model_rule': 'one response per generator_batch_id (already frozen)',
    'cross_cell_reuse': False,
    'estimand': 'response-average marginal FNR/FPR over retained independent primary analysis units',
    'status': 'candidate_feasibility_only_not_frozen',
}

print('CANDIDATE CLEAN-INDEPENDENCE DESIGN')
for k,v in candidate.items():
    print(f"{k}: {v}")
print()

# -----------------------------------------------------------------------------
# 1. Primary two-sample FNR NI candidate: Farrington-Manning-style constrained
#    score test for independent binomial samples. We validate its boundary error
#    empirically before using it in power calculations.
# -----------------------------------------------------------------------------
Z_ALPHA = float(norm.ppf(alpha_safety))


def fm_reject_vec(x_ref, n_ref, x_shift, n_shift, delta=margin, alpha=alpha_safety):
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_shift = np.asarray(x_shift, dtype=np.float64)
    eps = 1e-10

    # Constrained MLE under H0 boundary p_shift = p_ref + delta.
    p = (x_ref + x_shift - n_shift * delta) / (n_ref + n_shift)
    p = np.clip(p, eps, 1.0 - delta - eps)

    for _ in range(35):
        p_shift = p + delta
        score = (
            x_ref / p - (n_ref - x_ref) / (1.0 - p)
            + x_shift / p_shift - (n_shift - x_shift) / (1.0 - p_shift)
        )
        deriv = (
            -x_ref / p**2 - (n_ref - x_ref) / (1.0 - p)**2
            -x_shift / p_shift**2 - (n_shift - x_shift) / (1.0 - p_shift)**2
        )
        p_new = np.clip(p - score / deriv, eps, 1.0 - delta - eps)
        if np.max(np.abs(p_new - p)) < 1e-12:
            p = p_new
            break
        p = p_new

    p_shift = p + delta
    var0 = p * (1.0 - p) / n_ref + p_shift * (1.0 - p_shift) / n_shift
    z = ((x_shift / n_shift - x_ref / n_ref) - delta) / np.sqrt(var0)
    return z < norm.ppf(alpha)


def cp_upper_lookup(n: int, alpha: float):
    x = np.arange(n + 1)
    out = np.ones(n + 1, dtype=np.float64)
    mask = x < n
    out[mask] = beta_dist.ppf(1.0 - alpha, x[mask] + 1, n - x[mask])
    return out


def mc_ci(successes: int, reps: int):
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(0.025, successes, reps-successes+1))
    hi = 1.0 if successes == reps else float(beta_dist.ppf(0.975, successes+1, reps-successes))
    return lo, hi

# -----------------------------------------------------------------------------
# 2. Boundary type-I calibration in the scientifically relevant low-FNR region.
# -----------------------------------------------------------------------------
TYPEI_REPS = 100_000
TYPEI_N = [600, 1000, 1400, 1800]
TYPEI_PREF = [0.01, 0.03, 0.05, 0.07]  # shifted boundary is +0.03 => <=0.10
rng_master = np.random.default_rng(20260919)

typei_rows = []
print('=== BOUNDARY TYPE-I CALIBRATION: RELATIVE FNR NI ===')
for n in TYPEI_N:
    for p_ref in TYPEI_PREF:
        p_shift = p_ref + margin
        seed = int(rng_master.integers(1, 2**31-1))
        rng = np.random.default_rng(seed)
        xr = rng.binomial(n, p_ref, TYPEI_REPS)
        xs = rng.binomial(n, p_shift, TYPEI_REPS)
        rej = fm_reject_vec(xr, n, xs, n)
        k = int(rej.sum())
        rate = k / TYPEI_REPS
        lo, hi = mc_ci(k, TYPEI_REPS)
        clear_fail = lo > alpha_safety
        row = {
            'n_per_cell': n,
            'p_ref': p_ref,
            'p_shift_boundary': p_shift,
            'rejection_rate': rate,
            'mc95_lower': lo,
            'mc95_upper': hi,
            'alpha': alpha_safety,
            'clear_anti_conservative': clear_fail,
        }
        typei_rows.append(row)
        print(
            f"n={n:4d} p_ref={p_ref:.2f} p_shift={p_shift:.2f} "
            f"typeI={rate:.4f} MC95=[{lo:.4f},{hi:.4f}] "
            f"{'FAIL' if clear_fail else 'OK'}"
        )
print()

# Exact one-sample CP tests are finite-sample conservative by construction.
# Still simulate boundary behavior at representative n to document operating error.
print('=== BOUNDARY TYPE-I CALIBRATION: ABSOLUTE FNR / FPR ===')
one_sample_typei = []
for n in TYPEI_N:
    for family, p0, a in [('absolute_FNR', fnr_ceiling, alpha_safety), ('FPR', fpr_ceiling, alpha_fpr)]:
        rng = np.random.default_rng(20260919 + n + (1 if family == 'FPR' else 0))
        x = rng.binomial(n, p0, TYPEI_REPS)
        U = cp_upper_lookup(n, a)
        passed = U[x] <= p0
        k = int(passed.sum())
        rate = k / TYPEI_REPS
        lo, hi = mc_ci(k, TYPEI_REPS)
        clear_fail = lo > a
        one_sample_typei.append({
            'family': family,
            'n_per_cell': n,
            'boundary': p0,
            'rejection_rate': rate,
            'mc95_lower': lo,
            'mc95_upper': hi,
            'alpha': a,
            'clear_anti_conservative': clear_fail,
        })
        print(
            f"{family:12s} n={n:4d} typeI={rate:.4f} "
            f"MC95=[{lo:.4f},{hi:.4f}] alpha={a:.3f} "
            f"{'FAIL' if clear_fail else 'OK'}"
        )
print()

# -----------------------------------------------------------------------------
# 3. Joint power under the actual all-must-pass structure for one monitor.
#    Primary planning truth retained from the repo's earlier prespecified design:
#      FNR=0.05 in all validation cells (true degradation 0)
#      FPR=0.025 in all validation cells
#    This is a planning alternative, not an observed claim.
# -----------------------------------------------------------------------------
POWER_REPS = 50_000
N_GRID = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
TRUE_FNR = 0.05
TRUE_FPR = 0.025
power_rows = []

print('=== JOINT POWER GRID: SINGLETON INDEPENDENT PRIMARY UNITS ===')
print('Planning truth: FNR=0.05 in A-val/T/S/F/SF; FPR=0.025 in all five validation cells.')
print('W1 safety = 3 relative NI + absolute FNR in T/S/F/SF.')
print('Temporal safety = A-val->T NI + absolute FNR in A-val/T.')
print('Full preservation = all four NI comparisons + absolute FNR in all five cells.')
print('Availability = FPR<=0.05 in all five validation cells.')
print()

for n in N_GRID:
    rng = np.random.default_rng(880000 + n)
    y1 = rng.binomial(n, TRUE_FNR, size=(POWER_REPS, 5))
    A, T, S, F, SF = y1.T

    U_abs = cp_upper_lookup(n, alpha_safety)
    abs_A = U_abs[A] <= fnr_ceiling
    abs_T = U_abs[T] <= fnr_ceiling
    abs_S = U_abs[S] <= fnr_ceiling
    abs_F = U_abs[F] <= fnr_ceiling
    abs_SF = U_abs[SF] <= fnr_ceiling

    ni_S = fm_reject_vec(T, n, S, n)
    ni_F = fm_reject_vec(T, n, F, n)
    ni_SF = fm_reject_vec(T, n, SF, n)
    ni_time = fm_reject_vec(A, n, T, n)

    w1 = abs_T & abs_S & abs_F & abs_SF & ni_S & ni_F & ni_SF
    temporal = abs_A & abs_T & ni_time
    full = abs_A & abs_T & abs_S & abs_F & abs_SF & ni_S & ni_F & ni_SF & ni_time

    y0 = rng.binomial(n, TRUE_FPR, size=(POWER_REPS, 5))
    U_fpr = cp_upper_lookup(n, alpha_fpr)
    availability = np.all(U_fpr[y0] <= fpr_ceiling, axis=1)

    row = {'n_per_cell': n}
    for name, arr in [
        ('w1_safety_power', w1),
        ('temporal_safety_power', temporal),
        ('full_preservation_power', full),
        ('five_cell_availability_power', availability),
    ]:
        k = int(arr.sum())
        rate = k / POWER_REPS
        lo, hi = mc_ci(k, POWER_REPS)
        row[name] = rate
        row[name + '_mc95_lower'] = lo
        row[name + '_mc95_upper'] = hi
    power_rows.append(row)

    print(
        f"n={n:4d} | W1={row['w1_safety_power']:.3f} "
        f"(L95={row['w1_safety_power_mc95_lower']:.3f}) | "
        f"temporal={row['temporal_safety_power']:.3f} "
        f"(L95={row['temporal_safety_power_mc95_lower']:.3f}) | "
        f"full={row['full_preservation_power']:.3f} "
        f"(L95={row['full_preservation_power_mc95_lower']:.3f}) | "
        f"FPR5={row['five_cell_availability_power']:.3f} "
        f"(L95={row['five_cell_availability_power_mc95_lower']:.3f})"
    )
print()

# Find first design whose MC lower bound meets the existing 80% planning target.
def first_meeting(field):
    for r in power_rows:
        if r[field + '_mc95_lower'] >= 0.80:
            return r['n_per_cell']
    return None

selected = {
    'W1_safety_n': first_meeting('w1_safety_power'),
    'temporal_safety_n': first_meeting('temporal_safety_power'),
    'full_preservation_n': first_meeting('full_preservation_power'),
    'five_cell_availability_n': first_meeting('five_cell_availability_power'),
}

relative_typei_clear_failures = sum(r['clear_anti_conservative'] for r in typei_rows)
one_sample_clear_failures = sum(r['clear_anti_conservative'] for r in one_sample_typei)

print('=== SYNTHESIS ===')
print(f"Relative-NI clear anti-conservative boundary rows: {relative_typei_clear_failures}/{len(typei_rows)}")
print(f"One-sample FNR/FPR clear anti-conservative boundary rows: {one_sample_clear_failures}/{len(one_sample_typei)}")
print('First n/cell with MC95 lower >= 0.80:')
for k,v in selected.items():
    print(f"  {k}: {v}")
print()

if selected['full_preservation_n'] is not None:
    n_full = selected['full_preservation_n']
    print(
        'FEASIBILITY WARNING: under the clean singleton-block candidate, the full '
        f'preservation claim needs about {n_full} eligible independent Y=1 primary '
        'units PER validation cell at the FNR=0.05 planning truth.'
    )
    print(
        'For human cells this means the same order of magnitude of distinct independent '
        'authors yielding eligible Y=1 responses; raw recruitment would be larger because '
        'labels/eligibility are not guaranteed. This is a feasibility result, not a quota freeze.'
    )
else:
    print('FEASIBILITY WARNING: the tested grid did not reach 80% full-preservation power.')

print()
print('DECISION BOUNDARY')
print('- Do NOT start W0.')
print('- Do NOT freeze quotas from this candidate study.')
print('- Do NOT revive the failed clustered CR2/WCR candidates.')
print('- If this clean independent-unit design is infeasible, the next scientific decision is whether to')
print('  accept stronger model assumptions / a different estimand, or narrow the confirmatory claim.')
print('- No repo files were modified by this run.')

payload = {
    'study_id': 'external_validation_independent_block_feasibility_v1',
    'status': 'candidate_feasibility_only_not_frozen',
    'repo_head_observed': __import__('subprocess').check_output(['git','rev-parse','HEAD'], text=True).strip(),
    'candidate_design': candidate,
    'frozen_claim_inputs': {
        'relative_FNR_NI_margin': margin,
        'absolute_FNR_ceiling': fnr_ceiling,
        'safety_alpha': alpha_safety,
        'FPR_ceiling': fpr_ceiling,
        'FPR_alpha': alpha_fpr,
        'planning_power_target': 0.80,
        'planning_true_FNR': TRUE_FNR,
        'planning_true_FPR': TRUE_FPR,
    },
    'relative_NI_typeI': typei_rows,
    'one_sample_typeI': one_sample_typei,
    'power_grid': power_rows,
    'first_n_meeting_80pct_lower_MC_bound': selected,
    'W0_authorized': False,
    'repo_modified': False,
}
OUT_JSON.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

# Compact CSV for power grid.
import csv
with OUT_CSV.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(power_rows[0].keys()))
    writer.writeheader()
    writer.writerows(power_rows)

print()
print(f'WROTE {OUTDIR / "independent_block_feasibility_v1.txt"}')
print(f'WROTE {OUT_JSON}')
print(f'WROTE {OUT_CSV}')
PY

echo
echo "=== REPO STATUS (should be unchanged by this script) ===" | tee -a "$OUTDIR/independent_block_feasibility_v1.txt"
git status --short --branch | tee -a "$OUTDIR/independent_block_feasibility_v1.txt"

echo
echo "=== COMPLETE ==="
echo "Upload this file back to ChatGPT:"
echo "$OUTDIR/independent_block_feasibility_v1.txt"
