#!/usr/bin/env python3
"""Phase 4D: cheap CPU-only falsification of optional-monitor score value.

Post-hoc development-only ablation, not confirmatory inference or a deployed
system. Keeps historic classifier D fixed and treats cached optional scores as
unavailable except on the exact cases selected for simulated acquisition.

Outputs aggregate results and per-repetition SCALAR errors (never examples,
labels, original predictions or private data) into a NEW versioned folder.
"""
from __future__ import annotations
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_BRANCH = 'stage-a-paired-transport-20260921'
SOURCE = 'reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet'
FOLDS = 'reports/decision_value_real_data/development_outer_fold_assignments.csv'
PROTOCOL = 'configs/decision_value_real_data_protocol_v1.json'
BOUNDARY = 'configs/exact_cost_risk_cascade_protocol_v2.json'
OUT = 'results/phase4d_optional_score_falsification_v1'
ARTIFACT = 'phase4d_optional_score_falsification_v1'
# Exact hashes recorded in the prior development-only runs. Fail closed on changes.
HASHES = {
    SOURCE: '0041158b10e7ee0dd1189a0611c5b07f18a44d88355e3c150ac044e4112fe6ec',
    FOLDS: '62a2d8df0ba55b047d7e4685488b4d4d0ef77d480bcdb26cc34e5586e935d2dd',
    PROTOCOL: '7c82846f9de712a56260d74fbc1901b084262f2a7a745007547a75eab3d5f914',
    BOUNDARY: '96a0bfdf1a0954d9313ecd7a2ae1272a07f0df4ba6197eede2f0e0afd9f1c1c7',
}
# Compare at exactly the same human + optional-call budgets, with repeated
# identical acquired sets within each run. Never select an arm from test results.
CALLS = (80, 320)
REVIEWS = (80, 160)
REPS = 400
SEED = 20261009
BOOT = 1000
ARMS = ('cheap_only', 'acquisition_indicator', 'permuted_optional', 'real_optional')
METRICS = ('miss', 'FNR')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def downloads():
    try:
        s = subprocess.check_output(
            ['powershell.exe', '-NoProfile', '-Command',
             '[Environment]::GetFolderPath("UserProfile")'],
            text=True, timeout=8).strip().replace('\r', '')
        p = Path(subprocess.check_output(['wslpath', '-u', s],
                                         text=True, timeout=5).strip()) / 'Downloads'
        if p.is_dir():
            return p
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    p = Path('/mnt/c/Users/user/Downloads')
    return p if p.is_dir() else None


def acquire(uncertainty, ids, m, rng):
    """4C priority acquisition: 80% highest CHEAP uncertainty + 20% random."""
    n = len(ids)
    top_n = m - int(math.ceil(0.2 * m))
    order = np.lexsort((ids, -uncertainty))
    chosen = order[:top_n]
    rest = np.setdiff1d(np.arange(n), chosen, assume_unique=True)
    sentinel = rng.choice(rest, size=m - top_n, replace=False)
    flags = np.zeros(n, dtype=bool)
    flags[chosen] = True
    flags[sentinel] = True
    assert int(flags.sum()) == m
    return flags


def make_groups(d, cheap, acquired=None, optional=None, mode='cheap_only'):
    """Stratification depends only on observed cheap signals and paid scores.

    cheap_only: D x four cheap-score quartiles (up to 8 strata).
    acquisition_indicator: D x acquired x two cheap-score bands.
    real/permuted optional: D x acquired x two bands; on acquired rows
    sort by the specified (real/shuffled) purchased score, else cheap.
    """
    groups, weights = [], []
    n = len(d)
    if mode == 'cheap_only':
        for dd in (0, 1):
            indices = np.flatnonzero(d == dd)
            ordered = indices[np.argsort(cheap[indices], kind='stable')]
            for g in np.array_split(ordered, 4):
                if len(g):
                    groups.append(g)
                    weights.append(1.0)
    else:
        assert acquired is not None and len(acquired) == n
        for dd in (0, 1):
            for bought in (False, True):
                indices = np.flatnonzero((d == dd) & (acquired == bought))
                if not len(indices):
                    continue
                if bought and mode in ('real_optional', 'permuted_optional'):
                    assert optional is not None
                    order = np.argsort(optional[indices], kind='stable')
                else:
                    order = np.argsort(cheap[indices], kind='stable')
                # Match the Phase 4C split convention: lower floor-half,
                # upper ceil-half (important for odd-sized acquired groups).
                ordered = indices[order]
                cut = len(ordered) // 2
                halves = (ordered[:cut], ordered[cut:])
                for band, g in enumerate(halves):
                    if len(g):
                        groups.append(g)
                        weights.append(2.0 if bought and band == 1 else 1.0)
    flat = np.concatenate(groups)
    assert len(flat) == n and len(np.unique(flat)) == n
    return groups, np.asarray(weights)


def allocate(groups, budget, weights):
    """4C's bounded proportional/tilted greedy stratum allocation."""
    sizes = np.asarray([len(g) for g in groups], dtype=int)
    assigned = np.minimum(2, sizes)
    if budget < int(assigned.sum()) or budget > int(sizes.sum()):
        raise ValueError('invalid review budget or too many tiny strata')
    utility = sizes * weights
    while assigned.sum() < budget:
        possible = np.flatnonzero(assigned < sizes)
        chosen = possible[np.argmax(utility[possible] / (assigned[possible] + 1))]
        assigned[chosen] += 1
    assert int(sum(assigned)) == budget
    return assigned


def draw_estimate(y, d, groups, budget, weights, seed):
    alloc = allocate(groups, budget, weights)
    rng = np.random.default_rng(seed)
    n = len(y)
    miss, harm = 0., 0.
    for indices, nh in zip(groups, alloc):
        observed = rng.choice(indices, size=int(nh), replace=False)
        y_s = y[observed]
        d_s = d[observed]
        miss += len(indices) * float(np.mean(y_s * (1 - d_s)))
        harm += len(indices) * float(np.mean(y_s))
    miss, harm = miss / n, harm / n
    return miss, miss / harm if harm > 0 else float('nan')


def one_setup(df, name, ms, repetitions=REPS, call_counts=CALLS, budgets=REVIEWS):
    g = df.sort_values('example_id')
    ids = g.example_id.astype(str).to_numpy()
    y = g.y.astype(int).to_numpy()
    d = g.base_prediction.astype(int).to_numpy()
    cheap = g.base_uncertainty.astype(float).to_numpy()
    opt = g.optional_monitor_score.astype(float).to_numpy()
    assert len(y) == 1687 and len(set(ids)) == len(ids)
    assert set(np.unique(y)).issubset({0, 1}) and set(np.unique(d)).issubset({0, 1})
    assert np.isfinite(cheap).all() and np.isfinite(opt).all()
    assert np.all((opt >= 0) & (opt <= 1))
    truth_miss = float(np.mean(y * (1 - d)))
    truth_fnr = float(np.sum(y * (1 - d)) / np.sum(y))
    setup_code = 0 if name == 'compact_after_rule' else 1
    cheap_groups, cheap_weights = make_groups(d, cheap, mode='cheap_only')
    rows = []
    for m in call_counts:
        if m >= len(y):
            raise ValueError('monitor budget must be less than the population')
        for r in range(repetitions):
            acq_rng = np.random.default_rng(np.random.SeedSequence([SEED, setup_code, m, r, 0]))
            acquired = acquire(cheap, ids, m, acq_rng)
            # Placebo: permute the purchased scores within decision groups only.
            # No Y / nonpurchased optional score is accessed to make partitions.
            permuted = np.full(len(y), np.nan, dtype=float)
            perm_rng = np.random.default_rng(np.random.SeedSequence([SEED, setup_code, m, r, 1]))
            for dd in (0, 1):
                ix = np.flatnonzero((d == dd) & acquired)
                permuted[ix] = perm_rng.permutation(opt[ix])
            groups_by_arm = {
                'cheap_only': (cheap_groups, cheap_weights),
                'acquisition_indicator': make_groups(d, cheap, acquired, mode='acquisition_indicator'),
                'permuted_optional': make_groups(d, cheap, acquired, permuted, mode='permuted_optional'),
                'real_optional': make_groups(d, cheap, acquired, opt, mode='real_optional'),
            }
            for b in budgets:
                # Same number of labels; deterministic coupled sampling RNG seeds.
                base_seed = np.random.SeedSequence([SEED, setup_code, m, r, b, 2])
                for arm in ARMS:
                    parts, w = groups_by_arm[arm]
                    miss, fnr = draw_estimate(y, d, parts, b, w, base_seed)
                    rows.append({
                        'setup_id': name, 'monitor_calls': m, 'human_reviews': b,
                        'repetition': r, 'arm': arm,
                        'charge_monitor_calls': 0 if arm == 'cheap_only' else m,
                        'monitor_cost_ms_legacy': 0.0 if arm == 'cheap_only' else float(m * ms),
                        'miss_error': miss - truth_miss,
                        'FNR_error': fnr - truth_fnr if math.isfinite(fnr) else float('nan'),
                    })
    return rows, {'miss': truth_miss, 'FNR': truth_fnr}


def summarize(frame):
    outputs = []
    for keys, g in frame.groupby(['setup_id', 'monitor_calls', 'human_reviews', 'arm'], sort=True):
        row = dict(zip(['setup_id', 'monitor_calls', 'human_reviews', 'arm'], keys))
        row.update({'repetitions': len(g),
                    'charged_monitor_calls': int(g.charge_monitor_calls.iloc[0]),
                    'legacy_cost_seconds': float(g.monitor_cost_ms_legacy.iloc[0] / 1000)})
        for metric in METRICS:
            e = g[f'{metric}_error'].to_numpy(dtype=float)
            finite = np.isfinite(e)
            row[metric + '_fraction_estimable'] = float(finite.mean())
            row[metric + '_bias'] = float(e[finite].mean()) if finite.any() else float('nan')
            row[metric + '_RMSE'] = float(np.sqrt(np.mean(e[finite] ** 2))) if finite.any() else float('nan')
        outputs.append(row)
    return pd.DataFrame(outputs)


def contrasts(frame, boot=BOOT):
    rows = []
    for (setup, calls, reviews), g in frame.groupby(['setup_id', 'monitor_calls', 'human_reviews'], sort=True):
        pivots = {metric: g.pivot(index='repetition', columns='arm', values=metric+'_error')
                  for metric in METRICS}
        for metric, table in pivots.items():
            for control in ('cheap_only', 'acquisition_indicator', 'permuted_optional'):
                a = table['real_optional'].to_numpy()
                c = table[control].to_numpy()
                valid = np.isfinite(a) & np.isfinite(c)
                a, c = a[valid], c[valid]
                if len(a) < 100:
                    raise ValueError('too few estimable paired repetitions for meaningful contrasts')
                delta = math.sqrt(float(np.mean(a*a))) - math.sqrt(float(np.mean(c*c)))
                rng = np.random.default_rng(np.random.SeedSequence([SEED, calls, reviews,
                    0 if setup=='compact_after_rule' else 1, 0 if metric=='miss' else 1,
                    ['cheap_only','acquisition_indicator','permuted_optional'].index(control), 9]))
                idx = rng.integers(len(a), size=(boot, len(a)))
                bs = np.sqrt(np.mean(a[idx]**2, axis=1)) - np.sqrt(np.mean(c[idx]**2, axis=1))
                rows.append({'setup_id': setup, 'monitor_calls': calls, 'human_reviews': reviews,
                             'metric': metric, 'comparison': 'real_optional_minus_'+control,
                             'n_valid_paired': len(a), 'delta_RMSE': delta,
                             'mc_bootstrap_lo95': float(np.quantile(bs, 0.025)),
                             'mc_bootstrap_hi95': float(np.quantile(bs, 0.975))})
    return pd.DataFrame(rows)


def self_test():
    y=np.array([1,0,1,1,0,1,0,0,1,0,0,1],dtype=int)
    d=np.array([0,0,0,0,0,0,1,1,1,1,1,1],dtype=int)
    cheap=np.linspace(.05,.95,12)
    opt=np.linspace(.92,.08,12)
    ids=np.array([f'case{i:03d}' for i in range(12)])
    flags=acquire(cheap,ids,5,np.random.default_rng(3))
    assert sum(flags)==5
    for arm in ARMS:
        groups,w=make_groups(d,cheap,flags,opt,mode=arm)
        assert len(np.concatenate(groups))==12
        if arm!='cheap_only':
            groups1,w1=make_groups(d,cheap,flags,opt+999 * (~flags),mode=arm)
            if arm != 'acquisition_indicator':
                assert all(np.array_equal(a,b) for a,b in zip(groups,groups1))
        if arm=='cheap_only':
            sample=draw_estimate(y,d,groups,12,w,np.random.SeedSequence([2]))
            assert abs(sample[0]-float(np.mean(y*(1-d))))<1e-12
    assert math.isclose(float(np.mean(y*(1-d))), 4/12)
    print('PHASE4D_SELF_TEST_PASS')


def main():
    self_test()
    if '--self-test' in sys.argv:
        return 0
    root=Path.cwd().resolve()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=8).strip()
    if branch!=ROOT_BRANCH:
        print('STOP: wrong branch',branch)
        return 2
    out=root/OUT
    if out.exists():
        print('STOP: versioned output already exists; preserved:',out)
        return 2
    missing=[p for p in HASHES if not (root/p).is_file()]
    if missing:
        print('STOP: missing permitted local input(s)',missing)
        return 2
    for p,expect in HASHES.items():
        if sha256(root/p)!=expect:
            raise ValueError('frozen source/protocol hash mismatch: '+p)
    protocol=json.loads((root/PROTOCOL).read_text())
    boundary=json.loads((root/BOUNDARY).read_text())
    allowed=boundary.get('data_boundary',{}).get('permitted_existing_development_artifacts',[])
    if not {SOURCE,FOLDS}.issubset(set(allowed)):
        raise ValueError('development boundary does not permit input')
    columns=['example_id','setup_id','outer_fold','y','base_prediction',
             'base_uncertainty','optional_monitor_score']
    df=pd.read_parquet(root/SOURCE,columns=columns)
    folds=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
    n=protocol['scope']['expected_development_rows']
    if len(folds)!=n or folds.example_id.nunique()!=n:
        raise ValueError('development fold assignment is not the expected population')
    expected={x['setup_id']:x for x in protocol['optional_monitor_setups']}
    if len(df)!=n*len(expected) or set(df.setup_id)!=set(expected):
        raise ValueError('development population/setup mismatch')
    if df.groupby('example_id').y.nunique().max()!=1:
        raise ValueError('cross-setup labels disagree')
    all_rows=[]; totals={}
    for setup, item in expected.items():
        sub=df[df.setup_id==setup]
        joined=sub[['example_id','outer_fold']].merge(
            folds,on='example_id',validate='one_to_one',suffixes=('_data','_frozen'))
        if len(joined)!=n or not (joined.outer_fold_data.astype(str)==joined.outer_fold_frozen.astype(str)).all():
            raise ValueError('cross-fitted folds changed for '+setup)
        if len(sub)!=n or sub.example_id.nunique()!=n or int(sub.y.sum())!=291:
            raise ValueError('development counts incorrect for '+setup)
        print('RUN', setup, '2 budgets x 2 monitor counts x',REPS,'paired repetitions...',flush=True)
        rows, t=one_setup(sub, setup, item['measured_mean_cost_ms'])
        all_rows.extend(rows);totals[setup]=t
    frame=pd.DataFrame(all_rows)
    summary=summarize(frame)
    checks=contrasts(frame)
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'paired_scalar_errors.csv',index=False,float_format='%.12g',lineterminator='\n')
    summary.to_csv(out/'ablation_summary.csv',index=False,float_format='%.12g',lineterminator='\n')
    checks.to_csv(out/'paired_rmse_contrasts.csv',index=False,float_format='%.12g',lineterminator='\n')
    report=['# Phase 4D: Does the purchased optional-monitor SCORE add information?', '',
      'Post-hoc exploratory, development-only. D is an archived cross-fitted classifier, not a deployed guard.',
      'Scores are used only on records where simulated extra-monitor acquisition occurs.',
      'Matched human review budgets across arms. At fixed monitor count, acquisition and randomization are shared.',
      'The no-monitor control incurs zero optional calls; the other three arms are charged identically.',
      '', '## FNR RMSE (smaller is better)', '',
      '| Setup | Monitor calls | Human reviews | Cheap only (0 calls) | Acquisition only | Shuffled scores | Real scores |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for (setup,call,review),grp in summary.groupby(['setup_id','monitor_calls','human_reviews']):
        lookup={x.arm:x.FNR_RMSE for x in grp.itertuples()}
        report.append(f'| {setup} | {call} | {review} | '+ ' | '.join(f'{lookup[a]:.4f}' for a in ARMS)+' |')
    report+=['', '## Interpretation and restrictions', '',
      '- Negative real-minus-control RMSE means better real-score performance in this Monte Carlo simulation.',
      '- Paired bootstrap intervals quantify Monte Carlo uncertainty only; they are not validation or deployment intervals.',
      '- Shuffling nulls the per-case link between paid scores and outcomes, retaining decision-group score distribution.',
      '- The acquisition-only arm retains the allocation effects of prioritizing cheap-uncertain queries.',
      '- Review labels are historical, simulated as purchased; no new independently labeled cases or fresh confirmatory data.',
      '- FNR and FPR are ratios and do not inherit the conditional unbiasedness of the linear stratum mean estimators.',
      '- Optional monitor latency uses historical batch-mean per-call cost; no new wall-clock timing or human wage cost.',
      '- This specific four-arm ablation was devised *after* reviewing Phase 4C. It must not be sold as pre-registered confirmation.',
      '- No algorithm selection, new model training, or evidence of deployment superiority follows from this pilot.', '']
    (out/'README.md').write_text('\n'.join(report),encoding='utf8')
    manifest={'artifact_id':ARTIFACT,'status':'posthoc_development_only_optional_score_ablation',
      'source_branch':branch,'source_sha256':{p:sha256(root/p) for p in HASHES},
      'historical_fixed_decision':'base_prediction cross-fitted classifier, NOT a deployed native guard',
      'response_labels':'historical development labels, NOT independent adjudication',
      'original_stage_a_screen':'0 of 4 signals remains unchanged',
      'optional_monitor_call_cost':'historical protocol mean milliseconds per check; not newly measured',
      'sampling':'80% high cheap-uncertainty priority + 20% random, identical acquired indices across comparator arms',
      'design_arms':list(ARMS),'human_review_budgets':list(REVIEWS),
      'optional_monitor_calls':list(CALLS),'repetitions':REPS,'seed':SEED,
      'paired_MC_bootstrap_draws':BOOT,'development_population_totals':totals,
      'no_protected_rows_accessed':True,'no_raw_records_exported':True,
      'review_cost':'not monetized',
      'selection_status':'posthoc falsification motivated by observing Phase 4C results, not confirmatory'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf8')
    archive=out/(ARTIFACT+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for x in ('ablation_summary.csv','paired_rmse_contrasts.csv','paired_scalar_errors.csv','README.md','manifest.json'):
            z.write(out/x,arcname=x)
    dest=downloads()
    if dest:
        target=dest/archive.name
        if target.exists():
            print('Preserved existing Downloads file. New archive remains in repo:',archive)
        else:
            shutil.copy2(archive,target)
            print('WINDOWS_DOWNLOADS:',target)
    else:
        print('Windows Downloads unresolved; archive remains:',archive)
    print('PHASE4D_SUCCESS',len(frame),'error rows,',len(summary),'summary rows,',len(checks),'paired contrasts')
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('PHASE4D_STOP:',type(exc).__name__,str(exc))
        sys.exit(2)
