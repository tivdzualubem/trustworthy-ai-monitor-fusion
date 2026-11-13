#!/usr/bin/env python3
"""Population-level robustness audit for the existing phase8_cost_matched module.

Put next to the EXISTING `phase8_cost_matched.py` in the WSL research folder.
Uses the original implementation unchanged, resampling the synthetic population,
proxy realization, reviewer votes, case selection and human-active-time costs.

No human text, labels, case identifiers or credentials are read. As in Phase8A,
only recorded (rater, active_seconds) are read read-only from the reviewer DB.

Important: the original Phase8A module varies the population seed and draws
synthetic proxy sensitivity/specificity at fixed assumptions; it does NOT
measure real proxy behavior nor implement published full MultiPPI/AM-PPI.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import t
import numpy as np
import phase8_cost_matched as p8

BASE = 'allowed_srs_single'
METHODS = p8.METHODS
SCENARIOS = tuple(p8.SCENARIOS)

def validate_result(obj, geometries, budget):
    for geo in geometries:
        if geo not in obj['geometries']:
            raise ValueError('Missing geometry ' + geo)
        for s in SCENARIOS:
            key=f'{s}|budget_{int(budget)}s'
            for m in METHODS:
                entry=obj['geometries'][geo]['summary'][key][m]
                if not (0<=entry['pooled_rmse_vs_latent_Y']<=1):
                    raise ValueError(f'Invalid RMSE {geo} {key} {m}')
                if entry['mean_spent_seconds']>budget+1.e-8 or entry['max_spent_seconds']>budget+1.e-8:
                    raise ValueError('Hard budget violated')
                if entry['mean_reserved_seconds']>budget+1.e-8 or entry['max_reserved_seconds']>budget+1.e-8:
                    raise ValueError('Hard reserved budget violated')
    return True

def summarize_population_runs(runs, population_diagnostics, budget, bootstrap_reps, bootstrap_seed):
    """Runs must be iid populations with equal within-population repetition counts.

    Returns Monte Carlo confidence intervals clustered by population, not pooled
    as if 30 repeated votes on one population were 30 independent populations.
    """
    out={'synthetic_population_count':len(runs),
         'proxy_population_diagnostics':{}, 'summary':{}, 'paired_contrasts_vs_allowed_srs':{}}
    if not runs:
        raise ValueError('No runs')
    diagnostic_keys=('proxy_positive_allowed','proxy_negative_allowed','harmful_allowed',
                     'harmful_allowed_in_proxy_positive','harmful_allowed_in_proxy_negative')
    for k in diagnostic_keys:
        arr=np.asarray([float(d[k]) for d in population_diagnostics])
        out['proxy_population_diagnostics'][k]={'mean':float(arr.mean()),'min':int(arr.min()),
                                                'max':int(arr.max())}
    perfect=np.array([d['harmful_allowed_in_proxy_negative']==0 for d in population_diagnostics])
    out['proxy_population_diagnostics']['fraction_of_populations_with_all_harm_in_proxy_positive']=float(perfect.mean())
    out['proxy_population_diagnostics']['near_census_fraction_diagnostic']=(
        'Each run resamples the proxy-positive stratum. Inspect its size against roughly half the '
        'selected allowed_proxy_stratified_equal responses; do not assume a single population is representative.')
    rng=np.random.default_rng(bootstrap_seed)
    P=len(runs)
    sample_indices=rng.integers(0,P,size=(bootstrap_reps,P),dtype=np.int32) if bootstrap_reps else None
    keys=tuple(runs[0]['summary'])
    for key in keys:
        rows={}
        sq_err_by_method={}
        for method in METHODS:
            dat=[r['summary'][key][method] for r in runs]
            sq=np.asarray([float(d['rmse_midpoint_vs_latent_Y'])**2 for d in dat])
            sq_err_by_method[method]=sq
            rmse=float(np.sqrt(sq.mean()))
            row={'pooled_rmse_vs_latent_Y':rmse,
                 'rmse_across_populations_min':float(np.sqrt(sq.min())),
                 'rmse_across_populations_max':float(np.sqrt(sq.max())),
                 'mean_ci_width':float(np.mean([d['mean_ci_width'] for d in dat])),
                 'mean_coverage_for_latent_Y_diagnostic_only':float(np.mean([d['ci_coverage_against_latent_Y_diagnostic_only'] for d in dat])),
                 'mean_coverage_for_simulated_reviewer_endpoint':float(np.mean([d['ci_coverage_reviewer_completion_set'] for d in dat])),
                 'mean_distinct_cases':float(np.mean([d['mean_unique_responses'] for d in dat])),
                 'mean_reviewer_actions':float(np.mean([d['mean_reviewer_actions'] for d in dat])),
                 'mean_spent_seconds':float(np.mean([d['mean_spent_seconds'] for d in dat])),
                 'max_spent_seconds':float(max(d['max_spent_seconds'] for d in dat)),
                 'mean_reserved_seconds':float(np.mean([d['mean_reserved_seconds'] for d in dat])),
                 'max_reserved_seconds':float(max(d['max_reserved_seconds'] for d in dat)),
                 'mean_point_identification_width':float(np.mean([d['mean_point_identification_width'] for d in dat])),
            }
            if sample_indices is not None:
                boot=np.sqrt(np.mean(sq[sample_indices],axis=1))
                row['between_population_bootstrap_95_interval']=[float(x) for x in np.quantile(boot,[.025,.975])]
            rows[method]=row
        out['summary'][key]=rows
        contrasts={}
        for method in METHODS:
            if method==BASE:continue
            x,y=sq_err_by_method[method],sq_err_by_method[BASE]
            difference=float(np.sqrt(x.mean())-np.sqrt(y.mean()))
            record={'rmse_difference_method_minus_allowed_srs':difference,
                    'interpretation':'Negative favors method; positive favors allowed SRS',
                    'uncertainty_unit':'independently generated synthetic population'}
            if sample_indices is not None:
                d=np.sqrt(np.mean(x[sample_indices],axis=1))-np.sqrt(np.mean(y[sample_indices],axis=1))
                record['between_population_bootstrap_95_interval']=[float(q) for q in np.quantile(d,[.025,.975])]
            contrasts[method]=record
        out['paired_contrasts_vs_allowed_srs'][key]=contrasts
    return out

def run(args):
    if args.populations < 4 or args.repetitions_per_population < 2:
        raise ValueError('Use >=4 independent populations and >=2 reps/population')
    if args.budget_seconds <= 0 or args.bootstrap_reps<0:
        raise ValueError('Invalid budget/bootstrap')
    # The existing study must remain private. We only read its observed durations.
    pools,cost_source=p8.cost_pools(args.db,'empirical')
    original_script=Path(p8.__file__).resolve()
    script_hash=hashlib.sha256(original_script.read_bytes()).hexdigest()
    output={
       'study':'PHASE8B_POPULATION_VARIABILITY_AUDIT',
       'scientific_status':'Synthetic robustness check, not independently validated empirical risk',
       'underlying_phase8a_python_sha256':script_hash,
       'population_seed_base':args.seed,
       'population_seeds':[int(args.seed+1009*i) for i in range(args.populations)],
       'within_population_repetitions':args.repetitions_per_population,
       'human_review_cost_source':cost_source,
       'budget_human_active_seconds':args.budget_seconds,
       'confidence_interval_scope':'Between independently generated synthetic populations (bootstrap). Not frequentist guarantees for real safety labels.',
       'geometries':{}
    }
    for j,geometry in enumerate(args.geometry):
        runs=[];diagnostics=[]
        for i in range(args.populations):
            seed=int(args.seed+1009*i)
            # same population seeds can be matched across geometries, without confounding
            # method comparisons with different latent truth in one experiment.
            y,d,h=p8.make_population(geometry,seed+100)
            allowed=(d==0);hp=(h==1)&allowed;hn=(h==0)&allowed
            diagnostics.append({
              'harmful_allowed':int(np.sum((y==1)&allowed)),
              'proxy_positive_allowed':int(np.sum(hp)),
              'proxy_negative_allowed':int(np.sum(hn)),
              'harmful_allowed_in_proxy_positive':int(np.sum((y==1)&hp)),
              'harmful_allowed_in_proxy_negative':int(np.sum((y==1)&hn)),
            })
            item=p8.simulate(pools,cost_source,geometry,[args.budget_seconds],
                             args.repetitions_per_population,seed,args.alpha)
            runs.append(item)
            print(f'{geometry}: population {i+1}/{args.populations}; proxy-positive={diagnostics[-1]["proxy_positive_allowed"]}; harm in proxy-negative={diagnostics[-1]["harmful_allowed_in_proxy_negative"]}',flush=True)
        output['geometries'][geometry]=summarize_population_runs(
            runs,diagnostics,args.budget_seconds,args.bootstrap_reps,args.seed+7100+j)
        output['geometries'][geometry]['latent_true_missed_harm_prevalence'] = float(runs[0]['latent_true_missed_harm_prevalence'])
    validate_result(output,args.geometry,args.budget_seconds)
    output['limitations']=[
        'Independent synthetic population variation reduces one-seed selection artifacts but does not identify real-world reviewer error.',
        'Proxy predictions are generated at zero accounted acquisition cost; published fair cost comparison must add proxy inference and calibration.',
        'Reviewer durations are resampled independently of case difficulty and labels: an unverified assumption.',
        'R4 triage reserves a fourth review for every selected case and therefore underuses actual budget.',
        'These methods are NOT faithful full MultiPPI, AM-PPI, nor fully tuned StratPPI.',
        'The benchmark still holds proxy sensitivity/specificity and synthetic harm prevalence fixed across populations.',
        'No population label gold standard from real study. No prospective native-monitor certification.',
    ]
    if args.out.exists():raise FileExistsError(f'Existing file preserved: {args.out}')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
    print('\nCOMPLETE:',args.out)
    for geometry,data in output['geometries'].items():
        print('\n'+geometry)
        print('Fraction of generated populations with no harmful H-negative cases:',
            data['proxy_population_diagnostics']['fraction_of_populations_with_all_harm_in_proxy_positive'])
        for scenario in ('perfect_binary','noisy_independent','noisy_correlated','correlated_with_abstention'):
            key=f'{scenario}|budget_{int(args.budget_seconds)}s'
            print(scenario)
            for method in (BASE,'allowed_proxy_stratified_equal', 'allowed_one_proxy_ppi_fixed_lambda1',
                           'allowed_three_vote_majority','allowed_three_plus_r4_reserve_triage'):
                r=data['summary'][key][method]
                print(f'  {method:41s} RMSE={r["pooled_rmse_vs_latent_Y"]:.5f} CI-width={r["mean_ci_width"]:.4f} Review-seconds={r["mean_spent_seconds"]:.1f}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--geometry',nargs='+',default=['phase6_5000','development_footprint_1687'],
                    choices=['phase6_5000','development_footprint_1687'])
    ap.add_argument('--budget-seconds',default=4800.,type=float)
    ap.add_argument('--populations',default=16,type=int)
    ap.add_argument('--repetitions-per-population',default=30,type=int)
    ap.add_argument('--seed',default=20261113,type=int)
    ap.add_argument('--alpha',default=.05,type=float)
    ap.add_argument('--bootstrap-reps',default=1500,type=int)
    args=ap.parse_args()
    run(args)
if __name__=='__main__':main()
