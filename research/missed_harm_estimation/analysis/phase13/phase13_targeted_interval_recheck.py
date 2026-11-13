#!/usr/bin/env python3
"""Targeted *interval-only* Phase 11 recheck. Same estimator code as frozen P11.

No new models, no policy rejudging, no private response / vote access.
We regenerate synthetic individual trials because P11 exported only aggregates and
confidence intervals cannot be recomputed from aggregated means.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import phase11_reference_unmodified as p
import phase13_methods as m

ORIGINAL_RESULT=p.result
ORIGINAL_OPTIMIZER=p.optimize_multi


def run_trial_with_interval_check(pop,scenario,budget,durations,cheap,expensive,seed):
    results=[]
    for method_idx,name in enumerate(p.METHODS):
        rng=np.random.default_rng(seed+193*method_idx)
        ledger=p.Ledger(budget,durations,cheap,expensive,rng)
        certificates=[]
        def instrumented_result(*args,**kw):
            item=ORIGINAL_RESULT(*args,**kw)
            return m.augment_original_result(item,pop.q)
        def instrumented_optimizer(Sigma,costs,seed=0):
            fr,success=ORIGINAL_OPTIMIZER(Sigma,costs,seed)
            cert=m.multi_optimality_certificate(Sigma,costs,fr)
            cert.pop('gradient')
            cert['original_solver_success']=success
            certificates.append(cert)
            return fr,success
        p.result=instrumented_result
        p.optimize_multi=instrumented_optimizer
        try:
            r=p.METHOD_FUN[name](pop,scenario,ledger,rng)
        finally:
            p.result=ORIGINAL_RESULT
            p.optimize_multi=ORIGINAL_OPTIMIZER
        if certificates:
            r['restricted_multi_solver_certificate']=certificates[0]
        r['scenario']=scenario
        r['population_index']=None
        r['rep_index']=None
        if ledger.spent>ledger.budget+1e-7:raise AssertionError('Budget exceeded')
        results.append(r)
    return results


def aggregate(raw):
    out={}
    for sc in p.SCENARIOS:
        out[sc]={}
        for name in p.METHODS:
            rows=[r for r in raw if r['scenario']==sc and r['method']==name]
            good=[r for r in rows if r['valid']]
            if not good:
                out[sc][name]={'runs':len(rows),'valid':0};continue
            def avg(fn):return float(np.mean([fn(r) for r in good]))
            entries={
                'runs':len(rows),'valid':len(good),
                'mean_original_interval_width':avg(lambda r:r['ci_width']),
                'mean_clipped_conditional_interval_width':avg(lambda r:r['feasible_interval_95']['width'] if r['feasible_interval_95']['nonempty'] else 0),
                'empty_clipped_interval_count':sum(not r['feasible_interval_95']['nonempty'] for r in good),
                'coverage_reviewer_Z_original':avg(lambda r:r['covers_reviewer_Z']),
                'coverage_reviewer_Z_clipped':avg(lambda r:r['feasible_interval_covers_Z']),
                'rmse_synthetic_latent_Y':math.sqrt(avg(lambda r:r['squared_error_Y'])),
                'rmse_reviewer_endpoint_Z':math.sqrt(avg(lambda r:r['squared_error_Z'])),
                'max_total_cost':max(r['ledger']['actual_total_equivalent_units'] for r in good),
                'hard_budget_failures':sum(not r['ledger']['hard_budget_ok'] for r in rows),
                'interval_note':'Asymptotic intervals are exploratory and not certified',
            }
            if name not in ('allowed_srs','whole_srs'):
                entries['mean_clipped_wald_exploratory_interval_width']=avg(
                    lambda r:r['paper_style_wald_exploratory_95']['width'] if r['paper_style_wald_exploratory_95']['nonempty'] else 0)
                entries['empty_wald_exploratory_interval_count']=sum(
                    not r['paper_style_wald_exploratory_95']['nonempty'] for r in good)
                entries['wald_exploratory_coverage_reviewer_Z']=avg(lambda r:r['wald_exploratory_covers_Z'])
            if name=='multipppi_subset_gls':
                cert=[r['restricted_multi_solver_certificate'] for r in good]
                entries['restricted_convex_optimizer_certificate']={
                  'certificate_count':len(cert),
                  'original_solver_success_count':sum(x['original_solver_success'] for x in cert),
                  'maximum_relative_objective_gap_upper_bound':max(x['relative_gap_upper'] for x in cert),
                  'mean_relative_gap_upper_bound':float(np.mean([x['relative_gap_upper'] for x in cert])),
                  'relative_gap_upper_below_1pct_count':sum(x['relative_gap_upper']<.01 for x in cert),
                  'note':'Certifies only Phase11 restricted convex covariance-information budget allocation, not full published MultiPPI or coverage.'}
            out[sc][name]=entries
    return out


def render_md(meta,summary):
    lines=['# Phase 13 — Targeted interval and restricted solver fidelity recheck','',
        '**Scope:** new postprocessing of unmodified Phase 11 estimators. NO new harmfulness evidence.','',
        f"Synthetic geometry: N={meta['N']}, allowed={meta['allowed_N']}. Review durations: {meta['duration_source']}. Budget={meta['budget_equivalent_units']}.",'',
        'Feasible prevalence domain is [0, allowed/N]. Raw conditional interval intersected with this domain. Coverage is preserved for any target within that domain.','',
        '**Exploratory normal intervals are not finite-sample guarantees or identical to the author papers\' intervals.**','',
        '| Scenario | Method | Raw width | Clipped width | Clipped Z coverage | Exploratory Wald width | Wald Z coverage | RMSE vs synthetic Y |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for sc in p.SCENARIOS:
        for k,d in summary[sc].items():
            def f(v):return f'{v:.4f}' if isinstance(v,(int,float)) else '—'
            lines.append(f'| {sc} | {k} | {f(d.get("mean_original_interval_width"))} | {f(d.get("mean_clipped_conditional_interval_width"))} | {f(d.get("coverage_reviewer_Z_clipped"))} | {f(d.get("mean_clipped_wald_exploratory_interval_width"))} | {f(d.get("wald_exploratory_coverage_reviewer_Z"))} | {f(d.get("rmse_synthetic_latent_Y"))} |')
    lines+=['','## Strict evidence boundaries','',
       '- These are new interval checks, not new estimator constructions. Phase 11 estimator code was copied exactly and preserved; estimator point estimates and budget logic are unchanged.',
       '- Clipping does NOT fix unknown reviewer misclassification or latent Y identification. Synthetic latent-Y coverage is NOT a certificate.',
       '- Original conservative intervals rely on iid-with-replacement fixed-population sampling, bounded variables, frozen independent pilot, and review-duration independence.',
       '- Exploratory Wald intervals can severely under-cover for rare harms or weak pilots; report empirical coverage alongside width; no author-equivalence claims.',
       '- Multi global-optimality gap is a convex objective **restricted to Phase11 GLS surrogate** with mandatory 12% gold-budget constraint; it is not official MultiPPI solver or inference.',
       '- Phase13 includes a separate exact Bernoulli propensity-corrected mean identity for AM-PPI Eq.(1). It is not the full jointly optimized AM-PPI algorithm, and Bernoulli sampling controls **expected** label cost, not a deterministic hard cap.',
       '- Proxy scores and simulated outcomes are synthetic; human durations come from actual prior review only if a verified local DB is supplied.',
       '- No raw private DB entries are exported. No files are written to GitHub.','']
    return '\n'.join(lines)


def run(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',type=Path,default=None,help='SQLite read-only, takes only active_seconds from ratings')
    ap.add_argument('--geometry',choices=['development_1687','phase6_5000'],default='development_1687')
    ap.add_argument('--populations',type=int,default=2)
    ap.add_argument('--repetitions',type=int,default=10)
    ap.add_argument('--budget',type=float,default=4800.)
    ap.add_argument('--cheap-unit-equivalent',type=float,default=.04565729060156249)
    ap.add_argument('--expensive-unit-equivalent',type=float,default=1.5975605696953123)
    ap.add_argument('--seed',type=int,default=20261112)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    if args.out_dir.exists():raise FileExistsError(f'Refusing overwrite: {args.out_dir}')
    if min(args.populations,args.repetitions)<1 or args.budget<=200 or args.cheap_unit_equivalent<0 or args.expensive_unit_equivalent<=0:ap.error('Invalid run configuration')
    duration,source=p.read_review_durations(args.db)
    N,M=(1687,1557) if args.geometry=='development_1687' else (5000,4000)
    raw=[]
    for pp in range(args.populations):
        pop=p.make_population(args.seed+1009*pp+101,N,M)
        for rep in range(args.repetitions):
            for iscen,sc in enumerate(p.SCENARIOS):
                trial=run_trial_with_interval_check(pop,sc,args.budget,duration,args.cheap_unit_equivalent,args.expensive_unit_equivalent,
                    args.seed+100000*pp+1000*rep+13*iscen+71)
                for row in trial:row['population_index']=pp;row['rep_index']=rep
                raw.extend(trial)
        print(f'Completed {pp+1}/{args.populations} populations',flush=True)
    summary=aggregate(raw)
    meta={'status':'SIMULATED_OUTCOMES_AND_REVIEWER_ENDPOINTS', 'N':N,'allowed_N':M,
          'populations':args.populations,'repetitions':args.repetitions,
          'budget_equivalent_units':args.budget,'duration_source':source,
          'duration_n':len(duration),'cheap_assumed_unit':args.cheap_unit_equivalent,
          'expensive_assumed_unit':args.expensive_unit_equivalent,
          'phase11_unmodified_source_sha256':hashlib.sha256((Path(__file__).parent/'phase11_reference_unmodified.py').read_bytes()).hexdigest(),
          'generated_utc':datetime.now(timezone.utc).isoformat()}
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'phase13_interval_summary.json').write_text(json.dumps({'metadata':meta,'summary':summary},indent=2,sort_keys=True)+'\n')
    (args.out_dir/'phase13_interval_summary.md').write_text(render_md(meta,summary)+'\n')
    print('DONE -- only aggregate JSON/MD:', args.out_dir)

if __name__=='__main__':run()
