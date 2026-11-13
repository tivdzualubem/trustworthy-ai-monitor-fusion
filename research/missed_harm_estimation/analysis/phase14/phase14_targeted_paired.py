#!/usr/bin/env python3
"""Paired original vs repaired restricted MultiPPI allocator on synthetic Y,Z only.

Real DB, if passed, supplies active review durations exclusively.
No raw prompts/responses, case IDs or votes are read or exported.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
import numpy as np
import phase11_reference_unmodified as p
import phase13_methods as m
from phase14_certified_solver import repaired_optimize


class UncertifiedSolver(RuntimeError): pass


def paired_run(pop,scenario,budget,durations,c0,c1,seed,tolerance):
    trials=[];captured={}
    original=p.optimize_multi
    for label in ('original','repaired'):
        rng=np.random.default_rng(seed)
        ledger=p.Ledger(budget,durations,c0,c1,rng)
        def callback(Sigma,costs,seed=0):
            old,success=original(Sigma,costs,seed)
            if label=='original':
                captured[label]={'certificate':m.multi_optimality_certificate(Sigma,costs,old),'success':success}
                return old,success
            sol=repaired_optimize(Sigma,costs,target_relative_gap=tolerance,original_optimizer=original)
            captured[label]={'certificate':sol['after_certificate'],'success':sol['certified'],
                             'pre_certificate':sol['original_certificate']}
            if not sol['certified']:raise UncertifiedSolver('Repair failed to certify; no point estimate accepted')
            return sol['fractions'],True
        p.optimize_multi=callback
        try:
            r=p.multipppi(pop,scenario,ledger,rng)
            if not ledger.report()['hard_budget_ok']:raise AssertionError('Hard budget exceeded')
        except UncertifiedSolver:
            r={'valid':False,'reason':'uncertified solver allocation'}
        finally:p.optimize_multi=original
        r.update({'arm':label,'solver_audit':captured.get(label)})
        trials.append(r)
    return trials


def summarize(records):
    summary={}
    for sc in p.SCENARIOS:
        rs=[d for d in records if d['scenario']==sc]
        out={'paired_total':len(rs),
             'certified_repaired':sum(bool(d['repaired']['solver_audit'] and d['repaired']['solver_audit']['success']) for d in rs),
             'repaired_uncertified_runs':sum(not (d['repaired']['solver_audit'] and d['repaired']['solver_audit']['success']) for d in rs),
             'original_solver_reported_success':sum(bool(d['original']['solver_audit'] and d['original']['solver_audit']['success']) for d in rs)}
        for label in ('original','repaired'):
            candidates=[d[label] for d in rs if d[label]['valid']]
            out[label]={
               'valid':len(candidates),
               'rmse_vs_synthetic_Y':float(math.sqrt(np.mean([x['squared_error_Y'] for x in candidates]))) if candidates else None,
               'rmse_vs_reviewer_Z':float(math.sqrt(np.mean([x['squared_error_Z'] for x in candidates]))) if candidates else None,
               'mean_ci_width':float(np.mean([x['ci_width'] for x in candidates])) if candidates else None,
               'mean_realized_cost':float(np.mean([x['ledger']['actual_total_equivalent_units'] for x in candidates])) if candidates else None,
               'max_realized_cost':float(max(x['ledger']['actual_total_equivalent_units'] for x in candidates)) if candidates else None,
               'hard_budget_violations':sum(not x['ledger']['hard_budget_ok'] for x in candidates),
            }
            gaps=[d[label]['solver_audit']['certificate']['relative_gap_upper'] for d in rs if d[label].get('solver_audit')]
            out[label]['max_certified_relative_gap_upper']=float(max(gaps)) if gaps else None
            out[label]['median_certified_relative_gap_upper']=float(np.median(gaps)) if gaps else None
            out[label]['fraction_below_0_5pct']=float(np.mean(np.array(gaps)<.005)) if gaps else None
        paired=[d for d in rs if d['original']['valid'] and d['repaired']['valid']]
        out['paired_valid']=len(paired)
        out['difference_squared_error_Y_repaired_minus_original']=float(np.mean([d['repaired']['squared_error_Y']-d['original']['squared_error_Y'] for d in paired])) if paired else None
        out['all_pilot_continuous_objectives_not_worse']=bool(all(d['repaired']['solver_audit']['certificate']['objective']<=d['original']['solver_audit']['certificate']['objective']*(1+1e-9) for d in rs if d['repaired']['solver_audit'] and d['original']['solver_audit']))
        summary[sc]=out
    return summary


def render(result):
    z=result['metadata']; summary=result['summary']; lines=[
     '# Phase 14 — Targeted restricted MultiPPI optimizer repair', '',
     '**Scope: synthetic outcomes and votes; resampled real review durations only. Not an author MultiPPI replication.**','',
     f"N={z['N']}, allowed={z['allowed_N']}, budget={z['budget']}, populations={z['populations']}, repetitions={z['repetitions']}; reviewer duration source={z['duration_source']}",'',
     '| Scenario | Arm | RMSE vs synthetic Y | RMSE vs reviewer Z | Mean CI width | Max convex relative-gap upper bound | Valid pairs |',
     '|---|---|---:|---:|---:|---:|---:|']
    for sc,d in summary.items():
        for arm in ('original','repaired'):
            e=d[arm]
            f=lambda x: 'NA' if x is None else f'{x:.5f}'
            lines.append(f"| {sc} | {arm} | {f(e['rmse_vs_synthetic_Y'])} | {f(e['rmse_vs_reviewer_Z'])} | {f(e['mean_ci_width'])} | {f(e['max_certified_relative_gap_upper'])} | {d['paired_valid']} |")
    lines+=['','## Scientific boundaries','',
       '- Original and repaired estimators have distinct randomized acquisitions after their common pilot; comparisons are paired at population/seed but not matched acquired observations.',
       '- The optimizer target is the original restricted 7-subset covariance-GLS continuous objective with a 12% human-observation budget floor, NOT official MultiPPI optimal allocation.',
       '- Numerical certificate applies to the pilot covariance objective; it cannot prove final RMSE or confidence interval coverage, and does not certify integer realization.',
       '- Repaired result is rejected if the relative first-order convexity-gap upper bound exceeds the declared tolerance.',
       '- Methods have different human action counts and actual realized budget charges; all cost components are reported without equating GPU seconds with money.',
       '- Synthetic Y and reviewer Z are not real policy-adjudicated safety outcomes. The private DB is accessed only for prior reviewer active durations.',
       '- Further general simulations are not a substitute for prospective reference-label validation.', '']
    return '\n'.join(lines)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',type=Path)
    ap.add_argument('--geometry',choices=['development_1687','phase6_5000'],default='development_1687')
    ap.add_argument('--populations',type=int,default=2)
    ap.add_argument('--repetitions',type=int,default=5)
    ap.add_argument('--budget',type=float,default=4800.)
    ap.add_argument('--cheap-unit-equivalent',type=float,default=.04565729060156249)
    ap.add_argument('--expensive-unit-equivalent',type=float,default=1.5975605696953123)
    ap.add_argument('--tolerance',type=float,default=.005)
    ap.add_argument('--seed',type=int,default=20261112)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    if args.out_dir.exists():raise FileExistsError('Refusing overwrite: '+str(args.out_dir))
    if args.populations<1 or args.repetitions<1 or args.budget<=250 or not 0<args.tolerance<1:ap.error('Invalid parameters')
    durations,source=p.read_review_durations(args.db)
    N,M=(1687,1557) if args.geometry=='development_1687' else (5000,4000)
    records=[]
    for pp in range(args.populations):
        pop=p.make_population(args.seed+1009*pp+101,N,M)
        for rr in range(args.repetitions):
            for iscen,sc in enumerate(p.SCENARIOS):
                seed=args.seed+100000*pp+1000*rr+13*iscen+71+193*4
                old,rep=paired_run(pop,sc,args.budget,durations,args.cheap_unit_equivalent,args.expensive_unit_equivalent,seed,args.tolerance)
                records.append({'scenario':sc,'original':old,'repaired':rep})
        print(f'Completed {pp+1}/{args.populations} synthetic populations',flush=True)
    output={'metadata':{'study':'PHASE14_RESTRICTED_CONTINUOUS_GLS_SOLVER_FIX',
               'N':N,'allowed_N':M,'geometry':args.geometry,'populations':args.populations,
               'repetitions':args.repetitions,'budget':args.budget,
               'duration_source':source,'duration_n':len(durations),
               'cheap_cost_assumption':args.cheap_unit_equivalent,'expensive_cost_assumption':args.expensive_unit_equivalent,
               'relative_gap_certification_threshold':args.tolerance,
               'simulated_outcomes_only':True,'method_fidelity':'RESTRICTED_NOT_AUTHOR_MULTI_PPI',
               'generated_utc':datetime.now(timezone.utc).isoformat(),
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'summary':summarize(records)}
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'phase14_aggregate.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    (args.out_dir/'phase14_aggregate.md').write_text(render(output))
    print('PHASE14 TARGETED COMPARISON COMPLETE',args.out_dir)

if __name__=='__main__':main()
