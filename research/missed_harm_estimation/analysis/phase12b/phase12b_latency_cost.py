#!/usr/bin/env python3
"""Read historical *public repository* CSV timing summaries, certify the source inputs,
and optionally rerun a synthetic benchmark with explicit GPU/human cost RATIO.

No private cases, votes, identifiers or response content are read by this module.
The child Phase11 experiment, if requested, reads reviewer durations in sqlite read-only.
This is a controlled SENSITIVITY ANALYSIS, NOT an empirical economic cost equivalence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'phase12b_latency_provenance_v1'
LATENCY = Path('reports/final_v3_policy_timing/latency_summary.csv')
STAGES = Path('reports/final_v3_policy_timing/qwen_stage_latency_summary.csv')
POLICIES = Path('reports/evaluation_measurement_pilot_v1/t4/policy_latency_summary.csv')

class EvidenceError(ValueError): pass


def table(path: Path, required: set[str]) -> list[dict]:
    if not path.is_file():
        raise EvidenceError(f'Missing required historical timing summary: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise EvidenceError(f'Unexpected timing schema: {path.name}, columns={reader.fieldnames}')
        data = list(reader)
    if not data:
        raise EvidenceError(f'Empty timing summary: {path}')
    return data


def metric(row:dict, col:str) -> float:
    try:
        x = float(row[col])
    except (ValueError, TypeError, KeyError) as ex:
        raise EvidenceError(f'Invalid {col} in timing row') from ex
    if not math.isfinite(x) or x < 0:
        raise EvidenceError(f'Nonfinite or negative {col} in timing row')
    return x


def pick(rows:list[dict], filters:dict[str,str], name:str) -> dict:
    hits = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if len(hits) != 1:
        raise EvidenceError(f'Expected exactly one row for {name}; found {len(hits)}')
    row = hits[0]
    n = metric(row, 'n') if 'n' in row else metric(row,'direct_n')
    mean_col = 'mean_ms' if 'mean_ms' in row else 'direct_mean_ms'
    p95_col = 'p95_ms' if 'p95_ms' in row else 'direct_p95_ms'
    mean = metric(row, mean_col)
    p95 = metric(row, p95_col)
    if int(n) != n or n < 1 or mean <= 0 or p95 < mean:
        raise EvidenceError(f'Inconsistent mean/p95 or sample size for {name}')
    return {'name': name, 'n':int(n), 'mean_ms':mean, 'p95_ms':p95,
            'mean_sec':mean / 1000, 'p95_sec':p95 / 1000}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_provenance(repo:Path, factor:float) -> dict:
    if not math.isfinite(factor) or factor <= 0:
        raise EvidenceError('GPU/human marginal cost ratio must be finite and positive')
    r, st, pl = [repo / p for p in (LATENCY, STAGES, POLICIES)]
    rows = table(r, {'component','mode_or_policy','n','mean_ms','p95_ms'})
    stage = table(st, {'component','mode_or_policy','n','mean_ms','p95_ms'})
    policy = table(pl, {'stack','policy_kind','direct_n','direct_mean_ms','direct_p95_ms'})
    model = {
        name:pick(rows, {'component':comp,'mode_or_policy':'prompt_response'},name)
        for name,comp in (
            ('rule_filter_v1','rule_filter_v1'),
            ('cheap_koala_text_moderation','koala_text_moderation'),
            ('expensive_qwen3guard_official','qwen3guard_official'))
    }
    q=pick(stage,{'component':'qwen3guard_official_stage','mode_or_policy':'full_information_always_on:total_latency_ms'},'qwen_stage_total')
    if abs(q['mean_ms']-model['expensive_qwen3guard_official']['mean_ms']) > max(0.05,model['expensive_qwen3guard_official']['mean_ms']*.001):
        raise EvidenceError('Qwen stage total and model timing disagree; do not combine')
    policy_comp = {
        'cheap_only_direct':pick(policy,{'stack':'compact_to_qwen','policy_kind':'cheap_only'},'compact_to_qwen_cheap_only'),
        'full_information_direct':pick(policy,{'stack':'compact_to_qwen','policy_kind':'full_information'},'compact_to_qwen_full_information'),
    }
    # Never subtract two differently batched policy measurements to call it a Qwen call.
    c0=model['cheap_koala_text_moderation']['mean_sec']*factor
    c1=model['expensive_qwen3guard_official']['mean_sec']*factor
    return {
        'study':VERSION,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'HISTORICAL_INFERENCE_LATENCY_MEASURED_COST_RATIO_ASSUMED',
        'privacy':'Public repository timing summaries only; no raw prompt/response or reviewer data',
        'source_csv_sha256':{str(p):sha256(repo/p) for p in (LATENCY,STAGES,POLICIES)},
        'component_benchmark':model,
        'qwen_stage_consistency':q,
        'different_t4_policy_benchmark':policy_comp,
        'relative_gpu_cost_per_second_to_human_cost_per_second_ASSUMED':factor,
        'latency_based_budget_units':{'cheap_proxy_unit':c0,'expensive_proxy_unit':c1},
        'preserved_separate_evidence':{'component_benchmark_n':model['cheap_koala_text_moderation']['n'],
            't4_policy_benchmark_n':policy_comp['cheap_only_direct']['n']},
        'limitations':[
            'Measured latency is historical and may not transfer to different hardware, batch size, caching, input lengths or current software.',
            'Unknown GPU/human economic price ratio. The chosen ratio is an explicit sensitivity parameter, NOT a measured wage or hourly hardware price.',
            'Benchmark proxies are synthetic scores; treating them as Koala/Qwen for unit-cost scaling is a mapping assumption, not a logged predictor run.',
            'No real policy adjudication/training/organizer time or independent gold-label verification is included.',
            'Component timings and the 363-case T4 policy timing experiment are NOT pooled; their measurement protocols differ.',
            'If GPU processing is parallel/async, summing GPU elapsed seconds and human attention seconds is not a wall-clock deployment latency claim.',
            'Repricing an estimator changes its cost model; it does not improve label validity or establish a new theorem.'
        ],
    }


def render_md(out:dict)->str:
    m=out['component_benchmark']; pol=out['different_t4_policy_benchmark']
    lines=['# Phase 12B — Historical proxy-latency evidence and normalized-cost sensitivity','','**Status:** Archived measured component timing; economic exchange ratio assumed. NOT an empirical cost-parity certificate.','','## Measured historical components','','| Component | n | Mean (ms) | p95 (ms) |','|---|---:|---:|---:|']
    for v in m.values():
        lines.append(f"| {v['name']} | {v['n']} | {v['mean_ms']:.3f} | {v['p95_ms']:.3f} |")
    lines +=['','## Separate T4 policy timing experiment','', '| Direct policy | n | Mean (ms) | p95 (ms) |','|---|---:|---:|---:|']
    for v in pol.values():lines.append(f"| {v['name']} | {v['n']} | {v['mean_ms']:.3f} | {v['p95_ms']:.3f} |")
    f=out['relative_gpu_cost_per_second_to_human_cost_per_second_ASSUMED'];c=out['latency_based_budget_units']
    lines+=['','## Sensitivity conversion (NOT measured prices)','',
         r'For assumed relative unit-cost rate $\rho$, the normalized cost in human-cost-seconds of one GPU model call is $\rho\times t_{\mathrm{GPU,sec}}$.',
         f'Chosen **illustrative** ratio $\rho={f:g}$.',
         f"- Cheap model: {c['cheap_proxy_unit']:.8f} normalized units/call",
         f"- Expensive model: {c['expensive_proxy_unit']:.8f} normalized units/call",
         '- Historical Phase 11 prices were 0.08 and 2.5 normalized units/call and were **not** empirically calibrated.','','## Caveats']
    lines +=['- '+x for x in out['limitations']]
    lines+=['','## Reproducibility','','Input-file SHA256 identifiers are in `phase12b_cost_provenance.json`. Repriced Phase 11 results retain synthetic Y, label noise, and actual review-duration resampling when the private DB is passed.','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--relative-gpu-cost',type=float,default=1.0,
        help='ASSUMPTION: GPU marginal economic cost per sec divided by human cost per sec; 1.0 is ONLY a sensitivity scenario')
    p.add_argument('--run-phase11',type=Path,help='Path to previously tested Phase11 simulation script; if set runs both geometries')
    p.add_argument('--db',type=Path,help='Private reviewer DB (read by Phase11 only in read-only mode); REQUIRED for --run-phase11')
    p.add_argument('--populations',type=int,default=8)
    p.add_argument('--repetitions',type=int,default=25)
    p.add_argument('--budget',type=float,default=4800.0)
    a=p.parse_args(argv)
    repo=a.repo.expanduser().resolve()
    outdir=a.out_dir.expanduser().resolve()
    if (a.run_phase11 is not None) and (a.db is None or not a.db.is_file()):
        p.error('--run-phase11 requires existing --db')
    if a.populations<1 or a.repetitions<1 or a.budget <=0:p.error('Positive populations, repetitions and budget required')
    try: evid=read_provenance(repo,a.relative_gpu_cost)
    except EvidenceError as e:p.error(str(e))
    outdir.mkdir(parents=True,exist_ok=True)
    j=outdir/'phase12b_cost_provenance.json'; md=outdir/'phase12b_cost_provenance.md'
    j.write_text(json.dumps(evid,indent=2)+'\n',encoding='utf8')
    md.write_text(render_md(evid),encoding='utf8')
    print('MEASURED TIMING PROVENANCE OK')
    for x in evid['component_benchmark'].values():print(x['name'], 'mean_ms=',round(x['mean_ms'],3),'n=',x['n'])
    print('ASSUMED relative GPU/human price ratio:',a.relative_gpu_cost)
    print('Model-call normalized costs:',evid['latency_based_budget_units'])
    print('Evidence:', j)
    if a.run_phase11:
        c=evid['latency_based_budget_units']
        for geo in ('development_1687','phase6_5000'):
            cmd=[sys.executable, str(a.run_phase11.expanduser().resolve()),'--db',str(a.db.expanduser().resolve()),'--geometry',geo,
                 '--populations',str(a.populations),'--repetitions',str(a.repetitions),'--budget',str(a.budget),
                 '--cheap-unit-equivalent',str(c['cheap_proxy_unit']),
                 '--expensive-unit-equivalent',str(c['expensive_proxy_unit']),
                 '--out-dir',str(outdir/geo)]
            print('RUNNING SYNTHETIC COST SENSITIVITY:',geo,flush=True)
            subprocess.run(cmd,check=True)
        print('PHASE 12B SYNTHETIC REPRICING COMPLETE (ratio assumed, outcomes synthetic)')

if __name__=='__main__':main()
