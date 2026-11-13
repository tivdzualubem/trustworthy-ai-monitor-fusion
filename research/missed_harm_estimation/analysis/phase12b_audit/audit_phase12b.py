#!/usr/bin/env python3
"""Aggregate-only audit of uploaded Phase 11 and Phase 12B JSON files.

Scope: synthetic latent outcomes, historical duration resampling and measured
historical inference latency with ASSUMED relative GPU/human price ratio.
No case records or reviewer IDs are accessed. No new scientific claims inferred.
"""
import argparse, json, math, hashlib
from pathlib import Path
SCENARIOS=('perfect','independent_errors','correlated_errors')
METHODS=('whole_srs','allowed_srs','ppi_plus_tuned','stratppi_mean','multipppi_subset_gls','active_aipw_restricted')
LABELS={'whole_srs':'Whole-population SRS','allowed_srs':'Allowed-only SRS','ppi_plus_tuned':'PPI++ mean','stratppi_mean':'StratPPI mean','multipppi_subset_gls':'MultiPPI subset-GLS adaptation','active_aipw_restricted':'Restricted active AIPW'}
GEO={'development':(1687,1557),'phase6':(5000,4000)}

def load(p):
    with open(p,'r',encoding='utf8') as f: return json.load(f)

def inspect(p11,p12,prov,geo):
    N,M=GEO[geo]
    a,b=load(p11),load(p12)
    for x in (a,b):
        m=x['metadata']
        assert (m['N'],m['allowed_N'])==(N,M)
        assert m['duration_n']==336 and m['duration_source']=='OBSERVED_336_BROWSER_ACTIVE_SECONDS_RESAMPLED'
        assert (m['populations'],m['repetitions_per_population'])==(8,25)
        assert m['budget_units']==4800 and m['data_scope']=='SYNTHETIC_Y_AND_VOTES_ONLY'
        assert m['full_author_AM_PPI_replication'] is False and m['official_MultiPPI_solver_certified'] is False
        assert set(x['methods'])==set(METHODS) and set(x['scenarios'])==set(SCENARIOS)
    assert math.isclose(b['metadata']['cheap_unit_equivalent'],prov['latency_based_budget_units']['cheap_proxy_unit'],rel_tol=1e-10)
    assert math.isclose(b['metadata']['expensive_unit_equivalent'],prov['latency_based_budget_units']['expensive_proxy_unit'],rel_tol=1e-10)
    assert (a['metadata']['cheap_unit_equivalent'],a['metadata']['expensive_unit_equivalent'])==(.08,2.5)
    assert a['metadata']['seed']==b['metadata']['seed']
    out={}
    for scenario in SCENARIOS:
        out[scenario]={}
        for method in METHODS:
            prev=a['summary'][scenario][method];cur=b['summary'][scenario][method]
            assert all(z['n_runs']==200 and z['valid_runs']==200 and z['failed_runs']==0 and z['hard_budget_violation_count']==0 for z in (prev,cur))
            assert cur['max_total_equivalent_cost']<=4800.0+1e-6
            assert math.isclose(cur['mean_total_equivalent_cost'],cur['mean_human_active_seconds']+cur['mean_proxy_cost_units'],rel_tol=1e-9,abs_tol=1e-7)
            assert cur['mean_ci_width']>=0 and 0<=cur['empirical_ci_coverage_reviewer_Z']<=1
            if method in ('whole_srs','allowed_srs'):
                for key in ['rmse_vs_synthetic_latent_Y','rmse_vs_reviewer_endpoint_Z','mean_ci_width','mean_total_equivalent_cost']:
                    assert math.isclose(prev[key],cur[key],abs_tol=1e-11),f'{geo}:{scenario}:{method}:{key} changed despite zero proxy calls'
            if method=='allowed_srs':
                assert cur['mean_cheap_proxy_calls']==0 and cur['mean_expensive_proxy_calls']==0
            val={
                'phase11_rmse_Y':prev['rmse_vs_synthetic_latent_Y'],
                'phase12b_rmse_Y':cur['rmse_vs_synthetic_latent_Y'],
                'change_in_rmse_Y':cur['rmse_vs_synthetic_latent_Y']-prev['rmse_vs_synthetic_latent_Y'],
                'phase12b_rmse_Z':cur['rmse_vs_reviewer_endpoint_Z'],
                'mean_interval_width_unclipped':cur['mean_ci_width'],
                'reviewer_Z_coverage':cur['empirical_ci_coverage_reviewer_Z'],
                'synthetic_Y_coverage_diagnostic':cur['empirical_ci_coverage_latent_Y_DIAGNOSTIC_ONLY'],
                'mean_cost_units':cur['mean_total_equivalent_cost'],
                'mean_human_seconds':cur['mean_human_active_seconds'],
                'mean_cheap_calls':cur['mean_cheap_proxy_calls'],
                'mean_expensive_calls':cur['mean_expensive_proxy_calls'],
            }
            if method!='allowed_srs':
                ci=cur['paired_population_bootstrap95_diff_Y']
                val['paired_diff_rmse_Y_vs_allowed_srs']=cur['paired_rmse_diff_Y_vs_allowed_srs']
                val['paired_bootstrap95']=ci
                val['bootstrap_sign']='above_zero' if ci[0]>0 else 'below_zero' if ci[1]<0 else 'crosses_zero'
            out[scenario][method]=val
    return out

def make_report(prov,report):
    lines=['# Phase 12B — Verified synthetic-cost sensitivity audit','',
    '**Evidence status:** Simulated harmfulness/reviewer responses; actual 336 historical reviewer active times resampled; proxy compute costs based on measured historical inference latency but an **assumed GPU-to-human cost conversion**. Not native outcome validation, not a monetized total cost and not full official author-method fidelity.','',
    '## What was verified','',
    '- Historical 128-case mean latency: Koala **45.6573 ms**, Qwen3Guard **1597.5606 ms**; distinct 363-case T4 policy timing evidence was not pooled.',
    '- Conversion parameter **ρ=1** was assumed, not measured; model-call costs were **0.0456573** and **1.5975606** human-cost-second-equivalent units.',
    '- Both population geometries (1,687/1,557 allowed; 5,000/4,000 allowed), three reviewer scenarios, six methods, 8 independent synthetic populations × 25 repetitions: **7,200 method evaluations** in Phase 12B.',
    '- **Zero reported hard-budget violations and zero failed runs**, conditional on historical observed maximum review-time reservation and assumed cost model.',
    '- Inputs have matching generator seeds and common population geometries. Repricing changes sampling/estimator random draws for methods using proxy calls, so differences between Phase 11 and Phase 12B are *not exact within-trial counterfactual paired effects*.',
    '- SRS (no proxy calls) outputs are identical across repricings; calibration labels are recorded as charged in the benchmark.',
    '', '## RMSE against synthetic latent Y (lower is better)', '']
    for geo,values in report.items():
        lines += [f'### {"Development" if geo=="development" else "Phase 6"} geometry — N={GEO[geo][0]}', '',
        '| Method | Perfect | Independent errors | Correlated errors |', '|---|---:|---:|---:|']
        for m in METHODS:
            lines.append('| '+LABELS[m]+' | '+' | '.join(f'{values[s][m]["phase12b_rmse_Y"]:.4f}' for s in SCENARIOS)+' |')
        lines += ['', '### Paired RMSE difference vs allowed-only SRS (Δ=method−baseline)','', '| Scenario | Method | ΔRMSE | Population-bootstrap interval | Diagnostic |','|---|---|---:|---:|---|']
        for s in SCENARIOS:
            for m in METHODS:
                if m=='allowed_srs':continue
                d=values[s][m]
                a,b=d['paired_bootstrap95']
                lines.append(f'| {s} | {LABELS[m]} | {d["paired_diff_rmse_Y_vs_allowed_srs"]:+.5f} | [{a:+.5f}, {b:+.5f}] | {d["bootstrap_sign"].replace("_"," ")} |')
        lines += ['', '### Interval informativeness (reviewer-defined Z)', '',
        '| Scenario | Allowed SRS width | StratPPI width | MultiPPI width | Active AIPW width |', '|---|---:|---:|---:|---:|']
        for s in SCENARIOS:
            lines.append('| '+s+' | '+' | '.join(f'{values[s][m]["mean_interval_width_unclipped"]:.4f}' for m in ('allowed_srs','stratppi_mean','multipppi_subset_gls','active_aipw_restricted'))+' |')
        lines.append('')
    lines += ['## Mathematical and methodological limitations','',
    '1. **Latent Y remains unidentified in the historical study.** The simulated Y is known to the generator, not the original human reviewers; reviewer-defined endpoint CIs do not imply truth-level coverage under unrestricted human error.',
    '2. **The PPI intervals are broad and use untruncated bounds** (the current code does not intersect its CI with the known feasible range [0, q]). Raw mean-width comparisons are thus not final fair interval comparisons; clipping (and testing correct coverage) should precede presentation of interval efficiency.',
    '3. **Only 8 independent synthetic populations** were generated per geometry. Repeated trials within each population are not independent population replicates; bootstrap uncertainty must be clustered by population, as in the existing summary.',
    '4. **No direct cost parity certificate.** Historical T4 latency on 128 cases and recorded foreground reviewer time are separate resources. The cost-ratio ρ=1 is hypothetical; no measured labor rate, real GPU electricity/rental rate, retraining, independent adjudication or parallel scheduling cost is established.',
    '5. **Method fidelity remains partial.** MultiPPI uses a covariance-information subset GLS adaptation with a numerical SLSQP solver and a custom CI; AM-PPI uses restricted AIPW and does not solve the published joint KKT routing/sampling/weighting problem. Do not call either a complete official benchmark.',
    '6. **Case-level proxy-score validity is untested.** Synthetic predictor scores were never obtained from real Koala/Qwen outputs on this simulated population. Matching their unit prices to historical models does not calibrate predictor accuracy.',
    '', '## Decisions (stop generic sweeps)', '',
    '**Close the existing ρ=1 synthetic repricing analysis.** The hypothesis that simply substituting measured latency values makes a sophisticated estimator consistently superior is not supported.',
    'The next work is scientific fidelity and identifiability, not another broad synthetic simulation:',
    '- Implement/verify author-equivalent MultiPPI optimization and coverage; verify StratPPI theoretical mean specialization; AM-PPI if author implementation available, otherwise label it restricted.',
    '- Fix and validate confidence-interval construction/comparison (including feasible-range clipping and uncertainty accounting for cost/calibration).',
    '- Decide the actual target: reviewer-defined Z (estimable with measured noise) versus latent true Y (requires externally supported error assumptions or independently validated reference labels).',
    '- Acquire genuinely independent response-level reference outcomes under a predeclared probability protocol if claiming latent missed harm; no private data, R2-only disputes or synthetic labels can substitute.',
    '- Preserve reproducibility and provenance records and exclude private cases, ratings, policy credentials and individual review times from public artifacts.',
    '', '## Verified papers', '',
    '- StratPPI: Fisch et al. NeurIPS 2024, https://arxiv.org/abs/2406.04291',
    '- MultiPPI: Cowen-Breen et al. ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/hash/d79ead018586772763ee1e486bd12b3a-Abstract-Conference.html',
    '- AM-PPI: Brawand et al. 2026, https://arxiv.org/abs/2605.08429',
    '', 'All conclusions conditional on the uploaded aggregate inputs and methods as implemented.']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser()
    for flag in ['phase11_development','phase11_phase6','phase12b_development','phase12b_phase6','provenance']:
        p.add_argument('--'+flag.replace('_','-'),type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    prov=load(a.provenance)
    assert prov['status']=='HISTORICAL_INFERENCE_LATENCY_MEASURED_COST_RATIO_ASSUMED'
    assert prov['relative_gpu_cost_per_second_to_human_cost_per_second_ASSUMED']==1.0
    assert prov['component_benchmark']['cheap_koala_text_moderation']['n']==128
    assert prov['component_benchmark']['expensive_qwen3guard_official']['n']==128
    assert prov['different_t4_policy_benchmark']['cheap_only_direct']['n']==363
    report={'development':inspect(a.phase11_development,a.phase12b_development,prov,'development'),
            'phase6':inspect(a.phase11_phase6,a.phase12b_phase6,prov,'phase6')}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'phase12b_verified_crosscheck.json').write_text(json.dumps({'study':'PHASE12B_AGGREGATE_CHECK','status':'VERIFIED_CONDITIONAL_SYNTHETIC','verified_7200_trials':True,'evidence':report},indent=2)+'\n')
    (a.out_dir/'PHASE12B_RESEARCH_DECISION.md').write_text(make_report(prov,report))
    print('VERIFIED: 2 geometries, 3 scenarios, 6 methods, 7200 total trials, zero reported budget violations')
    print('REPORT:',a.out_dir/'PHASE12B_RESEARCH_DECISION.md')

if __name__=='__main__':main()
