#!/usr/bin/env python3
"""Reconcile completed Trustworthy AI research evidence without treating unverified Y as gold.

Inputs: aggregate Phase8B and Phase9B JSON, Phase5B1/B2/B3 zipped aggregates.
Does not access case text, passwords, individual reviewer judgments, protected test splits,
or external network. Generates a descriptive evidence-completeness assessment.
"""
import argparse,csv,hashlib,io,json,math,zipfile
from pathlib import Path

METHODS_B2=('allowed_only_srs','stratppi_proportional_fixed1','stratppi_proportional_pilot_lambda','stratppi_allowed70_pilot_lambda')

def read_zip(p,n):
    with zipfile.ZipFile(p) as z:
        if n not in z.namelist(): raise ValueError(f'{n} missing in {p}')
        return z.read(n).decode('utf-8')

def read_manifest(p):return json.loads(read_zip(p,'manifest.json'))
def read_csv_zip(p,n):return list(csv.DictReader(io.StringIO(read_zip(p,n))))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def float_or_nan(s):
    try:return float(s)
    except (ValueError,TypeError):return float('nan')

def analyze(p8,p9,p5b1,p5b2,p5b3):
    d8=json.loads(Path(p8).read_text()); d9=json.loads(Path(p9).read_text())
    b1=read_manifest(p5b1);b2=read_manifest(p5b2);b3=read_manifest(p5b3)
    if d8.get('study')!='PHASE8B_POPULATION_VARIABILITY_AUDIT':raise ValueError('Wrong Phase8B audit')
    if d9.get('audit_id')!='PHASE9B_READ_ONLY_THREE_RATER_R4_DESIGN_COST_AUDIT':raise ValueError('Wrong Phase9B audit')
    if d9['review_counts']['total_ratings']!=336 or d9['review_counts']['distinct_cases']!=96:raise ValueError('Reviewer integrity mismatch')
    if d8['geometries']['development_footprint_1687']['synthetic_population_count']!=16:raise ValueError('Population count mismatch')
    m2=read_csv_zip(p5b2,'stratppi_comparison.csv'); m3=read_csv_zip(p5b3,'comparison.csv')
    rows2={r['method']:r for r in m2 if r['setup_id']=='qwen_after_rule_compact' and r['human_budget']=='160'}
    if not all(k in rows2 for k in METHODS_B2):raise ValueError('Phase 5B2 method missing')
    metrics2={m:{'A_RMSE':float(rows2[m]['A_RMSE']),'calibration_labels':int(rows2[m]['calibration_labels'])} for m in METHODS_B2}
    qwenb1=b1['populations']['qwen_after_rule_compact']; compb1=b1['populations']['compact_after_rule'];N=qwenb1['N']
    assert N==compb1['N']==1687
    delta_count=compb1['allowed_harmful_count']-qwenb1['allowed_harmful_count']
    assert delta_count==1,'Unexpected archived Qwen effect'
    scenario_out={}
    for geometry,g in d8['geometries'].items():
        sc={}
        for scenario in ['perfect_binary','noisy_independent','noisy_correlated','correlated_with_abstention']:
            key=scenario+'|budget_4800s'
            if key not in g['summary']: raise ValueError('Missing '+key)
            x=g['summary'][key]
            b=x['allowed_srs_single'];major=x['allowed_three_vote_majority'];ppi=x['allowed_proxy_stratified_equal']
            paired=g['paired_contrasts_vs_allowed_srs'][key]
            sc[scenario]={
                'srs_rmse':b['pooled_rmse_vs_latent_Y'],
                'majority_rmse':major['pooled_rmse_vs_latent_Y'],
                'stratified_rmse':ppi['pooled_rmse_vs_latent_Y'],
                'majority_delta_rmse_ci95':paired['allowed_three_vote_majority']['between_population_bootstrap_95_interval'],
                'strat_delta_rmse_ci95':paired['allowed_proxy_stratified_equal']['between_population_bootstrap_95_interval'],
                'majority_latent_ci_coverage_diagnostic_only':major['mean_coverage_for_latent_Y_diagnostic_only'],
                'srs_latent_ci_coverage_diagnostic_only':b['mean_coverage_for_latent_Y_diagnostic_only'],
            }
        scenario_out[geometry]=sc
    costs=d9['review_timing']
    group=[];N=d9['stratification']['population_N']
    for r in ('R1','R2','R3'):
        A=0.;v=0.;q=0
        for s in d9['stratification']['strata']:
            if s['archived_decision']!='allowed':continue
            nh=s['sample_count'];Nh=s['population_count'];cnt=s['primary_label_counts_by_role'][r]
            if nh<2:raise ValueError('Insufficient stratum')
            ph=cnt.get('harmful',0)/nh;w=Nh/N
            A+=w*ph
            v+=w*w*(1-nh/Nh)*ph*(1-ph)/(nh-1)
            q+=1
        se=math.sqrt(v)
        group.append({'reviewer':r,'harmful_endpoint_A':A,'design_se_approx':se,
                      'approx_normal_ci95':[max(0,A-1.96*se),min(1,A+1.96*se)]})
    extra=[]
    for r in m3:
        if r['setup_id']=='qwen_after_rule_compact' and r['human_budget_reference']=='160' and r['optional_call_budget_hint']=='80' and r['hypothetical_review_seconds']=='30':
            extra.append({'method':r['method'],'A_RMSE':float(r['A_RMSE']),
                          'fraction_over_budget':float(r['total_cost_over_budget_mean']),
                          'mean_total_cost_seconds':float(r['actual_total_cost_seconds_mean'])})
    if len(extra)!=3:raise ValueError('Unexpected restricted AM-PPI rows')
    return {'provenance':{str(x.name):sha(x) for x in (p8,p9,p5b1,p5b2,p5b3)},
           'phase5_old_labels':{'setup':'qwen_after_rule_compact','budget_count':160,'qwen_effect_allowed_harm_reduction_cases':delta_count,
                                  'qwen_effect_rate_per_historical_case':delta_count/N,'methods':metrics2,
                                  'amppi_restricted_substudy':extra,
                                  'caveats':['Y historical unverified labels','D cross-fitted historical, not native deployment',
                                             'Phase5B2 is a mean specialization, not full StratPPI CI',
                                             'Phase5B3 is a restricted AM-PPI adaptation, not faithful author implementation',
                                             'Phase5B3 30sec cost is hypothetical; some designs overspend']},
           'phase8b_synthetic':scenario_out,
           'phase9b_real_review':{'weighted_reviewer_endpoints':group,
              'three_reviewers_primary_n':d9['review_counts']['primary_ratings'],
              'selected_r4_n':d9['review_counts']['r4_ratings'],
              'real_active_seconds':costs['all_active_total_seconds'],
              'real_wall_seconds':costs['all_wall_total_seconds'],
              'r4_active_seconds':costs['r4_active_total_seconds'],
              'r4_share_of_active_seconds':costs['r4_share_of_all_active_time']},
            'formal_status':{'estimand':'A=P(Y=1,D=0) (post-response, pre-delivery historical monitor)',
                'Y_identified_from_existing_reviewers':False,
                'native_deployment_or_fresh_confirmatory_data_available':False,
                'full_published_baselines_verified':False,
                'full_cost_validated':False,
                'prospective_blinded_study_prepared':True,
                'gold_adjudication_or_error_bounds_validated':False},
           'not_github_commit_ready':True}

def markdown(a):
    x=a['phase5_old_labels'];s=a['phase8b_synthetic'];h=a['phase9b_real_review'];f=a['formal_status']
    out=['# Phase 10 — Completed evidence reconciliation / work-completion gate',
         '', '> Evidence inventory only; does not establish a deployment safety certificate.',
         '', '## Primary estimand and frame',
         '', 'The primary estimand is **A=P(Y=1,D=0)** where Y is actual policy-defined assistant-response harmfulness and D=0 is allowed by a frozen post-generation, pre-delivery monitor. The historical 1,687-case development D is a cross-fitted classifier decision, not live deployed guard behavior. FNR is secondary.',
         '', '## 1. Previously completed historical baseline comparisons',
         '', '| 160-label method (Qwen setup) | A RMSE | Labels spent tuning |',
         '|---|---:|---:|']
    for name,row in x['methods'].items():out.append(f'| `{name}` | {row["A_RMSE"]:.5f} | {row["calibration_labels"]} |')
    out +=['','**Result:** Allowed-only random sampling is better than these limited StratPPI mean estimators on the archived historical-label experiment. This is **not** a claim against full StratPPI. The old labels and classifier are not externally validated.',
          '',f'Archived Qwen improvement (one historical setup): **{x["qwen_effect_allowed_harm_reduction_cases"]} fewer allowed harmful response(s) out of 1,687**, equivalent to {100*x["qwen_effect_rate_per_historical_case"]:.4f} percentage points in A; one more blocked harmful response. Not a new monitor deployment result.',
          '', '### Earlier restricted AM-PPI cost-validity audit',
          '| Method | A RMSE | Actual simulation over-budget frequency |',
          '|---|---:|---:|']
    for row in x['amppi_restricted_substudy']:out.append(f'| `{row["method"]}` | {row["A_RMSE"]:.5f} | {100*row["fraction_over_budget"]:.0f}% |')
    out+=['','**Not a paper-faithful AM-PPI comparison:** Costs used hypothetical 30s reviews and historical proxy means; expected-budget designs can exceed a realized hard budget. These results are negative exploratory evidence, not the final published-method comparison.',
          '', '## 2. Independent-population synthetic robustness evidence',
          '', '16 synthetic populations ×30 repetitions/geometry/scenario, human active-time budget 4,800 seconds, proxy acquisition cost set to zero (limitation). RMSE is against synthetic Y, not real Y.',
          '', '| Geometry | Scenario | SRS RMSE | Proxy-stratified RMSE | 3-review majority RMSE | Majority–SRS 95% paired population-bootstrap CI |',
          '|---|---|---:|---:|---:|---|']
    for geo,sc in s.items():
        for name,z in sc.items():out.append(f'| {geo} | {name} | {z["srs_rmse"]:.5f} | {z["stratified_rmse"]:.5f} | {z["majority_rmse"]:.5f} | [{z["majority_delta_rmse_ci95"][0]:+.5f}, {z["majority_delta_rmse_ci95"][1]:+.5f}] |')
    out+=['','**Conditional result:** better proxy stratification with perfect truth; majority voting smaller RMSE with specified noisy raters. CIs for simulated reviewer outcomes are not valid intervals for latent Y unless the reviewer-error model is justified. Real reviewer errors are uncalibrated.',
          '', '## 3. Completed real blinded review',
          '',f'96 historical cases, {h["three_reviewers_primary_n"]} primary assessments, {h["selected_r4_n"]} selected fourth ratings; {h["real_active_seconds"]:.1f} active seconds and {h["real_wall_seconds"]:.1f} wall seconds. The fourth reviewer contributed {h["r4_active_seconds"]:.1f} active seconds ({100*h["r4_share_of_active_seconds"]:.1f}%). These are observed browser-active timings, **not** total economic costs.',
          '', '| Reviewer-defined binary Harmful endpoint | Weighted A | Approx. SE | Approx. 95% normal CI |',
          '|---|---:|---:|---|']
    for q in h['weighted_reviewer_endpoints']:
        lo,hi=q['approx_normal_ci95'];out.append(f'| {q["reviewer"]} | {q["harmful_endpoint_A"]:.4f} | {q["design_se_approx"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
    out += ['', 'These CIs assume the six development source×decision samples were SRS without replacement within strata and approximate normality; several strata are tiny. They target each **reviewer’s judgment**, not true harmfulness. Cross-reviewer comparisons are paired on the same cases; do not use independent-difference standard errors. Full latent-Y inference remains impossible without extra error assumptions or validated reference labels.',
            '', '## 4. Mathematical limits and prior-method overlap',
            '', 'Without any restriction on P(R1,R2,R3,R4|Y,D), Y is not identified by review votes: any observed voting distribution can coexist with multiple latent-Y prevalences. This is a classical, **not novel**, nonidentification argument. Three conditionally independent binary reviewers may generically identify a two-class mixture up to label switching, but conditional independence is not validated by the current 96-case pilot; the three-reviewer 2×2×2 binary model is just-identified. The selected R4 vote is an additional opinion, not gold truth.',
            '', 'MultiPPI/AM-PPI optimize predictor/label acquisition under verified-outcome assumptions. StratPPI handles stratified PPI. This project must not rename their allocation formulas, inverse probability weights, difference estimates, or elementary bounded-label inequalities as new mathematics.',
            '', '## 5. Evidence completeness',
            '', '| Requirement | Evidence state | Remaining definitive action |',
            '|---|---|---|',
            '| Primary estimand / monitor timing | Defined | Freeze native prospective D before making deployment claim |',
            '| Explain Qwen allowed/blocked contrast | Historical decomposition complete | Verify on truly native frozen monitor / fresh data if claimed generally |',
            '| Strong simple sampling | Historical Phase5B1, B2 completed | Confirm on credible independent Y, under equal full costs |',
            '| Published StratPPI / MultiPPI / AM-PPI | Mean specialization / restricted AM only | Reproduce relevant **published** estimator, tuning, CIs, costs under same assumptions and data; no fidelity claim otherwise |',
            '| Mathematical impossibility/necessary conditions | Classical no-identification regime and conditional-bounds mapped | Derive/verify any additional claimed theorem against prior literature; if none, choose empirical contribution |',
            '| Real independent label collection and uncertainty | 96×3 + selected 48 completed | Qualified independent reference procedure or predeclared sensitivity bounds for latent Y; **do not** re-open ad hoc R2 QC |',
            '| Cost: calibration, humans, adjudication, inference | Active timings measured, historic inference available | Per-case native inference and calibration/adjudication overhead; distinguish acquisition from wall-clock/cost |',
            '| Confidence coverage / distribution shift | Synthetic robustness with 16 independent generated populations | Reference-truth coverage / fresh native population remains unavailable; cannot certify |',
            '| Reproducible freeze | Hashes produced below | Freeze WSL execution code, native data source, predictions, groups, labels, timing logs; run tests and do privacy audit |',
            '| Publication of reproducible research | Pending evidence review | Document method/cost/reliability limits before public release |',
            '', '### Decision rule',
            '', '**Do not prolong the 12-case R2 diagnosis.** The next high-value step is one carefully frozen, genuinely independent native-monitor/reference-label study with measured per-component costs and a fair published-method benchmark. If qualifying new labels are infeasible, frame only an honest development-stage methodology result; do not claim full completion, true latent risk, or paper-ready novelty.',
            '', '## 6. Provenance / SHA256', '']
    for name,dig in a['provenance'].items():out.append(f'- `{name}`: `{dig}`')
    out += ['', 'All inputs are aggregated or archived historical developmental summaries. No private case records were read.','']
    return '\n'.join(out)

def main():
    p=argparse.ArgumentParser()
    for x in ('phase8b','phase9b','phase5b1','phase5b2','phase5b3'):
        p.add_argument('--'+x,required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    args=p.parse_args()
    a=analyze(args.phase8b,args.phase9b,args.phase5b1,args.phase5b2,args.phase5b3)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    jf=args.out_dir/'phase10_evidence_gate.json'; mf=args.out_dir/'phase10_evidence_gate.md'
    if jf.exists() or mf.exists():raise FileExistsError('Refusing overwrite: '+str(args.out_dir))
    jf.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');mf.write_text(markdown(a))
    print('Evidence reconciliation PASSED; no private data accessed; no label truth claimed.')
    print('Gate:',a['formal_status']);print('REPORT:',mf);print('JSON:',jf)
if __name__=='__main__':main()
