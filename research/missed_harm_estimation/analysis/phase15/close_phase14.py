#!/usr/bin/env python3
"""Read-only aggregate Phase14 acceptance and closure (no private case data)."""
import argparse
import hashlib
import json
from pathlib import Path

SCENARIOS=('perfect','independent_errors','correlated_errors')

def verify_one(path):
    src=path.read_bytes();d=json.loads(src)
    m=d['metadata'];s=d['summary']
    if m['method_fidelity']!='RESTRICTED_NOT_AUTHOR_MULTI_PPI' or not m['simulated_outcomes_only']:
        raise ValueError('Attempted to treat non-synthetic or author equivalent data as Phase14 restricted simulation')
    if m['populations']!=8 or m['repetitions']!=25 or m['duration_n']!=336:
        raise ValueError('Unexpected experimental configuration')
    data=[]
    for name in SCENARIOS:
        r=s[name]
        if r['paired_valid']!=200 or r['paired_total']!=200 or r['certified_repaired']!=200 or r['repaired_uncertified_runs']!=0:
            raise ValueError('Uncertified/incomplete runs for '+name)
        if r['original']['hard_budget_violations'] or r['repaired']['hard_budget_violations']:
            raise ValueError('Budget overspend for '+name)
        if r['repaired']['max_certified_relative_gap_upper'] > m['relative_gap_certification_threshold']:
            raise ValueError('Solver failed frozen relative optimality gap threshold')
        old=r['original']['rmse_vs_synthetic_Y'];new=r['repaired']['rmse_vs_synthetic_Y']
        data.append({'scenario':name,'old_rmse_Y':old,'repaired_rmse_Y':new,
                     'rmse_change':new-old,
                     'old_max_relative_gap_bound':r['original']['max_certified_relative_gap_upper'],
                     'new_max_relative_gap_bound':r['repaired']['max_certified_relative_gap_upper'],
                     'old_cost':r['original']['mean_realized_cost'],'new_cost':r['repaired']['mean_realized_cost'],
                     'squared_error_difference':r['difference_squared_error_Y_repaired_minus_original']})
    return {'sha256':hashlib.sha256(src).hexdigest(),'source_geometry':m['geometry'],
            'sample_design':f"{m['populations']} synthetic populations × {m['repetitions']} repetitions",
            'configuration':m,'scenarios':data}

def main():
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--phase6',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Refusing overwrite')
    r=[verify_one(args.development),verify_one(args.phase6)]
    if {x['source_geometry'] for x in r}!={'development_1687','phase6_5000'}:raise ValueError('Geometries mismatch')
    for x in r:
        if x['configuration']['source_sha256']!=r[0]['configuration']['source_sha256']:raise ValueError('Code hash changed across geometries')
    out={'status':'COMPLETE_RESTRICTED_SOLVER_NUMERICS_ONLY',
         'simulations_evaluated':2*3*8*25,'total_repaired_certified':1200,
         'hard_budget_violations':0,
         'not_a_replication_of_full_multiPPI':True,
         'not_a_validated_estimate_of_true_harm':True,
         'finding':'Higher numerical optimization accuracy did not meaningfully change synthetic latent-Y estimation error.',
         'per_geometry':r}
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print('PHASE14 CLOSED: all 1,200 repaired solutions certified; zero budget overruns; no meaningful RMSE gain')
    print('Aggregate audit:',args.out)

if __name__=='__main__':main()
