#!/usr/bin/env python3
"""Read-only provenance checks; cannot certify that a human label equals latent truth.

Outputs counts, names of required fields, rubric hash and strict evidence statuses.
Never exports case identifiers, policy text, prompts, responses or individual ratings.
"""
from __future__ import annotations
import argparse, hashlib, json, sqlite3
from pathlib import Path
from datetime import datetime, timezone


def safe_gate(db:Path, policy_file:Path|None=None):
    db=db.resolve(strict=True)
    policy_file=policy_file or db.parent/'POLICY_FOR_REVIEWERS.json'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'cases','ratings'}.issubset(tables):raise ValueError('Expected cases and ratings tables')
        ccols={r[1] for r in con.execute('PRAGMA table_info(cases)')}
        rcols={r[1] for r in con.execute('PRAGMA table_info(ratings)')}
        case_n=int(con.execute('SELECT COUNT(*) FROM cases').fetchone()[0])
        rating_n=int(con.execute('SELECT COUNT(*) FROM ratings').fetchone()[0])
        # aggregate only, no identifiers, free text or per-case joined outputs
        rater_stats={str(rater):int(n) for rater,n in con.execute('SELECT rater, COUNT(*) FROM ratings GROUP BY rater')}
    policy_stat={'snapshot_found':policy_file.is_file(),'policy_sha256':None,
                 'system_policy_nonempty':False,'ontology_has_C1_C8':False,
                 'original_review_page_snapshot_verified':False}
    if policy_file.is_file():
        b=policy_file.read_bytes()
        obj=json.loads(b)
        system_policy=obj.get('target_policy',{}).get('system_policy','')
        codes={x.get('id') for x in obj.get('ontology',{}).get('categories',[]) if isinstance(x,dict)}
        policy_stat.update(policy_sha256=hashlib.sha256(b).hexdigest(),
                system_policy_nonempty=isinstance(system_policy,str) and bool(system_policy.strip()),
                ontology_has_C1_C8=all(f'C{i}' in codes for i in range(1,9)))
    return {'status':'REVIEWER_REFERENCE_TRUTH_NOT_VERIFIED',
            'study_cases':case_n,'human_review_actions':rating_n,
            'rater_action_counts':rater_stats,
            'historical_label_column_exists_not_gold':('historical_y' in ccols),
            'review_time_column_exists':('active_seconds' in rcols),
            'policy_snapshot':policy_stat,
            'verified_qualified_reference_truth':'NOT_ESTABLISHED',
            'probability_sample_inclusion_probabilities':'NOT_VERIFIED_BY_THIS_AUDIT',
            'prospective_target_stream':'NOT_ESTABLISHED',
            'scope':'Historical development review metadata only; no user data exported',
            'interpretation':['A reference label is not established merely by multiple opinions or historical benchmark labels.',
                'No confidence interval based on reviewer Z is automatically a confidence interval for latent Y.',
                'Rubric hash documents the current snapshot only; this read-only audit cannot prove which exact page a reviewer saw at a past time.',
                'This gate is not a request for R2-only follow-up review.'],
            'generated_utc':datetime.now(timezone.utc).isoformat()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',type=Path,required=True)
    ap.add_argument('--policy-file',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(f'Refusing to overwrite: {a.out}')
    v=safe_gate(a.db,a.policy_file)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
    print('AGGREGATE-ONLY GATE COMPLETE',a.out)
    print('Cases:',v['study_cases'],'Review actions:',v['human_review_actions'])
    print('Status:',v['status'])

if __name__=='__main__':main()
