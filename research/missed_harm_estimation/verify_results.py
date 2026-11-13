#!/usr/bin/env python3
"""Aggregate-only consistency checks. No private database access."""
from pathlib import Path
import json,math,hashlib
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
for name in ['phase14_development.json','phase14_phase6.json']:
 d=json.loads((R/name).read_text())
 assert d['metadata']['simulated_outcomes_only'] is True
 for scen, s in d['summary'].items():
  assert s['certified_repaired']==200 and s['repaired_uncertified_runs']==0
  assert s['repaired']['hard_budget_violations']==0
  assert s['repaired']['fraction_below_0_5pct']==1
for name in ['phase11_development_20261112.json','phase11_phase6_20261112.json','phase12b_development_20261112.json','phase12b_phase6_20261112.json']:
 d=json.loads((R/name).read_text())
 assert d['metadata']['data_scope']=='SYNTHETIC_Y_AND_VOTES_ONLY'
 for scenario,vals in d['summary'].items():
  for method,v in vals.items():
   assert v['hard_budget_violation_count']==0 and v['failed_runs']==0
   assert v['valid_runs']==200
   assert v['max_total_equivalent_cost']<=4800+1e-6
for name in ['phase13_development.json','phase13_phase6.json']:
 d=json.loads((R/name).read_text())
 for scen,methods in d['summary'].items():
  for key,v in methods.items():
   assert v['coverage_reviewer_Z_clipped']==v['coverage_reviewer_Z_original']
   assert v['mean_clipped_conditional_interval_width']<=v['mean_original_interval_width']+1e-9
   assert v['hard_budget_failures']==0
p=json.loads((R/'phase13_reference_gate.json').read_text())
assert p['human_review_actions']==336 and p['verified_qualified_reference_truth']=='NOT_ESTABLISHED'
print('VERIFICATION OK: 2 phase14, 4 phase11/12B, 2 phase13 and reference gate checked')
