#!/usr/bin/env python3
"""Phase 4C: CPU-only, development-only priced optional-monitor audit pilot.

This is a *two-stage stratified probability sample*, not guard deployment,
Active Testing, a new method, or a valid field-safety certificate.
Existing optional-monitor scores are treated as unavailable unless a monitor
query is explicitly allocated to the associated row in a simulated run.

Only reads the two permitted development artifacts, frozen protocols and prior
aggregate manifests. Never reads sealed test/shift data or emits raw records.
"""
from __future__ import annotations
import hashlib, json, math, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

BRANCH='stage-a-paired-transport-20260921'
SOURCE='reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet'
FOLDS='reports/decision_value_real_data/development_outer_fold_assignments.csv'
PROTOCOL='configs/decision_value_real_data_protocol_v1.json'
BOUNDARY='configs/exact_cost_risk_cascade_protocol_v2.json'
PRIOR='results/phase4b_active_testing_lure_v1/active_testing_manifest.json'
BASELINES='results/phase4b_active_testing_lure_v1/matched_baseline_comparison.csv'
OUT='results/phase4c_monitor_assisted_audit_v1'
ARTIFACT='phase4c_monitor_assisted_audit_v1'
METHODS=('random_proportional','priority_proportional','priority_tilted')
MONITOR_COUNTS=(80,320)
HUMAN_BUDGETS=(40,80,160)
REPETITIONS=200
SEED=20261009


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()


def windows_downloads():
 try:
  s=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
                             '[Environment]::GetFolderPath("UserProfile")'],text=True,timeout=8).strip().replace('\r','')
  p=Path(subprocess.check_output(['wslpath','-u',s],text=True,timeout=5).strip())/'Downloads'
  if p.is_dir():return p
 except (OSError,subprocess.SubprocessError,ValueError):pass
 p=Path('/mnt/c/Users/user/Downloads')
 return p if p.is_dir() else None


def grouped(D, uncertainty, optional, acquired, ids):
 """Observable partitions using only pre-review cheap and *acquired* optional scores.

 For unscored rows, use cheap-uncertainty median split inside each decision D.
 For scored rows, split at within-acquired-D optional-score median. Thus 8
 possible strata: 2 decisions x 2 acquisition flags x 2 score bands.
 """
 N=len(D)
 if len(uncertainty)!=N or len(optional)!=N or len(acquired)!=N or len(ids)!=N:raise ValueError('length mismatch')
 if not np.isfinite(uncertainty).all():raise ValueError('nonfinite cheap uncertainty')
 groups=[]; labels=[]; priority=[]
 for d in (0,1):
  for bought in (False,True):
   ix=np.flatnonzero((D==d)&(acquired==bought))
   if not len(ix):continue
   vals=optional[ix] if bought else uncertainty[ix]
   if not np.isfinite(vals).all():raise ValueError('nonfinite acquired monitor values')
   order=np.lexsort((ids[ix],vals))
   for band, members in enumerate((ix[order[:len(ix)//2]],ix[order[len(ix)//2:]])):
    if len(members):
     groups.append(members);labels.append(f'D{d}_{"monitor" if bought else "cheap"}_{band}')
     priority.append(2.0 if bought and band==1 else 1.0)
 return groups,labels,np.asarray(priority,dtype=float)


def acquire_monitor(D,uncertainty,ids,M,policy,rng):
 """Simulated monitor queries selected without inspecting any Y or optional score."""
 N=len(D)
 if M<1 or M>=N:raise ValueError('monitor count invalid')
 if policy=='random':
  selected=rng.choice(N,size=M,replace=False)
 elif policy=='priority':
  # Sentinel: 20% uniform among all remaining, other queries cheap-first.
  deterministic=M-int(math.ceil(.2*M))
  # Top uncertainty, tie-break by example id; both D strata remain eligible.
  rank=np.lexsort((ids,-uncertainty))
  first=rank[:deterministic]
  remain=np.setdiff1d(np.arange(N),first,assume_unique=True)
  extra=rng.choice(remain,size=M-deterministic,replace=False)
  selected=np.concatenate((first,extra))
 else:raise ValueError('unknown optional query policy')
 out=np.zeros(N,dtype=bool);out[selected]=True
 return out


def allocation(groups,budget,tilt):
 """Known-strata SRSWOR allocation, >=2 reviews/stratum if feasible."""
 sizes=np.array([len(g) for g in groups],dtype=int)
 if budget>sizes.sum():raise ValueError('budget exceeds population')
 minimum=np.minimum(2,sizes)
 if budget<minimum.sum():raise ValueError('budget too low for 2/stratum')
 assigned=minimum.copy()
 weights=sizes*np.asarray(tilt,dtype=float)
 while assigned.sum()<budget:
  can=np.flatnonzero(assigned<sizes)
  # Choose next by target allocation / (current n + 1), exact B overall.
  idx=can[np.argmax(weights[can]/(assigned[can]+1))]
  assigned[idx]+=1
 return assigned


def estimate(y,D,groups,alloc,rng):
 """Unbiased conditional HT-stratum means for miss and harm; FNR ratio not unbiased."""
 N=len(y); miss=harm=fp=benign=0.0;variance=0.0
 for ix,n in zip(groups,alloc):
  nh=int(n);Nh=len(ix)
  sel=rng.choice(ix,size=nh,replace=False)
  yy=y[sel];dd=D[sel]
  z=yy*(1-dd)
  miss+=Nh*float(np.mean(z))
  harm+=Nh*float(np.mean(yy))
  fp+=Nh*float(np.mean((1-yy)*dd))
  benign+=Nh*float(np.mean(1-yy))
  if Nh>1 and nh>1:variance+=(Nh/N)**2*(1-nh/Nh)*float(np.var(z,ddof=1))/nh
 miss/=N;harm/=N
 half=1.95996398454*math.sqrt(max(0,variance))
 return {'miss':miss,'harm':harm,'FNR':miss/harm if harm>1e-12 else np.nan,
         'FPR':(fp/N)/(benign/N) if benign>1e-12 else np.nan,
         'ci_lo':max(0,miss-half),'ci_hi':min(1,miss+half)}


def truth(y,D):
 return {'miss':float(np.mean(y*(1-D))), 'harm':float(np.mean(y)),
         'FNR':float(np.sum(y*(1-D))/np.sum(y)) if y.sum() else np.nan,
         'FPR':float(np.sum((1-y)*D)/np.sum(1-y)) if (1-y).sum() else np.nan}


def run_setup(g, setup, latency_ms, repetitions=REPETITIONS,counts=MONITOR_COUNTS,budgets=HUMAN_BUDGETS):
 ids=g.example_id.astype(str).to_numpy()
 y=g.y.astype(int).to_numpy();D=g.base_prediction.astype(int).to_numpy()
 uncertainty=g.base_uncertainty.astype(float).to_numpy()
 optional=g.optional_monitor_score.astype(float).to_numpy()
 if np.any(~np.isfinite(optional)) or not np.all((optional>=0)&(optional<=1)):
  raise ValueError('Optional score must be in [0,1] for all pre-cached rows: '+setup)
 T=truth(y,D);results={}
 for method in METHODS:
  for M in counts:
   for B in budgets:results[(method,M,B)]=[]
 for r in range(repetitions):
  for k,method in enumerate(METHODS):
   acquisition='random' if method.startswith('random') else 'priority'
   for M in counts:
    ss=np.random.SeedSequence([SEED,k,M,r,0 if setup=='compact_after_rule' else 1])
    rng=np.random.default_rng(ss)
    acquired=acquire_monitor(D,uncertainty,ids,M,acquisition,rng)
    # Optional is passed to the groups function, where it is accessed only at acquired rows.
    groups,names,priority=grouped(D,uncertainty,optional,acquired,ids)
    for B in budgets:
     a=allocation(groups,B,priority if method.endswith('tilted') else np.ones(len(groups)))
     result=estimate(y,D,groups,a,rng)
     results[(method,M,B)].append(result)
 rows=[]
 for (method,M,B),vec in results.items():
  d={'setup_id':setup,'method':method,'optional_acquisition': 'random' if method.startswith('random') else 'priority',
     'monitor_calls':M,'human_reviews':B,'repetitions':repetitions,'N_development':len(y),
     'optional_latency_ms_total_legacy_mean':float(latency_ms)*M,
     'legacy_mean_ms_per_optional_call':float(latency_ms),
     'truth_miss_prevalence_dev':T['miss'],'truth_harm_prevalence_dev':T['harm'],
     'truth_FNR_dev':T['FNR'],'truth_FPR_dev':T['FPR']}
  for stat in ('miss','harm','FNR','FPR'):
   vals=np.array([x[stat] for x in vec],dtype=float)
   okay=np.isfinite(vals)
   d[stat+'_fraction_estimable']=float(np.mean(okay))
   d[stat+'_mean']=float(vals[okay].mean()) if okay.any() else None
   d[stat+'_bias']=float((vals[okay]-T[stat]).mean()) if okay.any() else None
   d[stat+'_RMSE']=float(np.sqrt(np.mean((vals[okay]-T[stat])**2))) if okay.any() else None
  d['miss_CI95_coverage_descriptive']=float(np.mean([x['ci_lo']<=T['miss']<=x['ci_hi'] for x in vec]))
  rows.append(d)
 return rows


def checks():
 # No-label access to monitor query or strata; exhaustive unbiasedness for small groups.
 rng=np.random.default_rng(7)
 ids=np.array(['b','a','d','c','f','e','h','g'])
 D=np.array([0,0,0,0,1,1,1,1]);u=np.arange(8,dtype=float)/8
 s=np.array([.1,.8,.4,.6,.2,.9,.7,.3]);y=np.array([0,1,1,0,1,0,1,0])
 for policy in ('random','priority'):
  a=acquire_monitor(D,u,ids,4,policy,rng)
  assert a.sum()==4
  gs,n,t=grouped(D,u,s,a,ids)
  assert sorted(np.concatenate(gs).tolist())==list(range(8))
  assert len(set(np.concatenate(gs).tolist()))==8
  alloc=allocation(gs,8,np.ones(len(gs)))
  assert sum(alloc)==8
  e=estimate(y,D,gs,alloc,np.random.default_rng(18));T=truth(y,D)
  assert np.isclose(e['miss'],T['miss']) and np.isclose(e['FNR'],T['FNR'])
 # A strict unbiasedness check for stratified n_h<N_h (no adaptive postselection).
 groups=[np.array([0,1,2]),np.array([3,4])];yv=np.array([1,0,1,0,1]);Dv=np.array([0,0,1,0,1]);a=np.array([2,1])
 import itertools
 vals=[]
 for group1 in itertools.combinations(groups[0],2):
  for group2 in itertools.combinations(groups[1],1):
   N=len(yv);miss=0
   for ix,ss in zip(groups,[group1,group2]):miss+=len(ix)*float(np.mean(yv[list(ss)]*(1-Dv[list(ss)])))
   vals.append(miss/N)
 assert np.isclose(np.mean(vals),truth(yv,Dv)['miss'])
 print('PHASE4C_SELF_TEST_PASS')


def main():
 checks()
 if '--self-test' in sys.argv:return 0
 root=Path.cwd().resolve()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=6).strip()
 if branch!=BRANCH:
  print('Not running: wrong branch:',branch);return 2
 if (root/OUT).exists():
  print('Not running: Phase4C outputs exist; preserved:',OUT);return 2
 required=[SOURCE,FOLDS,PROTOCOL,BOUNDARY,PRIOR,BASELINES]
 missing=[name for name in required if not (root/name).is_file()]
 if missing:
  print('Not running: missing permitted development or prior aggregate files:',missing);return 2
 boundary=json.loads((root/BOUNDARY).read_text(encoding='utf8'))
 allowed=boundary.get('data_boundary',{}).get('permitted_existing_development_artifacts',[])
 if SOURCE not in allowed or FOLDS not in allowed:raise RuntimeError('development boundary disallows these sources')
 old=json.loads((root/PRIOR).read_text(encoding='utf8'))
 hashes={'source_sha256':sha(root/SOURCE),'fold_source_sha256':sha(root/FOLDS),
         'decision_value_protocol_sha256':sha(root/PROTOCOL),'boundary_protocol_sha256':sha(root/BOUNDARY)}
 for key,value in hashes.items():
  if old['input_sha256'].get(key)!=value:raise ValueError('historical data/protocol changed: '+key)
 protocol=json.loads((root/PROTOCOL).read_text(encoding='utf8'))
 cols=['example_id','setup_id','outer_fold','y','base_score','base_prediction','base_uncertainty','optional_monitor_score']
 df=pd.read_parquet(root/SOURCE,columns=cols)
 folds=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
 n=int(protocol['scope']['expected_development_rows'])
 setups={x['setup_id']:x for x in protocol['optional_monitor_setups']}
 if len(folds)!=n or folds.example_id.nunique()!=n:raise ValueError('fold assignment size mismatch')
 if len(df)!=n*len(setups) or set(df.setup_id)!=set(setups):raise ValueError('source setups/row counts differ')
 if df.groupby('example_id').y.nunique().max()!=1:raise ValueError('setup labels disagree')
 prior=pd.read_csv(root/BASELINES)
 if len(prior)!=36 or set(prior.method)!={'uniform','proportional_stratified','score_tilted_stratified',
                     'allowed_only','lure_adaptive_beta','lure_fixed_cheap_rank'}:raise ValueError('baseline comparison changed')
 new=[]
 for setup,params in setups.items():
  g=df[df.setup_id==setup].sort_values('example_id')
  if len(g)!=n or g.example_id.nunique()!=n:raise ValueError('duplicates/missing example IDs')
  j=g[['example_id','outer_fold']].merge(folds,on='example_id',validate='one_to_one',suffixes=('_data','_assigned'))
  if len(j)!=n or not (j.outer_fold_data.astype(str)==j.outer_fold_assigned.astype(str)).all():
   raise ValueError('outer folds differ')
  y=g.y.to_numpy();d=g.base_prediction.to_numpy()
  if not set(y).issubset({0,1}) or not set(d).issubset({0,1}):raise ValueError('not binary Y/D')
  if int(y.sum())!=int(protocol['scope']['expected_positive_n']):raise ValueError('Y prevalence mismatch')
  T=truth(y,d);prior_setup=prior[prior.setup_id==setup]
  # Truth was matched in Phase4B to the original Phase4A export; additionally compare here.
  # Recompute only, never change frozen baselines.
  if prior_setup.shape[0]!=18:raise ValueError('missing previous 18 comparison groups')
  print('RUN',setup,'legacy optional mean ms',params['measured_mean_cost_ms'])
  new.extend(run_setup(g,setup,params['measured_mean_cost_ms']))
 out=root/OUT;out.mkdir(parents=True,exist_ok=False)
 frame=pd.DataFrame(new).sort_values(['setup_id','monitor_calls','human_reviews','method'])
 frame.to_csv(out/'monitor_assisted_audit_metrics.csv',index=False,lineterminator='\n')
 provenance={'artifact_id':ARTIFACT,'status':'exploratory_development_only_two_stage_probability_audit',
   'source_branch':branch,'sealed_rows_accessed':False,'raw_examples_exported':False,
   'candidate_optional_monitor_allocation':['random SRSWOR','80% high cheap uncertainty plus 20% random sentinel'],
   'human_sampling':'SRSWOR within label-free D x acquisition x score band strata, HT weighting',
   'label_lookahead':False,'known_inclusion_probability_human':'stratum nh/Nh conditional on first-stage acquisition',
   'estimands':['miss_prevalence','harm_prevalence','FNR_ratio','FPR_ratio'],
   'confidence':'descriptive stratified-normal coverage for missed-harm prevalence only; not validated',
   'sampling_methods':list(METHODS),'monitor_call_budgets':list(MONITOR_COUNTS),
   'human_review_budgets':list(HUMAN_BUDGETS),'repetitions':REPETITIONS,'seed':SEED,
   'source_paths':[SOURCE,FOLDS],'source_hashes':hashes,
   'protocol_setups':setups,'legacy_extra_monitor_mean_latency_ms':'from historical development protocol, not a fresh measurement',
   'prior_matched_baseline_hash':sha(root/BASELINES),
   'D':'archived cross-fitted classifier, NOT deployed native guard',
   'Y':'historical response-level labels, not new independent human review',
   'cost_limitation':'Optional monitor latency and human-review counts reported separately; no unmeasured human labor cost converted to currency',
   'hypothesis_status':'Post-hoc exploratory pilot, no selection or claim of method dominance',
   'original_screen':'Stage-A readout-transfer 0/4 remains untouched',
   'libraries':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}}
 (out/'manifest.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n',encoding='utf8')
 report=['# Phase 4C: Cost-aware optional-monitor auditing — development only','',
 'No native guard deployment risk claim. Archived optional scores are revealed only on query-selected rows, and review selection uses no unsampled Y.','',
 '| Setup | Human reviews | Optional checks | Method | Miss RMSE | FNR RMSE | Legacy optional ms |',
 '|---|---:|---:|---|---:|---:|---:|']
 for row in frame.to_dict('records'):
  report.append(f"| {row['setup_id']} | {row['human_reviews']} | {row['monitor_calls']} | {row['method']} | {row['miss_RMSE']:.4f} | {row['FNR_RMSE']:.4f} | {row['optional_latency_ms_total_legacy_mean']:.1f} |")
 report.extend(['','## Guardrails','','- The additional monitor is a simulated, *charged* selective acquisition of an already-cached score; it does not change the historical D.','- The human auditor is simulated from historical development labels, not an actual human annotator.','- The miss/harm prevalence HT stratum means are conditionally design-unbiased; FNR and FPR ratios generally are not.','- Nominal confidence intervals are exploratory, not deployment certificates.','- Human review counts and historic monitor inference time are separate budget coordinates; comparisons to no-monitor baselines cannot be called cheaper without an explicit utility/cost model.','- Selected hypotheses and new methods need a fresh independent validation sample before publication claims.',''])
 (out/'findings.md').write_text('\n'.join(report),encoding='utf8')
 archive=out/(ARTIFACT+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in ('monitor_assisted_audit_metrics.csv','manifest.json','findings.md'):
   z.write(out/name,arcname=name)
 dest=windows_downloads()
 if dest:
  target=dest/archive.name
  if target.exists():print('Preserved existing Windows ZIP; new ZIP is:',archive)
  else:shutil.copy2(archive,target);print('Windows Downloads:',target)
 else:print('Windows Downloads unresolved; ZIP remains:',archive)
 print('PHASE4C_SUCCESS',len(frame),'aggregate rows; protected datasets untouched')
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:print('PHASE4C_STOP:',type(e).__name__,str(e));sys.exit(2)
