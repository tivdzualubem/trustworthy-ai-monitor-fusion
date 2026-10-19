#!/usr/bin/env python3
"""Phase 5B3: restricted, development-only AM-PPI Algorithm 1/2 adaptation.

Preserves the frozen response-level development data and the archived classifier D.
The two candidate predictor subsets are [1,cheap] and [1,cheap,optional].
Both response-decision groups use separate, randomized burn-in calibration data;
one Bernoulli active-auditing design follows each calibration.  This is NOT a
full theorem/CI reproduction of Brawand et al., 2026, nor a prospective result.
The cost sensitivity is hypothetical: NOT measured review-time economics.
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
INPUT_HASHES={
 SOURCE:'0041158b10e7ee0dd1189a0611c5b07f18a44d88355e3c150ac044e4112fe6ec',
 FOLDS:'62a2d8df0ba55b047d7e4685488b4d4d0ef77d480bcdb26cc34e5586e935d2dd',
 PROTOCOL:'7c82846f9de712a56260d74fbc1901b084262f2a7a745007547a75eab3d5f914',
 BOUNDARY:'96a0bfdf1a0954d9313ecd7a2ae1272a07f0df4ba6197eede2f0e0afd9f1c1c7',
}
OUT='results/phase5b3_amppi_restricted_v1'
ARTIFACT='phase5b3_amppi_restricted_v1'
SEED=20261014
HUMAN_BUDGETS=(80,160)
OPTIONAL_BUDGET_HINTS=(80,320)
# Human review duration is unknown, so 30/120 seconds are sensitivity assumptions,
# never measurements.  Historical mean optional-call times likewise are not fresh.
ASSUMED_REVIEW_SECONDS=(30.,120.)
REPETITIONS=100
PI_FLOOR=0.007
METHODS=('blocked_allowed_srs','cheap_only_active_ppi','restricted_amppi_2predictor')
PAPER='https://arxiv.org/abs/2605.08429'


def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()


def downloads():
 try:
  s=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
   '[Environment]::GetFolderPath("UserProfile")'],text=True,timeout=8).strip().replace('\r','')
  p=Path(subprocess.check_output(['wslpath','-u',s],text=True,timeout=5).strip())/'Downloads'
  if p.is_dir():return p
 except (OSError,ValueError,subprocess.SubprocessError):pass
 p=Path('/mnt/c/Users/user/Downloads')
 return p if p.is_dir() else None


def linfit(X,y,weight=None):
 """Ridge-stabilized WLS; coefficients depend only on calibration labels."""
 X=np.asarray(X,float); y=np.asarray(y,float)
 weight=np.ones(len(y)) if weight is None else np.asarray(weight,float)
 weight=np.maximum(0.,weight)
 A=X.T@(X*weight[:,None]);rhs=X.T@(y*weight)
 reg=np.eye(X.shape[1])*0.02
 reg[0,0]=0.001
 return np.linalg.solve(A+reg,rhs)


def design(cheap,optional=None):
 f=np.asarray(cheap,float)
 return np.column_stack([np.ones(len(f)),f]) if optional is None else np.column_stack([np.ones(len(f)),f,np.asarray(optional,float)])


def calibrate(y,f,o,unc,pilot,allow_optional):
 yp=y[pilot];fp=f[pilot];up=unc[pilot]
 feats=np.column_stack([np.ones(len(pilot)),fp,up])
 out=[]
 for use_opt in range(2 if allow_optional else 1):
  X=design(fp,o[pilot] if use_opt else None)
  beta=linfit(X,yp)
  residual=np.abs(yp-X@beta)
  u_beta=linfit(feats,residual)
  out.append((beta,u_beta))
 return out


def residual_uncertainty(cheap,unc,u_beta):
 x=np.column_stack([np.ones(len(cheap)),cheap,unc]);return np.clip(x@u_beta,.05,1.)


def route_policy(cheap,unc,models,extra_cost,label_cost,total_budget,optional_enabled):
 """Paper Eq. (5), (8), (9); one shared dual multiplier for all records.

 No optional outcome score is used for routing: uncertainty regressors depend
 only on CHEAP pre-query features.  Budget is in assumed seconds.
 """
 n=len(cheap)
 if n<1 or total_budget<=0:raise ValueError('bad routing population/budget')
 u=np.array([residual_uncertainty(cheap,unc,t[1]) for t in models])
 costs=np.array([0.,extra_cost] if optional_enabled else [0.])
 def compute(mu):
  p=np.clip(u/np.sqrt(n*mu*label_cost),PI_FLOOR,1.)
  loss=u*u/n*(1/p-1)+mu*(costs[:,None]+p*label_cost)
  chosen=np.argmin(loss,axis=0)
  selpi=p[chosen,np.arange(n)]
  cost=float(np.sum(costs[chosen]+label_cost*selpi))
  return chosen,selpi,cost
 lo,hi=1e-15,1e12
 if compute(lo)[2]<total_budget:
  return compute(lo)
 if compute(hi)[2]>total_budget:
  return compute(hi)
 for _ in range(55):
  mid=math.sqrt(lo*hi)
  if compute(mid)[2]>total_budget:lo=mid
  else:hi=mid
 return compute(hi)


def one_group(y,f,o,unc,whole,pilot,arm,budget,optional_sec,label_sec,rng):
 """Conditional AIPW estimate for the portion not in pilot.

 The sampled pilot labels are included EXACTLY in the total rather than
 discarded. Only the sampled Bernoulli evaluation labels are accessed.
 """
 rem=np.setdiff1d(whole,pilot,assume_unique=True)
 if len(rem)==0:raise ValueError('no evaluation population')
 allow_opt=arm=='restricted_amppi_2predictor'
 models=calibrate(y,f,o,unc,pilot,allow_opt)
 # A tiny burn-in permits a first-pass model, not the paper's CI theorem.
 chosen,pi,expected=route_policy(f[rem],unc[rem],models,optional_sec,label_sec,budget,allow_opt)
 selected=rng.random(len(rem))<pi
 if allow_opt:
  # Crucially only consult optional scores on actively routed records.
  preds=design(f[rem])@models[0][0]
  pay=np.flatnonzero(chosen==1)
  if len(pay):preds[pay]=design(f[rem][pay],o[rem][pay])@models[1][0]
 else:preds=design(f[rem])@models[0][0]
 # y is accessed for reviewed rows ONLY in the estimator; elsewhere only in
 # evaluation truth after all sampling has completed, in one_setup().
 correction=np.zeros(len(rem))
 if np.any(selected): correction[selected]=(y[rem[selected]]-preds[selected])/pi[selected]
 total=float(np.sum(y[pilot])+np.sum(preds+correction))
 purchased=int(np.sum(chosen==1)) if allow_opt else 0
 labels=len(pilot)+int(np.sum(selected))
 realized=labels*label_sec+(purchased+len(pilot)*(1 if allow_opt else 0))*optional_sec
 return total,{'actual_labels':labels,'calibration_labels':len(pilot),'new_labels':int(np.sum(selected)),
   'actual_optional_calls':purchased+len(pilot)*(1 if allow_opt else 0),
   'expected_deployment_cost_seconds':expected,
   'actual_total_cost_seconds':realized,
   'min_selection_probability':float(pi.min()),
   'max_selection_probability':float(pi.max()),
   'routed_optional_deployment':purchased}


def one_setup(df,setup,optional_sec,reps=REPETITIONS,budgets=HUMAN_BUDGETS,
              hints=OPTIONAL_BUDGET_HINTS,review_seconds=ASSUMED_REVIEW_SECONDS):
 g=df.sort_values('example_id')
 ids=g.example_id.astype(str).to_numpy()
 y=g.y.to_numpy(dtype=int);d=g.base_prediction.to_numpy(dtype=int)
 f=g.base_score.to_numpy(dtype=float);o=g.optional_monitor_score.to_numpy(dtype=float)
 uncertainty=g.base_uncertainty.to_numpy(dtype=float)
 n=len(y)
 if len(set(ids))!=n or n!=1687 or sum(y)!=291 or not np.isfinite(np.column_stack([f,o,uncertainty])).all():
  raise ValueError('bad development input')
 if not set(y).issubset({0,1}) or not set(d).issubset({0,1}):raise ValueError('not binary')
 groups=[np.flatnonzero(d==0),np.flatnonzero(d==1)]
 truth_a=float(np.mean(y*(1-d))); truth_b=float(np.mean(y*d));truth_f=truth_a/(truth_a+truth_b)
 result=[]
 setupcode=0 if setup=='compact_after_rule' else 1
 for reviews in budgets:
  # 20% predeclared calibration labels, equally divided between D groups.
  cal=round(.2*reviews);cal0=cal//2;cal1=cal-cal0
  for hint in hints:
   for label_sec in review_seconds:
    total_budget=reviews*label_sec+hint*optional_sec
    # 70/30 budget division is fixed BEFORE simulation; not optimized on Y.
    remaining_by_group=np.array([.7,.3])*total_budget
    for rep in range(reps):
     rng=np.random.default_rng(np.random.SeedSequence([SEED,setupcode,reviews,hint,int(label_sec),rep]))
     pilot=[rng.choice(groups[0],cal0,replace=False),rng.choice(groups[1],cal1,replace=False)]
     for arm in METHODS:
      if arm=='blocked_allowed_srs':
       # Spend all available budget on human reviews; no labeled tuning stage.
       count=min(int(total_budget//label_sec),n)
       n0=int(round(.7*count));n0=max(2,min(n0,len(groups[0])));n1=count-n0
       if n1<2 or n1>len(groups[1]):raise ValueError('bad SRS allocation')
       s0=rng.choice(groups[0],n0,replace=False);s1=rng.choice(groups[1],n1,replace=False)
       total0=len(groups[0])*float(np.mean(y[s0]));total1=len(groups[1])*float(np.mean(y[s1]))
       costs={'actual_labels':count,'calibration_labels':0,'new_labels':count,
        'actual_optional_calls':0,'actual_total_cost_seconds':count*label_sec,
        'expected_deployment_cost_seconds':count*label_sec,'min_selection_probability':float('nan'),
        'max_selection_probability':float('nan'),'routed_optional_deployment':0}
      else:
       totals=[];part=[]
       for k in range(2):
        cal_cost=len(pilot[k])*(label_sec+(optional_sec if arm=='restricted_amppi_2predictor' else 0))
        budget_remaining=float(remaining_by_group[k]-cal_cost)
        if budget_remaining<=0:raise ValueError('calibration cost exceeds group budget')
        t,c=one_group(y,f,o,uncertainty,groups[k],pilot[k],arm,budget_remaining,
                      optional_sec,label_sec,rng)
        totals.append(t);part.append(c)
       total0,total1=totals
       costs={'actual_labels':sum(c['actual_labels'] for c in part),
        'calibration_labels':cal,'new_labels':sum(c['new_labels'] for c in part),
        'actual_optional_calls':sum(c['actual_optional_calls'] for c in part),
        'actual_total_cost_seconds':sum(c['actual_total_cost_seconds'] for c in part),
        'expected_deployment_cost_seconds':sum(c['expected_deployment_cost_seconds'] for c in part),
        'min_selection_probability':min(c['min_selection_probability'] for c in part),
        'max_selection_probability':max(c['max_selection_probability'] for c in part),
        'routed_optional_deployment':sum(c['routed_optional_deployment'] for c in part)}
      A=total0/n;B=total1/n;F=A/(A+B) if A+B>1e-12 else float('nan')
      row={'setup_id':setup,'human_budget_reference':reviews,'optional_call_budget_hint':hint,
       'hypothetical_review_seconds':label_sec,'repetition':rep,'method':arm,
       'total_budget_seconds_hypothetical':total_budget,
       'A_error':A-truth_a,'B_error':B-truth_b,
       'FNR_error':F-truth_f if math.isfinite(F) else float('nan'),
       'FNR_out_of_range':bool(math.isfinite(F) and (F<0 or F>1)),
       'FNR_estimable':math.isfinite(F)}
      row.update(costs);row['total_cost_over_budget']=row['actual_total_cost_seconds']>total_budget+1e-7
      result.append(row)
 return result,{'N':n,'A':truth_a,'B':truth_b,'FNR':truth_f,
                'allowed_population':len(groups[0]),'blocked_population':len(groups[1]),'positives':int(y.sum())}


def summarize(df):
 out=[]
 keys=['setup_id','human_budget_reference','optional_call_budget_hint','hypothetical_review_seconds','method']
 for group,g in df.groupby(keys,sort=True):
  row=dict(zip(keys,group));row['repetitions']=len(g)
  for m in ('A','B','FNR'):
   e=g[m+'_error'].to_numpy(float);good=np.isfinite(e)
   row[m+'_estimable_fraction']=float(good.mean())
   row[m+'_RMSE']=float(np.sqrt(np.mean(e[good]**2))) if good.any() else None
   row[m+'_bias']=float(e[good].mean()) if good.any() else None
  for k in ('actual_labels','calibration_labels','actual_optional_calls','actual_total_cost_seconds',
            'total_cost_over_budget','FNR_out_of_range'):
   row[k+'_mean']=float(g[k].mean())
  out.append(row)
 return pd.DataFrame(out)


def self_test():
 rng=np.random.default_rng(5)
 n=1687
 f=rng.uniform(.1,.8,n);o=np.clip(f+.10*rng.standard_normal(n),0,1)
 u=np.abs(f-.45);y=np.zeros(n,int);y[rng.choice(n,291,replace=False)]=1
 d=np.zeros(n,int);d[:130]=1
 tmp=pd.DataFrame({'example_id':[f'zz{i:05d}' for i in range(n)],'y':y,'base_prediction':d,
                   'base_score':f,'base_uncertainty':u,'optional_monitor_score':o})
 for mode in ('cheap_only_active_ppi','restricted_amppi_2predictor'):
  j=np.flatnonzero(d==0);pilot=j[:10];remain=j[10:]
  c=calibrate(y,f,o,u,pilot,mode=='restricted_amppi_2predictor')
  group,pi,spent=route_policy(f[remain],u[remain],c,1.5,60.,2000.,mode=='restricted_amppi_2predictor')
  assert len(group)==len(remain) and np.all((pi>=PI_FLOOR)&(pi<=1)) and spent>0
  # AIPW conditional unbiasedness for fixed y and f, using expectations of Bernoulli xi.
  truth=np.sum(y[j]);base=np.sum(y[pilot]);pred=design(f[remain])@c[0][0]
  if mode=='restricted_amppi_2predictor':
   ix=np.flatnonzero(group==1)
   if len(ix):pred[ix]=design(f[remain][ix],o[remain][ix])@c[1][0]
  expected=base+np.sum(pred+(y[remain]-pred))
  assert abs(expected-truth)<1e-9
 # Two synthetic repetitions; output never contains case-level rows.
 rows,truth=one_setup(tmp,'compact_after_rule',1.5,reps=2,budgets=(80,),hints=(80,),review_seconds=(60.,))
 assert len(rows)==2*3 and len(summarize(pd.DataFrame(rows)))==3
 assert all(t['actual_labels']==t['calibration_labels']+t['new_labels'] for t in rows)
 print('PHASE5B3_SELF_TEST_PASS',len(rows),'scalar-only rows')


def main():
 self_test()
 if '--self-test' in sys.argv:return 0
 root=Path.cwd().resolve()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=8).strip()
 if branch!=BRANCH:print('STOP: unexpected branch',branch);return 2
 out=root/OUT
 if out.exists():print('STOP: output exists; preserved',out);return 2
 for f,h in INPUT_HASHES.items():
  p=root/f
  if not p.is_file() or sha256(p)!=h:raise ValueError('missing/changed permitted input '+f)
 frozen=json.loads((root/BOUNDARY).read_text(encoding='utf8'))
 permitted=set(frozen['data_boundary']['permitted_existing_development_artifacts'])
 if SOURCE not in permitted or FOLDS not in permitted:raise ValueError('development whitelist rejects sources')
 protocol=json.loads((root/PROTOCOL).read_text(encoding='utf8'))
 expected={s['setup_id']:s['measured_mean_cost_ms']/1000 for s in protocol['optional_monitor_setups']}
 cols=['example_id','setup_id','outer_fold','y','base_prediction','base_score',
       'base_uncertainty','optional_monitor_score']
 df=pd.read_parquet(root/SOURCE,columns=cols)
 folds=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
 if len(folds)!=1687 or folds.example_id.nunique()!=1687:raise ValueError('fold file mismatch')
 if len(df)!=3374 or set(df.setup_id)!=set(expected):raise ValueError('data setup mismatch')
 if df.groupby('example_id').y.nunique().max()!=1:raise ValueError('cross-setup response labels disagree')
 rows=[];truths={}
 for setup,c in expected.items():
  group=df[df.setup_id==setup].copy()
  if len(group)!=1687:raise ValueError('setup length mismatch')
  join=group[['example_id','outer_fold']].merge(folds,on='example_id',validate='one_to_one',suffixes=('_data','_frozen'))
  if len(join)!=1687 or not (join.outer_fold_data.astype(str)==join.outer_fold_frozen.astype(str)).all():
   raise ValueError('frozen fold integrity failure')
  print('RUN',setup,'repetitions',REPETITIONS,flush=True)
  r,t=one_setup(group,setup,c)
  rows.extend(r);truths[setup]=t
 frame=pd.DataFrame(rows);summary=summarize(frame)
 out.mkdir(parents=True,exist_ok=False)
 frame.to_csv(out/'repetition_scalar_errors_and_costs.csv',index=False,float_format='%.12g',lineterminator='\n')
 summary.to_csv(out/'comparison.csv',index=False,float_format='%.12g',lineterminator='\n')
 manifest={'artifact_id':ARTIFACT,'status':'exploratory_restricted_amppi_algorithm_adaptation',
  'paper':PAPER,'repetitions':REPETITIONS,'methods':list(METHODS),
  'source_sha256':INPUT_HASHES,'development_truth':truths,
  'human_budgets':list(HUMAN_BUDGETS),'optional_budget_hints':list(OPTIONAL_BUDGET_HINTS),
  'assumed_review_seconds':list(ASSUMED_REVIEW_SECONDS),
  'cost_validity':'Human review seconds are HYPOTHETICAL; inference seconds frozen historical means. Results are not full-cost validation.',
  'budget_semantics':'Single combined expected budget. Bernoulli review counts can exceed expected cost on a realization; over-budget rate recorded.',
  'calibration':'20% of reference human labels, charged; expensive predictor score on every calibration item also charged',
  'estimator':'two-subset AM-PPI paper Eq1/5/9 structure; tiny calibration + cheap-feature absolute residual regression; not full author benchmark, CI or guarantees',
  'primary':'A=P(Y=1,D=0)','secondary':'FNR=A/(A+B)',
  'known_limitation':'D is historical cross-fitted classifier; Y historical response labels; no independent review or native deployment',
  'no_protected_rows_read':True,'no_new_model_inference':True,'original_frozen_screen_unchanged':'0/4',
  'export':'aggregate and per-repetition scalar errors/costs only; no raw example records'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf8')
 lines=['# Phase 5B3 - restricted AM-PPI feasibility adaptation','',
  'Reference: '+PAPER,'',
  'This is NOT a full author-code replication and does NOT establish the AM-PPI paper guarantees.','',
  'Uses paper AIPW Eq (1), model-dependent Bernoulli review probabilities Eq (5), two-subset selection Eq (9), and burn-in calibration.','Uncertainty is fit from tiny historical calibration samples using pre-review cheap features. Group budgets are fixed 70% allowed/30% blocked, not globally optimal.','',
  '## Key restrictions','',
  '- Hypothetical review seconds 30 and 120 are sensitivity values, NOT actual reviewer time.','- Optional monitor cost uses historical mean seconds. No new Qwen calls occurred.','- Fixed historical classifier D is not deployed native monitor.','- All human reviews are simulated on historical response labels; no independently reviewed responses.','- Calibration labels and expensive calibration monitor scores are fully charged.','- Single total expected budget; realized stochastic label spend can exceed it (recorded as violation).','- A and B are linear population components, FNR is an unstable ratio and may leave [0,1]. Raw errors are kept.','- Comparison uses no sealed final-test or shift data.','- Do not interpret any improvement as confirmatory evidence or deployable cost savings.','',
  'Interpret the CSV and over-budget rates BEFORE choosing any follow-up.']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 archive=out/(ARTIFACT+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in ('comparison.csv','repetition_scalar_errors_and_costs.csv','manifest.json','README.md'):
   z.write(out/f,arcname=f)
 dest=downloads()
 if dest and not (dest/archive.name).exists():
  shutil.copy2(archive,dest/archive.name);print('WINDOWS_DOWNLOADS:',dest/archive.name)
 else:print('Archive in repository:',archive)
 print('PHASE5B3_SUCCESS',len(frame),'records',len(summary),'aggregate rows')
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:
  print('PHASE5B3_STOP',type(exc).__name__,str(exc));sys.exit(2)