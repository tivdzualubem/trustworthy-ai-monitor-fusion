#!/usr/bin/env python3
"""Post hoc Stage-A Phase 3: frozen-pair lexical baselines and source-probe cutoff check."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sklearn

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results/stage_a_readout_transfer_kill_v1'
DEST = ROOT / 'results/stage_a_phase3_shortcut_threshold_v1'
EXPECTED_PROMPT_SHA = 'e9d4c6047c19cc75c1d3b1badd6abc0fa6cf3c234507efd96353597dfd044a8b'
EXPECTED_SCREEN_SHA = 'c163e061ce45efbec211e9d0ee999bd965b6c545da5e56f04d95b63f91b4a1ed'
GUARDS = ['llama_guard_3_1b', 'shieldgemma_2b', 'granite_guardian_3_3_8b', 'qwen3guard_gen_4b']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checks(df):
    assert len(df) == 152 and df['experiment_example_id'].is_unique
    assert set(df['representation']) == {'direct', 'O2'} and set(df['fold_id']) == set(range(5))
    assert df['pair_id'].nunique() == 38 and df['semantic_case_id'].nunique() == 76
    assert all(g['fold_id'].nunique() == 1 and len(g) == 4 for _,g in df.groupby('pair_id'))
    assert all(g['representation'].nunique() == 2 and len(g) == 2 for _,g in df.groupby('semantic_case_id'))
    assert all(len(df[(df.fold_id==f)&(df.representation=='direct')&(df.label_harmful==v)]) in (7,8) for f in range(5) for v in (0,1))

def clf():
    return LogisticRegression(C=1.0, solver='liblinear', max_iter=5000)

def threshold(y, s):
    y = np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    assert len(y) and set(y)=={0,1} and np.isfinite(s).all()
    vals=np.unique(s)
    candidates=[np.nextafter(vals.min(),-np.inf)]
    candidates += ((vals[:-1]+vals[1:])/2).tolist()
    candidates += [np.nextafter(vals.max(),np.inf)]
    return float(min(candidates,key=lambda t:(-balanced_accuracy_score(y,(s>=t).astype(int)),abs(t-0.5),t)))

def metrics(df):
    out=[]
    for (method, guard, rep),g in df.groupby(['method','guard','representation'],sort=True):
        y=g.y.to_numpy(int);p=g.prediction.to_numpy(int);s=g.score.to_numpy(float)
        out.append(dict(method=method,guard=guard,representation=rep,n=len(g),n_positive=int(y.sum()),n_negative=int((1-y).sum()),balanced_accuracy=float(balanced_accuracy_score(y,p)),auroc=float(roc_auc_score(y,s)),harmful_proxy_recall=float(p[y==1].mean()),benign_proxy_specificity=float((1-p[y==0]).mean()),tp=int(((y==1)&(p==1)).sum()),fn=int(((y==1)&(p==0)).sum()),fp=int(((y==0)&(p==1)).sum()),tn=int(((y==0)&(p==0)).sum())))
    return pd.DataFrame(out)

def rows_for(df, idx, method, guard, rep, scores, cut):
    ix=np.flatnonzero(idx);s=np.asarray(scores,dtype=float)
    assert len(ix)==len(s)
    return [dict(method=method,guard=guard,representation=rep,fold_id=int(df.iloc[i].fold_id),pair_id=df.iloc[i].pair_id,semantic_case_id=df.iloc[i].semantic_case_id,experiment_example_id=df.iloc[i].experiment_example_id,y=int(df.iloc[i].label_harmful),score=float(v),prediction=int(v>=cut),threshold=float(cut)) for i,v in zip(ix,s)]

def load_hidden(path, df):
    with np.load(path,allow_pickle=False) as z:
        ids=z['experiment_example_id'].astype(str)
        hidden=z['hidden'].astype('float32')
    assert len(ids)==152 and len(set(ids))==152 and hidden.shape[0]==152 and np.isfinite(hidden).all()
    by_id={v:i for i,v in enumerate(ids)}
    assert set(by_id)==set(df.experiment_example_id)
    return hidden[[by_id[v] for v in df.experiment_example_id]]

def run(root):
    global ROOT, SOURCE, DEST
    ROOT=root;SOURCE=root/'results/stage_a_readout_transfer_kill_v1';DEST=root/'results/stage_a_phase3_shortcut_threshold_v1'
    p=SOURCE/'readout_probe_prompts.csv';screen=SOURCE/'analysis/screening_decision.csv'
    assert digest(p)==EXPECTED_PROMPT_SHA, 'Frozen prompt input differs; stopping'
    assert digest(screen)==EXPECTED_SCREEN_SHA, 'Frozen screen differs; stopping'
    assert not DEST.exists(),f'New result directory already exists: {DEST}. Refusing overwrite.'
    df=pd.read_csv(p,keep_default_na=False);checks(df)
    rep=df.representation.to_numpy();fold=df.fold_id.to_numpy();y=df.label_harmful.to_numpy(int)
    preds=[]; thresholds=[]; equivalences=[]
    # Fixed configuration, no hyperparameter selection based on held-out metrics.
    specs=[('word_tfidf','word',(1,2)),('char_tfidf','char',(2,5))]
    for name,analyzer,ngrams in specs:
        for source_rep,target_rep in [('direct','direct'),('direct','O2'),('O2','O2')]:
            method=f'{name}__train_{source_rep}'
            for f in range(5):
                tr=(fold!=f)&(rep==source_rep);te=(fold==f)&(rep==target_rep)
                v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngrams,lowercase=True)
                ztr=v.fit_transform(df.loc[tr,'prompt_text']);model=clf().fit(ztr,y[tr])
                sc=model.predict_proba(v.transform(df.loc[te,'prompt_text']))[:,1]
                preds+=rows_for(df,te,method,'text_only',target_rep,sc,0.5)
    for guard in GUARDS:
        h=load_hidden(SOURCE/'features'/f'{guard}_probe_features.npz',df)
        for f in range(5):
            train_d=(fold!=f)&(rep=='direct')
            train_o=(fold!=f)&(rep=='O2')
            test_d=(fold==f)&(rep=='direct')
            test_o=(fold==f)&(rep=='O2')
            scaler=StandardScaler().fit(h[train_d]);model=clf().fit(scaler.transform(h[train_d]),y[train_d])
            train_scores=model.predict_proba(scaler.transform(h[train_o]))[:,1]
            cut=threshold(y[train_o],train_scores)
            train_d_scores=model.predict_proba(scaler.transform(h[train_d]))[:,1]
            source_cut=threshold(y[train_d],train_d_scores)
            test_scores=model.predict_proba(scaler.transform(h[test_o]))[:,1]
            preds+=rows_for(df,test_o,'source_probe_default',guard,'O2',test_scores,0.5)
            preds+=rows_for(df,test_o,'source_probe_O2_calibrated',guard,'O2',test_scores,cut)
            preds+=rows_for(df,test_o,'source_probe_direct_calibrated',guard,'O2',test_scores,source_cut)
            # Control: direct-source reproduction for original OOF values.
            direct_test_sc=model.predict_proba(scaler.transform(h[test_d]))[:,1]
            preds+=rows_for(df,test_d,'source_probe_default',guard,'direct',direct_test_sc,0.5)
            thresholds.append(dict(guard=guard,fold_id=f,threshold_O2_train=cut,threshold_direct_train=source_cut,training_O2_n=int(train_o.sum()),training_direct_n=int(train_d.sum()),heldout_O2_n=int(test_o.sum())))
            # Mathematical equivalence of paired mean shift to intercept adjustment.
            left=df.loc[train_d,['semantic_case_id']].copy();left['ix']=np.flatnonzero(train_d)
            right=df.loc[train_o,['semantic_case_id']].copy();right['ix']=np.flatnonzero(train_o)
            paired=left.merge(right,on='semantic_case_id',suffixes=('_d','_o'),validate='one_to_one')
            delta=(h[paired.ix_d.to_numpy()]-h[paired.ix_o.to_numpy()]).mean(axis=0)
            direct_logit=model.decision_function(scaler.transform(h[test_o]))
            shifted_logit=model.decision_function(scaler.transform(h[test_o]+delta))
            standardized_delta=delta/scaler.scale_
            offset=float(np.dot(model.coef_[0],standardized_delta))
            gap=float(np.max(np.abs(shifted_logit-(direct_logit+offset))))
            if gap>1e-3: raise AssertionError(f'Mean-shift mathematical check failed: {guard}/{f}: {gap}')
            equivalences.append(dict(guard=guard,fold_id=f,linear_logit_offset=offset,max_abs_discrepancy=gap))
    out=pd.DataFrame(preds);agg=metrics(out)
    # Guard reproduction against frozen archived per-example predictions (no label tuning).
    original=pd.read_csv(SOURCE/'analysis/oof_predictions.csv')
    check=original[(original.method=='source_probe')&(original.representation.isin(['direct','O2']))]
    new=out[(out.method=='source_probe_default')]
    matched=check.merge(new,on=['guard','fold_id','experiment_example_id','semantic_case_id','representation'],validate='one_to_one')
    assert len(matched)==608
    worst=float(np.max(np.abs(matched.score_x-matched.score_y)))
    # scikit-learn versions may differ; require close numerical reproducibility, not exact float identity.
    if worst>1e-3: raise AssertionError(f'Source probe reproduction mismatch: {worst}')
    archive_metrics=pd.read_csv(SOURCE/'analysis/probe_metrics.csv')
    historical=archive_metrics[(archive_metrics.method.isin(['native_default','native_target_threshold','target_probe','paired_mean_shift_repair','source_probe'])) & (archive_metrics.representation=='O2')].copy()
    historical.insert(0,'provenance','frozen_original_oct07')
    DEST.mkdir(parents=True)
    out.to_csv(DEST/'oof_predictions.csv',index=False)
    agg.to_csv(DEST/'new_metrics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(DEST/'source_probe_cutoffs.csv',index=False)
    pd.DataFrame(equivalences).to_csv(DEST/'linear_offset_equivalence.csv',index=False)
    historical.to_csv(DEST/'original_comparators.csv',index=False)
    details={'study':'post_hoc_exploratory_CPU_only_phase3','task':'case_type_prompt_proxy_not_response_harm','frozen_screen':'0_of_4_failed_unchanged','source_probe_test_score_reproduction_max_abs_error':worst,'source_input_sha256':digest(p),'frozen_screen_sha256':digest(screen),'script_sha256':digest(Path(__file__)),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'folds':5,'text_baselines':{'word':(1,2),'char':(2,5)},'target_labeled_calibration':'O2 training-fold labels used for cutoff only; no O2 test-fold labels used to select cutoff','original_results_modified':False,'limitations':['weak length-matched benign controls','policy variation and uncertain cases','no independent multi-rater review','training-pair label proxy only','post hoc after original screen','no response-level FNR','cross-fold AUROC uses scores from different fold-trained estimators']}
    (DEST/'summary.json').write_text(json.dumps(details,indent=2)+'\n')
    archive=DEST/'stage_a_phase3_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for child in sorted(DEST.iterdir()):
            if child.is_file() and child!=archive: z.write(child,child.name)
    print('Phase 3 CPU diagnostics complete. Existing screen preserved. Original source-probe maximum score difference:',worst)
    print(agg[['guard','method','representation','balanced_accuracy','auroc','harmful_proxy_recall','benign_proxy_specificity']].to_string(index=False))
    # Convenient Windows Downloads delivery; works in user's WSL, optional elsewhere.
    import subprocess
    try:
        winhome=subprocess.check_output(['powershell.exe','-NoProfile','-Command','[Environment]::GetFolderPath("UserProfile")'],text=True,stderr=subprocess.DEVNULL).strip()
        dl=Path(subprocess.check_output(['wslpath','-u',winhome],text=True,stderr=subprocess.DEVNULL).strip())/'Downloads'
        if not dl.is_dir(): dl=Path('/mnt/c/Users/user/Downloads')
        if dl.is_dir():
            import shutil
            shutil.copy2(archive,dl/archive.name)
            print('Ready in Windows Downloads:',dl/archive.name)
        else: print('Downloads not found; archive remains:',archive)
    except (FileNotFoundError,subprocess.CalledProcessError):print('No WSL Windows-Downloads integration; archive remains:',archive)
    return agg

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=ROOT)
    run(parser.parse_args().repo.resolve())
