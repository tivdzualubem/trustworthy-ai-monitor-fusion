import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
p=Path(__file__).with_name('stage_a_phase3_cpu.py')
s=importlib.util.spec_from_file_location('phase3',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
assert abs(m.threshold([0,0,1,1],[.1,.2,.8,.9])-.5)<.01
assert m.threshold([0,1],[.7,.8])>.7
rows=[]
for j in range(38):
 for cls in ['harmful','benign']:
  for rep in ['direct','O2']:
   rows.append(dict(pair_id=f'FU{j:02}',semantic_case_id=f'FU{j:02}::{cls}',experiment_example_id=f'FU{j:02}::{cls}::{rep}',representation=rep,fold_id=j%5,label_harmful=int(cls=='harmful')))
df=pd.DataFrame(rows)
m.checks(df)
mask=(df.fold_id==0)&(df.representation=='O2')
preds=m.rows_for(df,mask,'dummy','guard','O2',np.tile([.8,.2],8),.5)
assert len(preds)==16
summ=m.metrics(pd.DataFrame(preds))
assert summ.iloc[0].n==16
print('PASS: thresholds, pair-ID split invariants, row predictions, metrics')
