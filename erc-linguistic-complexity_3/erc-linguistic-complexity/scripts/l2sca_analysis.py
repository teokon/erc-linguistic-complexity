"""
L2SCA Per-Metric Bin Analysis
Τρεξε τοπικα μετα απο L2SCA inference.
Απαιτει: rq3_results_with_l2sca.csv
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

df = pd.read_csv('rq3_results_with_l2sca.csv')

LABELS   = ['neutral','frustration','sadness','anger','excited','happiness']
label2id = {l:i for i,l in enumerate(LABELS)}
BINS     = ['Low','Medium-Low','Medium-High','High']

df['label_id']    = df['label'].map(label2id)
df['pred_ctx_id'] = df['pred_ctx'].map(label2id)
df['pred_sng_id'] = df['pred_sng'].map(label2id)
df['gain_per_utt'] = df['correct_ctx'] - df['correct_sng']

l2sca_metrics = {'MLT': 'MLT_l2sca', 'MLS': 'MLS_l2sca', 'DC/C': 'DC_C_l2sca'}

for name, col in l2sca_metrics.items():
    try:
        df['bin_tmp'] = pd.qcut(df[col], q=4, labels=BINS, duplicates='drop')
    except Exception as e:
        print(name + ': cannot bin - ' + str(e))
        continue

    print('='*55)
    print('Metric: ' + name)
    print('Bin              Single    EmoBERTa  Gain    n')
    print('-'*55)
    for b in BINS:
        sub = df[df['bin_tmp'] == b]
        if len(sub) < 10: continue
        fc = f1_score(sub['label_id'], sub['pred_ctx_id'], average='weighted', zero_division=0)
        fs = f1_score(sub['label_id'], sub['pred_sng_id'], average='weighted', zero_division=0)
        print(b.ljust(15) + '  ' + str(round(fs,3)) + '     ' + str(round(fc,3)) + '     ' + str(round(fc-fs,3)) + '  ' + str(len(sub)))

    df['rank_tmp'] = df['bin_tmp'].map({'Low':1,'Medium-Low':2,'Medium-High':3,'High':4})
    valid = df['rank_tmp'].notna()
    r, p = stats.pearsonr(df.loc[valid,'rank_tmp'], df.loc[valid,'gain_per_utt'])
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
    print('Correlation r=' + str(round(r,3)) + '  p=' + str(round(p,4)) + '  ' + sig)
    print()

print('Done.')
