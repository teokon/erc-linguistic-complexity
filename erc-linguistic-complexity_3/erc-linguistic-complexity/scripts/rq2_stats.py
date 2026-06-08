"""
RQ2 Statistical Tests - Blind Spots per Emotion
Mann-Whitney U test: Low vs High complexity bin per emotion.
Απαιτει: rq3_results_with_l2sca.csv
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('rq3_results_with_l2sca.csv')

LABELS   = ['neutral','frustration','sadness','anger','excited','happiness']
label2id = {l:i for i,l in enumerate(LABELS)}
BINS     = ['Low','Medium-Low','Medium-High','High']

df['label_id']    = df['label'].map(label2id)
df['pred_ctx_id'] = df['pred_ctx'].map(label2id)
df['correct_ctx'] = (df['pred_ctx'] == df['label']).astype(int)

print('='*60)
print('RQ2: BLIND SPOTS - MANN-WHITNEY TEST PER EMOTION')
print('='*60)

for emo in LABELS:
    sub = df[df['label'] == emo].copy()
    if len(sub) < 20:
        print(emo + ': too few samples (' + str(len(sub)) + ')')
        continue

    try:
        sub['bin'] = pd.qcut(df.loc[sub.index,'composite_full'], q=4, labels=BINS, duplicates='drop')
    except Exception as e:
        print(emo + ': cannot bin - ' + str(e))
        continue

    low  = sub[sub['bin']=='Low']['correct_ctx'].values
    high = sub[sub['bin']=='High']['correct_ctx'].values

    if len(low) < 5 or len(high) < 5:
        print(emo + ': not enough per bin')
        continue

    stat, p = stats.mannwhitneyu(low, high, alternative='greater')
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))

    print()
    print(emo.upper() + ' (n=' + str(len(sub)) + ')')
    print('  Accuracy Low:  ' + str(round(low.mean(),3)) + '  (n=' + str(len(low)) + ')')
    print('  Accuracy High: ' + str(round(high.mean(),3)) + '  (n=' + str(len(high)) + ')')
    print('  Delta:        ' + str(round(low.mean()-high.mean(),3)))
    print('  Mann-Whitney: U=' + str(round(stat,1)) + '  p=' + str(round(p,4)) + '  ' + sig)

print()
print('Done.')
