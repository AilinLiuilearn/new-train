"""Cached-prediction patient routing simulation; not new inference/training.

MCAR: all slices from a case choose the same path. Each seeded permutation
gives nested missing-case sets across rates. Reports actual case/slice rates.
"""
import argparse,json
from pathlib import Path
import numpy as np


def simulate(rows,rates=(0,.25,.5,.75,1),repeats=20,seed=2023):
    if not rows or repeats<1 or any(not 0<=r<=1 for r in rates):raise ValueError('Invalid rates/repeats/samples')
    ids=[r['image_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image_id')
    cases=sorted({r['case_id'] for r in rows});indices={c:i for i,c in enumerate(cases)}
    values=np.zeros((len(cases),2,5),dtype=float)
    for row in rows:
        for j,mode in enumerate(('full','missing')):
            v=row[mode]
            values[indices[row['case_id']],j]+= [v['tp'],v['fp'],v['fn'],v['hd95'],1]
    rng=np.random.RandomState(seed);runs=[]
    for trial in range(repeats):
        order=rng.permutation(len(cases))
        for rate in rates:
            n=int(np.floor(rate*len(cases)+.5));missing=np.zeros(len(cases),dtype=bool);missing[order[:n]]=True
            totals=values[np.arange(len(cases)),missing.astype(int)].sum(0)
            tp,fp,fn,hd,count=totals;denom=2*tp+fp+fn
            runs.append({'trial':trial,'requested_case_missing_rate':rate,'actual_case_missing_rate':n/len(cases),
                'actual_slice_missing_rate':float(values[missing,1,4].sum()/count),
                'dice':float(2*tp/denom) if denom else 1.,'hd95':float(hd/count),
                'tp':int(tp),'fp':int(fp),'fn':int(fn),'missing_cases':[cases[i] for i in order[:n]]})
    summary=[]
    for rate in rates:
        group=[r for r in runs if r['requested_case_missing_rate']==rate]
        summary.append({'requested_case_missing_rate':rate,'actual_case_missing_rate':group[0]['actual_case_missing_rate'],
                        **{name+'_'+stat:float(fn([r[name] for r in group])) for name in ('dice','hd95','actual_slice_missing_rate')
                           for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]}})
    return {'source':'cached paired predictions; no new inference','policy':'MCAR, case-level; nested missing sets per trial',
            'cases':len(cases),'slices':len(rows),'repeats':repeats,'seed':seed,'summary':summary,'runs':runs,
            'limits':['Routing sensitivity, not training seeds or independent test evidence.',
                      'Missing rate changes path assignment, not CT-only model strength.',
                      'Global Dice is computed from routed TP/FP/FN; never interpolate endpoint Dice.',
                      'Assumes cached CT branch equals Missing inference, as guaranteed by the current isolated model path.',
                      'Does not simulate within-image/partial PET loss or nonrandom clinical missingness.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=20)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=simulate([json.loads(x) for x in a.samples.read_text().splitlines()],repeats=a.repeats)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
