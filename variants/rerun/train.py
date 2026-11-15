"""Explicit three-recipe reruns, sharing the historical numerical training functions."""
import json
import traceback
from runtime import parser,REPO
from variants.tsbci import train as engine
from variants.concat_single.model import ConcatSingle

LABELS={'m1':'Original dual alignment + M1 + GN',
        'concat_single':'CT-only pre-alignment + raw PET + concat/compress + GN',
        'tsbci':'Original 50% TSBCI + M1 + GN'}
ACTIVE=None


def results(rows,complete=False):
    if not rows:return
    best=max(rows,key=lambda r:r['validation']['full']['dice'])
    metrics=best['validation']['full']
    engine.dump('RESULTS.json',{'complete':complete,'variant':ACTIVE,'label':LABELS[ACTIVE],
                              'best_epoch':best['epoch'],'weights':best['weights'],'metrics':metrics,
                              'selection':'maximum validation Dice; HD95 from that same checkpoint',
                              'patience':3,'scope':'Single seed exploratory reused development set',
                              'comparison_limit':'Concat removes PET pre-alignment as well as changing fusion; its difference from M1 is not an isolated M1 effect'})
    lines=['# '+LABELS[ACTIVE],'','|Epoch|Weights|Dice %|Same-epoch HD95|','|---|---|---:|---:|']
    for row in rows:
        v=row['validation']['full'];lines.append(f"|{row['epoch']}|{row['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|")
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def concat_diagnostics(m,o,epoch,step,update):
    cap=o['capture']
    engine.log('diagnostics.jsonl',{'epoch':epoch,'step':step,'AMP_gradient':update,
              'scales':{str(i+1):{n:engine.stats(cap[n][i]) for n in ['ct_raw','ct_aligned','pet_raw','fused']} for i in range(4)},
              'alignment':'CT-only pre-concat alignment; PET raw, existing concat1x1 compressor afterwards'})


if __name__=='__main__':
    p=parser('tsbci');p.add_argument('--variant',choices=list(LABELS),required=True);p.set_defaults(config=None)
    args=p.parse_args();ACTIVE=args.variant
    args.config=args.config or str(REPO/'configs'/f'rerun_{ACTIVE}_p3.json')
    args.arm='m1' if ACTIVE=='m1' else 'full'
    try:
        cfg=json.loads(open(args.config).read())
        if cfg['early_stopping_patience']!=3:raise ValueError('This rerun recipe requires patience=3')
        engine.main(args,variant=ACTIVE,
                    model_factory=ConcatSingle if ACTIVE=='concat_single' else None,
                    results_fn=results,diagnostics_fn=concat_diagnostics if ACTIVE=='concat_single' else None)
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
