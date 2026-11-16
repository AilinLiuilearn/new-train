import json
from pathlib import Path
import traceback

from runtime import REPO,parser,local_paths
from variants.tsbci import train as engine
from utils.run_common import module_grad_norm
from .model import Model
from .losses import objective
from .metrics import JointMetrics

REFERENCE=None
RECIPE=None


def diagnostics(model,output,epoch,step,update):
    cap=output['capture']
    record={'epoch':epoch,'step':step,'AMP_gradient':update,
            'scales':{str(i+1):{name:engine.stats(cap[name][i]) for name in ['ct','pet','fused']} for i in range(4)},
            'predictions':{k:engine.stats(output[k]) for k in ['logits','ct_logits','pet_logits']},
            'scope':'Training snapshots; not isolated task contribution'}
    if update['success']:
        record['gradient_groups']={n:module_grad_norm(v) for n,v in model.named_children()}
        if hasattr(model,'semantic'):
            record['semantic_projection_gradients']=[module_grad_norm(v) for v in model.semantic.projections]
    engine.log('diagnostics.jsonl',record)


def results(rows,complete=False):
    if not rows:return
    best=max(rows,key=lambda r:r['validation']['full']['dice']);m=best['validation']['full']
    result={'complete':complete,'recipe':RECIPE,'best_epoch':best['epoch'],'weights':best['weights'],
            'metrics':m,'diagnostics':best['validation'].get('auxiliary_diagnostics'),
            'selection':'Fusion validation Dice best; all reported metrics from that epoch',
            'scope':'Single-seed reused dev; auxiliary outputs are not independently selected experts',
            'target_788_hd17_met':m['dice']>=.788 and m['hd95']<17,
            'next_experiments':[]}
    if REFERENCE and REFERENCE.exists():
        ref=json.loads(REFERENCE.read_text());result['reference_M1']={'path':str(REFERENCE),'result':ref}
        result['delta_M1_dice_pp']=100*(m['dice']-ref['metrics']['dice'])
        result['delta_M1_hd95']=m['hd95']-ref['metrics']['hd95']
    engine.dump('RESULTS.json',result)
    lines=['# GT-VCL '+str(RECIPE),'','|Epoch|Weights|Fusion Dice %|Paired HD95|Sensitivity %|Precision %|',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        v=row['validation']['full']
        lines.append(f"|{row['epoch']}|{row['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|{100*v['sensitivity']:.4f}|{100*v['precision']:.4f}|")
    lines+=['','Each recipe starts fresh from generic encoder pretraining; no task checkpoint continuation.',
            'One validation traversal; only fusion Dice selects best. Text-semantic efficacy is not isolated by E4 alone.']
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=parser('gt_vcl_E1');p.add_argument('--recipe',choices=['E1','E2','E4'],required=True);p.set_defaults(config=None)
    args=p.parse_args();RECIPE=args.recipe
    args.config=args.config or str(REPO/'configs'/f'gt_vcl_{RECIPE}.json')
    try:
        cfg=json.loads(Path(args.config).read_text())
        if cfg['recipe']!=RECIPE:raise ValueError('Recipe/config mismatch')
        REFERENCE=Path(local_paths(args.paths)['output_root'])/cfg['comparison_m1_run']/'RESULTS.json'
        engine.main(args,variant='gt_vcl_'+RECIPE,model_factory=Model,results_fn=results,
                    diagnostics_fn=diagnostics,model_objective_fn=objective,extra_metrics_factory=JointMetrics,
                    protocol_extra={'loss':'Three BCE+Dice terms weight1; E2 adds balanced CT-reference0.1; E4 adds region text0.1',
                        'recipe':RECIPE,'aux_decoder':'One shared GN decoder called separately for CT/PET; fusion decoder independent',
                        'reference':'Current stop-gradient CT prediction, no frozen teacher or reference decoder',
                        'validation':'One traversal, shared encodings and three decoder outputs; fusion alone selects best',
                        'inference':'Encoders -> M1 -> fusion decoder only; auxiliary decoder and text removed',
                        'scope':'Exploratory E1/E2/E4 ablation; no independent test or semantic causality claim'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
