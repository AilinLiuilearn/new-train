"""OriginalT architecture, original training engine, one extra final-logit loss."""
import json
from pathlib import Path
import traceback
import torch
from runtime import REPO,parser
from variants.tsbci import train as engine
from .model import Model
from .metrics import ErrorMetrics
from .train import diagnostics
from .tssda import TSSDA

LOSS=None
TRAIN_CALLS=0


def objective(model,output,target,criterion):
    global LOSS,TRAIN_CALLS
    if LOSS is None:LOSS=TSSDA(scale_mode=model.cfg['tssda_scale'])
    primary,parts=criterion(output['logits'].float(),target)
    extra,stats=LOSS(output['logits'],target)
    weight=model.cfg['tssda_weight']
    total=primary+weight*extra
    if torch.is_grad_enabled():
        TRAIN_CALLS+=1
        if TRAIN_CALLS==1 or TRAIN_CALLS%100==0:
            base_grad=torch.autograd.grad(primary,output['logits'],retain_graph=True)[0].float()
            aux_grad=torch.autograd.grad(weight*extra,output['logits'],retain_graph=True)[0].float()
            engine.log('loss_gradients.jsonl',{'training_call':TRAIN_CALLS,
                'base_logit_grad_l2':float(base_grad.norm()),
                'weighted_tssda_logit_grad_l2':float(aux_grad.norm()),
                'ratio':float(aux_grad.norm()/base_grad.norm().clamp_min(1e-12)),
                'cosine':float((base_grad*aux_grad).sum()/(base_grad.norm()*aux_grad.norm()).clamp_min(1e-12)),
                'scope':'Output-logit gradient diagnostics before AMP scaling; not proof of task benefit.'})
    return total,{'total':float(total.detach()),'bce':float(parts['loss_bce']),
        'dice_loss':float(parts['loss_dice']),'segmentation':float(primary.detach()),
        'tssda':float(extra.detach()),'tssda_weighted':float((weight*extra).detach()),
        **{'tssda_'+k:v for k,v in stats.items()}}


def results(rows,complete=False):
    if not rows:return
    row=max(rows,key=lambda r:r['validation']['full']['dice'])
    metrics=row['validation']['full']
    reference={'dice':.7894926861949604,'hd95':16.169107096291484}
    result={'complete':complete,'best_epoch':row['epoch'],'weights':row['weights'],'metrics':metrics,
        'error_diagnostics':row['validation'].get('auxiliary_diagnostics'),
        'original_T':reference,'delta_original_T_dice_pp':100*(metrics['dice']-reference['dice']),
        'delta_original_T_hd95':metrics['hd95']-reference['hd95'],
        'both_original_T_metrics_improved':metrics['dice']>reference['dice'] and metrics['hd95']<reference['hd95'],
        'scope':'Reused dev; same-seed stochastic rerun differs. One candidate does not establish loss novelty or isolated efficacy.',
        'next_experiments':[]}
    engine.dump('RESULTS.json',result)
    lines=['# OriginalT + TSSDA, directional scale, weight0.01','',
        '|Epoch|Weights|Dice %|Paired HD95|Sensitivity %|Precision %|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation']['full']
        lines.append(f"|{r['epoch']}|{r['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|{100*v['sensitivity']:.4f}|{100*v['precision']:.4f}|")
    lines+=['','Fresh task modules; generic encoder pretraining only. Best-only, patience3. No automatic tuning/ablation queue.']
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=parser('scgr_T_tssda');args=p.parse_args()
    try:
        engine.main(args,variant='scgr_T_tssda',model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=objective,extra_metrics_factory=ErrorMetrics,
            protocol_extra={'loss':'Original BCE+Dice +0.01TSSDA;16 fixed directions,1px mass-conserving linear bins, GT directional std floor1.',
                'initialization':'Only generic encoder pretraining; no originalT or interrupted repeat task weights.',
                'inference':'Exactly originalT; same checkpoint loader/format. Loss has no saved model parameters.',
                'limits':'Whole-slice GT scale includes inter-region separation. Finite directions, probability normalization and confident-error gradients have known limits.',
                'scope':'One authorized full candidate, no automatic new search or hyperparameter changes.'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
