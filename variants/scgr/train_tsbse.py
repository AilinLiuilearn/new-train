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
from .tsbse import TSBSE

LOSS=None
TRAIN_CALLS=0


class TSBSEErrorMetrics(ErrorMetrics):
    @torch.no_grad()
    def update(self,logits,target):
        super().update(logits,target)
        predictions=logits.float().sigmoid().cpu().numpy()[:,0]>.5
        targets=target.cpu().numpy()[:,0]>.5
        for p,y in zip(predictions,targets):
            a=int(y.sum());tp=int((p&y).sum());fp=int((p&~y).sum());fn=a-tp
            self.counts['slice_count']+=1
            if a:
                self.counts['positive_slices']+=1
                if not tp:
                    self.counts['zero_overlap_slices']+=1
                    self.counts['zero_overlap_nonempty_prediction' if p.any() else 'zero_overlap_empty_prediction']+=1
            group='le256' if a<=256 else ('257_1024' if a<=1024 else 'gt1024')
            for key,value in [('slices',1),('tp',tp),('fp',fp),('fn',fn)]:
                self.counts[group+'_'+key]+=value

    def compute(self):
        result=super().compute();c=self.counts;groups={}
        for name in ['le256','257_1024','gt1024']:
            tp,fp,fn=[c.get(name+'_'+k,0) for k in ['tp','fp','fn']]
            den=2*tp+fp+fn
            groups[name]={'slices':c.get(name+'_slices',0),'tp':tp,'fp':fp,'fn':fn,
                'pooled_pixel_dice':2*tp/den if den else None}
        result['gt_area_groups']=groups
        return result


def objective(model,output,target,criterion):
    global LOSS,TRAIN_CALLS
    if LOSS is None:
        LOSS = TSBSE(
            coverage_scale=model.cfg.get('tsbse_cover_scale', 'global'),
            coverage_reference=model.cfg.get('tsbse_cover_reference', 'predicted_foreground'),
        )
    primary,parts=criterion(output['logits'].float(),target)
    extra,stats,terms=LOSS(output['logits'],target)
    weight=getattr(model,'_tsbse_effective_weight',model.cfg['tsbse_weight'])
    total=primary+weight*extra
    if torch.is_grad_enabled():
        TRAIN_CALLS+=1
        if TRAIN_CALLS==1 or TRAIN_CALLS%100==0:
            base_grad=torch.autograd.grad(primary,output['logits'],retain_graph=True)[0].float()
            aux_grad=torch.autograd.grad(weight*extra,output['logits'],retain_graph=True)[0].float()
            cg=torch.autograd.grad(.5*weight*terms['cover'],output['logits'],retain_graph=True)[0].float()
            sg=torch.autograd.grad(.5*weight*terms['spill'],output['logits'],retain_graph=True)[0].float()
            engine.log('loss_gradients.jsonl',{'training_call':TRAIN_CALLS,
                'epoch':getattr(model,'_tsbse_epoch',None),
                'effective_weight':weight,
                'base_logit_grad_l2':float(base_grad.norm()),
                'weighted_tsbse_logit_grad_l2':float(aux_grad.norm()),
                'ratio':float(aux_grad.norm()/base_grad.norm().clamp_min(1e-12)),
                'weighted_cover_grad_l2':float(cg.norm()),
                'weighted_spill_grad_l2':float(sg.norm()),
                'cosine':float((base_grad*aux_grad).sum()/(base_grad.norm()*aux_grad.norm()).clamp_min(1e-12)),
                'scope':'Output-logit gradient diagnostics before AMP scaling; not proof of task benefit.'})
    return total,{'total':float(total.detach()),'bce':float(parts['loss_bce']),
        'dice_loss':float(parts['loss_dice']),'segmentation':float(primary.detach()),
        'tsbse_effective_weight':weight,'tsbse':float(extra.detach()),'tsbse_weighted':float((weight*extra).detach()),
        **{'tsbse_'+k:v for k,v in stats.items()}}


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
    lines=['# OriginalT + revised TSBSE, see resolved config for audited weight','',
        '|Epoch|Weights|Dice %|Paired HD95|Sensitivity %|Precision %|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation']['full']
        lines.append(f"|{r['epoch']}|{r['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|{100*v['sensitivity']:.4f}|{100*v['precision']:.4f}|")
    lines+=['','Fresh task modules; generic encoder pretraining only. Best-only, patience3. No automatic tuning/ablation queue.']
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=parser('scgr_T_tsbse');args=p.parse_args()
    try:
        engine.main(args,variant='scgr_T_tsbse',model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=objective,extra_metrics_factory=TSBSEErrorMetrics,
            protocol_extra={'loss':'Original BCE+Dice plus audited-weight TSBSE; detached exact EDT, bounded d/(d+r), logit softplus, GT-budgeted background top-K.',
                'initialization':'Only generic encoder pretraining; no originalT or interrupted repeat task weights.',
                'inference':'Exactly originalT; same checkpoint loader/format. Loss has no saved model parameters.',
                'limits':'Detached threshold geometry and hard-negative tail; not a fully differentiable distance metric. Whole-slice scale and label ambiguity remain limitations.',
                'scope':'One authorized full candidate, no automatic new search or hyperparameter changes.'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
