import json
import traceback
from runtime import REPO,parser
from variants.tsbci import train as engine
from utils.run_common import module_grad_norm
from .model import Model


def diagnostics(m,o,epoch,step,u):
    cap=o['capture'];scales={}
    for i,d in enumerate(cap['interaction']):
        s={k:engine.stats(cap[k][i]) for k in ['ct_aligned','pet_aligned','ct_interacted','pet_interacted','fused']}
        s['interaction']={k:engine.stats(v) for k,v in d.items() if k not in ['ct_attention','pet_attention','ct_score','pet_score']}
        s['attention']={n:{k:float(v.detach()) for k,v in d[n+'_attention'].items()} for n in ['ct','pet']}
        s['gradient_groups']={k:module_grad_norm(v) for k,v in m.front.blocks[i].named_children()} if u['success'] else {}
        for n in ['ct','pet']:
            score=d[n+'_score']
            if score.grad is not None and u['success']:
                grad=score.grad.detach().float()/u['backward_scale'];chosen=d[n+'_mask'].bool()
                s[n+'_score_grad']={'selected_abs_mean':float(grad[chosen].abs().mean()),
                                    'unselected_abs_mean':float(grad[~chosen].abs().mean())}
            x=cap[n+'_aligned'][i].detach().float();delta=d[n+'_delta'].detach().float()
            s[n+'_relative_delta_rms']=float(delta.square().mean().sqrt()/(x.square().mean().sqrt()+1e-8))
            p=getattr(m.front.blocks[i],'to_'+n).beta
            s[n+'_beta_grad']=float(p.grad.detach()) if p.grad is not None and u['success'] else None
        if m.front.text_enabled:s['text_gradient_norm']=module_grad_norm(m.front.conditions[i]) if u['success'] else None
        scales[str(i+1)]=s
    engine.log('diagnostics.jsonl',{'epoch':epoch,'step':step,'AMP_gradient':u,'scales':scales})


def results(rows,complete=False):
    if not rows:return
    best=max(rows,key=lambda r:r['validation']['full']['dice'])
    engine.dump('RESULTS.json',{'complete':complete,'best_epoch':best['epoch'],'weights':best['weights'],
                'metrics':best['validation']['full'],'scope':'Full candidate with or without fixed text; reused dev, single seed; no isolated core/semantic claim'})
    lines=['|Epoch|Weights|Dice %|Same-epoch HD95|','|---|---|---:|---:|']
    for r in rows:
        v=r['validation']['full'];lines.append(f"|{r['epoch']}|{r['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|")
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=parser('ccscmi_visual');p.add_argument('--text',action='store_true');p.set_defaults(config=None)
    args=p.parse_args();variant='ccscmi_text' if args.text else 'ccscmi_visual'
    args.config=args.config or str(REPO/'configs'/f'{variant}.json')
    try:
        cfg=json.loads(open(args.config).read())
        if bool(cfg['text_enabled'])!=args.text:raise ValueError('Text flag/config mismatch')
        engine.main(args,variant=variant,model_factory=Model,results_fn=results,diagnostics_fn=diagnostics)
    except Exception as e:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(e))
        raise SystemExit(1)
