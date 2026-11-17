"""One discarded real-batch preflight plus implementation invariants; no checkpoint."""
import argparse,json,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,resolve_config,parser
require_myenv()
import torch
from variants.scgr.tsbse import TSBSE


def synthetic():
    torch.set_num_threads(2);module=TSBSE();cases={}
    gt=torch.zeros(1,1,32,32);gt[:,:,8:12,8:12]=1
    for name,y,z in [('correct',gt,torch.where(gt.bool(),20.,-20.)),
        ('empty_pred',gt,torch.full_like(gt,-1000.)),('full_pred',gt,torch.full_like(gt,1000.)),
        ('empty_gt',torch.zeros_like(gt),torch.zeros_like(gt)),
        ('full_gt',torch.ones_like(gt),torch.full_like(gt,-20.)),
        ('near_threshold',gt,torch.full_like(gt,-.001))]:
        z=z.clone().requires_grad_(True);loss,stats,_=module(z,y);g=torch.autograd.grad(loss,z)[0]
        assert torch.isfinite(loss) and torch.isfinite(g).all()
        assert float(g.abs().sum())<=1.00001
        if name=='empty_pred':assert bool((g[y.bool()]<0).all())
        if name=='full_pred':assert bool((g[~y.bool()]>0).any())
        cases[name]={'loss':float(loss),'gradient_l1':float(g.abs().sum()),**stats}
    # Batch averaging includes zero-valued skipped slices.
    z=torch.where(gt.bool(),-10.,10.).requires_grad_(True)
    single=module(z,gt)[0]
    combined=module(torch.cat([z,z]),torch.cat([gt,torch.zeros_like(gt)]))[0]
    assert torch.allclose(combined,.5*single)
    # Frozen geometry and stable ranks: directional finite difference.
    z=torch.linspace(-5,-.1,1024).reshape_as(gt).requires_grad_(True)
    direction=torch.linspace(.01,.05,1024).reshape_as(gt)
    loss=module(z,gt)[0];g=torch.autograd.grad(loss,z)[0];eps=.01
    numerical=(module(z.detach()+eps*direction,gt)[0]-module(z.detach()-eps*direction,gt)[0])/(2*eps)
    analytic=(g*direction).sum();assert abs(float(numerical-analytic))<.0001
    return {'cases':cases,'finite_difference':float(numerical),'analytic':float(analytic),'passed':True}


def gpu(output):
    from utils.run_common import seed_everything
    from utils.seg_losses import BCEDiceLoss
    from datasets.staged import loaders
    from variants.scgr.model import Model
    args=parser('scgr_T_tsbse').parse_args(['--run','tsbse_preflight'])
    cfg,_,locations=resolve_config(args,'scgr_T_tsbse')
    seed_everything(2023);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    model=Model(vars(cfg),load_generic=True).cuda().train()
    train,_,_=loaders(cfg,output)
    batch=next(iter(train));ct,pet,y=[batch[k].cuda() for k in ('ct','pet','mask')]
    opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    begin=time.time()
    with torch.autocast('cuda',dtype=torch.float16):o=model(ct,pet)
    weight=vars(cfg)['tsbse_weight']
    base,_=BCEDiceLoss()(o['logits'].float(),y)
    extra,info,terms=TSBSE()(o['logits'],y)
    bg=torch.autograd.grad(base,o['logits'],retain_graph=True)[0].float()
    ag=torch.autograd.grad(weight*extra,o['logits'],retain_graph=True)[0].float()
    ratio=float(ag.norm()/bg.norm().clamp_min(1e-12))
    cg=torch.autograd.grad(.5*weight*terms['cover'],o['logits'],retain_graph=True)[0].float()
    sg=torch.autograd.grad(.5*weight*terms['spill'],o['logits'],retain_graph=True)[0].float()
    scaler.scale(base+weight*extra).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    assert torch.isfinite(norm),'Nonfinite real-batch preflight gradient'
    scaler.step(opt);scaler.update();torch.cuda.synchronize()
    assert model.decoder_full and all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.decoder_full.modules())
    return {'batch':len(y),'image_shape':list(ct.shape),'segmentation_loss':float(base),
        'tsbse_loss':float(extra),'weighted_tsbse':float(weight*extra),'logit_gradient_ratio':ratio,'weight':weight,
        'weighted_cover_gradient_ratio':float(cg.norm()/bg.norm()),'weighted_spill_gradient_ratio':float(sg.norm()/bg.norm()),'unscaled_gradient_norm':float(norm),'max_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'seconds':time.time()-begin,'stats':info,'all_trainable':True,'decoder':'GN','finite_backward':True,
        'note':'One real augmented training batch; optimizer discarded. Formal experiment retains original AMP scaler settings.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'synthetic':synthetic()}
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.gpu:result['gpu']=gpu(out)
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
