"""Frozen preflight evidence; no task checkpoints saved or loaded."""
import argparse,json,os,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,REPO,resolve_config,parser
require_myenv()
import torch
from variants.scgr.tssda import TSSDA


def evaluate(z,y):
    z=z.clone().requires_grad_(True)
    value,stats=TSSDA()(z,y)
    g=torch.autograd.grad(value,z)[0]
    assert torch.isfinite(value) and torch.isfinite(g).all()
    return {'loss':float(value),'gradient_max':float(g.abs().max()),
        'gt_gradient_mean':float(g[y.bool()].mean()) if y.any() else None,**stats}


def synthetic():
    cases={}
    y=torch.zeros(1,1,16,16);y[:,:,5:9,5:9]=1
    z=torch.where(y.bool(),torch.full_like(y,12),torch.full_like(y,-20))
    cases['correct']=evaluate(z,y)
    cases['correct_unequal_confidence']=evaluate(z+5*y,y)
    for shift in [1,3,6]:
        shifted=torch.roll(y,(shift,shift),(-2,-1))
        cases[f'translation_{shift}']=evaluate(torch.where(shifted.bool(),torch.full_like(y,8),torch.full_like(y,-8)),y)
    assert cases['translation_1']['loss']<cases['translation_3']['loss']<cases['translation_6']['loss']
    for shape in ['thin','irregular','separated','single']:
        target=torch.zeros_like(y)
        if shape=='thin':target[:,:,3:13,7]=1
        if shape=='irregular':target[:,:,3:11,4]=1;target[:,:,9,4:12]=1
        if shape=='separated':target[:,:,2:4,2:4]=1;target[:,:,12:14,12:14]=1
        if shape=='single':target[:,:,3,3]=1
        cases[shape]=evaluate(torch.where(target.bool(),torch.full_like(y,8),torch.full_like(y,-10)),target)
    target=torch.zeros_like(y);target[:,:,3,3]=1
    for magnitude in [20.,1000.]:
        z=torch.full_like(target,-magnitude);z[:,:,12,12]=magnitude
        cases[f'confident_wrong_{magnitude}']=evaluate(z,target)
    cases['all_low']=evaluate(torch.full_like(y,-1000),y)
    cases['all_high']=evaluate(torch.full_like(y,1000),y)
    cases['empty']=evaluate(torch.zeros_like(y),torch.zeros_like(y))
    y=torch.zeros(1,1,512,512);y[:,:,255,240:273]=1
    p=torch.where(y.bool(),torch.full_like(y,.99),torch.full_like(y,.001))
    cases['diffuse_background_512']=evaluate(p.logit(),y)
    # Direct bin-mass conservation, endpoint accounting and finite differences.
    module=TSSDA();q=torch.rand(2,16*16);q/=q.sum(1,keepdim=True)
    max_error=0.
    for pos,index,w,bins in module._geometry(16,16,'cpu'):
        projected=q.new_zeros(2,bins);idx=index[None].expand(2,-1)
        projected.scatter_add_(1,idx,q*(1-w));projected.scatter_add_(1,idx+1,q*w)
        assert torch.allclose(projected,module.project(q,index,w),atol=1e-7)
        max_error=max(max_error,float((projected.sum(1)-1).abs().max()))
        assert torch.allclose((projected*torch.arange(bins)).sum(1),(q*(pos-pos.min())).sum(1),atol=1e-5)
    assert max_error<1e-6
    target=torch.zeros(1,1,5,5);target[:,:,2,2]=1
    z=torch.linspace(-2,2,25).reshape_as(target).requires_grad_(True)
    value,_=module(z,target);grad=torch.autograd.grad(value,z)[0]
    direction=torch.linspace(.1,1,25).reshape_as(z);eps=.001
    plus=module(z.detach()+eps*direction,target)[0];minus=module(z.detach()-eps*direction,target)[0]
    numerical=float((plus-minus)/(2*eps));analytic=float((grad*direction).sum())
    assert abs(numerical-analytic)<.003
    return {'cases':cases,'mass_max_error':max_error,'finite_difference':numerical,'analytic':analytic,
        'limitations':'Low-probability background and saturation remain real limitations, not passed efficacy claims.',
        'passed_implementation_checks':True}


def gpu(output):
    from utils.run_common import seed_everything
    from utils.seg_losses import BCEDiceLoss
    from datasets.staged import loaders
    from variants.scgr.model import Model
    args=parser('scgr_T_tssda').parse_args(['--run','tssda_preflight'])
    cfg,_,locations=resolve_config(args,'scgr_T_tssda')
    seed_everything(2023);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    model=Model(vars(cfg),load_generic=True).cuda().train()
    train,_,_=loaders(cfg,output)
    batch=next(iter(train));ct,pet,y=[batch[k].cuda() for k in ('ct','pet','mask')]
    opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    begin=time.time()
    with torch.autocast('cuda',dtype=torch.float16):o=model(ct,pet)
    base,_=BCEDiceLoss()(o['logits'].float(),y)
    extra,info=TSSDA()(o['logits'],y)
    bg=torch.autograd.grad(base,o['logits'],retain_graph=True)[0].float()
    ag=torch.autograd.grad(.01*extra,o['logits'],retain_graph=True)[0].float()
    ratio=float(ag.norm()/bg.norm().clamp_min(1e-12))
    scaler.scale(base+.01*extra).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    assert torch.isfinite(norm),'Nonfinite real-batch preflight gradient'
    scaler.step(opt);scaler.update();torch.cuda.synchronize()
    assert model.decoder_full and all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in model.decoder_full.modules())
    return {'batch':len(y),'image_shape':list(ct.shape),'segmentation_loss':float(base),
        'tssda_loss':float(extra),'weighted_tssda':float(.01*extra),'logit_gradient_ratio':ratio,
        'unscaled_gradient_norm':float(norm),'max_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
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
