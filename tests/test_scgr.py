import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO
from variants.scgr.model import Model
from variants.scgr.module import MultiScaleRefiner
from variants.scgr.losses import score_loss,objective
from variants.scgr.inference import load_model
from variants.scgr.metrics import ErrorMetrics
from variants.tsbci.model import SHCIPETCT
from utils.seg_losses import BCEDiceLoss

torch.set_num_threads(2)


def cfg(recipe):return json.loads((REPO/'configs'/f'scgr_{recipe}.json').read_text())


def inputs():return [torch.randn(2,c,r,r,requires_grad=True) for c,r in [(64,32),(128,16),(320,8),(512,4)]]


def test_common_initialization_rng_identity_and_inference(tmp_path):
    torch.manual_seed(2023);base=SHCIPETCT(cfg('V'),arm='m1',load_generic=False).eval()
    state={k:v.clone() for k,v in base.state_dict().items()};rng=torch.get_rng_state().clone()
    ct,pet=torch.randn(1,3,256,256),torch.randn(1,3,256,256)
    with torch.no_grad():expected=base(ct,pet)['logits']
    common=None
    for recipe in ['V','T','S']:
        torch.manual_seed(2023);m=Model(cfg(recipe),load_generic=False).eval()
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(m.state_dict()[k],v) for k,v in state.items())
        if common is None:common={k:v.clone() for k,v in m.state_dict().items()}
        else:assert all(torch.equal(m.state_dict()[k],v) for k,v in common.items())
        with torch.no_grad():actual=m(ct,pet)['logits']
        assert torch.equal(actual,expected)
        assert all(p.requires_grad for p in m.parameters())
        assert not hasattr(m,'decoder_ct') and not hasattr(m,'decoder_sep')
        if recipe=='S':
            ckpt=tmp_path/'best.pt'
            torch.save({'format':'scgr_support_conditioned_v1','config':cfg(recipe),'weights':m.state_dict()},ckpt)
            deployed=load_model(ckpt,'cpu')
            with torch.no_grad():restored=deployed(ct,pet)['logits']
            assert torch.equal(actual,restored)


def test_score_targets_native_grid_and_empty_masks():
    scores=[(torch.randn(2,1,s,s,requires_grad=True),torch.randn(2,1,max(s//2,1),max(s//2,1),requires_grad=True)) for s in (16,8,4,2)]
    for foreground in (False,True):
        y=torch.ones(2,1,32,32) if foreground else torch.zeros(2,1,32,32)
        if foreground:y[:,:,4:24,5:25]=0
        loss,_=score_loss(scores,y)
        expected=torch.stack([F.binary_cross_entropy_with_logits(z,F.interpolate(y,size=z.shape[-2:],mode='area')) for pair in scores for z in pair]).mean()
        torch.testing.assert_close(loss,expected)
        grads=torch.autograd.grad(loss,[z for p in scores for z in p])
        assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)


def test_only_S_adds_score_loss_and_native_head_is_shared():
    y=torch.zeros(2,1,128,128);y[:,:,30:80,40:90]=1
    for recipe in ['V','T','S']:
        m=Model(cfg(recipe),load_generic=False)
        scores=[(torch.randn(2,1,s,s,requires_grad=True),torch.randn(2,1,max(s//2,1),max(s//2,1),requires_grad=True)) for s in (16,8,4,2)]
        out={'logits':torch.randn_like(y,requires_grad=True),'support_logits':scores}
        primary=BCEDiceLoss()(out['logits'],y)[0]
        loss,_=objective(m,out,y,BCEDiceLoss())
        expected=primary+.1*score_loss(scores,y)[0] if recipe=='S' else primary
        torch.testing.assert_close(loss,expected)
        grads=torch.autograd.grad(loss,[z for p in scores for z in p],allow_unused=True)
        assert all(g is not None for g in grads) if recipe=='S' else all(g is None for g in grads)
        called=[];hook=m.refiner.blocks[0].support.register_forward_hook(lambda mod,args,out:called.append(id(mod)))
        m.refiner(inputs());hook.remove()
        assert called==[id(m.refiner.blocks[0].support)]*2


def test_content_and_condition_inputs_are_separate_and_trainable():
    anchor=torch.randn(512);anchor=anchor/anchor.norm()
    for enabled in (False,True):
        m=MultiScaleRefiner(anchor,enabled);opt=torch.optim.AdamW(m.parameters(),lr=8e-5)
        seen={};calls=[]
        hook=m.blocks[0].value.register_forward_pre_hook(lambda mod,args:calls.append(args[0].shape[1]))
        for step in range(3):
            features=inputs();opt.zero_grad(set_to_none=True)
            output,scores,details=m(features,capture=True)
            if step==0:assert all(torch.equal(x,y) for x,y in zip(features,output))
            # A task surrogate supplies nonconstant spatial and channel gradients.
            loss=sum((x-torch.randn_like(x)).square().mean() for x in output)
            loss.backward()
            for s,b in enumerate(m.blocks):
                for n in ('support','local','context','accept','value','exit','channel','text_projection'):
                    val=sum(float(p.grad.abs().sum()) for p in getattr(b,n).parameters() if p.grad is not None)
                    seen[(s,n)]=max(seen.get((s,n),0),val)
            opt.step()
        hook.remove();assert set(calls)=={64} # only L/K, never score maps
        assert all(v>0 for (s,n),v in seen.items() if n!='text_projection' or enabled)
        if not enabled:assert all(v==0 for (s,n),v in seen.items() if n=='text_projection')
        assert all(0<=float(d['acceptance'].min())<=float(d['acceptance'].max())<=1 for d in details.values())


def test_spatial_diagnostics_known_regions():
    y=torch.zeros(1,1,512,512);y[:,:,100:110,100:110]=1
    z=torch.full_like(y,-10);z[:,:,100:110,100:110]=10;z[:,:,300:302,300:302]=10
    m=ErrorMetrics();m.update(z,y);c=m.compute()['counts']
    assert c['tp']==100 and c['fn']==0 and c['fp_far12']==4 and c['fp_detached']==4
    m=ErrorMetrics();m.update(torch.full_like(y,-10),y);c=m.compute()['counts']
    assert c['fn']==100 and c['fn_inner3']==84 and c['fn_deep']==16
    empty=torch.zeros_like(y);m=ErrorMetrics();m.update(z,empty);c=m.compute()['counts']
    assert c['fp_empty_gt']==104 and c['fp_outer3']==c['fp_outer3_12']==c['fp_far12']==0
