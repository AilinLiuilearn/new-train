import copy
import json
from pathlib import Path
import sys
import torch
import pytest
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO
from variants.gt_vcl.model import Model
from variants.gt_vcl.losses import calibration_loss,objective
from variants.gt_vcl.semantic import SemanticRegularizer
from variants.gt_vcl.inference import load_model
from variants.tsbci.model import SHCIPETCT
from models.components.group_unet_decoder import UNetStyleDecoder
from utils.seg_losses import BCEDiceLoss

torch.set_num_threads(2)


def config(recipe):
    return json.loads((REPO/'configs'/('gt_vcl_'+recipe+'.json')).read_text())


def test_calibration_stops_ct_gradient_and_moves_full_toward_gt():
    y=torch.tensor([[[[1.,0.]]]])
    c=torch.logit(torch.tensor([[[[.9,.1]]]])).requires_grad_()
    f=torch.logit(torch.tensor([[[[.1,.9]]]])).requires_grad_()
    loss,_=calibration_loss(c,f,y)
    gc,gf=torch.autograd.grad(loss,(c,f),allow_unused=True)
    assert gc is None and gf[0,0,0,0]<0 and gf[0,0,0,1]>0
    assert abs(float(loss)-.75)<2e-6
    wrong_c=-c.detach()
    zero,_=calibration_loss(wrong_c,f,y)
    assert float(zero)==0 and torch.isfinite(zero)


def test_calibration_balances_classes_and_uses_current_online_reference():
    def loss_with_bg(n):
        y=torch.cat([torch.ones(1),torch.zeros(n)]).reshape(1,1,1,-1)
        pc=torch.where(y.bool(),.9,.1);pf=torch.where(y.bool(),.6,.4)
        return calibration_loss(torch.logit(pc),torch.logit(pf),y)[0]
    torch.testing.assert_close(loss_with_bg(1),loss_with_bg(1000),atol=2e-6,rtol=0)
    y=torch.ones(1,1,1,1);f=torch.logit(torch.full_like(y,.7))
    low,_=calibration_loss(torch.logit(torch.full_like(y,.55)),f,y)
    high,_=calibration_loss(torch.logit(torch.full_like(y,.95)),f,y)
    assert low==0 and high>0


def test_semantic_chunk_matches_dense_values_and_gradients():
    torch.manual_seed(9)
    chunked=SemanticRegularizer(torch.randn(2,512),chunk_size=3).train()
    dense=copy.deepcopy(chunked)
    a=torch.randn(2,64,4,4,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    y=torch.zeros(2,1,8,8);y[:,:,1:5,2:6]=1
    f,bg,valid=chunked.region_vectors(a,y,0)
    z=F.normalize(dense.projections[0](b.flatten(2).transpose(1,2)),dim=-1,eps=1e-6)
    weights=F.interpolate(y,size=(4,4),mode='area').flatten(1).unsqueeze(-1)
    df=F.normalize((z*weights).sum(1)/weights.sum(1),dim=-1,eps=1e-6)
    db=F.normalize((z*(1-weights)).sum(1)/(1-weights).sum(1),dim=-1,eps=1e-6)
    torch.testing.assert_close(f,df,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(bg,db,atol=1e-6,rtol=1e-6)
    ga=torch.autograd.grad((f.square()+bg).sum(),(a,chunked.projections[0].weight))
    gb=torch.autograd.grad((df.square()+db).sum(),(b,dense.projections[0].weight))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-5)
    empty=torch.zeros_like(y)
    loss,stats=chunked((a,a),empty)
    assert loss==0 and stats['ct_sem_valid_fraction']==0
    assert torch.isfinite(torch.autograd.grad(loss,a)[0]).all()


def test_decoder_feature_flag_and_checkpoint_recomputation_are_identical():
    cfg=config('E4');torch.manual_seed(2023)
    model=Model(cfg,load_generic=False).train()
    features=[torch.randn(2,d,s,s,requires_grad=True) for d,s in [(64,32),(128,16),(320,8),(512,4)]]
    plain=model.decoder_sep(features,(128,128))['logits']
    with_feature=model.decoder_sep(features,(128,128),return_features=True)
    assert torch.equal(plain,with_feature['logits'])
    checkpointed,h=model.decode_single(features,(128,128))
    torch.testing.assert_close(checkpointed,plain,atol=0,rtol=0)
    a=torch.autograd.grad(checkpointed.square().mean()+h.square().mean(),features,retain_graph=False)
    b=torch.autograd.grad(plain.square().mean()+with_feature['features'].square().mean(),features)
    for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-5)


def test_initialization_shared_decoder_and_inference_pruning(tmp_path):
    torch.manual_seed(2023);base=SHCIPETCT(config('E1'),arm='m1',load_generic=False).eval()
    common={k:v.clone() for k,v in base.state_dict().items()};rng=torch.get_rng_state().clone()
    ct,pet=torch.randn(1,3,256,256),torch.randn(1,3,256,256)
    with torch.no_grad():reference=base(ct,pet)['logits']
    auxiliary=None
    for recipe in ['E1','E2','E4']:
        torch.manual_seed(2023);model=Model(config(recipe),load_generic=False).eval()
        assert torch.equal(torch.get_rng_state(),rng)
        state=model.state_dict()
        assert all(torch.equal(state[k],v) for k,v in common.items())
        sep={k:v for k,v in state.items() if k.startswith('decoder_sep.')}
        if auxiliary is None:auxiliary={k:v.clone() for k,v in sep.items()}
        assert all(torch.equal(auxiliary[k],v) for k,v in sep.items())
        assert not ({p.data_ptr() for p in model.decoder_sep.parameters()}&{p.data_ptr() for p in model.decoder_full.parameters()})
        calls=[];hook=model.decoder_sep.register_forward_pre_hook(lambda module,args:calls.append(id(module)))
        with torch.no_grad():out=model(ct,pet)
        hook.remove()
        assert calls==[id(model.decoder_sep)]*2
        assert torch.equal(out['logits'],reference)
        if recipe=='E4':
            path=tmp_path/'best.pt'
            torch.save({'format':'gt_vcl_inference_v1','config':model.cfg,'weights':model.state_dict()},path)
            deployed=load_model(path,'cpu')
            assert not hasattr(deployed,'decoder_sep') and not hasattr(deployed,'semantic')
            with torch.no_grad():actual=deployed(ct,pet)['logits']
            assert torch.equal(actual,reference)


def test_three_task_losses_have_exact_unit_weights_and_branch_gradient_routes():
    model=Model(config('E1'),load_generic=False).train()
    ct,pet=torch.randn(2,3,256,256),torch.randn(2,3,256,256)
    y=torch.zeros(2,1,256,256);y[:,:,60:180,60:180]=1
    out=model(ct,pet);criterion=BCEDiceLoss()
    loss,stats=objective(model,out,y,criterion)
    expected=sum(criterion(out[k].float(),y)[0] for k in ['logits','ct_logits','pet_logits'])
    torch.testing.assert_close(loss,expected)
    p_ct=next(model.enc_ct.parameters());p_pet=next(model.enc_pet.parameters())
    p_sep=next(model.decoder_sep.parameters());p_f=next(model.decoder_full.parameters())
    single=criterion(out['ct_logits'].float(),y)[0]
    grads=torch.autograd.grad(single,(p_ct,p_pet,p_sep,p_f),allow_unused=True,retain_graph=True)
    assert grads[0] is not None and grads[0].abs().sum()>0
    assert grads[1] is None and grads[3] is None and grads[2].abs().sum()>0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
