import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO
from variants.v31.model import Corrector,CorrectedModel,decode_with_feature
from variants.v31.losses import targets,balanced_gate,protection,objective
from models.components.group_unet_decoder import UNetStyleDecoder


def test_targets_empty_full_and_small_lesion():
    for full,sign in [(False,-1),(True,1)]:
        t=targets(np.full((20,20),full))
        assert (t['ct_target']==sign).all() and not t['near_bg'].any()
        assert (t['ct_weight']==1).all()
    y=np.zeros((20,20),bool);y[8:12,8:12]=True;t=targets(y)
    assert t['pet_target'].sum()>y.sum()
    mask=torch.from_numpy(y)
    assert (t['ct_target'][0][mask]>0).all() and (t['ct_target'][0][~mask]<0).all()
    assert (t['near_bg'][0][mask]==0).all()


def test_balanced_losses_no_empty_mask_nan():
    z=torch.zeros(2,1,8,8,requires_grad=True)
    for error in [torch.zeros_like(z,dtype=torch.bool),torch.ones_like(z,dtype=torch.bool)]:
        loss=balanced_gate(z,error);assert torch.isfinite(loss);loss.backward()
    zero=protection(z,torch.zeros_like(z),torch.zeros_like(z),torch.zeros_like(z))
    assert zero==0;zero.backward();assert torch.isfinite(z.grad).all()


def test_decoder_feature_extraction_matches_existing_forward():
    torch.set_num_threads(2);d=UNetStyleDecoder().eval()
    xs=[torch.randn(1,c,s,s) for c,s in zip([64,128,320,512],[32,16,8,4])]
    with torch.no_grad():a=d(xs,(128,128))['logits'];b,f=decode_with_feature(d,xs,(128,128))
    assert torch.equal(a,b) and f.shape==(1,64,32,32)


def test_zero_init_and_consensus_stop_gradient():
    torch.set_num_threads(2);m=Corrector()
    args=[torch.randn(2,64,16,16),torch.randn(2,1,64,64),torch.randn(2,1,64,64),torch.randn(2,1,64,64).tanh()]
    o=m(*args);assert torch.equal(o['logits'],args[1])
    o['gate_logits'].sum().backward()
    assert all(p.grad is None for p in m.consensus.parameters())
    assert any(p.grad is not None for p in m.gate.parameters())
    m.zero_grad(set_to_none=True);torch.nn.init.normal_(m.residual[-1].weight,std=.01)
    o=m(*args);b={'mask':(torch.rand_like(args[1])>.7).float(),'near_bg':torch.ones_like(args[1])}
    loss,_=objective(o,b,'correction');loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.consensus.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_frozen_model_stays_eval_and_has_no_gt_input():
    import inspect
    class Frozen(torch.nn.Module):
        def __init__(self):super().__init__();self.bn=torch.nn.BatchNorm2d(1)
    m=CorrectedModel(Frozen());m.train()
    assert not m.base.training and not m.base.bn.training
    assert not any(p.requires_grad for p in m.base.parameters())
    assert 'gt' not in inspect.signature(m.forward).parameters


def test_role_and_correction_counters_use_actual_predictions():
    from variants.v31.metrics import RoleMetrics
    y=torch.zeros(1,1,16,16);y[:,:,4:12,4:12]=1
    b={k:v.unsqueeze(0) for k,v in targets(y.numpy()).items()};b['mask']=y
    after=(y*2-1)*5;before=after.clone();before[:,:,0,0]=5
    gate=torch.full_like(y,.1);gate[:,:,0,0]=.9
    o={'logits':after,'base_logits':before,'gate':gate,'delta':after-before,'change':after-before,
       'ct_distance':b['ct_target'],'pet_logits':after}
    metric=RoleMetrics();metric.update(o,b);r=metric.compute()
    assert r['all']['net_repaired_pixels']==1 and r['all']['harmful_rate']==0
    assert r['pet_candidate_precision']==1 and r['pet_candidate_recall']==1
    assert r['ct_band_distance_MAE_pixels']==0 and r['ct_boundary_F1_tolerance3']==1
    assert r['gate_pr_auc_hist2048']==1
