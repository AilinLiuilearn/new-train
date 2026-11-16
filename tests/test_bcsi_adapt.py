"""BCSI-Adapt: true channel-token axis, correct scatter and trainable hard route."""
from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from variants.bcsi_adapt.front import BCSIAdapt, Direction, Router, Scale
from variants.bcsi_adapt.metrics import SmallTargetMetrics

torch.set_num_threads(2)


@pytest.mark.parametrize('width,selected', [(64,32),(128,64),(320,160),(512,256)])
def test_50_percent_and_unchanged_unselected_channels(width, selected):
    torch.manual_seed(2023)
    layer = Scale(width)
    ct, pet = torch.randn(2,width,8,8), torch.randn(2,width,8,8)
    cc, pp, info = layer(ct,pet,capture=True)
    for modality, raw, updated in [('ct',ct,cc),('pet',pet,pp)]:
        indices=info[modality+'_indices']
        assert indices.shape==(2,selected)
        assert all(len(torch.unique(row))==selected for row in indices)
        mask=torch.zeros(2,width,dtype=torch.bool).scatter(1,indices,True)
        assert torch.equal(raw[~mask],updated[~mask])
        assert info[modality+'_delta'][~mask].abs().max()==0
        attention=info[modality+'_attention']['attention']
        assert attention.shape==(2,selected,selected)
        torch.testing.assert_close(attention.sum(-1),torch.ones(2,selected))
    assert not torch.equal(cc,ct), 'This recipe deliberately has direct nonzero residual initialization'
    layer.eval()
    with torch.no_grad():
        evalc,evalp,_=layer(ct,pet)
    assert torch.equal(cc,evalc) and torch.equal(pp,evalp)


def test_router_st_keeps_hard_values_but_reaches_selected_and_unselected_scores():
    torch.manual_seed(17)
    router=Router(32)
    x=torch.randn(3,32,8,8,requires_grad=True)
    selected,indices,weights,scores=router(x,capture=True)
    reference=x.gather(1,indices[:,:,None,None].expand(-1,-1,8,8))
    assert torch.equal(selected,reference) and (weights==1).all()
    (selected*torch.randn_like(selected)).sum().backward()
    mask=torch.zeros_like(scores,dtype=torch.bool).scatter(1,indices,True)
    assert scores.grad[mask].abs().sum()>0
    assert scores.grad[~mask].abs().sum()>0
    assert sum(float(p.grad.abs().sum()) for p in router.parameters() if p.grad is not None)>0
    assert torch.isfinite(scores.grad).all()


def test_attention_operates_between_channels_and_preserves_full_spatial_values():
    layer=Direction()
    with torch.no_grad():
        layer.q.weight.zero_();layer.q.bias.zero_();layer.k.weight.zero_();layer.k.bias.zero_()
    receiver=torch.randn(2,3,16,24)
    source=torch.randn(2,5,16,24)
    # A high-frequency value that 8x8 Q/K pooling cannot preserve on its own.
    source[:,:,:,::2]+=3
    message,info=layer(receiver,source,capture=True)
    torch.testing.assert_close(message,source.mean(1,keepdim=True).expand(-1,3,-1,-1))
    assert message.shape==receiver.shape and info['attention'].shape==(2,3,5)


def test_scatter_uses_receiver_channel_ids_not_sender_ids():
    layer=Scale(16)
    with torch.no_grad():
        for router in [layer.ct_router,layer.pet_router]:
            for p in router.parameters():p.zero_()
        layer.ct_router.score[-1].bias.copy_(torch.arange(16.))
        layer.pet_router.score[-1].bias.copy_(-torch.arange(16.))
        for direction in [layer.to_ct,layer.to_pet]:
            direction.q.weight.zero_();direction.q.bias.zero_()
            direction.k.weight.zero_();direction.k.bias.zero_()
    ct=torch.randn(2,16,4,4)
    pet=torch.arange(16.).reshape(1,16,1,1).expand(2,16,4,4)
    cc,pp,info=layer(ct,pet,capture=True)
    assert (info['ct_indices']>=8).all() and (info['pet_indices']<8).all()
    torch.testing.assert_close(cc[:,8:]-ct[:,8:],torch.full_like(ct[:,8:],3.5))
    assert torch.equal(cc[:,:8],ct[:,:8])


def test_both_directions_and_routers_receive_task_gradients_without_auxiliary_loss():
    torch.manual_seed(2023)
    layer=Scale(32)
    opt=torch.optim.AdamW(layer.parameters(),lr=8e-5,weight_decay=0)
    ct,pet=torch.randn(2,32,8,8),torch.randn(2,32,8,8)
    goalc,goalp=torch.randn_like(ct),torch.randn_like(pet)
    seen={}
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        cc,pp,_=layer(ct,pet)
        loss=(cc-goalc).square().mean()+(pp-goalp).square().mean()
        loss.backward()
        for name, module in layer.named_children():
            norm=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            seen[name]=max(norm,seen.get(name,0))
        assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
        opt.step()
    assert all(v>0 for v in seen.values()),seen


def test_only_minimal_modules_exist():
    model=BCSIAdapt()
    for block in model.blocks:
        assert set(dict(block.named_children()))=={'ct_router','pet_router','to_ct','to_pet'}
        assert set(dict(block.to_ct.named_children()))=={'q','k'}


def test_small_component_metrics_and_no_positive_case():
    metric=SmallTargetMetrics(max_area=8)
    y=torch.zeros(1,1,16,16)
    y[:,:,1:3,1:3]=1
    y[:,:,8:12,8:12]=1
    z=torch.full_like(y,-10)
    z[:,:,1,1:3]=10
    metric.update(z,y)
    result=metric.compute()
    assert result['small_2d_component_count']==1
    assert result['small_2d_pixel_sensitivity']==.5
    assert result['small_2d_half_coverage_recall']==1.
    assert SmallTargetMetrics().compute()['small_2d_any_overlap_recall'] is None
