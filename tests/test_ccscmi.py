import json
import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO
from variants.ccscmi.front import Scale,TextCondition,CCSCMI
from third_party.ktb_sampling import sample_union


def test_cell_center_sampling_and_identity():
    torch.set_num_threads(2);block=Scale(64,True)
    feature=torch.arange(64.).reshape(1,1,8,8)
    sampled=sample_union(feature,block.grid,block.grid)
    assert torch.equal(sampled[0,:64,0],feature.flatten())
    assert torch.equal(sampled[:,:64],sampled[:,64:])
    c=torch.randn(2,64,16,16);p=torch.randn_like(c)
    cc,pp,d=block(c,p,capture=True)
    assert torch.equal(cc,c) and torch.equal(pp,p)
    assert (d['ct_mask'].sum(1)==32).all() and (d['pet_mask'].sum(1)==32).all()
    assert d['ct_points'].abs().max()<1 and d['ct_attention']['attention_channel_std']==0


@pytest.mark.parametrize('text',[False,True])
def test_task_gradients_open_after_real_updates(text):
    torch.set_num_threads(2);torch.manual_seed(2023)
    block=Scale(64,True);condition=TextCondition(64) if text else None
    roles=torch.randn(2,512);c=torch.randn(2,64,16,16);p=torch.randn_like(c)
    params=list(block.parameters())+(list(condition.parameters()) if condition else [])
    opt=torch.optim.AdamW(params,lr=8e-5,weight_decay=0)
    goalc=torch.randn_like(c);goalp=torch.randn_like(p)
    seen={}
    for step in range(6):
        opt.zero_grad(set_to_none=True)
        extra=condition(c,p,roles) if condition else None
        cc,pp,d=block(c,p,extra,capture=True)
        loss=(cc-goalc).square().mean()+(pp-goalp).square().mean();loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in params if v.grad is not None)
        for name in ['ct_router','pet_router','ct_offset','pet_offset','to_ct','to_pet']:
            norm=sum(float(v.grad.abs().sum()) for v in getattr(block,name).parameters() if v.grad is not None)
            seen[name]=max(seen.get(name,0),norm)
        for name in ['to_ct','to_pet']:
            grad=getattr(block,name).beta.grad;seen[name+'_beta']=max(seen.get(name+'_beta',0),float(grad.abs()))
        for name in ['ct','pet']:
            score=d[name+'_score'];mask=d[name+'_mask'].bool()
            assert (score.grad[~mask].abs()>0).any() if step>0 else True
            assert not d[name+'_delta'][~mask[:,:,None,None].expand_as(d[name+'_delta'])].any()
        if condition:
            seen['condition']=max(seen.get('condition',0),sum(float(v.grad.abs().sum()) for v in condition.parameters() if v.grad is not None))
        opt.step()
    assert all(v>0 for v in seen.values()),seen
    assert (cc-c).abs().max()>0 and (pp-p).abs().max()>0


def test_visual_text_common_initialization_and_rng():
    roles=torch.tensor(json.loads((REPO/'resources/biomedclip_roles.json').read_text())['embeddings'])
    torch.manual_seed(99);a=CCSCMI(roles,False);rng=torch.get_rng_state()
    torch.manual_seed(99);b=CCSCMI(roles,True)
    assert torch.equal(rng,torch.get_rng_state())
    sa,sb=a.state_dict(),b.state_dict()
    assert all(torch.equal(sa[k],sb[k]) for k in sa)
