import io
import torch
from models.ct_full_matrix import MatrixScale


def scale(initialization='full'):
    s=MatrixScale(4,value_mode='lowrank',value_rank=2,value_init=initialization,
        clusters=2,capacity=32,min_candidates=8,min_cluster_samples=2,decay=.5)
    gen=torch.Generator().manual_seed(12)
    x=torch.randn(32,4,generator=gen);x[:16,0]+=4;x[16:,1]+=4
    for b in s.banks:
        b.add(x,1.5*x,torch.ones(32));assert b.refresh()
    return s


def test_identity_gradient():
    s=scale('identity');x=torch.randn(2,4,4,4,requires_grad=True)
    assert torch.allclose(s(x),x,atol=1e-6)
    s(x).square().mean().backward()
    assert x.grad is not None
    for b in s.banks:
        assert b.value_a.grad.abs().sum()>0
        assert b.value_b.grad is not None
        assert not b.keys.requires_grad


def test_learning_and_refresh():
    s=scale();optimizer=torch.optim.AdamW(s.parameters(),lr=.01)
    x=torch.randn(2,4,4,4)
    before=s.foreground_bank.value_a.detach().clone()
    loss=(s(x)-2*x).square().mean();loss.backward()
    for b in s.banks:
        assert b.value_a.grad.abs().sum()>0 and b.value_b.grad.abs().sum()>0
    optimizer.step()
    assert not torch.equal(s.foreground_bank.value_a,before)
    parameters={k:v.detach().clone() for k,v in s.named_parameters()}
    for b in s.banks:assert b.refresh()
    assert all(torch.equal(parameters[k],v) for k,v in s.named_parameters())
    # Raw buffers remain diagnostic snapshots, inference reads trainable factors.
    b=s.foreground_bank
    with torch.no_grad():b.transforms.zero_()
    assert not torch.equal(s(x),x)
    buffer=io.BytesIO();torch.save(dict(model=s.state_dict(),optimizer=optimizer.state_dict()),buffer);buffer.seek(0)
    restored=scale();state=torch.load(buffer,weights_only=False);restored.load_state_dict(state['model'])
    opt=torch.optim.AdamW(restored.parameters(),lr=.01);opt.load_state_dict(state['optimizer'])
    assert torch.equal(s(x),restored(x))
    assert torch.equal(s(x,read_mode='off'),x)


def test_all_scales():
    from models.ct_full_matrix import CTFullMatrixMemory
    memory=CTFullMatrixMemory((4,8,12,16),value_mode='lowrank',value_rank=2,value_init='identity',
        warmup_steps=0,clusters=2,capacity=32,min_candidates=8,min_cluster_samples=2)
    for s in memory.scales:
        for b in s.banks:
            b.keys.copy_(torch.nn.functional.normalize(torch.randn_like(b.keys),dim=1));b.valid.fill_(True)
    inputs=[torch.randn(2,c,4,4,requires_grad=True) for c in (4,8,12,16)]
    sum(x.square().mean() for x in memory(inputs)).backward()
    assert all(s.foreground_bank.value_a.grad is not None for s in memory.scales)
    memory.eval();before={k:v.clone() for k,v in memory.state_dict().items()}
    memory(inputs)
    assert all(torch.equal(v,before[k]) for k,v in memory.state_dict().items())


if __name__=='__main__':
    test_identity_gradient();test_learning_and_refresh();test_all_scales()
    print('low-rank gradient, initialization, refresh, four-scale and resume tests passed')
