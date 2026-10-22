import copy
import json
import numpy as np
import torch
from torch import nn
from scripts.compare_raw_ema import state_difference
from scripts.analyze_raw_ema import dice, paired_bootstrap
from utils.ema import ModelEMA


def test_ema_is_side_copy_and_does_not_change_raw_optimizer_trajectory():
    torch.manual_seed(2023)
    raw=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Linear(4,1))
    plain=copy.deepcopy(raw)
    ema=ModelEMA(raw,decay=.999)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-3) for m in (raw,plain)]
    for _ in range(5):
        x,y=torch.randn(8,3),torch.randn(8,1)
        for model,opt in zip((raw,plain),optimizers):
            opt.zero_grad();(model(x)-y).square().mean().backward();opt.step()
        ema.update(raw)
        assert all(torch.equal(v,plain.state_dict()[k]) for k,v in raw.state_dict().items())
    assert not ema.model.training
    assert all(not p.requires_grad and p.grad is None for p in ema.model.parameters())
    assert any(not torch.equal(v,ema.state_dict()[k]) for k,v in raw.state_dict().items())


def test_ema_float_buffers_averaged_and_integer_buffers_copied():
    source=nn.BatchNorm1d(2)
    ema=ModelEMA(source,decay=.9,warmup=False)
    with torch.no_grad():
        source.weight.fill_(3);source.running_mean.fill_(10);source.num_batches_tracked.fill_(7)
    before={k:v.clone() for k,v in source.state_dict().items()}
    ema.update(source)
    assert torch.allclose(ema.model.weight,torch.full((2,),1.2))
    assert torch.equal(ema.model.running_mean,torch.ones(2))
    assert ema.model.num_batches_tracked.item()==7
    assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
    diff=state_difference(source,source.state_dict(),ema.state_dict())
    assert diff['parameters']['changed_elements']>0
    assert diff['floating_buffers']['changed_elements']>0
    assert diff['integer_buffers']['changed_elements']==0


def test_paired_patient_bootstrap_known_improvement(tmp_path):
    report={'results':{}}
    for tag,tp,fp,fn,score in [('raw',1,1,1,.5),('ema',2,0,0,1.)]:
        rows=[{'image_id':f'{patient}_slice','case_id':patient,
               **{mode:dict(tp=tp,fp=fp,fn=fn) for mode in ('full','missing')}} for patient in ('a','b')]
        (tmp_path/(tag+'-samples.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report['results'][tag]={mode:{'dice':score} for mode in ('full','missing')}
    result=paired_bootstrap(tmp_path,report,repeats=100)
    assert result['patients']==2
    assert result['ema_minus_raw_percentile_95ci']['joint']==[.5,.5]
    assert dice(np.array([0.,0.,0.]))==1
