"""Exercise both full trainer control paths with a tiny CPU fixture, no GPU contention."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO
from variants.v31 import train as runner
from variants.v31.losses import collate
from utils.ema import ModelEMA


@pytest.mark.parametrize('phase,generic',[('joint',False),('joint',True),('correction',False)])
def test_complete_stage_control_path(phase,generic,tmp_path,monkeypatch):
    torch.set_num_threads(2)
    class TinyJoint(torch.nn.Module):
        def __init__(self,*a,**kw):
            super().__init__();self.base=torch.nn.Conv2d(3,1,1)
            self.ct_expert=torch.nn.Conv2d(3,1,1);self.pet_expert=torch.nn.Conv2d(3,1,1)
            self.base.enc_ct=torch.nn.Identity();self.base.enc_pet=torch.nn.Identity()
        def import_m1(self,ck):pass
        def forward(self,ct,pet,with_feature=False,capture=False):
            return {'logits':self.base(ct+pet),'ct_distance':self.ct_expert(ct).tanh(),
                    'pet_logits':self.pet_expert(pet),'feature':ct.mean(1,keepdim=True).expand(-1,64,-1,-1),'scales':{}}
    init=tmp_path/'input.pt';torch.save({'fixture':True},init)
    out=tmp_path/'run';out.mkdir()
    cfg=json.loads((REPO/'configs/v31.json').read_text());cfg.update(epochs=1,mixed_precision=False,ct_pretrained_path="fixture",pet_pretrained_path="fixture")
    monkeypatch.setattr(runner,'prepare_run',lambda *a:(out,argparse.Namespace(**cfg),{}))
    monkeypatch.setattr(runner,'JointModel',TinyJoint)
    monkeypatch.setattr(runner,'load_joint',lambda p:(TinyJoint(),{}))
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self:self)
    monkeypatch.setattr(runner,'ModelEMA',lambda m,*a,**kw:ModelEMA(m,.999,device='cpu'))
    monkeypatch.setattr(runner,'move',lambda b,device:b)
    monkeypatch.setattr(runner,'autocast_context',lambda *a:nullcontext())
    monkeypatch.setattr(runner,'gpu_lock_path',lambda:tmp_path/'gpu.lock')
    monkeypatch.setattr(runner,'evaluate',lambda *a,**kw:{'full':{'dice':.75,'hd95':12.}})
    items=[{'ct':torch.randn(3,16,16),'pet':torch.randn(3,16,16),
            'mask':(torch.rand(1,16,16)>.8).float(),'is_full':True} for _ in range(2)]
    b=collate(items)
    class Loader:
        def __len__(self):return 1
        def __iter__(self):yield b
    monkeypatch.setattr(runner,'loaders',lambda *a:(Loader(),Loader(),Loader()))
    import models.components.backbones as backbones
    monkeypatch.setattr(backbones,'load_local_weights_safe',lambda *a,**kw:None)
    args=argparse.Namespace(init='generic' if generic else str(init),phase=phase)
    runner.main(args)
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert json.loads((out/'RESULTS.json').read_text())['complete']
    ck=torch.load(str(out/'best.pt'),map_location='cpu',weights_only=False)
    assert ck['format']==('v31_joint_v1' if phase=='joint' else 'v31_correction_v1')
    assert not any(k in ck for k in ['optimizer','scheduler','scaler','rng','model_ema'])
    assert not (out/'last.pt').exists()
    if phase=='correction':assert all(not k.startswith('base.') for k in ck['weights'])
