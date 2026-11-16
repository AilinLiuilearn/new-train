"""Regression for the real trainer control path, with tiny CPU tensors and model."""
import argparse
import json
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from variants.tsbci import train as engine
from runtime import REPO


@pytest.mark.parametrize('with_auxiliary', [False, True])
def test_one_epoch_control_path_writes_logs_and_best_only(tmp_path,monkeypatch,with_auxiliary):
    cfg=json.loads((REPO/'configs/rerun_m1_p3.json').read_text())
    cfg.update(epochs=1,device='cpu',mixed_precision=False,ct_pretrained_path=None,pet_pretrained_path=None)
    cfg=argparse.Namespace(**cfg)
    monkeypatch.setattr(engine,'DEVICE',torch.device('cpu'))
    monkeypatch.setattr(engine,'prepare_run',lambda args,variant:(tmp_path,cfg,{'generic_weights':{}}))
    monkeypatch.setattr(engine,'gpu_lock_path',lambda:tmp_path/'gpu.lock')
    monkeypatch.setattr(engine,'split_signature',lambda cfg:{'train':'fixture','val':'fixture'})
    class Tiny(torch.nn.Module):
        def __init__(self,*a,**kw):
            super().__init__();self.decoder_full=torch.nn.Sequential(torch.nn.Conv2d(3,1,1),torch.nn.GroupNorm(1,1))
        def forward(self,ct,pet,capture=False):return {'logits':self.decoder_full(ct+pet)}
    # The unchanged official HD95 implementation deliberately requires 512x512.
    batch={'ct':torch.randn(2,3,512,512),'pet':torch.randn(2,3,512,512),
           'mask':torch.zeros(2,1,512,512),'is_full':torch.ones(2,dtype=torch.bool)}
    batch['mask'][:,:,128:384,128:384]=1
    monkeypatch.setattr(engine,'loaders',lambda cfg,out:([batch],[batch],[batch]))
    def write_results(rows,complete=False):engine.dump('RESULTS.json',{'complete':complete,'rows':rows})
    calls=[]
    def custom_objective(o,y,criterion):
        calls.append(torch.is_grad_enabled())
        loss,parts=engine.objective(o,y,criterion)
        auxiliary=o['logits'].float().square().mean()
        total=loss+0.1*auxiliary
        return total,{**parts,'total':float(total.detach()),'test_auxiliary':float(auxiliary.detach())}
    args=argparse.Namespace(arm='m1')
    engine.main(args,variant='m1',model_factory=Tiny,results_fn=write_results,
                diagnostics_fn=lambda *a:engine.log('diagnostics.jsonl',{'ok':True}),
                objective_fn=custom_objective if with_auxiliary else None,
                protocol_extra={'loss':'test hook'} if with_auxiliary else None)
    assert json.loads((tmp_path/'status.json').read_text())['state']=='complete'
    assert json.loads((tmp_path/'RESULTS.json').read_text())['complete']
    assert (tmp_path/'best.pt').exists() and (tmp_path/'train.jsonl').exists()
    assert (tmp_path/'protocol.json').exists() and (tmp_path/'diagnostics.jsonl').exists()
    assert not (tmp_path/'last.pt').exists()
    if with_auxiliary:
        assert calls==[True,False], 'Custom objective must run in both train and validation'
        assert json.loads((tmp_path/'protocol.json').read_text())['loss']=='test hook'
        row=json.loads((tmp_path/'train.jsonl').read_text())
        assert 'test_auxiliary' in row['train'] and 'test_auxiliary' in row['validation']['losses']
