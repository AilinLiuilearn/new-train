"""Regression for the real trainer control path, with tiny CPU tensors and model."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from variants.tsbci import train as engine
from runtime import REPO


def test_one_epoch_control_path_writes_logs_and_best_only(tmp_path,monkeypatch):
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
    args=argparse.Namespace(arm='m1')
    engine.main(args,variant='m1',model_factory=Tiny,results_fn=write_results,
                diagnostics_fn=lambda *a:engine.log('diagnostics.jsonl',{'ok':True}))
    assert json.loads((tmp_path/'status.json').read_text())['state']=='complete'
    assert json.loads((tmp_path/'RESULTS.json').read_text())['complete']
    assert (tmp_path/'best.pt').exists() and (tmp_path/'train.jsonl').exists()
    assert (tmp_path/'protocol.json').exists() and (tmp_path/'diagnostics.jsonl').exists()
    assert not (tmp_path/'last.pt').exists()
