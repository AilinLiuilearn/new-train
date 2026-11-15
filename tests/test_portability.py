"""Check machine separation and best-only policy, rather than model architecture claims."""
from pathlib import Path
import argparse
import ast
import importlib
import json
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime


def test_relocation_and_shared_split(tmp_path):
    paths=tmp_path/'paths.local.json'
    paths.write_text(json.dumps({'data_root':str(tmp_path/'images'),'pretrained_root':str(tmp_path/'weights'),
                                'output_root':str(tmp_path/'results')}))
    args=argparse.Namespace(paths=str(paths),config=str(runtime.REPO/'configs/tsbci.json'),
                            run='card2-tsbci',output=None,arm='full')
    cfg,out,loc=runtime.resolve_config(args,'tsbci')
    assert out==tmp_path/'results/card2-tsbci'
    assert cfg.ct_pretrained_path==str(tmp_path/'weights/convnextv2_nano')
    assert cfg.val_split_file==cfg.test_split_file
    assert cfg.random_state==2023 and cfg.learning_rate==8e-5 and cfg.image_size_2d==512
    assert cfg.save_last_checkpoint is False
    args.run='../old_run'
    with pytest.raises(ValueError):runtime.resolve_config(args,'tsbci')


@pytest.mark.parametrize('variant',['tsbci','a2'])
def test_checkpoint_policy(variant,tmp_path,monkeypatch):
    trainer=importlib.import_module('variants.'+variant+'.train')
    monkeypatch.setattr(trainer,'OUT',tmp_path)
    trainer.save('best.pt',{'weights':{'dummy':torch.ones(1)}})
    assert torch.equal(torch.load(tmp_path/'best.pt',weights_only=True)['weights']['dummy'],torch.ones(1))
    with pytest.raises(RuntimeError):trainer.save('last.pt',{})
    tree=ast.parse((runtime.REPO/'variants'/variant/'train.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save']
    assert calls and all(n.args[0].value=='best.pt' for n in calls)
    assert not (tmp_path/'last.pt').exists()


def test_two_role_vectors_are_frozen_inputs():
    raw=json.loads((runtime.REPO/'resources/biomedclip_roles.json').read_text())
    text=torch.tensor(raw['embeddings'])
    assert text.shape==(2,512) and torch.isfinite(text).all()
    assert not text.requires_grad
    assert torch.allclose(text.norm(dim=1),torch.ones(2),atol=1e-5)


def test_reject_base_python(monkeypatch):
    monkeypatch.setattr(sys,'prefix','/anywhere/miniconda3')
    with pytest.raises(RuntimeError,match='myenv'):runtime.require_myenv()
