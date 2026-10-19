"""Dual Nano must preserve the MiT run protocol and use independent real encoders."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from models.build_mdt_seg import build_ct_only_model, build_dual_model
from models.components.add_fusion import AddFusion
from models.components.backbones import TimmFeatureBackbone

ROOT = Path(__file__).resolve().parents[1]


def test_dual_nano_launcher_matches_mit_protocol(monkeypatch, tmp_path):
    capture = tmp_path / 'args.json'
    shim = tmp_path / 'python'
    shim.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(dict(argv=sys.argv[1:],omp=os.environ["OMP_NUM_THREADS"],mkl=os.environ["MKL_NUM_THREADS"])))\n')
    shim.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               CAPTURE_ARGS=str(capture))
    from run_full_add_baseline import FullAddConfig
    configs = []
    for encoder in ('mitb1', 'convnextnano'):
        subprocess.run(['bash', str(ROOT/'scripts'/f'run_full_add_{encoder}_lr4e5.sh'),
                        '--hash', 'controlled-run'], env=env, check=True)
        data = json.loads(capture.read_text())
        assert data['omp'] == data['mkl'] == '1'
        assert data['argv'][0] == 'run_full_add_baseline.py'
        monkeypatch.setattr(sys, 'argv', ['x'] + data['argv'][1:])
        configs.append(vars(FullAddConfig.parse_arguments()).copy())
    mit, nano = configs
    assert nano['ct_backbone'] == nano['pet_backbone'] == 'convnextv2_nano'
    assert nano['ct_pretrained_path'] == nano['pet_pretrained_path']
    assert nano['ct_pretrained_path'].endswith('/pretrained/convnextv2_nano')
    for name in ('ct_backbone', 'pet_backbone', 'ct_pretrained_path', 'pet_pretrained_path'):
        mit.pop(name)
        nano.pop(name)
    assert nano == mit
    assert nano['learning_rate'] == 4e-5 and nano['pretrained']
    assert nano['train_batch_mode'] == 'full'
    assert not nano['use_deep_supervision'] and not nano['deep_supervision']


def test_dual_nano_loads_same_source_into_separate_encoders(tmp_path):
    cfg = SimpleNamespace(ct_backbone='convnextv2_nano', pet_backbone='convnextv2_nano',
        pretrained=False, decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False)
    source = build_ct_only_model(cfg)['model'].enc_ct
    path = tmp_path/'nano.pth'
    torch.save(source.state_dict(), path)
    cfg.pretrained = True
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = str(path)
    model = build_dual_model(cfg)['model']
    assert isinstance(model.enc_ct, TimmFeatureBackbone)
    assert isinstance(model.enc_pet, TimmFeatureBackbone)
    assert model.enc_ct is not model.enc_pet
    assert {p.data_ptr() for p in model.enc_ct.parameters()}.isdisjoint(
        {p.data_ptr() for p in model.enc_pet.parameters()})
    for name, value in source.state_dict().items():
        for encoder in (model.enc_ct, model.enc_pet):
            torch.testing.assert_close(encoder.state_dict()[name], value, atol=0, rtol=0)
    assert isinstance(model.fusion, AddFusion)
    assert model.enc_ct.feature_info.channels() == model.enc_pet.feature_info.channels() == [80, 160, 320, 640]
    assert all(p.requires_grad for p in model.parameters())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in model.decoder.modules())
    model.eval()
    with torch.no_grad():
        ct = torch.randn(2, 3, 64, 64)
        output = model(ct, pet=torch.randn_like(ct), forward_mode='full')
    assert output['logits'].shape == (2, 1, 64, 64)
    assert torch.isfinite(output['logits']).all()
    assert output['aux'] == {}
