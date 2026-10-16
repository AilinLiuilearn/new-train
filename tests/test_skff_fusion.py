"""Contract tests for original SKFF selection and the existing Full runner."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import run_full_cmx_fusion as runner
from run_full_skff_fusion import FullSKFFConfig
from configs.full_cmx import FullCMXConfig
from eval_full_cmx_fusion import load_full_checkpoint
from models.components.skff_fusion import SKFFFusion, STAGE_CHANNELS, UPSTREAM_BLOB
from models.components.skff_source import SKFF
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_cmx_fusion import FullCMXPETCT, build_full_cmx_model
from tasks.mdt_seg import MDTSegTeacher


def cfg_for(monkeypatch, tmp_path, config_cls=FullSKFFConfig):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false', '--require_cuda', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', 'skff-test', '--batch_size', '2',
        '--epochs', '1', '--cosine_warmup', '0', '--mixed_precision', 'false', '--hd95_mode', 'surface'])
    cfg = config_cls.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = cfg.model_arch = 'full_cmx_fusion'
    cfg.fusion_type = 'SKFFFusion' if cfg.fusion_backend == 'skff' else 'CMXFusion'
    return cfg


def data():
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:32, 20:32] = 1
    return dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=mask)


def test_upstream_source_and_license_byte_identity():
    root = Path(__file__).resolve().parents[1]
    for name, expected in {
        'models/components/skff_source.py': UPSTREAM_BLOB,
        'third_party/MIRNetv2_LICENSE.md': '610e48de827921d0590f7709da8e358d1554de3f'}.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected


def test_original_initialization_parameters_and_no_extra_layers():
    torch.manual_seed(2023)
    fusion = SKFFFusion()
    torch.manual_seed(2023)
    originals = nn.ModuleList(SKFF(c, height=2, reduction=8, bias=False) for c in STAGE_CHANNELS)
    for block, original in zip(fusion.skffs, originals):
        for key, value in original.state_dict().items():
            assert torch.equal(value, block.state_dict()[key])
    assert sum(p.numel() for p in fusion.parameters()) == 144384
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in fusion.modules())
    assert all(m.bias is None for m in fusion.modules() if isinstance(m, nn.Conv2d))


def test_actual_feature_scales_exact_original_output_and_input_gradients():
    fusion = SKFFFusion()
    reference = copy.deepcopy(fusion.skffs)
    cs = [torch.randn(2, c, s, s, requires_grad=True) for c, s in zip(STAGE_CHANNELS, (128,64,32,16))]
    ps = [torch.randn_like(c, requires_grad=True) for c in cs]
    cr = [c.detach().clone().requires_grad_() for c in cs]
    pr = [p.detach().clone().requires_grad_() for p in ps]
    actual = fusion(cs, ps)
    expected = [m([c,p]) for m,c,p in zip(reference,cr,pr)]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert a.shape == b.shape
    sum(x.square().mean() for x in actual).backward()
    sum(x.square().mean() for x in expected).backward()
    for a,b in zip(cs+ps,cr+pr):
        torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    for m in fusion.skffs:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(float(p.grad.abs().sum()) > 0 for p in m.parameters())


def test_uniform_selection_is_average_not_sum_and_has_no_external_residual():
    fusion = SKFFFusion()
    with torch.no_grad():
        for m in fusion.skffs:
            for fc in m.fcs: fc.weight.zero_()
    cs = [torch.ones(1,c,4,4) for c in STAGE_CHANNELS]
    ps = [torch.full_like(c,3) for c in cs]
    for x in fusion(cs,ps):
        torch.testing.assert_close(x,torch.full_like(x,2),atol=0,rtol=0)


def test_pet_interpolation_matches_baseline():
    fusion = SKFFFusion()
    cs = [torch.randn(1,c,8,8) for c in STAGE_CHANNELS]
    ps = [torch.randn(1,c,4,4) for c in STAGE_CHANNELS]
    for a,m,c,p in zip(fusion(cs,ps),fusion.skffs,cs,ps):
        expected = m([c,F.interpolate(p,size=c.shape[-2:],mode='bilinear',align_corners=False)])
        torch.testing.assert_close(a,expected,atol=0,rtol=0)


@pytest.mark.parametrize('bad', ['count','channel','dtype','empty'])
def test_invalid_features_rejected(bad):
    fusion = SKFFFusion()
    cs = [torch.randn(1,c,4,4) for c in STAGE_CHANNELS]
    ps = [torch.randn_like(c) for c in cs]
    if bad == 'count': ps.pop()
    if bad == 'channel': ps[0] = torch.randn(1,63,4,4)
    if bad == 'dtype': ps[0] = ps[0].double()
    if bad == 'empty': ps[0] = ps[0][...,:0]
    with pytest.raises(ValueError): fusion(cs,ps)


def test_shared_baseline_weights_rng_and_selected_backend():
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullCMXPETCT(pretrained=False, fusion_backend='skff')
    assert isinstance(model.fusion,SKFFFusion)
    assert torch.equal(baseline_rng,torch.get_rng_state())
    for k,v in baseline.state_dict().items():
        assert torch.equal(v,model.state_dict()[k]), k
    assert sum(p.numel() for p in model.parameters()) == 32587025


def test_cli_defaults_selection_and_incompatible_frm(monkeypatch,tmp_path):
    cfg = cfg_for(monkeypatch,tmp_path)
    assert cfg.fusion_backend == 'skff' and not cfg.cmx_ffm_enabled and not cfg.cmx_frm_enabled
    assert cfg.skff_output_scale == 1 and cfg.skff_initialization == 'official'
    cfg = cfg_for(monkeypatch,tmp_path,FullCMXConfig)
    assert cfg.fusion_backend == 'cmx' and cfg.cmx_ffm_enabled
    args = sys.argv + ['--fusion_backend','skff']
    monkeypatch.setattr(sys,'argv',args)
    selected = FullCMXConfig.parse_arguments()
    assert selected.fusion_backend == 'skff' and not selected.cmx_ffm_enabled
    monkeypatch.setattr(sys,'argv',args+['--cmx_frm_enabled','true'])
    with pytest.raises(ValueError,match='replaces CMX'): FullCMXConfig.parse_arguments()


def test_actual_two_optimizer_updates_main_loss_only(monkeypatch,tmp_path):
    cfg = cfg_for(monkeypatch,tmp_path)
    task = MDTSegTeacher(build_full_cmx_model(cfg),cfg)
    ids = {id(p) for group in task.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in task.model.parameters())
    batch = data()
    first = [p.detach().clone() for p in task.model.fusion.parameters()]
    for _ in range(2):
        task.optimizer.zero_grad(set_to_none=True)
        loss,out,stats = runner.train_step_full(task,batch)
        main,_ = task.criterion(out['logits'],batch['mask'])
        torch.testing.assert_close(loss,main,atol=0,rtol=0)
        assert not out['aux'] and stats['num_samples'] == 2
        loss.backward()
        for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder','fusion'):
            grads = [p.grad for p in getattr(task.model,name).parameters()]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads),name
            assert sum(float(g.abs().sum()) for g in grads) > 0,name
        task.optimizer.step()
    assert all(not torch.equal(a,b) for a,b in zip(first,task.model.fusion.parameters()))


def test_standalone_train_val_ema_final_and_strict_reload(monkeypatch,tmp_path):
    cfg = cfg_for(monkeypatch,tmp_path)
    cfg.ema_enabled,cfg.ema_start_epoch = True,0
    monkeypatch.setattr(FullSKFFConfig,'parse_arguments',classmethod(lambda cls:cfg))
    batch = data()
    monkeypatch.setattr(runner,'_loaders',lambda cfg:([batch],[batch],[batch]))
    runner.main(config_cls=FullSKFFConfig)
    folder = Path(cfg.checkpoint_dir)
    result = json.loads((folder/'final_test_full.json').read_text())
    assert result['fusion_backend'] == 'skff' and result['fusion_type'] == 'SKFFFusion'
    assert not result['cmx_ffm_enabled'] and result['weights'] == 'ema'
    with (folder/'train_log.csv').open() as f:
        row = next(csv.DictReader(f))
    assert float(row['grad_fusion']) > 0 and float(row['grad_pet_align']) > 0
    path = folder/'ckpt.best_full.pth.tar'
    for choice in (None,False):
        loaded,saved,ckpt,tag = load_full_checkpoint(str(path),use_ema=choice)
        assert isinstance(loaded.fusion,SKFFFusion) and saved.skff_output_scale == 1
        key = 'model_ema' if tag == 'ema' else 'model'
        for k,v in ckpt[key].items(): assert torch.equal(v,loaded.state_dict()[k])
    ckpt['config']['skff_output_scale'] = 2
    torch.save(ckpt,path)
    with pytest.raises(ValueError,match='SKFF architecture/source'): load_full_checkpoint(str(path))


def test_legacy_cmx_checkpoint_without_backend_is_still_cmx(monkeypatch,tmp_path):
    cfg = cfg_for(monkeypatch,tmp_path,FullCMXConfig)
    del cfg.fusion_backend
    task = MDTSegTeacher(build_full_cmx_model(cfg),cfg)
    path = tmp_path/'legacy.pth.tar'
    task.save_checkpoint(str(path),1,.5,1,{'dice':.5})
    loaded,saved,_,_ = load_full_checkpoint(str(path))
    assert saved.fusion_backend == loaded.fusion_backend == 'cmx'
    assert loaded.fusion.ffm_enabled


@pytest.mark.parametrize('mode,state,pet', [('missing',None,True),('auto',[1,0],True),
    ('full',[0,1],True),('full',None,False)])
def test_missing_rejected_before_both_encoders(mode,state,pet):
    model = FullCMXPETCT(pretrained=False,fusion_backend='skff')
    def forbidden(*args): raise AssertionError('encoder must not run')
    model.enc_ct.register_forward_pre_hook(forbidden)
    model.enc_pet.register_forward_pre_hook(forbidden)
    x = torch.randn(2,3,64,64)
    with pytest.raises(ValueError): model(x,pet=x if pet else None,forward_mode=mode,pet_available=state)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_amp_source_forward_backward():
    fusion = SKFFFusion().cuda()
    cs = [torch.randn(2,c,s,s,device='cuda',requires_grad=True)
          for c,s in zip(STAGE_CHANNELS,(128,64,32,16))]
    ps = [torch.randn_like(c,requires_grad=True) for c in cs]
    with torch.autocast('cuda',dtype=torch.float16):
        loss = sum(x.float().square().mean() for x in fusion(cs,ps))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
