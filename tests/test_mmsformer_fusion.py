"""Native MMSFormer FusionBlock source, gradients and integration contracts."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import run_full_cmx_fusion as runner
from configs.full_mmsformer import FullMMSFormerConfig
from eval_full_mmsformer import load_full_checkpoint
from models.components.mmsformer_fusion import MMSFormerFusion, STAGE_CHANNELS, UPSTREAM_BLOB, LICENSE_BLOB
from models.components.mmsformer_source import FusionBlock
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_mmsformer_fusion import FullMMSFormerPETCT, build_full_mmsformer_model


def features(actual=False):
    sizes = (128, 64, 32, 16) if actual else (8, 4, 2, 2)
    cs = [torch.randn(2, c, s, s, requires_grad=True) for c, s in zip(STAGE_CHANNELS, sizes)]
    return cs, [torch.randn_like(c, requires_grad=True) for c in cs]


def archived_reference():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'third_party/MMSFormer/mmsformer.py').read_text())
    wanted = {'ChannelAttentionBlock', 'DWConv', 'CustomDWConv', 'CustomPWConv', 'MLP', 'MixFFN', 'FusionBlock'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert len(nodes) == len(wanted)
    scope = dict(torch=torch, nn=nn, Tensor=Tensor, F=F, init=nn.init)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<archived MMSFormer>', 'exec'), scope)
    return scope


def test_archive_license_and_extracted_class_identity():
    root = Path(__file__).resolve().parents[1]
    for name, blob in {'third_party/MMSFormer/mmsformer.py': UPSTREAM_BLOB,
                       'third_party/MMSFormer/LICENSE': LICENSE_BLOB}.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob
    original = ast.parse((root / 'third_party/MMSFormer/mmsformer.py').read_text())
    used = ast.parse((root / 'models/components/mmsformer_source.py').read_text())
    definitions = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.ClassDef)}
    for node in used.body:
        if isinstance(node, ast.ClassDef): assert ast.dump(node) == definitions[node.name]


def test_all_actual_scales_native_initialization_forward_and_gradient_parity():
    torch.manual_seed(2023)
    fusion = MMSFormerFusion()
    torch.manual_seed(2023)
    reference = archived_reference()['FusionBlock'](list(STAGE_CHANNELS), reduction=16, num_modals=2)
    assert type(fusion.block) is FusionBlock
    for k, v in reference.state_dict().items(): assert torch.equal(v, fusion.block.state_dict()[k]), k
    cs, ps = features(actual=True)
    cr, pr = [x.detach().clone().requires_grad_() for x in cs], [x.detach().clone().requires_grad_() for x in ps]
    actual = fusion(cs, ps)
    expected = [reference([c, p], layer_idx=i) for i, (c, p) in enumerate(zip(cr, pr))]
    probes = [torch.randn_like(x) for x in actual]
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum((a * z).mean() for a, z in zip(actual, probes)).backward()
    sum((a * z).mean() for a, z in zip(expected, probes)).backward()
    for a, b in zip(cs + ps, cr + pr):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        assert torch.isfinite(a.grad).all() and float(a.grad.abs().sum()) > 0
    for a, b in zip(fusion.block.parameters(), reference.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert any(not torch.equal(a, c + p) for a, c, p in zip(actual, cs, ps))


def test_disabled_mode_is_exact_add_without_parameters():
    fusion = MMSFormerFusion(enabled=False)
    cs, ps = features()
    for a, c, p in zip(fusion(cs, ps), cs, ps): torch.testing.assert_close(a, c+p, atol=0, rtol=0)
    assert sum(p.numel() for p in fusion.parameters()) == 0


def test_spatial_resize_and_strict_roundtrip():
    fusion = MMSFormerFusion().eval()
    other = copy.deepcopy(fusion)
    other.load_state_dict(fusion.state_dict(), strict=True)
    cs, ps = features()
    ps = [p[..., ::2, ::2] for p in ps]
    for i, (a, b, c, p) in enumerate(zip(fusion(cs, ps), other(cs, ps), cs, ps)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        direct = fusion.block([c, F.interpolate(p, size=c.shape[-2:], mode='bilinear', align_corners=False)], i)
        torch.testing.assert_close(a, direct, atol=0, rtol=0)


@pytest.mark.parametrize('bad', ['count', 'channel', 'batch', 'dtype', 'empty', 'nonfinite'])
def test_invalid_features_rejected(bad):
    fusion = MMSFormerFusion()
    cs, ps = features()
    if bad == 'count': ps.pop()
    if bad == 'channel': ps[0] = ps[0][:, :63]
    if bad == 'batch': ps[0] = ps[0][:1]
    if bad == 'dtype': ps[0] = ps[0].double()
    if bad == 'empty': ps[0] = ps[0][..., :0]
    if bad == 'nonfinite': ps[0] = torch.full_like(ps[0], float('nan'))
    with pytest.raises((ValueError, RuntimeError)): fusion(cs, ps)


@pytest.mark.parametrize('enabled', [True, False])
def test_shared_baseline_initialization_rng_and_disabled_logits(enabled):
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullMMSFormerPETCT(pretrained=False, mmsformer_enabled=enabled).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in baseline.state_dict().items(): assert torch.equal(v, model.state_dict()[k]), k
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        actual = model(ct, pet, forward_mode='full')['logits']
        expected = baseline(ct, pet, forward_mode='full')['logits']
    if enabled: assert not torch.equal(actual, expected)
    else: torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_missing_and_auto_bypass_absent_pet_and_fusion():
    model = FullMMSFormerPETCT(pretrained=False).eval()
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    def forbidden(*args): raise AssertionError('Missing must bypass PET encoder/fusion')
    hooks = [model.enc_pet.register_forward_pre_hook(forbidden), model.fusion.register_forward_pre_hook(forbidden)]
    with torch.no_grad():
        a = model(ct, pet=None, forward_mode='missing')['logits']
        b = model(ct, pet=torch.full_like(pet, float('nan')), forward_mode='missing')['logits']
    for h in hooks: h.remove()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    with torch.no_grad():
        a = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
        pet[1].fill_(float('nan'))
        b = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def cfg_for(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false', '--require_cuda', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', 'mmsformer-test', '--mmsformer_enabled', str(enabled),
        '--batch_size', '2', '--epochs', '1', '--cosine_warmup', '0', '--mixed_precision', 'false', '--hd95_mode', 'surface'])
    return FullMMSFormerConfig.parse_arguments()


@pytest.mark.parametrize('enabled', [True, False])
def test_two_updates_training_ema_final_and_raw_ema_checkpoint(monkeypatch, tmp_path, enabled):
    cfg = cfg_for(monkeypatch, tmp_path, enabled)
    assert cfg.learning_rate == 4e-5 and cfg.norm_mode == 'imagenet' and cfg.early_stop_patience == 60
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    monkeypatch.setattr(FullMMSFormerConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:32, 20:32] = 1
    batch = dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=mask)
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([batch, batch], [batch], [batch]))
    runner.main(config_cls=FullMMSFormerConfig, model_builder=build_full_mmsformer_model,
                expected_fusion=MMSFormerFusion, experiment_type='full_mmsformer_fusion',
                result_config_fields=('mmsformer_enabled',))
    folder = Path(cfg.checkpoint_dir)
    result = json.loads((folder / 'final_test_full.json').read_text())
    assert result['mmsformer_enabled'] == enabled and result['weights'] == 'ema'
    path = folder / 'ckpt.best_full.pth.tar'
    for choice in (None, False):
        model, saved, checkpoint, tag = load_full_checkpoint(path, use_ema=choice)
        weights = checkpoint['model_ema'] if tag == 'ema' else checkpoint['model']
        for k, v in weights.items(): assert torch.equal(v, model.state_dict()[k])
        assert saved.mmsformer_enabled == enabled
    checkpoint['config']['mmsformer_output_mode'] = 'extra_add_residual'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='architecture'): load_full_checkpoint(path)


def test_main_loss_only_and_all_model_parts_learn(monkeypatch, tmp_path):
    from tasks.mdt_seg import MDTSegTeacher
    cfg = cfg_for(monkeypatch, tmp_path)
    task = MDTSegTeacher(build_full_mmsformer_model(cfg), cfg)
    batch = dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=torch.ones(2, 1, 64, 64))
    loss, outputs, _ = runner.train_step_full(task, batch)
    main, _ = task.criterion(outputs['logits'], batch['mask'])
    torch.testing.assert_close(loss, main, atol=0, rtol=0)
    assert not outputs['aux']
    loss.backward()
    for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
        grads = [p.grad for p in getattr(task.model, name).parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(float(g.abs().sum()) for g in grads) > 0, name


def test_existing_run_is_rejected_without_overwrite(monkeypatch, tmp_path):
    cfg = cfg_for(monkeypatch, tmp_path)
    path = Path(cfg.checkpoint_dir) / 'config_args.json'
    path.write_text('original')
    with pytest.raises(FileExistsError): FullMMSFormerConfig.parse_arguments()
    assert path.read_text() == 'original'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp_native_forward_backward():
    fusion = MMSFormerFusion().cuda()
    cs, ps = features(actual=True)
    cs = [x.detach().cuda().requires_grad_() for x in cs]
    ps = [x.detach().cuda().requires_grad_() for x in ps]
    with torch.autocast('cuda', dtype=torch.float16):
        loss = sum(x.float().square().mean() for x in fusion(cs, ps))
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in cs + ps)
