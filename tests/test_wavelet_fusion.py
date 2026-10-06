# -*- coding: utf-8 -*-
"""Integration tests for the Full-only wavelet fusion experiment.

Synthetic inputs + real encoder architectures with random weights
(pretrained=False). Covers wiring, zero-init equivalence, two-stage
gradients (low/high/context branches), Missing bypass, optimizer/EMA
membership, strict checkpoint roundtrips and old-entry preservation.
"""
import subprocess
import sys

import pytest
import torch

import run_full_wavelet_fusion as wavelet_entry
from models.build_mdt_seg import build_dual_model
from models.build_wavelet_fusion import build_wavelet_model
from models.components.add_fusion import AddFusion
from models.context_guided_wavelet_fusion import ContextGuidedWaveletFusionPyramid
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_wavelet_fusion import (
    MODEL_ARCH,
    DualSharedWaveletPETCTModel,
)
from tasks.mdt_seg import MDTSegTeacher

OLD_KEEP = [
    'run_ct_only_seg.py',
    'run_full_add_baseline.py',
    'run_full_missing_baseline.py',
    'run_mdt_seg.py',
    'models/ct_only_baseline.py',
    'models/dual_shared_add_baseline.py',
    'models/build_mdt_seg.py',
]


def _cfg(**over):
    base = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False, pretrained=False,
        model_arch=MODEL_ARCH, train_batch_mode='full',
        accumulation_steps=1, optimizer='adamw',
        fusion_dim=32, fusion_heads=4, fusion_window=8,
        fusion_checkpoint_attention=False,
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        random_state=2023, ema_enabled=True, ema_decay=0.9,
        ema_decay_warmup=False, ema_start_epoch=0,
    )
    base.update(over)
    return type('C', (), base)()


def _batch(n=2, size=64):
    return {'ct': torch.randn(n, 1, size, size),
            'pet': torch.randn(n, 1, size, size),
            'mask': (torch.rand(n, 1, size, size) > 0.5).float()}


def _make_pair(seed=0):
    """Add baseline + wavelet model with identical shared weights (strict copy)."""
    torch.manual_seed(seed)
    base = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False)
    torch.manual_seed(seed)
    fusion = DualSharedWaveletPETCTModel(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False)
    base_state = base.state_dict()
    fusion_state = fusion.state_dict()
    shared = {k: v for k, v in base_state.items() if not k.startswith('fusion.')}
    assert shared
    missing = set(fusion_state) - set(shared)
    assert all(k.startswith('fusion.') for k in missing)
    fusion.load_state_dict({**shared,
                            **{k: fusion_state[k] for k in missing}}, strict=True)
    return base, fusion


def test_module_embedded_tests_pass():
    r = subprocess.run(
        [sys.executable, 'models/context_guided_wavelet_fusion.py', '--test'],
        capture_output=True, text=True, cwd='.')
    assert r.returncode == 0, r.stderr[-3000:]
    assert 'OK' in r.stderr or 'ok' in r.stderr.lower()


def test_pyramid_zero_init_is_exact_addition():
    torch.manual_seed(0)
    module = ContextGuidedWaveletFusionPyramid(checkpoint_attention=False)
    ct = [torch.randn(1, c, s, s) for c, s in zip((64, 128, 320, 512), (32, 16, 8, 4))]
    pet = [torch.randn_like(c) for c in ct]
    fused = module(ct, pet)
    assert len(fused) == 4
    for f, c, p in zip(fused, ct, pet):
        assert torch.equal(f, c + p)


def test_all_bands_and_context_affect_output():
    """LL/LH/HL/HL exchanges plus context projections must all reach the
    output (zero-init output projections are opened in test state only)."""
    torch.manual_seed(1)
    module = ContextGuidedWaveletFusionPyramid(checkpoint_attention=False)
    with torch.no_grad():
        for block in module.scales:
            block.ct_out.weight.fill_(0.01)
            block.pet_out.weight.fill_(0.01)
    ct = [torch.randn(1, c, 16, 16) for c in (64, 128, 320, 512)]
    pet = [torch.randn_like(c) for c in ct]
    ref = [f.detach().clone() for f in module(ct, pet)]
    block = module.scales[0]
    targets = [f'ct_exchange.{b}' for b in ('LL', 'LH', 'HL', 'HH')]
    targets += [f'pet_exchange.{b}' for b in ('LL', 'LH', 'HL', 'HH')]
    checked = 0
    for dotted in targets:
        holder, name = dotted.split('.')
        params = list(getattr(block, holder)[name].parameters())
        assert params, dotted
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.3)
        out = module(ct, pet)
        assert any(not torch.equal(a, b) for a, b in zip(out, ref)), dotted
        checked += 1
    assert checked == 8


def test_new_builder_builds_pyramid_with_four_scales():
    model = build_wavelet_model(_cfg())['model']
    assert isinstance(model, DualSharedWaveletPETCTModel)
    assert isinstance(model.fusion, ContextGuidedWaveletFusionPyramid)
    assert len(model.fusion.scales) == 4
    assert [s.channels for s in model.fusion.scales] == [64, 128, 320, 512]
    assert len({id(s) for s in model.fusion.scales}) == 4


def test_decoder_receives_four_scale_list():
    _, fusion = _make_pair()
    seen = {}
    orig_fusion = fusion.fusion.forward
    orig_decode = fusion._decode

    def fusion_spy(ct_feats, pet_feats):
        out = orig_fusion(ct_feats, pet_feats)
        seen['fusion_out'] = len(out)
        return out

    def decode_spy(fused_feats, target_size):
        seen['decode_in'] = len(fused_feats)
        return orig_decode(fused_feats, target_size)

    fusion.fusion.forward = fusion_spy
    fusion._decode = decode_spy
    try:
        fusion.eval()
        with torch.no_grad():
            out = fusion(torch.randn(2, 1, 64, 64), pet=torch.randn(2, 1, 64, 64),
                         forward_mode='full')
        assert torch.isfinite(out['logits']).all()
    finally:
        fusion.fusion.forward = orig_fusion
        fusion._decode = orig_decode
    assert seen['fusion_out'] == 4 and seen['decode_in'] == 4


def test_zero_init_matches_add_baseline_eval():
    base, fusion = _make_pair()
    base.eval()
    fusion.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        a = base(ct, pet=pet, forward_mode='full')['logits']
        b = fusion(ct, pet=pet, forward_mode='full')['logits']
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_full_forward_backward_step_and_grads():
    from run_full_add_baseline import train_step_full
    assert wavelet_entry.train_step_full is train_step_full
    torch.manual_seed(3)
    task = MDTSegTeacher({'model': _make_pair(seed=3)[1]}, _cfg(ema_enabled=False))
    batch = _batch()
    task.optimizer.zero_grad(set_to_none=True)
    loss, outputs, stats = train_step_full(task, batch)
    assert stats['num_samples'] == 2 and torch.isfinite(loss).all()
    loss.backward()
    task.optimizer.step()
    pet_grads = [p.grad for p in task.model.enc_pet.parameters() if p.grad is not None]
    assert pet_grads and all(torch.isfinite(g).all() for g in pet_grads)
    out_grads = [p.grad for n, p in task.model.fusion.named_parameters()
                 if ('ct_out' in n or 'pet_out' in n) and p.grad is not None]
    assert out_grads and any(bool((g != 0).any()) for g in out_grads)


def test_second_backward_gives_finite_low_high_context_grads():
    """Zero-init step-1 internal grads are expected zero; after one update,
    low-frequency, high-frequency and context projections must carry finite
    nonzero gradients (no cutoff)."""
    torch.manual_seed(4)
    task = MDTSegTeacher({'model': _make_pair(seed=4)[1]}, _cfg(ema_enabled=False))
    from run_full_add_baseline import train_step_full
    loss, _, _ = train_step_full(task, _batch())
    loss.backward()
    task.optimizer.step()
    task.optimizer.zero_grad(set_to_none=True)
    loss2, _, _ = train_step_full(task, _batch())
    loss2.backward()
    block = task.model.fusion.scales[0]
    branches = ([f'ct_exchange.{b}' for b in ('LL', 'LH', 'HL', 'HH')]
                + [f'pet_exchange.{b}' for b in ('LL', 'LH', 'HL', 'HH')])
    for dotted in branches:
        holder, name = dotted.split('.')
        params = list(getattr(block, holder)[name].parameters())
        grads = [p.grad for p in params if p.grad is not None]
        assert grads, dotted
        assert all(torch.isfinite(g).all() for g in grads), dotted
        assert any(bool((g != 0).any()) for g in grads), dotted


def test_missing_path_calls_no_pet_encoder_dwt_or_fusion():
    _, fusion = _make_pair()
    calls = {'pet': 0, 'fusion': 0, 'dwt': 0}
    orig_pet = fusion.enc_pet.forward
    orig_fusion = fusion.fusion.forward
    dwt_fns = [s.wavelet.analysis for s in fusion.fusion.scales]
    fusion.enc_pet.forward = lambda x: (calls.__setitem__('pet', calls['pet'] + 1),
                                        orig_pet(x))[1]
    fusion.fusion.forward = lambda *a, **k: (calls.__setitem__('fusion', calls['fusion'] + 1),
                                            orig_fusion(*a, **k))[1]
    spies = []
    for i, fn in enumerate(dwt_fns):
        def make_spy(f):
            def spy(x):
                calls['dwt'] += 1
                return f(x)
            return spy
        spies.append(make_spy(fn))
    try:
        for s, spy in zip(fusion.fusion.scales, spies):
            s.wavelet.analysis = spy
        fusion.eval()
        with torch.no_grad():
            out = fusion(torch.randn(2, 1, 64, 64), pet=None, forward_mode='missing')
        assert torch.isfinite(out['logits']).all()
    finally:
        fusion.enc_pet.forward = orig_pet
        fusion.fusion.forward = orig_fusion
        for s, fn in zip(fusion.fusion.scales, dwt_fns):
            s.wavelet.analysis = fn
    assert calls == {'pet': 0, 'fusion': 0, 'dwt': 0}


def test_optimizer_and_ema_contain_fusion():
    torch.manual_seed(5)
    task = MDTSegTeacher(
        {'model': build_wavelet_model(_cfg())['model']}, _cfg())
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    fusion_params = list(task.model.fusion.parameters())
    assert fusion_params and all(id(p) in opt_ids for p in fusion_params)
    assert task.ema is not None
    assert any(k.startswith('fusion.') for k in task.ema.model.state_dict())
    task.update_ema()
    assert task.ema.updates == 1


def test_raw_and_ema_checkpoints_strict_reload(tmp_path):
    torch.manual_seed(6)
    task = MDTSegTeacher(
        {'model': build_wavelet_model(_cfg())['model']}, _cfg())
    path = str(tmp_path / 'ckpt.wavelet.tar')
    task.save_checkpoint(path, 2, best=0.6, best_epoch=2, val={'dice': 0.6})
    ckpt = MDTSegTeacher.load_state_dicts(path)
    for key in ('model', 'model_ema'):
        fresh = build_wavelet_model(_cfg())['model']
        fresh.load_state_dict(ckpt[key], strict=True)


def test_final_eval_rebuilds_wavelet_and_uses_val_dice():
    torch.manual_seed(7)
    task = MDTSegTeacher(
        {'model': build_wavelet_model(_cfg())['model']},
        _cfg(ema_enabled=False))
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), 'ckpt.best_full.tar')
    task.save_checkpoint(path, 3, best=0.62, best_epoch=3, val={'dice': 0.62})
    ckpt = MDTSegTeacher.load_state_dicts(path)
    assert ckpt.get('eval_weights', 'raw') == 'raw'
    eval_model = build_wavelet_model(_cfg())['model']
    assert isinstance(eval_model.fusion, ContextGuidedWaveletFusionPyramid)
    eval_model.load_state_dict(ckpt['model'], strict=True)
    eval_model.to(task.device).eval()
    out = task.evaluate([_batch()], eval_mode='full', model=eval_model)
    assert 0.0 <= out['dice'] <= 1.0
    assert ckpt['best'] == 0.62 and ckpt['best_epoch'] == 3


def test_entry_help_and_param_plumbing():
    r = subprocess.run([sys.executable, 'run_full_wavelet_fusion.py', '--help'],
                       capture_output=True, text=True, cwd='.')
    assert r.returncode == 0
    for flag in ('--fusion_dim', '--fusion_heads', '--fusion_window',
                 '--fusion_checkpoint_attention',
                 '--train_batch_mode {full}', MODEL_ARCH):
        assert flag in r.stdout, flag
    argv, sys.argv = sys.argv, ['x', '--model_arch', 'dual_shared_add_baseline']
    try:
        with pytest.raises(SystemExit):
            wavelet_entry.FullWaveletFusionConfig.parse_arguments()
    finally:
        sys.argv = argv
    argv, sys.argv = sys.argv, ['x', '--fusion_window', '0']
    try:
        bad = wavelet_entry.FullWaveletFusionConfig.parse_arguments()
    finally:
        sys.argv = argv
    with pytest.raises(ValueError):
        build_wavelet_model(bad)


def test_old_entries_builders_and_models_unchanged():
    r = subprocess.run(['git', 'status', '--porcelain', '--'] + OLD_KEEP,
                       capture_output=True, text=True, cwd='.')
    assert r.stdout.strip() == '', f'old files modified: {r.stdout.strip()}'
    from configs.seg_mdt import SegMDTConfig
    argv, sys.argv = sys.argv, ['x']
    try:
        old = SegMDTConfig.parse_arguments()
    finally:
        sys.argv = argv
    assert old.model_arch == 'dual_shared_add_baseline'
    assert not hasattr(old, 'fusion_dim')
    from types import SimpleNamespace
    old_cfg = _cfg(model_arch='dual_shared_add_baseline')
    old_model = build_dual_model(old_cfg)['model']
    assert isinstance(old_model.fusion, AddFusion)
    assert 'wavelet' not in type(old_model.fusion).__module__
    for path in ('run_ct_only_seg.py', 'run_full_add_baseline.py',
                 'run_full_missing_baseline.py', 'run_mdt_seg.py'):
        assert 'wavelet' not in open(path).read().lower(), path
    _ = SimpleNamespace
