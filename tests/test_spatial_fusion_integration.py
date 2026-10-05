# -*- coding: utf-8 -*-
"""Integration tests for the spatial-fusion experiment.

Scope: synthetic inputs + real encoder architectures with random weights
(pretrained=False). This verifies wiring, routing, equivalence, gradients,
optimizer/EMA membership and checkpoint roundtrips — NOT real-data training
or full-resolution CUDA behavior, which remain unvalidated (see delivery
notes).

A strict shared-weight protocol is used for equivalence: non-fusion weights
are copied tensor-by-tensor from the baseline into the spatial model (never
just "same seed"), and the copy is asserted strict-safe.
"""
import copy

import pytest
import torch
import torch.nn as nn

from models.build_mdt_seg import build_dual_model
from models.build_spatial_fusion_seg import build_spatial_fusion_model
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_spatial_fusion import (
    MODEL_ARCH,
    DualSharedSpatialFusionPETCTModel,
)
from models.spatial_bidirectional_fusion import MultiScaleSpatialBidirectionalFusion
from tasks.mdt_seg import MDTSegTeacher
from utils.run_common import build_balanced_pet_available, eval_missing_rates


def _base_cfg(**over):
    base = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False, pretrained=False,
        model_arch='dual_shared_add_baseline',
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        random_state=2023, ema_enabled=True, ema_decay=0.9,
        ema_decay_warmup=False, ema_start_epoch=0,
    )
    base.update(over)
    return type('C', (), base)()


def _spatial_cfg(**over):
    base = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False, pretrained=False,
        model_arch=MODEL_ARCH,
        fusion_enabled=True, fusion_mode='full', fusion_attention_dim=64,
        fusion_num_heads=4, fusion_local_kernel_size=5,
        fusion_max_axis_length=128, fusion_axis_chunk_size=32,
        fusion_use_checkpoint=False,
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        random_state=2023, ema_enabled=True, ema_decay=0.9,
        ema_decay_warmup=False, ema_start_epoch=0,
    )
    base.update(over)
    return type('C', (), base)()


def _make_pair(fusion_enabled=True, seed=0):
    """Baseline + spatial model with identical shared weights (strict copy)."""
    torch.manual_seed(seed)
    base = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False)
    torch.manual_seed(seed)
    spatial = DualSharedSpatialFusionPETCTModel(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False,
        fusion_enabled=fusion_enabled)
    base_state = base.state_dict()
    spatial_state = spatial.state_dict()
    shared = {k: v for k, v in base_state.items() if not k.startswith('fusion.')}
    assert shared, 'baseline has no shared weights to copy'
    missing_before = set(spatial_state) - set(shared)
    # fusion_enabled=False builds a parameter-free AddFusion, so the missing
    # set may be empty; otherwise every non-shared key must be a fusion key.
    assert all(k.startswith('fusion.') for k in missing_before)
    spatial.load_state_dict(shared, strict=False)
    msg = spatial.load_state_dict({**shared,
                                   **{k: spatial_state[k] for k in missing_before}},
                                  strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
    return base, spatial


def test_old_builder_still_builds_addfusion():
    model = build_dual_model(_base_cfg())['model']
    assert type(model).__name__ == 'DualSharedAddPETCTBaseline'
    assert isinstance(model.fusion, AddFusion)
    assert type(model.fusion).__name__ == 'AddFusion'


def test_old_config_and_entry_unchanged():
    from configs.seg_mdt import SegMDTConfig
    import sys
    argv, sys.argv = sys.argv, ['x']
    try:
        old = SegMDTConfig.parse_arguments()
    finally:
        sys.argv = argv
    assert old.model_arch == 'dual_shared_add_baseline'
    assert not hasattr(old, 'fusion_enabled')
    src = open('run_full_missing_baseline.py').read()
    assert 'build_dual_model' in src
    assert 'build_spatial_fusion_model' not in src
    assert 'MODEL_ARCH' not in src


def test_new_builder_builds_spatial_module():
    model = build_spatial_fusion_model(_spatial_cfg())['model']
    assert isinstance(model, DualSharedSpatialFusionPETCTModel)
    assert isinstance(model.fusion, MultiScaleSpatialBidirectionalFusion)
    assert len(model.fusion.blocks) == 4
    assert model.fusion.channels == (64, 128, 320, 512)
    assert type(model.enc_ct).__name__ == 'TimmFeatureBackbone'
    assert len(model.decoder.proj1) > 0


def test_fusion_disabled_matches_baseline_exactly():
    base, spatial = _make_pair(fusion_enabled=False)
    assert isinstance(spatial.fusion, AddFusion)
    spatial.eval()
    base.eval()
    ct = torch.randn(2, 1, 32, 32)
    pet = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        for mode, kw in (('full', {'pet': pet}),
                         ('missing', {'pet': None}),
                         ('auto', {'pet': pet, 'pet_available': torch.tensor([1, 0])})):
            a = base(ct, forward_mode=mode, **kw)['logits']
            b = spatial(ct, forward_mode=mode, **kw)['logits']
            assert torch.equal(a, b), mode


def test_zero_init_matches_add_baseline():
    base, spatial = _make_pair(fusion_enabled=True)
    base.eval()
    spatial.eval()
    ct = torch.randn(2, 1, 32, 32)
    pet = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        for mode, kw in (('full', {'pet': pet}),
                         ('missing', {'pet': None}),
                         ('auto', {'pet': pet, 'pet_available': torch.tensor([1, 0])})):
            a = base(ct, forward_mode=mode, **kw)['logits']
            b = spatial(ct, forward_mode=mode, **kw)['logits']
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), mode


def test_missing_path_calls_neither_pet_encoder_nor_fusion():
    _, spatial = _make_pair()
    calls = {'pet': 0, 'fusion': 0}
    orig_pet = spatial.enc_pet.forward
    orig_fusion = spatial.fusion.forward

    def pet_spy(x):
        calls['pet'] += 1
        return orig_pet(x)

    def fusion_spy(*a, **k):
        calls['fusion'] += 1
        return orig_fusion(*a, **k)

    spatial.enc_pet.forward = pet_spy
    spatial.fusion.forward = fusion_spy
    try:
        spatial.eval()
        with torch.no_grad():
            out = spatial(torch.randn(2, 1, 32, 32), pet=None, forward_mode='missing')
        assert torch.isfinite(out['logits']).all()
    finally:
        spatial.enc_pet.forward = orig_pet
        spatial.fusion.forward = orig_fusion
    assert calls == {'pet': 0, 'fusion': 0}


def test_auto_encodes_only_full_pet_rows_and_restores_order():
    _, spatial = _make_pair()
    seen = {}

    orig_pet = spatial._encode_pet

    def pet_spy(pet):
        seen['rows'] = pet.shape[0]
        return orig_pet(pet)

    spatial._encode_pet = pet_spy
    try:
        spatial.eval()
        ct = torch.randn(4, 1, 32, 32)
        pet = torch.randn(4, 1, 32, 32)
        state = torch.tensor([0, 1, 0, 1])
        with torch.no_grad():
            mixed = spatial(ct, pet=pet, pet_available=state,
                            forward_mode='auto')['logits']
            full_rows = spatial(ct[state.eq(1)], pet=pet[state.eq(1)],
                                forward_mode='full')['logits']
            missing_rows = spatial(ct[state.eq(0)], pet=None,
                                   forward_mode='missing')['logits']
    finally:
        spatial._encode_pet = orig_pet
    assert seen['rows'] == 2, 'PET encoder must see Full rows only'
    assert torch.allclose(mixed[state.eq(1)], full_rows, atol=1e-3)
    assert torch.allclose(mixed[state.eq(0)], missing_rows, atol=1e-3)


def test_missing_output_independent_of_missing_pet():
    _, spatial = _make_pair()
    spatial.eval()
    ct = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        a = spatial(ct, pet=torch.randn(2, 1, 32, 32),
                    pet_available=torch.tensor([1, 0]), forward_mode='auto')['logits']
        b = spatial(ct, pet=torch.randn(2, 1, 32, 32),
                    pet_available=torch.tensor([1, 0]), forward_mode='auto')['logits']
    assert torch.equal(a[1], b[1])
    assert not torch.equal(a[0], b[0])


def test_full_loss_updates_fusion_output_projection_and_zero_init_internals():
    """Zero-init: internal matching params get zero grad (expected — the
    residual branch is gated by zero output projections); the output
    projections themselves (out_c/out_p per step) receive real gradients."""
    torch.manual_seed(7)
    task = MDTSegTeacher({'model': _make_pair()[1]}, _base_cfg(ema_enabled=False))
    batch = {'ct': torch.randn(2, 1, 32, 32), 'pet': torch.randn(2, 1, 32, 32),
             'mask': (torch.rand(2, 1, 32, 32) > 0.5).float()}
    task.model.zero_grad(set_to_none=True)
    ct = batch['ct'].to(task.device)
    pet = batch['pet'].to(task.device)
    mask = batch['mask'].to(task.device)
    loss, _ = task.criterion(task.model(ct, pet=pet, forward_mode='full')['logits'], mask)
    loss.backward()
    proj_grads, internal_zero, internal_nonzero = [], [], []
    for n, p in task.model.fusion.named_parameters():
        g = p.grad
        is_out = ('out_c' in n) or ('out_p' in n)
        if is_out:
            proj_grads.append(bool((g != 0).any()))
        elif g is None or bool((g == 0).all()):
            internal_zero.append(n)
        else:
            internal_nonzero.append(n)
    assert proj_grads and all(proj_grads), 'full loss must update fusion output projections'
    assert internal_nonzero == [], f'zero-init internals must have zero grad: {internal_nonzero[:3]}'
    assert internal_zero, 'expected internal matching params to exist'


def test_missing_subset_loss_gives_no_pet_or_fusion_grads():
    torch.manual_seed(8)
    task = MDTSegTeacher({'model': _make_pair()[1]}, _base_cfg(ema_enabled=False))
    ct = torch.randn(4, 1, 32, 32).to(task.device)
    pet = torch.randn(4, 1, 32, 32).to(task.device)
    mask = (torch.rand(4, 1, 32, 32) > 0.5).float().to(task.device)
    state = torch.tensor([1, 1, 0, 0], device=task.device)
    task.model.zero_grad(set_to_none=True)
    logits = task.model(ct, pet=pet, pet_available=state, forward_mode='auto')['logits']
    missing_loss, _ = task.criterion(logits[state.eq(0)], mask[state.eq(0)])
    missing_loss.backward()
    for name, module in (('enc_pet', task.model.enc_pet), ('fusion', task.model.fusion)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads == [] or all(bool((g == 0).all()) for g in grads), name


def test_fusion_params_in_optimizer_and_ema():
    torch.manual_seed(9)
    task = MDTSegTeacher({'model': build_spatial_fusion_model(_spatial_cfg())['model']},
                         _spatial_cfg())
    opt_ids = {id(p) for group in task.optimizer.param_groups for p in group['params']}
    fusion_params = list(task.model.fusion.parameters())
    assert fusion_params, 'fusion must own parameters'
    assert all(id(p) in opt_ids for p in fusion_params)
    assert task.ema is not None
    ema_keys = set(task.ema.model.state_dict())
    assert any(k.startswith('fusion.') for k in ema_keys)
    task.update_ema()
    assert task.ema.updates == 1


def test_spatial_checkpoint_strict_roundtrip(tmp_path):
    torch.manual_seed(10)
    task = MDTSegTeacher({'model': build_spatial_fusion_model(_spatial_cfg())['model']},
                         _spatial_cfg())
    path = str(tmp_path / 'ckpt.spatial.pth.tar')
    task.save_checkpoint(path, 1, best=0.5, best_epoch=1, val={'joint_dice': 0.5})
    ckpt = MDTSegTeacher.load_state_dicts(path)
    fresh = build_spatial_fusion_model(_spatial_cfg())['model']
    fresh.load_state_dict(ckpt['model'], strict=True)
    fresh_ema = build_spatial_fusion_model(_spatial_cfg())['model']
    fresh_ema.load_state_dict(ckpt['model_ema'], strict=True)


def _tiny_loader(n=4, size=32, device='cpu'):
    torch.manual_seed(11)
    batches = [{
        'ct': torch.randn(2, 1, size, size),
        'pet': torch.randn(2, 1, size, size),
        'mask': (torch.rand(2, 1, size, size) > 0.5).float(),
        'case_id': [f'case{i}', f'case{i + 1}'],
    } for i in (0, 2)][:n // 2]
    return batches


def test_final_eval_builds_new_model_and_reports_joint(tmp_path):
    torch.manual_seed(12)
    model = build_spatial_fusion_model(_spatial_cfg())['model']
    assert isinstance(model, DualSharedSpatialFusionPETCTModel)
    task = MDTSegTeacher({'model': copy.deepcopy(model)}, _spatial_cfg(ema_enabled=False))
    loader = _tiny_loader()
    results, joint = eval_missing_rates(task, task.model, loader, seed=2023,
                                        checkpoint_dir=str(tmp_path),
                                        stem='probe', rates=[0.0, 1.0],
                                        weights_tag='raw')
    by_rate = {r['missing_rate']: r for r in results}
    assert abs(joint - 0.5 * (by_rate[0.0]['dice'] + by_rate[1.0]['dice'])) < 1e-9
    assert all(r['weights'] == 'raw' for r in results)
