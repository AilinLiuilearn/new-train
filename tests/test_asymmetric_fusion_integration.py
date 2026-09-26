# -*- coding: utf-8 -*-
"""Integration tests for the Full-only asymmetric fusion routing.

Covers: off switch, Full path, Missing path (zero fusion calls, CT identity),
mixed 8+8 (fusion sees exactly the 8 Full rows), PET sensitivity
(Missing invariant / Full sensitive), no in-place feature mutation,
gradient flow, optimizer coverage, EMA, and strict checkpoint round-trip.
"""
import pytest
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _cfg(**over):
    base = dict(learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
                loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
                ema_enabled=False, ema_decay=0.9, ema_decay_warmup=False, ema_start_epoch=0,
                train_batch_mode='mixed')
    base.update(over)
    return type('C', (), base)()


def _model(**over):
    kw = dict(ct_pretrained_path=None, pet_pretrained_path=None,
              asym_fusion_enabled=True, asym_use_text=False)
    kw.update(over)
    return DualSharedAddPETCTBaseline(**kw)


def _batch(n=8, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {'ct': torch.randn(n, 1, size, size, generator=g),
            'pet': torch.randn(n, 1, size, size, generator=g),
            'mask': (torch.rand(n, 1, size, size, generator=g) > 0.5).float()}


def _counting_wrapper(model):
    calls = []
    orig = model.fusion.forward

    def wrapped(ct_feats, pet_feats, **kwargs):
        calls.append((ct_feats[0].shape[0], kwargs.get('state')))
        return orig(ct_feats, pet_feats, **kwargs)

    model.fusion.forward = wrapped
    return calls


def test_off_switch_uses_addfusion():
    m = _model(asym_fusion_enabled=False)
    assert type(m.fusion).__name__ == 'AddFusion'
    assert sum(p.numel() for p in m.fusion.parameters()) == 0


def test_all_missing_zero_fusion_calls_and_ct_identity():
    torch.manual_seed(0)
    m = _model().eval()
    calls = _counting_wrapper(m)
    batch = _batch(n=4)
    with torch.no_grad():
        ct_feats = m._encode_ct(batch['ct'])
        out = m(batch['ct'], pet=batch['pet'],
                pet_available=torch.zeros(4, dtype=torch.long), forward_mode='auto')
    assert calls == []
    with torch.no_grad():
        ref = m._decode(list(ct_feats), (64, 64))['logits']
    assert torch.equal(out['logits'], ref)


def test_mixed_fusion_receives_only_full_rows():
    torch.manual_seed(0)
    m = _model().eval()
    calls = _counting_wrapper(m)
    batch = _batch(n=8)
    state = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
    with torch.no_grad():
        m(batch['ct'], pet=batch['pet'], pet_available=state, forward_mode='auto')
    assert len(calls) == 1
    assert calls[0] == (4, 'full')


def test_missing_rows_invariant_to_pet_full_rows_sensitive():
    torch.manual_seed(0)
    m = _model().eval()
    batch = _batch(n=4)
    state = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    with torch.no_grad():
        base = m(batch['ct'], pet=batch['pet'], pet_available=state,
                 forward_mode='auto')['logits']
        # Perturb only Missing rows' PET: logits must not move.
        pet2 = batch['pet'].clone()
        pet2[2:] = torch.randn_like(pet2[2:]) * 5.0 + 10.0
        out_missing_perturbed = m(batch['ct'], pet=pet2, pet_available=state,
                                  forward_mode='auto')['logits']
        # Perturb only Full rows' PET: logits must move.
        pet3 = batch['pet'].clone()
        pet3[:2] = torch.randn_like(pet3[:2]) * 5.0 + 10.0
        out_full_perturbed = m(batch['ct'], pet=pet3, pet_available=state,
                               forward_mode='auto')['logits']
    assert torch.equal(base, out_missing_perturbed)
    assert not torch.allclose(base, out_full_perturbed, atol=1e-6)


def test_no_inplace_feature_mutation():
    torch.manual_seed(0)
    m = _model().eval()
    batch = _batch(n=8)
    state = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
    with torch.no_grad():
        ct_feats = m._encode_ct(batch['ct'])
        pet_feats = m._encode_pet(batch['pet'])
        ct_clone = [f.clone() for f in ct_feats]
        pet_clone = [f.clone() for f in pet_feats]
        fused = m._fuse_features(ct_feats, pet_feats, state)
    for a, b in zip(ct_feats, ct_clone):
        assert torch.equal(a, b)
    for a, b in zip(pet_feats, pet_clone):
        assert torch.equal(a, b)
    # Fused outputs are new tensors, not aliases of the inputs.
    for f, c in zip(fused, ct_feats):
        assert f.data_ptr() != c.data_ptr()


def test_state_validation_rejects_float_truncation():
    torch.manual_seed(0)
    m = _model().eval()
    batch = _batch(n=4)
    with pytest.raises(ValueError):
        m(batch['ct'], pet=batch['pet'],
          pet_available=torch.tensor([1.0, 0.5, 0.0, 1.0]), forward_mode='auto')
    with pytest.raises(ValueError):
        m(batch['ct'], pet=batch['pet'],
          pet_available=torch.tensor([1, 2, 0, 1]), forward_mode='auto')


def test_mixed_gradients_and_optimizer_coverage():
    torch.manual_seed(0)
    m = _model()
    task = MDTSegTeacher({'model': m}, _cfg())
    batch = _batch(n=8)
    state = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, stats = task.train_step_mixed(batch, pet_available=state)
    loss.backward()
    assert stats['num_full'] == 4 and stats['num_missing'] == 4
    fused_params = [p for p in m.fusion.parameters() if p.requires_grad]
    assert len(fused_params) > 0
    assert all(p.grad is not None for p in fused_params)
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    assert all(id(p) in opt_ids for p in fused_params)
    task.optimizer.step()


def test_strict_checkpoint_roundtrip_with_asym():
    torch.manual_seed(0)
    import os
    import tempfile
    m = _model()
    task = MDTSegTeacher({'model': m}, _cfg(ema_enabled=True))
    batch = _batch(n=8)
    state = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, _ = task.train_step_mixed(batch, pet_available=state)
    loss.backward()
    task.optimizer.step()
    task.update_ema()
    path = os.path.join(tempfile.mkdtemp(), 'ckpt.pth.tar')
    task.save_checkpoint(path, epoch=1)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['ema_updates'] >= 1
    fresh = _model()
    msg = fresh.load_state_dict(ckpt['model'], strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
    msg_ema = fresh.load_state_dict(ckpt['model_ema'], strict=True)
    assert not msg_ema.missing_keys and not msg_ema.unexpected_keys
