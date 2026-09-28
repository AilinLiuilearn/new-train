# -*- coding: utf-8 -*-
"""Integration tests for the increment-fusion (inc) branch.

Covers selection/routing/gradients/EMA/checkpoint without datasets or
downloaded weights (backbone weights are randomly initialized).
"""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.petct_increment_fusion import PETCTIncrementFusion


def _model(**over):
    kw = dict(ct_pretrained_path=None, pet_pretrained_path=None,
              asym_fusion_enabled=True, asym_use_text=False, fusion_version='inc')
    kw.update(over)
    return DualSharedAddPETCTBaseline(**kw)


def test_inc_selected_without_text_encoder():
    import models.text_encoders as te
    calls = []
    real = te.load_text_embeddings
    te.load_text_embeddings = lambda *a, **k: (calls.append(1), (_ for _ in ()).throw(
        AssertionError('text encoder must not be called for inc')))
    try:
        from models.build_mdt_seg import build_mdt_seg_teacher
        cfg = type('C', (), dict(ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
            ct_pretrained_path=None, pet_pretrained_path=None,
            decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
            asym_fusion_enabled=True, asym_use_text=True, fusion_version='inc',
            train_batch_mode='full', ema_enabled=False))()
        m = build_mdt_seg_teacher(cfg)['model']
    finally:
        te.load_text_embeddings = real
    assert type(m.fusion).__name__ == 'PETCTIncrementFusion'
    assert calls == []


def test_decoder_receives_module_output_verbatim():
    torch.manual_seed(0)
    m = _model().eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    seen = {}
    real_decode = m._decode
    m._decode = lambda feats, ts: (seen.setdefault('feats', feats), real_decode(feats, ts))[1]
    with torch.no_grad():
        out = m(ct, pet, forward_mode='full')
        cf = m._encode_ct(ct)
        pf = m._encode_pet(pet)
        direct = m.fusion(cf, pf, state='full')
    for a, b in zip(seen['feats'], direct):
        assert torch.allclose(a, b, rtol=0, atol=0)
    assert out['logits'].shape == (2, 1, 64, 64)


def test_disabled_baseline_reproducible():
    torch.manual_seed(7)
    a = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    ka = {k: v.clone() for k, v in a.state_dict().items()}
    torch.manual_seed(7)
    b = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    for k in ka:
        assert torch.equal(ka[k], b.state_dict()[k]), k
    a.eval()
    b.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        oa = a(ct, pet, pet_available=torch.tensor([1, 0]), forward_mode='auto')['logits']
        ob = b(ct, pet, pet_available=torch.tensor([1, 0]), forward_mode='auto')['logits']
    assert torch.equal(oa, ob)


def test_full_backward_reaches_both_encoders():
    torch.manual_seed(0)
    m = _model()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out = m(ct, pet, forward_mode='full')
    out['logits'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None for p in m.fusion.parameters())
    assert any(p.grad is not None for p in m.enc_ct.parameters())
    assert any(p.grad is not None for p in m.enc_pet.parameters())
    assert any(p.grad is not None for p in m.decoder.parameters())
    opt_params = {id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in opt_params for p in m.fusion.parameters() if p.requires_grad)


def test_missing_error_and_zero_paths():
    import pytest
    torch.manual_seed(0)
    m = _model().eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with pytest.raises(RuntimeError, match='bank not connected'):
        with torch.no_grad():
            m(ct, pet, forward_mode='missing')
    mz = _model(inc_missing_policy='zero').eval()
    calls = []
    real_encode_pet = mz._encode_pet
    mz._encode_pet = lambda p: (calls.append(1), real_encode_pet(p))[1]
    with torch.no_grad():
        out = mz(ct, None, forward_mode='missing')
    assert calls == []
    assert out['logits'].shape == (2, 1, 64, 64) and torch.isfinite(out['logits']).all()


def test_mixed_zero_encodes_full_pet_only_and_eval_invariance():
    torch.manual_seed(0)
    m = _model(inc_missing_policy='zero').eval()
    ct = torch.randn(4, 1, 64, 64)
    pet = torch.randn(4, 1, 64, 64)
    seen_sizes = []
    real_encode_pet = m._encode_pet
    m._encode_pet = lambda p: (seen_sizes.append(p.shape[0]), real_encode_pet(p))[1]
    state = torch.tensor([1, 1, 0, 0])
    with torch.no_grad():
        out = m(ct, pet, pet_available=state, forward_mode='auto')
    assert seen_sizes == [2], seen_sizes
    assert out['num_full'] == 2 and out['num_missing'] == 2
    base = out['logits'].clone()
    pet_perturbed_missing = pet.clone()
    pet_perturbed_missing[2:] += 5.0
    with torch.no_grad():
        out2 = m(ct, pet_perturbed_missing, pet_available=state, forward_mode='auto')
    assert torch.allclose(base, out2['logits'], rtol=0, atol=0)
    pet_perturbed_full = pet.clone()
    pet_perturbed_full[:2] += 5.0
    with torch.no_grad():
        out3 = m(ct, pet_perturbed_full, pet_available=state, forward_mode='auto')
    assert not torch.allclose(base[:2], out3['logits'][:2], rtol=0, atol=1e-6)
    assert torch.allclose(base[2:], out3['logits'][2:], rtol=0, atol=0)


def test_ema_update_and_strict_roundtrip():
    from tasks.mdt_seg import MDTSegTeacher
    torch.manual_seed(0)
    cfg = type('C', (), dict(ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
        asym_fusion_enabled=True, asym_use_text=False, fusion_version='inc',
        decoder_norm='group', train_batch_mode='full',
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
        ema_enabled=True, ema_decay=0.999, ema_decay_warmup=False, ema_start_epoch=0))()
    from models.build_mdt_seg import build_mdt_seg_teacher
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    assert task.ema is not None
    task.begin_epoch(1)
    batch = {'ct': torch.randn(2, 1, 64, 64), 'pet': torch.randn(2, 1, 64, 64),
             'mask': (torch.rand(2, 1, 64, 64) > 0.5).float()}
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, _ = task.train_step(batch, forward_mode='full')
    assert torch.isfinite(loss)
    loss.backward()
    task.update_ema()
    assert task.ema.updates == 1
    assert all(p.grad is None for p in task.ema.model.parameters())
    sd = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
          for k, v in task.model.state_dict().items()}
    torch.manual_seed(1)
    task2 = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task2.model.load_state_dict(sd, strict=True)
    task2.model.eval()
    task.model.eval()
    dev = next(task.model.parameters()).device
    bc = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        o1 = task.model(bc['ct'], bc['pet'], forward_mode='full')['logits']
        o2 = task2.model(bc['ct'], bc['pet'], forward_mode='full')['logits']
    assert torch.allclose(o1, o2, rtol=0, atol=0)


def test_pgf_tests_untouched_by_inc():
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline as D
    m = D(ct_pretrained_path=None, pet_pretrained_path=None,
          asym_fusion_enabled=True, asym_use_text=False, fusion_version='pgf')
    assert type(m.fusion).__name__ == 'PETCTPairedGlobalFusion'
