# -*- coding: utf-8 -*-
"""Contracts for the two clean baselines.

Covers: shared CT encoder/align/decoder structure, GroupNorm decoder,
exact-half reproducible mixed batches, Full-encodes-PET / Missing-skips-PET,
auto order preservation, pet=None support, Missing PET-invariance, gradient
routing, output shapes/finiteness, and EMA checkpoint roundtrip.
"""
import copy

import pytest
import torch
import torch.nn as nn

from models.add_fusion import AddFusion
from models.ct_only_baseline import CTOnlySegmentationModel
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.group_unet_decoder import UNetStyleDecoder
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import build_balanced_pet_available


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'random_state': 2023,
        'ema_enabled': False,
        'ema_start_epoch': 0,
        'ema_decay': 0.999,
        'ema_decay_warmup': True,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _tiny_dual(**kwargs):
    kw = dict(ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
              ct_pretrained_path=None, pet_pretrained_path=None,
              use_deep_supervision=False, pretrained=False)
    kw.update(kwargs)
    return DualSharedAddPETCTBaseline(**kw)


def test_ct_only_has_no_pet_encoder():
    model = CTOnlySegmentationModel(ct_pretrained_path=None, pretrained=False)
    assert not hasattr(model, 'enc_pet')
    assert not hasattr(model, 'fusion')
    assert isinstance(model.decoder, UNetStyleDecoder)


def test_shared_structure_between_baselines():
    ct_only = CTOnlySegmentationModel(ct_pretrained_path=None, pretrained=False)
    dual = _tiny_dual()
    assert type(ct_only.enc_ct).__name__ == type(dual.enc_ct).__name__
    assert type(ct_only.ct_align).__name__ == type(dual.ct_align).__name__
    assert type(ct_only.decoder).__name__ == type(dual.decoder).__name__
    assert ct_only.decoder.proj1[0].in_channels == dual.decoder.proj1[0].in_channels == 64
    assert ct_only.decoder.proj4[0].in_channels == dual.decoder.proj4[0].in_channels == 512


def test_decoder_uses_groupnorm_only():
    for ctor in (lambda: CTOnlySegmentationModel(ct_pretrained_path=None, pretrained=False),
                 _tiny_dual):
        dec = ctor().decoder
        gn = [m for m in dec.modules() if isinstance(m, nn.GroupNorm)]
        bn = [m for m in dec.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(gn) > 0 and len(bn) == 0


def test_forward_shapes_and_finite():
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out_ct = CTOnlySegmentationModel(ct_pretrained_path=None, pretrained=False)(ct)
    assert out_ct['logits'].shape == (2, 1, 64, 64)
    assert torch.isfinite(out_ct['logits']).all()
    dual = _tiny_dual()
    for mode, kwargs in (('full', {'pet': pet}),
                         ('missing', {'pet': None}),
                         ('auto', {'pet': pet, 'pet_available': torch.tensor([1, 0])})):
        out = dual(ct, forward_mode=mode, **kwargs)
        assert out['logits'].shape == (2, 1, 64, 64), mode
        assert torch.isfinite(out['logits']).all(), mode


def test_mixed_batch_exact_half_and_reproducible():
    dev = torch.device('cpu')
    a = build_balanced_pet_available(8, 3, 2023, dev)
    b = build_balanced_pet_available(8, 3, 2023, dev)
    c = build_balanced_pet_available(8, 4, 2023, dev)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert int(a.sum()) == 4 and int((a == 0).sum()) == 4
    with pytest.raises(ValueError):
        build_balanced_pet_available(7, 0, 2023, dev)


def test_missing_allows_pet_none_and_auto_all_missing_allows_pet_none():
    dual = _tiny_dual()
    ct = torch.randn(2, 1, 32, 32)
    out = dual(ct, pet=None, forward_mode='missing')
    assert torch.isfinite(out['logits']).all()
    out = dual(ct, pet=None, pet_available=torch.zeros(2, dtype=torch.long),
               forward_mode='auto')
    assert out['num_full'] == 0 and out['num_missing'] == 2


def test_full_requires_pet():
    dual = _tiny_dual()
    with pytest.raises(ValueError):
        dual(torch.randn(1, 1, 32, 32), pet=None, forward_mode='full')
    with pytest.raises(ValueError):
        dual(torch.randn(1, 1, 32, 32), pet=None,
             pet_available=torch.ones(1, dtype=torch.long), forward_mode='auto')


def test_missing_output_independent_of_missing_pet_content():
    torch.manual_seed(0)
    dual = _tiny_dual()
    dual.eval()
    ct = torch.randn(2, 1, 32, 32)
    pet_a = torch.randn(2, 1, 32, 32)
    pet_b = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out_a = dual(ct, pet=pet_a, pet_available=torch.tensor([1, 0]),
                     forward_mode='auto')['logits']
        out_b = dual(ct, pet=pet_b, pet_available=torch.tensor([1, 0]),
                     forward_mode='auto')['logits']
    assert torch.equal(out_a[1], out_b[1])
    assert not torch.equal(out_a[0], out_b[0])


def test_auto_preserves_sample_order():
    torch.manual_seed(1)
    dual = _tiny_dual()
    dual.eval()
    ct = torch.randn(4, 1, 32, 32)
    pet = torch.randn(4, 1, 32, 32)
    state = torch.tensor([1, 0, 1, 0])
    with torch.no_grad():
        mixed = dual(ct, pet=pet, pet_available=state, forward_mode='auto')['logits']
        full_rows = dual(ct[state.eq(1)], pet=pet[state.eq(1)],
                         forward_mode='full')['logits']
        missing_rows = dual(ct[state.eq(0)], pet=None,
                            forward_mode='missing')['logits']
    assert torch.allclose(mixed[state.eq(1)], full_rows, atol=1e-3)
    assert torch.allclose(mixed[state.eq(0)], missing_rows, atol=1e-3)


def test_missing_loss_has_no_pet_encoder_grads():
    torch.manual_seed(2)
    task = MDTSegTeacher({'model': _tiny_dual()}, _make_cfg())
    batch = {'ct': torch.randn(4, 1, 32, 32), 'pet': torch.randn(4, 1, 32, 32),
             'mask': (torch.rand(4, 1, 32, 32) > 0.5).float()}
    state = torch.tensor([1, 1, 0, 0])
    task.model.zero_grad(set_to_none=True)
    ct = batch['ct'].to(task.device)
    pet = batch['pet'].to(task.device)
    mask = batch['mask'].to(task.device)
    outputs = task.model(ct, pet=pet, pet_available=state.to(task.device),
                         forward_mode='auto')
    logits = outputs['logits']
    # Backward the Missing-subset loss only: Missing rows never touched the
    # PET encoder, so it must receive no gradient.
    missing_loss, _ = task.criterion(logits[state.eq(0)], mask[state.eq(0)])
    missing_loss.backward()
    pet_grads = [p.grad for p in task.model.enc_pet.parameters() if p.grad is not None]
    assert pet_grads == [] or all(bool((g == 0).all()) for g in pet_grads)
    for module in (task.model.enc_ct, task.model.ct_align, task.model.decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, 'missing+full loss path must reach CT/align/decoder'


def test_full_loss_updates_all_four_parts():
    torch.manual_seed(3)
    task = MDTSegTeacher({'model': _tiny_dual()}, _make_cfg())
    batch = {'ct': torch.randn(2, 1, 32, 32), 'pet': torch.randn(2, 1, 32, 32),
             'mask': (torch.rand(2, 1, 32, 32) > 0.5).float()}
    task.model.zero_grad(set_to_none=True)
    ct = batch['ct'].to(task.device)
    pet = batch['pet'].to(task.device)
    mask = batch['mask'].to(task.device)
    out = task.model(ct, pet=pet, forward_mode='full')
    full_loss, _ = task.criterion(out['logits'], mask)
    full_loss.backward()
    for name, module in (('enc_ct', task.model.enc_ct), ('enc_pet', task.model.enc_pet),
                         ('ct_align', task.model.ct_align), ('decoder', task.model.decoder)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f'full loss must update {name}'


def test_add_fusion_strictness():
    fusion = AddFusion()
    ct = [torch.randn(2, c, 8, 8) for c in (64, 128, 320, 512)]
    pet = [torch.randn(2, c, 8, 8) for c in (64, 128, 320, 512)]
    assert len(fusion(ct, pet)) == 4
    with pytest.raises(ValueError):
        fusion(ct[:3], pet[:3])
    bad = [p.clone() for p in pet]
    bad[1] = torch.randn(2, 7, 8, 8)
    with pytest.raises(ValueError):
        fusion(ct, bad)
    nan_pet = [p.clone() for p in pet]
    nan_pet[0][0, 0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError):
        fusion(ct, nan_pet)


def test_invalid_state_rejected():
    dual = _tiny_dual()
    ct = torch.randn(2, 1, 32, 32)
    pet = torch.randn(2, 1, 32, 32)
    with pytest.raises(ValueError):
        dual(ct, pet=pet, pet_available=torch.tensor([1]), forward_mode='auto')
    with pytest.raises(ValueError):
        dual(ct, pet=pet, pet_available=torch.tensor([1.0, 0.0]), forward_mode='auto')
    with pytest.raises(ValueError):
        dual(ct, pet=pet, pet_available=torch.tensor([2, 0]), forward_mode='auto')
    with pytest.raises(ValueError):
        dual(ct, pet=pet, forward_mode='bogus')


def test_scheduler_state_dict_roundtrip():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_scheduler(opt, epochs=2, warmup_steps=1, min_lr=1e-6,
                                 steps_per_epoch=2, flat_ratio=0.3)
    state = copy.deepcopy(sched.state_dict())
    sched.step()
    sched.load_state_dict(state)
    assert isinstance(sched.state_dict(), dict)


def test_ema_checkpoint_roundtrip_and_eval(tmp_path):
    torch.manual_seed(4)
    task = MDTSegTeacher({'model': _tiny_dual()},
                         _make_cfg(ema_enabled=True, ema_decay=0.9,
                                   ema_decay_warmup=False, ema_start_epoch=0))
    assert task.ema is not None
    task.update_ema()
    path = tmp_path / 'ckpt.pth.tar'
    task.save_checkpoint(str(path), 1, best=0.5, best_epoch=1, val={'dice': 0.5})
    ckpt = MDTSegTeacher.load_state_dicts(str(path))
    assert ckpt['model_ema'] is not None
    fresh = _tiny_dual()
    fresh.load_state_dict(ckpt['model_ema'], strict=True)
    fresh.eval()
    with torch.no_grad():
        out = fresh(torch.randn(1, 1, 32, 32), pet=torch.randn(1, 1, 32, 32),
                    forward_mode='full')['logits']
    assert torch.isfinite(out).all()


def test_ct_only_dataset_has_no_pet(tmp_path):
    from datasets.pclt20k_seg import PCLT20KSegDataset
    import numpy as np
    from PIL import Image
    img = (np.random.rand(16, 16) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / 'caseA_000_CT.png')
    Image.fromarray((img > 128).astype(np.uint8) * 255).save(tmp_path / 'caseA_000_mask.png')
    records = [{'image_id': 'caseA_000', 'case_id': 'caseA', 'slice_id': 'caseA_000',
                'ct_path': str(tmp_path / 'caseA_000_CT.png'),
                'mask_path': str(tmp_path / 'caseA_000_mask.png')}]
    ds = PCLT20KSegDataset(records, image_size=16, ct_only=True)
    sample = ds[0]
    assert 'pet' not in sample
    assert sample['ct'].shape == (3, 16, 16)
