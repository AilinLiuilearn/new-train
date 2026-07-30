import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.build_mdt_seg import build_mdt_seg_teacher
from configs.seg_mdt import SegMDTConfig


def _make_cfg(arch='dual_shared_add_baseline'):
    cfg = SegMDTConfig()
    cfg.model_arch = arch
    cfg.ct_backbone = 'convnextv2_nano'
    cfg.pet_backbone = 'mit_b1'
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.decoder_channels = [64, 32, 16, 8]
    cfg.use_deep_supervision = False
    cfg.deep_supervision = False
    cfg.learning_rate = 8e-5
    cfg.weight_decay = 1e-4
    cfg.loss_smooth = 1.0
    cfg.bce_weight = 1.0
    cfg.dice_weight = 1.0
    cfg.mixed_precision = False
    cfg.pdtm_slots = 2
    cfg.pdtm_eps = 1e-4
    cfg.pdtm_max_pairs = 8
    return cfg


def _make_model(arch='dual_shared_add_baseline'):
    cfg = _make_cfg(arch)
    return build_mdt_seg_teacher(cfg)['model']


def _sample(batch=2, size=64):
    torch.manual_seed(0)
    ct = torch.randn(batch, 3, size, size)
    pet = torch.randn(batch, 3, size, size)
    return ct, pet


def _sync_models(baseline, pdtm):
    pdtm.load_state_dict(baseline.state_dict(), strict=False)


def test_baseline_full_and_pdtm_full_match_when_memory_empty():
    ct, pet = _sample()
    baseline = _make_model('dual_shared_add_baseline').eval()
    pdtm = _make_model('dual_shared_add_pdtm').eval()
    _sync_models(baseline, pdtm)
    with torch.no_grad():
        out1 = baseline(ct, pet=pet, forward_mode='full')['logits']
        out2 = pdtm(ct, pet=pet, forward_mode='full')['logits']
    assert torch.allclose(out1, out2)


def test_missing_empty_memory_matches_baseline():
    ct, pet = _sample()
    baseline = _make_model('dual_shared_add_baseline').eval()
    pdtm = _make_model('dual_shared_add_pdtm').eval()
    _sync_models(baseline, pdtm)
    with torch.no_grad():
        out1 = baseline(ct, pet=pet, forward_mode='missing')['logits']
        out2 = pdtm(ct, pet=pet, forward_mode='missing')['logits']
    assert torch.allclose(out1, out2)


def test_memory_build_and_reload_roundtrip():
    model = _make_model('dual_shared_add_pdtm').eval()
    ct, pet = _sample(batch=4)
    with torch.no_grad():
        model.collect_pdtm_pairs(ct, pet, case_ids=['a', 'b', 'c', 'd'])
        rep = model.finalize_pdtm_memory()
    assert rep['memory_ready']
    sd = model.state_dict()
    model2 = _make_model('dual_shared_add_pdtm').eval()
    model2.load_state_dict(sd)
    assert bool(model2.pdtm.memory_ready.item()) == bool(model.pdtm.memory_ready.item())
    assert int(model2.pdtm.valid_slots.item()) == int(model.pdtm.valid_slots.item())


def test_missing_forward_backward_has_gradients():
    model = _make_model('dual_shared_add_pdtm').train()
    ct, pet = _sample()
    with torch.no_grad():
        model.collect_pdtm_pairs(ct, pet, case_ids=['a', 'b'])
        model.finalize_pdtm_memory()
    ct.requires_grad_()
    out = model(ct, pet=pet, forward_mode='missing')['logits']
    loss = out.mean()
    loss.backward()
    assert ct.grad is not None
    assert torch.isfinite(ct.grad).all()
    assert not any(p.grad is not None for p in model.pdtm.parameters())


def test_auto_mixed_batch_runs():
    model = _make_model('dual_shared_add_pdtm').eval()
    ct, pet = _sample(batch=2)
    pet_available = torch.tensor([1, 0])
    with torch.no_grad():
        out = model(ct, pet=pet, pet_available=pet_available, forward_mode='auto')['logits']
    assert out.shape[:2] == (2, 1)
    assert torch.isfinite(out).all()
