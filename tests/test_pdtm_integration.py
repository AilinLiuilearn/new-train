import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from models.pdtm_runtime import PDTMRuntime


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


def test_pdtm_batch_retrieval_distinguishes_slots():
    runtime = PDTMRuntime(channels=4, slots=2, eps=1e-4)
    runtime.memory_ready.fill_(True)
    runtime.valid_slots.fill_(2)
    runtime.source_means[0] = torch.zeros(4)
    runtime.source_means[1] = torch.ones(4) * 5.0
    runtime.source_covariances[0] = torch.eye(4)
    runtime.source_covariances[1] = torch.eye(4)
    runtime.delta_means[0] = torch.ones(4) * 0.5
    runtime.delta_means[1] = torch.ones(4) * -0.5
    runtime.operators[0] = torch.eye(4) * 1.0
    runtime.operators[1] = torch.eye(4) * 2.0

    feat0 = torch.zeros(1, 4, 3, 3)
    feat1 = torch.ones(1, 4, 3, 3) * 5.0
    feats = torch.cat([feat0, feat1], dim=0).requires_grad_()
    transported, info = runtime(feats)

    assert transported.requires_grad
    assert transported.shape == feats.shape
    assert info['pdtm_memory_ready'] is True
    assert torch.isfinite(transported).all()
    assert len(runtime._nearest) == 2
    assert runtime._slot_hist.sum().item() == 2
    assert hasattr(runtime, '_last_selected_slots')
    assert runtime._last_selected_slots.shape == (2,)
    assert runtime._last_selected_slots[0].item() != runtime._last_selected_slots[1].item()

    with torch.no_grad():
        mean, cov = runtime._current_stats(feats)
        d0 = runtime._bw2(mean[0], cov[0], 0).item()
        d1 = runtime._bw2(mean[0], cov[0], 1).item()
        e0 = runtime._bw2(mean[1], cov[1], 0).item()
        e1 = runtime._bw2(mean[1], cov[1], 1).item()
    assert d0 < d1
    assert e1 < e0


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
    assert ct.grad.abs().sum().item() > 0
    for module in [model.enc_ct, model.ct_align, model.decoder, model.decoder.seg_head]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


def test_visualization_accepts_string_output_dir(tmp_path):
    model = _make_model('dual_shared_add_pdtm').eval()
    ct, pet = _sample(batch=2)
    with torch.no_grad():
        model.collect_pdtm_pairs(ct, pet, case_ids=['a', 'b'])
    output_dir = str(tmp_path / 'viz')
    paths = model.save_pdtm_visualizations(output_dir, 'epoch_001')
    assert paths
    assert any(str(path).endswith('.png') for path in paths)
    assert all(os.path.exists(path) for path in paths)
