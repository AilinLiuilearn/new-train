import json

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_add_cpbdm import DualSharedAddCPBDM


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'ct_backbone': 'convnextv2_nano',
        'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'decoder_channels': [64, 32, 16, 8],
        'use_deep_supervision': False,
        'deep_supervision': False,
        'cpbdm_k': 4,
        'cpbdm_query_dim': 8,
        'model_arch': 'dual_shared_add_baseline',
    }
    base.update(kwargs)
    return type('C', (), base)()


def test_cpbdm_import_and_builders():
    baseline = DualSharedAddPETCTBaseline(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8))
    cpbdm = DualSharedAddCPBDM(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8), cpbdm_k=4, cpbdm_query_dim=8)
    assert hasattr(cpbdm, 'cpbdm')
    assert baseline.decoder.use_deep_supervision is False


def test_builder_branching():
    out = build_mdt_seg_teacher(_make_cfg(model_arch='dual_shared_add_cpbdm'))
    assert 'model' in out
    assert isinstance(out['model'], DualSharedAddCPBDM)
    out2 = build_mdt_seg_teacher(_make_cfg(model_arch='dual_shared_add_baseline'))
    assert isinstance(out2['model'], DualSharedAddPETCTBaseline)


def test_full_path_matches_baseline():
    torch.manual_seed(0)
    baseline = DualSharedAddPETCTBaseline(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8))
    torch.manual_seed(0)
    cpbdm = DualSharedAddCPBDM(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8), cpbdm_k=4, cpbdm_query_dim=8)
    cpbdm.load_state_dict(baseline.state_dict(), strict=False)
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    a = baseline(ct, pet, forward_mode='full')['logits']
    b = cpbdm(ct, pet, forward_mode='full')['logits']
    assert torch.allclose(a, b)


def test_missing_matches_baseline_before_memory_ready():
    torch.manual_seed(1)
    baseline = DualSharedAddPETCTBaseline(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8))
    torch.manual_seed(1)
    cpbdm = DualSharedAddCPBDM(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8), cpbdm_k=4, cpbdm_query_dim=8)
    cpbdm.load_state_dict(baseline.state_dict(), strict=False)
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    a = baseline(ct, pet, forward_mode='missing')['logits']
    b = cpbdm(ct, pet, forward_mode='missing')['logits']
    assert torch.allclose(a, b)


def test_collect_finalize_and_diagnostics(tmp_path):
    model = DualSharedAddCPBDM(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8), cpbdm_k=4, cpbdm_query_dim=8)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    report = model.collect_cpbdm_candidates(ct, pet, mask)
    assert 'fit_selected' in report
    build = model.finalize_cpbdm_memory()
    assert build['memory_ready'] is True
    diag = model.cpbdm_diagnostics()
    assert abs(sum(diag['pi_zero']) + sum(diag['pi_positive']) + sum(diag['pi_negative']) - model.cpbdm.K) < 1e-3
    path = model.export_cpbdm_json(tmp_path, 'epoch_001', build_report=build)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data['diagnostics']['memory_ready'] is True


def test_auto_path_pet_available_switch():
    model = DualSharedAddCPBDM(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', decoder_channels=(64, 32, 16, 8), cpbdm_k=4, cpbdm_query_dim=8)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out = model(ct, pet, pet_available=torch.tensor([1, 0]), forward_mode='auto')
    assert out['logits'].shape == (2, 1, 64, 64)
    assert torch.isfinite(out['logits']).all()
