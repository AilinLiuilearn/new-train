import json

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.cpbdm_distribution_memory import CTConditionedPETBenefitDistributionMemory
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


def _make_memory(query_dim=4, K=3):
    mem = CTConditionedPETBenefitDistributionMemory(decoder_channels=4, query_dim=query_dim, K=K)
    with torch.no_grad():
        mem.keys.copy_(torch.eye(K, query_dim)[:K])
        mem.pi_zero.zero_()
        mem.pi_pos.fill_(0.5)
        mem.pi_neg.fill_(0.5)
        mem.mu_pos.copy_(torch.tensor([0.3, 0.2, 0.1]))
        mem.mu_neg.copy_(torch.tensor([-0.1, -0.2, -0.3]))
        mem.memory_ready.fill_(True)
    return mem


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


def test_jordan_equivalence_and_nonnegativity():
    torch.manual_seed(0)
    mem = _make_memory(query_dim=4, K=3)
    raw_weights_map = torch.softmax(torch.randn(2, 3, 5, 5), dim=1)
    positive_slot_value = (mem.pi_pos.float() * mem.mu_pos.float().clamp_min(0.0)).view(1, 3, 1, 1)
    negative_slot_value = (mem.pi_neg.float() * (-mem.mu_neg.float()).clamp_min(0.0)).view(1, 3, 1, 1)
    positive_mass_raw = raw_weights_map * positive_slot_value
    negative_mass_raw = raw_weights_map * negative_slot_value
    expected = (raw_weights_map * mem.slot_expected_delta().view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    jordan = positive_mass_raw.sum(dim=1, keepdim=True) - negative_mass_raw.sum(dim=1, keepdim=True)
    assert torch.all(positive_mass_raw >= 0)
    assert torch.all(negative_mass_raw >= 0)
    assert torch.allclose(expected, jordan, atol=1e-6, rtol=1e-5)


def test_mass_conserving_signed_diffusion_properties():
    torch.manual_seed(1)
    mem = _make_memory(query_dim=4, K=3)
    query = torch.randn(2, 4, 5, 5)
    weights_flat = torch.softmax(torch.randn(2, 25, 3), dim=2)
    out = mem._ct_guided_signed_benefit_diffusion(query, weights_flat)
    for key in ['positive_mass_raw', 'negative_mass_raw', 'positive_mass_final', 'negative_mass_final']:
        assert torch.all(torch.isfinite(out[key]))
        assert torch.all(out[key] >= 0)
    assert torch.allclose(out['positive_mass_raw'].sum(dim=(1, 2, 3)), out['positive_mass_final'].sum(dim=(1, 2, 3)), atol=1e-6, rtol=1e-4)
    assert torch.allclose(out['negative_mass_raw'].sum(dim=(1, 2, 3)), out['negative_mass_final'].sum(dim=(1, 2, 3)), atol=1e-6, rtol=1e-4)


def test_zero_mass_safety_and_uniform_invariance():
    torch.manual_seed(2)
    mem = _make_memory(query_dim=4, K=3)
    mem.pi_pos.zero_()
    mem.mu_pos.zero_()
    query = torch.randn(1, 4, 4, 4)
    weights_flat = torch.full((1, 16, 3), 1 / 3)
    out = mem._ct_guided_signed_benefit_diffusion(query, weights_flat)
    assert torch.all(out['positive_mass_final'] == 0)
    assert torch.isfinite(out['final_delta']).all()
    assert torch.allclose(out['final_delta'], out['raw_delta'], atol=1e-6, rtol=1e-5)


def test_query_boundary_protection_and_peak_spread():
    torch.manual_seed(3)
    mem = _make_memory(query_dim=4, K=3)
    query = torch.zeros(1, 4, 5, 5)
    query[:, :, :, :2] = 3.0
    query[:, :, :, 2:] = -3.0
    weights_flat = torch.zeros(1, 25, 3)
    weights_flat[:, :, 0] = 1.0
    out = mem._ct_guided_signed_benefit_diffusion(query, weights_flat)
    assert out['positive_mass_final'].sum() > 0
    assert out['positive_mass_final'].max() <= out['positive_mass_raw'].max() + 1e-6
    assert out['local_affinity'].shape[1] == 9
    assert out['final_delta'].abs().mean() <= out['raw_delta'].abs().mean() + 1e-6


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
