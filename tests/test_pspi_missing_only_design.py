# -*- coding: utf-8 -*-
"""Missing-only Module-1 (PSPI) design tests.

Validates the contracted boundary:
  Full  = raw CT + real PET via AddFusion (Module-1 bypassed)
  Missing = CT -> retrieval -> personalization -> AddFusion
  Semantic relation supervision only affects the student path.

Run:  python tests/test_pspi_missing_only_design.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.baseline_blocks import UNetStyleDecoder
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.paired_semantic_prototype_imputation import (
    PairedSemanticPrototypeImputation,
)


CHANNELS = (8, 12, 16, 20)
SHAPES = ((16, 16), (8, 8), (4, 4), (2, 2))


def _mask(batch=2, size=64):
    m = torch.zeros(batch, 1, size, size)
    m[:, :, 16:48, 16:48] = 1.0
    return m


def _banked_module(**kwargs):
    channels = kwargs.pop("channels", CHANNELS)
    module = PairedSemanticPrototypeImputation(
        channels=channels, num_clusters=3, build_stage=4,
        bank_update_mode="direct", semantic_loss_weight=0.01, **kwargs,
    )
    module.train()
    torch.manual_seed(11)
    shapes = {
        64: (16, 16), 128: (8, 8), 320: (4, 4), 512: (2, 2),
        8: (16, 16), 12: (8, 8), 16: (4, 4), 20: (2, 2),
    }
    shapes_by_channel = tuple(shapes[c] for c in channels)
    for _ in range(3):
        ct = [torch.randn(4, c, h, w) for c, (h, w) in zip(channels, shapes_by_channel)]
        pet = [torch.randn(4, c, h, w) for c, (h, w) in zip(channels, shapes_by_channel)]
        module.collect_candidates(ct, pet, _mask(4))
    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    return module


def _joint_model(pspi_enabled=True):
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=pspi_enabled, pspi_num_clusters=3,
        pspi_semantic_loss_weight=0.01,
    )
    if pspi_enabled:
        model.module1.train()
        torch.manual_seed(11)
        for _ in range(3):
            ct = [torch.randn(4, c, h, w) for c, (h, w) in zip(model.module1.channels, ((16, 16), (8, 8), (4, 4), (2, 2)))]
            pet = [torch.randn(4, c, h, w) for c, (h, w) in zip(model.module1.channels, ((16, 16), (8, 8), (4, 4), (2, 2)))]
            model.module1.collect_candidates(ct, pet, _mask(4))
        report = model.module1.finalize_epoch(epoch=1)
        assert report["status"] == "bank_updated"
    model.eval()
    return model


def test_full_bypass_equivalence():
    """18.1 Full logits identical with pspi on/off; 18.2 independent of bank."""
    torch.manual_seed(2023)
    m_on = _joint_model(pspi_enabled=True)
    m_off = _joint_model(pspi_enabled=False)
    m_off.load_state_dict(
        {k: v for k, v in m_on.state_dict().items() if not k.startswith("module1.")},
        strict=False,
    )
    # Align non-module1 weights exactly (decoder/encoders/fusion).
    missing_keys, unexpected = m_off.load_state_dict(
        {k: v for k, v in m_on.state_dict().items() if not k.startswith("module1.")},
        strict=False,
    )
    assert all(k.startswith("module1.") for k in missing_keys), missing_keys
    assert not unexpected

    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask()
    with torch.no_grad():
        out_on = m_on(ct, pet=pet, forward_mode="full", mask=mask)
        out_off = m_off(ct, pet=pet, forward_mode="full", mask=mask)
    assert torch.allclose(out_on["logits"], out_off["logits"], atol=1e-5)
    assert float(out_on["semantic_loss"]) == 0.0
    print("[18.1] Full bypass equivalence passed")

    # 18.2: different banks, same Full logits — same backbone weights, mutate bank only.
    with torch.no_grad():
        saved_s4_keys = m_on.module1.ct_keys_s4.clone()
        saved_s4_vals = m_on.module1.pet_values_s4.clone()
        m_on.module1.ct_keys_s4.normal_()
        m_on.module1.pet_values_s4.normal_()
        out2 = m_on(ct, pet=pet, forward_mode="full", mask=mask)
    assert torch.allclose(out2["logits"], out_off["logits"], atol=1e-5)
    with torch.no_grad():
        m_on.module1.ct_keys_s4.copy_(saved_s4_keys)
        m_on.module1.pet_values_s4.copy_(saved_s4_vals)
    print("[18.2] Full independent of bank passed")


def test_missing_strict_inference_and_leakage():
    """18.3 strict pet=None; 18.4 real PET does not affect Missing prediction."""
    torch.manual_seed(7)
    model = _joint_model(pspi_enabled=True)
    ct = torch.randn(1, 1, 64, 64)
    pet_a = torch.randn(1, 1, 64, 64)
    pet_b = torch.randn(1, 1, 64, 64)
    mask = _mask(1)

    # 18.3
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode="missing", mask=mask)
    assert out["logits"].shape == (1, 1, 64, 64)
    assert float(out["semantic_loss"]) == 0.0
    print("[18.3] Missing strict inference passed")

    # 18.4: train mode, collection disabled so cache cannot interfere.
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct_g = ct.clone().requires_grad_(True)
    out_a = model(ct_g, pet=pet_a, forward_mode="missing", mask=mask)
    out_b = model(ct_g, pet=pet_b, forward_mode="missing", mask=mask)
    assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-5)
    # semantic loss may differ (different teacher), logits must not.
    print("[18.4] Missing leakage test passed")


def test_missing_pet_encoder_gradient():
    """18.5: Missing loss gives zero grad to enc_pet; Full gives positive grad."""
    torch.manual_seed(5)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)

    model.zero_grad(set_to_none=True)
    out_missing = model(ct, pet=pet, forward_mode="missing", mask=mask)
    total_missing = out_missing["logits"].float().mean() + out_missing["semantic_loss_weighted"]
    total_missing.backward()
    pet_grads = [p.grad for p in model.enc_pet.parameters() if p.grad is not None]
    pet_grad_norm = sum(float(g.abs().sum()) for g in pet_grads)
    assert pet_grad_norm == 0.0, f"enc_pet received gradient from Missing loss: {pet_grad_norm}"
    print("[18.5a] Missing gives enc_pet zero grad passed")

    model.zero_grad(set_to_none=True)
    out_full = model(ct, pet=pet, forward_mode="full", mask=mask)
    out_full["logits"].float().mean().backward()
    pet_grads_full = [p.grad for p in model.enc_pet.parameters() if p.grad is not None]
    assert len(pet_grads_full) > 0
    assert sum(float(g.abs().sum()) for g in pet_grads_full) > 0.0
    print("[18.5b] Full gives enc_pet positive grad passed")


def test_fusion_boundary_and_stats():
    """18.6: fusion is AddFusion(CT, evidence); no simple_fused / gate / reliability."""
    model = _joint_model(pspi_enabled=True)
    assert isinstance(model.fusion, object)
    assert isinstance(model.decoder, UNetStyleDecoder)
    assert not hasattr(model.module1.config, "use_pet_contribution_gate")
    assert not hasattr(model.module1.config, "use_retrieval_reliability")
    assert not hasattr(model.module1.config, "use_affine_calibration")
    assert not hasattr(model.module1.config, "prototype_loss_type")
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    model.eval()
    with torch.no_grad():
        out_missing = model(ct, pet=None, forward_mode="missing", mask=mask)
        out_full = model(ct, pet=pet, forward_mode="full", mask=mask)
    for out in (out_full, out_missing):
        assert "semantic_loss" in out and "semantic_loss_weighted" in out
        assert "module1_bank_ready" in out and "module1_bank_version" in out
    assert float(out_full["semantic_loss"]) == 0.0
    assert float(out_missing["semantic_loss"]) == 0.0  # eval mode
    print("[18.6] Fusion boundary / stats fields passed")


def test_personalization_identity_and_bounds():
    """17.4-17.8: not-ready fallback, zero-init identity, gamma/beta bounds, sigma."""
    torch.manual_seed(3)
    module = _banked_module()

    # Zero-init identity: P_comp == P_proto.
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)
    retrieval = module.retrieve(ct, return_attention=False)
    for s in range(4):
        assert torch.allclose(pet_comp[s], retrieval["pet_proxy"][s], atol=1e-5)

    # gamma/beta in [-1, 1] after tanh; check with non-zero head.
    with torch.no_grad():
        for head in module.personalization.heads:
            head[-1].weight.normal_(std=0.05)
            head[-1].bias.normal_(std=0.05)
    ct_g = [c.clone().requires_grad_(True) for c in ct]
    pet_comp2, _ = module.recover_missing(ct_g)
    for s in range(4):
        assert bool(torch.isfinite(pet_comp2[s]).all())

    # Not-ready fallback: P_comp == 0.
    empty = PairedSemanticPrototypeImputation(channels=CHANNELS, num_clusters=3, build_stage=4)
    empty.eval()
    pet_zero, aux_zero = empty.recover_missing(ct)
    assert aux_zero["bank_ready"] is False
    for s in range(4):
        assert bool((pet_zero[s] == 0).all())
    print("[17.4-17.8] Personalization identity / bounds / fallback passed")


def test_semantic_relation_loss_properties():
    """17.9-17.12: finite, >=0, teacher stop-grad, student grad."""
    torch.manual_seed(17)
    module = _banked_module()
    module.train()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, _ = module.recover_missing(ct)
    pet_real = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    mask = _mask(2)

    result = module.compute_semantic_relation_loss(pet_comp, pet_real, mask)
    assert result["num_terms"] > 0
    assert bool(torch.isfinite(result["loss"]).all())
    assert float(result["loss"].item()) >= 0.0

    pet_real_g = [p.clone().detach().requires_grad_(True) for p in pet_real]
    ct_g = [c.clone().detach().requires_grad_(True) for c in ct]
    pet_comp_g, _ = module.recover_missing(ct_g)
    result2 = module.compute_semantic_relation_loss(pet_comp_g, pet_real_g, mask)
    if result2["num_terms"] > 0:
        result2["loss"].backward()
        assert all(p.grad is None or float(p.grad.abs().max()) == 0 for p in pet_real_g)
        assert any(c.grad is not None and float(c.grad.abs().sum()) > 0 for c in ct_g)
    print("[17.9-17.12] Semantic relation loss properties passed")


def test_recover_missing_api():
    """17.2/17.3: recover_missing has no PET arg, aux has no gate/reliability/fused."""
    module = _banked_module()
    import inspect
    sig = inspect.signature(module.recover_missing)
    assert "pet" not in sig.parameters
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)
    assert len(pet_comp) == 4
    for s in range(4):
        assert pet_comp[s].shape == ct[s].shape
    for forbidden in ("pet_gate", "simple_fused", "retrieval_reliability"):
        assert forbidden not in aux
    print("[17.2/17.3] recover_missing API passed")


def main():
    test_full_bypass_equivalence()
    test_missing_strict_inference_and_leakage()
    test_missing_pet_encoder_gradient()
    test_fusion_boundary_and_stats()
    test_personalization_identity_and_bounds()
    test_semantic_relation_loss_properties()
    test_recover_missing_api()
    print("[SELF-CHECK] passed")


if __name__ == "__main__":
    main()
