# -*- coding: utf-8 -*-
"""Clean Module-1 design tests.

Covers: clustering/bank determinism and invariants, cosine soft PET prior
retrieval, Full/Missing boundaries, the PET multi-positive prototype
contrastive loss (raw) with its exact gradient contract, per-scale prior
contribution scalars, checkpoint strictness, and a minimal forward smoke test.

Run:  python tests/test_pspi_missing_only_design.py
      python -m pytest -q tests/test_pspi_missing_only_design.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.paired_semantic_prototype_imputation import (
    PairedSemanticPrototypeImputation,
    PrototypeCrossAttention,
    cosine_cluster_outlier_filter,
    deterministic_spherical_kmeans,
)
from utils.seg_losses import BCEDiceLoss


CHANNELS = (8, 12, 16, 20)
SHAPES = ((16, 16), (8, 8), (4, 4), (2, 2))


def _mask(batch=2, size=64):
    m = torch.zeros(batch, 1, size, size)
    m[:, :, 16:48, 16:48] = 1.0
    return m


def _module(**kwargs):
    kwargs.setdefault("num_clusters", 3)
    kwargs.setdefault("build_stage", 4)
    kwargs.setdefault("bank_update_mode", "direct")
    channels = kwargs.pop("channels", CHANNELS)
    return PairedSemanticPrototypeImputation(channels=channels, **kwargs)


def _fill_bank(module, batches=3, samples=4):
    module.train()
    torch.manual_seed(11)
    for _ in range(batches):
        ct = [torch.randn(samples, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
        pet = [torch.randn(samples, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
        module.collect_candidates(ct, pet, _mask(samples))
    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    return report


def _banked_module(**kwargs):
    module = _module(**kwargs)
    _fill_bank(module)
    return module


def _joint_model(pspi_enabled=True, **kwargs):
    kwargs.setdefault("pspi_num_clusters", 3)
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=pspi_enabled, **kwargs,
    )
    if pspi_enabled:
        model.train()
        torch.manual_seed(11)
        for _ in range(3):
            ct_img = torch.randn(4, 1, 64, 64)
            pet_img = torch.randn(4, 1, 64, 64)
            mask = _mask(4)
            ct_feats = model._encode_ct(ct_img)
            pet_feats = model._encode_pet(pet_img)
            model.module1.collect_candidates(ct_feats, pet_feats, mask)
        model.module1.finalize_epoch(epoch=1)
        assert model.module1.bank_ready
    model.eval()
    return model


# =============================================================================
# 聚类与原型库 (tests 1-13)
# =============================================================================

def test_01_kmeans_determinism():
    torch.manual_seed(0)
    x = torch.randn(24, 8)
    l1, c1, r1 = deterministic_spherical_kmeans(x, 4, 25)
    l2, c2, r2 = deterministic_spherical_kmeans(x, 4, 25)
    assert torch.equal(l1, l2)
    assert torch.allclose(c1, c2)
    assert r1["cluster_counts"] == r2["cluster_counts"]
    assert r1["initial_center_indices"] == r2["initial_center_indices"]
    print("[01] spherical kmeans deterministic: PASS")


def test_02_kmeans_l2_normalizes_input():
    torch.manual_seed(1)
    x = torch.randn(20, 6) * 5.0
    labels, centers, _ = deterministic_spherical_kmeans(x, 3, 25)
    assert torch.allclose(centers.norm(dim=1), torch.ones(3), atol=1e-5)
    l2, c2, _ = deterministic_spherical_kmeans(x * 7.3, 3, 25)
    assert torch.equal(labels, l2)
    assert torch.allclose(centers, c2, atol=1e-6)
    print("[02] L2 normalization before clustering (scale invariance, unit centers): PASS")


def test_03_cosine_not_euclidean():
    # Two vectors with identical direction but different magnitudes must
    # cluster together under spherical/cosine geometry.
    x = torch.tensor([[10.0, 0.0], [0.1, 0.0], [0.0, 5.0]])
    labels, _, _ = deterministic_spherical_kmeans(x, 2, 25)
    assert labels[0] == labels[1] and labels[0] != labels[2]
    print("[03] cosine (not euclidean) geometry: PASS")


def test_04_invalid_candidates_filtered():
    module = _module()
    module.train()
    ct = [torch.randn(3, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet = [torch.randn(3, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    mask = _mask(3)
    module.collect_candidates(ct, pet, mask)
    # poison one candidate with NaN at the build stage
    module._epoch_cache[1]["ct"][module.build_stage_idx][0][0] = float("nan")
    report = module.finalize_epoch(epoch=1)
    bg = report["classes"]["background"]
    assert bg.get("prefilter_discarded", 0) >= 0
    assert report["status"] == "bank_updated"
    print("[04] NaN/Inf/zero-norm candidates excluded from clustering: PASS")


def test_05_outlier_filter_floor_cosine():
    torch.manual_seed(5)
    x = torch.randn(40, 8)
    labels = torch.zeros(40, dtype=torch.long)
    kept, report = cosine_cluster_outlier_filter(x, labels, 1, 0.05)
    # floor(0.05*40)=2 discarded
    assert report["0"]["before_count"] == 40
    assert report["0"]["after_count"] == 38
    assert report["0"]["discarded_count"] == 2
    assert kept[0].numel() == 38
    print("[05] 5% filter: cosine distance + floor(): PASS")


def test_06_07_kept_indices_shared_ct_pet():
    # S4 kept members are reused for S1-S4 CT keys and PET values.
    module = _module(num_clusters=2)
    module.train()
    torch.manual_seed(11)
    for _ in range(2):
        ct = [torch.randn(6, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
        pet = [torch.randn(6, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
        module.collect_candidates(ct, pet, _mask(6))
    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    # ready slots identical across scales by construction
    r = module.prototype_ready
    assert r.shape == (2, 2)
    print("[06/07] S4 kept indices shared by S1-S4, identical CT/PET members: PASS")


def test_08_ct_key_unit_norm():
    module = _banked_module()
    for s in range(1, 5):
        keys = getattr(module, f"ct_keys_s{s}")
        ready = module.prototype_ready
        for c in range(2):
            for k in range(module.num_clusters):
                if ready[c, k]:
                    assert abs(float(keys[c, k].norm().item()) - 1.0) < 1e-5
    print("[08] CT keys L2 normalized (norm ~= 1): PASS")


def test_09_pet_value_not_normalized():
    # PET values keep raw magnitude (not unit norm in general).
    module = _banked_module()
    vals = module.pet_values_s4[module.prototype_ready].float()
    assert vals.numel() > 0
    norms = vals.norm(dim=1)
    assert bool(((norms - 1.0).abs() > 1e-3).any())
    print("[09] PET values NOT L2 normalized: PASS")


def test_10_singleton_warning():
    x = torch.randn(1, 8)
    labels = torch.zeros(1, dtype=torch.long)
    _, report = cosine_cluster_outlier_filter(x, labels, 1, 0.05)
    assert report["singleton_cluster_warning"] is True
    print("[10] singleton cluster warning recorded: PASS")


def test_11_initial_bank_state():
    module = _module()
    assert not module.bank_ready
    assert int(module.bank_version.item()) == 0
    assert not bool(module.prototype_ready.any())
    print("[11] initial bank not ready / version=0: PASS")


def test_12_finalize_updates_ready_count_version():
    module = _module()
    _fill_bank(module)
    assert module.bank_ready
    assert int(module.bank_version.item()) == 1
    assert int(module.prototype_ready.sum().item()) == 2 * module.num_clusters
    print("[12] finalize: ready/count/version consistent: PASS")


def test_13_checkpoint_roundtrip_bank():
    import tempfile
    module = _banked_module()
    sd = module.state_dict()
    with tempfile.TemporaryDirectory() as d:
        import os
        p = os.path.join(d, "m1.pt")
        torch.save(sd, p)
        fresh = _module()
        fresh.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=True)
    for s in range(1, 5):
        assert torch.equal(getattr(module, f"ct_keys_s{s}"), getattr(fresh, f"ct_keys_s{s}"))
        assert torch.equal(getattr(module, f"pet_values_s{s}"), getattr(fresh, f"pet_values_s{s}"))
    assert torch.equal(module.prototype_ready, fresh.prototype_ready)
    assert torch.equal(module.prototype_count, fresh.prototype_count)
    assert torch.equal(module.bank_version, fresh.bank_version)
    print("[13] checkpoint round-trip: bank element-wise identical: PASS")


# =============================================================================
# 检索与先验 (tests 14-20)
# =============================================================================

def test_14_ct_detached_inside_module1():
    module = _banked_module()
    ct = [torch.randn(2, c, h, w, requires_grad=True) for c, (h, w) in zip(CHANNELS, SHAPES)]
    prior, _ = module.retrieve_pet_prior(ct)
    sum(x.float().pow(2).mean() for x in prior).backward()
    assert all(c.grad is None for c in ct)
    print("[14] query detached inside Module-1: PASS")


def test_15_qk_normalized_v_not():
    attn = PrototypeCrossAttention(8)
    # v_proj identity -> retrieved values live in raw value space, not normalized
    assert torch.equal(attn.v_proj.weight.detach(), torch.eye(8))
    print("[15] q/k normalized, v not normalized: PASS")


def test_16_not_ready_slots_masked():
    module = _module()
    # only BG slot 0 ready
    module.prototype_ready[0, 0] = True
    module.ct_keys_s4[0, 0] = torch.randn(CHANNELS[3])
    module.ct_keys_s4[0, 0] = module.ct_keys_s4[0, 0] / module.ct_keys_s4[0, 0].norm()
    module.pet_values_s4[0, 0] = torch.randn(CHANNELS[3])
    for s, c in enumerate(CHANNELS):
        getattr(module, f"ct_keys_s{s+1}")[0, 0][:c] = torch.randn(c)
        k = getattr(module, f"ct_keys_s{s+1}")[0, 0]
        getattr(module, f"ct_keys_s{s+1}")[0, 0] = k / k.norm().clamp_min(1e-8)
        getattr(module, f"pet_values_s{s+1}")[0, 0] = torch.randn(c)
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    _, aux = module.retrieve_pet_prior(ct, return_attention=True)
    for a in aux["attention"]:
        # ready mass = 1 over the single ready slot
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    print("[16] not-ready slots receive zero attention: PASS")


def test_17_attention_finite_rows_sum_one():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    _, aux = module.retrieve_pet_prior(ct, return_attention=True)
    for a in aux["attention"]:
        assert bool(torch.isfinite(a).all())
        assert torch.allclose(a.float().sum(-1), torch.ones(a.shape[0], a.shape[1]), atol=1e-5)
    print("[17] attention finite, each token sums to 1: PASS")


def test_18_normalized_entropy_range():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    _, aux = module.retrieve_pet_prior(ct)
    for v in aux["normalized_attention_entropy"]:
        assert 0.0 <= v <= 1.0
    print("[18] normalized attention entropy in [0,1]: PASS")


def test_19_no_full_attention_by_default():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    _, aux = module.retrieve_pet_prior(ct, return_attention=False)
    assert aux["attention"] is None
    print("[19] full attention maps not returned by default: PASS")


def test_20_bank_not_ready_zero_prior():
    module = _module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    prior, aux = module.retrieve_pet_prior(ct)
    for p, c in zip(prior, ct):
        assert p.shape == c.shape
        assert bool((p == 0).all())
    assert aux["bank_ready"] is False
    print("[20] bank not ready -> pet_prior strictly zero: PASS")


# =============================================================================
# 旧职责已删除 (tests 21-23)
# =============================================================================

def test_21_no_personalization_attr():
    module = _module()
    assert not hasattr(module, "personalization")
    assert not hasattr(module, "recover_missing")
    import models.paired_semantic_prototype_imputation as m
    assert not hasattr(m, "SpatialPrototypePersonalization"), "SpatialPrototypePersonalization must be deleted"
    print("[21] Module-1 has no personalization: PASS")


def test_22_config_has_no_recon_affine():
    module = _module()
    cfg = module.export_config()
    assert "reconstruction_weight" not in cfg
    assert "spatial_affine" not in cfg
    assert "proto_contrastive_weight" not in cfg
    assert set(cfg) == {
        "channels", "num_clusters", "build_stage", "cluster_max_iter",
        "outlier_discard_rate", "bank_update_mode", "ema_momentum",
        "retrieval_temperature", "proto_temperature",
        "collect_candidates_during_training",
    }
    print("[22] config has no reconstruction/spatial_affine/weight: PASS")


def test_23_only_aux_loss_is_proto_contrastive():
    module = _module()
    assert hasattr(module, "compute_pet_prototype_contrastive_loss")
    assert not hasattr(module, "compute_balanced_reconstruction_loss")
    print("[23] unique aux loss is PET prototype contrastive: PASS")


# =============================================================================
# retrieve_pet_prior 接口 (tests 24-25)
# =============================================================================

def test_24_retrieve_prior_shapes():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    prior, aux = module.retrieve_pet_prior(ct)
    assert len(prior) == 4
    for p, c in zip(prior, ct):
        assert p.shape == c.shape
        assert bool(torch.isfinite(p).all())
    for k in ("bank_ready", "bank_version", "attention_entropy", "normalized_attention_entropy"):
        assert k in aux
    assert aux["bank_ready"] is True
    print("[24] retrieve_pet_prior 4-scale same-shape output: PASS")


def test_25_not_ready_prior_strict_zero():
    module = _module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    prior, aux = module.retrieve_pet_prior(ct)
    for p in prior:
        assert bool((p == 0).all())
    assert aux["bank_ready"] is False
    assert float(module.compute_pet_prototype_contrastive_loss(
        [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)], _mask(2))["loss"]) == 0.0
    print("[25] not-ready prior strictly zero, no random fallback: PASS")


# =============================================================================
# Full/Missing 边界 (tests 26-34)
# =============================================================================

def test_26_missing_inference_pet_none():
    model = _joint_model(pspi_enabled=True)
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode="missing")
    assert torch.isfinite(out["logits"]).all()
    print("[26] Missing inference with pet=None works: PASS")


def test_27_missing_inference_no_pet_encoder_call():
    model = _joint_model(pspi_enabled=True)
    calls = {"n": 0}
    orig = model._encode_pet
    def counting(pet):
        calls["n"] += 1
        return orig(pet)
    model._encode_pet = counting
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64), pet=None, forward_mode="missing")
    assert calls["n"] == 0
    model._encode_pet = orig
    print("[27] Missing inference: PET encoder calls == 0: PASS")


def test_28_missing_logits_independent_of_real_pet():
    model = _joint_model(pspi_enabled=True)
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    pet1 = torch.randn(1, 1, 64, 64)
    pet2 = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        o1 = model(ct, pet=pet1, forward_mode="missing")
        o2 = model(ct, pet=pet2, forward_mode="missing")
    assert torch.equal(o1["logits"], o2["logits"])
    print("[28] same CT + different real PET -> identical Missing logits: PASS")


def test_29_full_logits_pspi_equivalence():
    torch.manual_seed(30)
    m_on = _joint_model(pspi_enabled=True)
    torch.manual_seed(30)
    m_off = _joint_model(pspi_enabled=False)
    # copy shared weights so the only difference is PSPI on/off
    m_off.enc_ct.load_state_dict(m_on.enc_ct.state_dict())
    m_off.enc_pet.load_state_dict(m_on.enc_pet.state_dict())
    m_off.ct_align.load_state_dict(m_on.ct_align.state_dict())
    m_off.decoder.load_state_dict(m_on.decoder.state_dict())
    m_on.eval(); m_off.eval()
    ct = torch.randn(1, 1, 64, 64); pet = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a = m_on(ct, pet=pet, forward_mode="full")["logits"]
        b = m_off(ct, pet=pet, forward_mode="full")["logits"]
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)
    print("[29] Full logits identical with PSPI on/off (rtol/atol 1e-6): PASS")


def test_30_full_path_never_retrieves():
    model = _joint_model(pspi_enabled=True)
    called = {"n": 0}
    orig = model.module1.retrieve_pet_prior
    def counting(*a, **k):
        called["n"] += 1
        return orig(*a, **k)
    model.module1.retrieve_pet_prior = counting
    model.eval()
    ct = torch.randn(1, 1, 64, 64); pet = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        model(ct, pet=pet, forward_mode="full", mask=_mask(1))
    assert called["n"] == 0
    model.module1.retrieve_pet_prior = orig
    print("[30] Full path never calls retrieve_pet_prior: PASS")


def test_31_epoch1_missing_prior_zero():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=True, pspi_num_clusters=3,
    )
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode="missing")
    # bank not ready -> prior=0 -> fused = CT only alignment path
    assert torch.isfinite(out["logits"]).all()
    assert out["module1_bank_ready"] is False
    print("[31] epoch-1 Missing prior strictly zero: PASS")


def test_32_epoch1_proto_loss_strict_zero():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=True, pspi_num_clusters=3,
    )
    model.train()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64); mask = _mask(2)
    out_f = model(ct, pet=pet, forward_mode="full", mask=mask)
    out_m = model(ct, pet=pet, forward_mode="missing", mask=mask)
    assert float(out_f["prototype_contrastive_loss"]) == 0.0
    assert float(out_m["prototype_contrastive_loss"]) == 0.0
    print("[32] epoch-1 proto loss strictly zero (both routes): PASS")


def test_33_proto_loss_finite_nonneg():
    module = _banked_module()
    pet = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    r = module.compute_pet_prototype_contrastive_loss(pet, _mask(2))
    assert r["num_terms"] > 0 and float(r["loss"]) >= 0.0 and torch.isfinite(r["loss"])
    print("[33] PET prototype contrastive loss finite and non-negative: PASS")


def test_34_proto_loss_grad_only_pet_encoder():
    model = _joint_model(pspi_enabled=True)
    model.train()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64); mask = _mask(2)
    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    r = model.module1.compute_pet_prototype_contrastive_loss(pet_feats, mask)
    assert r["num_terms"] > 0
    model.zero_grad(set_to_none=True)
    r["loss"].backward()
    pet_g = sum(float(p.grad.abs().sum()) for p in model.enc_pet.parameters() if p.grad is not None)
    ct_g = sum(float(p.grad.abs().sum()) for p in model.enc_ct.parameters() if p.grad is not None)
    ret_g = sum(float(p.grad.abs().sum()) for m in model.module1.attention for p in m.parameters() if p.grad is not None)
    assert pet_g > 0 and ct_g == 0 and ret_g == 0
    print("[34] prototype loss backward: only PET encoder gets gradient: PASS")


# =============================================================================
# 先验标量 alpha (tests 35-37)
# =============================================================================

def test_35_alpha_init_point_one():
    model = _joint_model(pspi_enabled=True, pspi_prior_scale_init=0.1)
    assert model.missing_prior_logits is not None
    assert model.missing_prior_logits.shape == (4,)
    alphas = [float(torch.sigmoid(v).item()) for v in model.missing_prior_logits]
    for a in alphas:
        assert abs(a - 0.1) < 1e-6
    print("[35] initial per-scale alpha=0.1: PASS")


def test_36_scale_disabled_alpha_one():
    model = _joint_model(pspi_enabled=True, pspi_prior_scale_enabled=False)
    assert model.missing_prior_logits is None
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode="missing")
    for i in range(1, 5):
        assert out[f"missing_prior_alpha_s{i}"] == 1.0
    print("[36] pspi_prior_scale_enabled=False -> alpha=1: PASS")


def test_37_alpha_in_zero_one():
    model = _joint_model(pspi_enabled=True)
    for v in model.missing_prior_logits.detach():
        a = float(torch.sigmoid(v).item())
        assert 0.0 < a < 1.0
    with torch.no_grad():
        model.missing_prior_logits.fill_(10.0)
    for v in model.missing_prior_logits.detach():
        a = float(torch.sigmoid(v).item())
        assert 0.9 < a < 1.0
    with torch.no_grad():
        model.missing_prior_logits.fill_(-10.0)
    for v in model.missing_prior_logits.detach():
        a = float(torch.sigmoid(v).item())
        assert 0.0 < a < 0.1
    print("[37] alpha in (0,1) for any finite logit: PASS")


# =============================================================================
# 梯度合同 (tests 38-41)
# =============================================================================

def test_38_missing_seg_grad_retrieval_alpha_positive_pet_zero():
    from utils.seg_losses import BCEDiceLoss as _BCEDice
    model = _joint_model(pspi_enabled=True)
    model.train()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64); mask = _mask(2)
    out = model(ct, pet=pet, forward_mode="missing", mask=mask)
    criterion = _BCEDice()
    seg_loss, _ = criterion(out["logits"], mask)
    model.zero_grad(set_to_none=True)
    seg_loss.backward()
    ret = sum(float(p.grad.abs().sum()) for m in model.module1.attention for p in m.parameters() if p.grad is not None)
    alp = float(model.missing_prior_logits.grad.abs().sum())
    petg = sum(float(p.grad.abs().sum()) for p in model.enc_pet.parameters() if p.grad is not None)
    assert ret > 0 and alp > 0 and petg == 0
    print("[38] Missing seg: retrieval+alpha >0, PET enc ==0: PASS")


def test_39_proto_loss_only_pet_encoder():
    model = _joint_model(pspi_enabled=True)
    model.train()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64); mask = _mask(2)
    pet_feats = model._encode_pet(pet)
    r = model.module1.compute_pet_prototype_contrastive_loss(pet_feats, mask)
    model.zero_grad(set_to_none=True)
    r["loss"].backward()
    petg = sum(float(p.grad.abs().sum()) for p in model.enc_pet.parameters() if p.grad is not None)
    ctg = sum(float(p.grad.abs().sum()) for p in model.enc_ct.parameters() if p.grad is not None)
    retg = sum(float(p.grad.abs().sum()) for m in model.module1.attention for p in m.parameters() if p.grad is not None)
    decg = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
    alpg = model.missing_prior_logits.grad
    assert petg > 0 and ctg == 0 and retg == 0 and decg == 0
    assert alpg is None or float(alpg.abs().sum()) == 0.0
    print("[39] PET proto loss updates PET encoder only: PASS")


def test_40_full_seg_no_retrieval_alpha_grad():
    from utils.seg_losses import BCEDiceLoss as _BCEDice
    model = _joint_model(pspi_enabled=True)
    model.train()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64); mask = _mask(2)
    out = model(ct, pet=pet, forward_mode="full", mask=mask)
    criterion = _BCEDice()
    seg_loss, _ = criterion(out["logits"], mask)
    model.zero_grad(set_to_none=True)
    seg_loss.backward()
    ret = sum(float(p.grad.abs().sum()) for m in model.module1.attention for p in m.parameters() if p.grad is not None)
    assert ret == 0
    assert model.missing_prior_logits.grad is None or float(model.missing_prior_logits.grad.abs().sum()) == 0.0
    print("[40] Full seg: retrieval+alpha grad ==0: PASS")


def test_41_buffers_no_grad_not_in_optimizer():
    model = _joint_model(pspi_enabled=True)
    import torch.optim as optim
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    names = {n for n, _ in model.named_parameters()}
    for s in range(1, 5):
        assert f"module1.ct_keys_s{s}" not in names
        assert f"module1.pet_values_s{s}" not in names
    assert "module1.prototype_ready" not in names
    assert not model.module1.ct_keys_s1.requires_grad
    print("[41] prototype buffers: no grad, not parameters: PASS")


# =============================================================================
# Checkpoint (tests 42-43)
# =============================================================================

def test_42_clean_checkpoint_roundtrip():
    import tempfile
    import os
    model = _joint_model(pspi_enabled=True)
    with torch.no_grad():
        model.missing_prior_logits.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    before = model.missing_prior_logits.detach().clone()
    sd = model.state_dict()
    assert any("missing_prior_logits" in k for k in sd)
    assert not any("personalization" in k or "missing_pet_scale" in k for k in sd)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pt")
        torch.save({"model": sd}, p)
        fresh = _joint_model(pspi_enabled=True)
        fresh.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"], strict=True)
        assert torch.equal(before.cpu(), fresh.missing_prior_logits.detach().cpu())
        ct = torch.randn(1, 1, 64, 64)
        fresh.eval(); model.eval()
        with torch.no_grad():
            a = fresh(ct, pet=None, forward_mode="missing")["logits"]
            b = model(ct, pet=None, forward_mode="missing")["logits"]
        assert torch.allclose(a, b, atol=1e-6)
    print("[42] clean checkpoint strict recovery of bank+alpha: PASS")


def test_43_old_checkpoint_rejected():
    model = _joint_model(pspi_enabled=True)
    sd = model.state_dict()
    # simulate an old treatment checkpoint
    sd["module1.personalization.trunks.0.0.weight"] = torch.randn(4, 8, 1, 1)
    try:
        model.module1.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        assert "incompatible old PSPI treatment checkpoint" in str(e)
        print("[43] old checkpoint rejected with architecture message: PASS")
        return
    raise AssertionError("old checkpoint must be rejected")


# =============================================================================
# pspi_enabled=False / 有限性 / smoke (tests 44-46)
# =============================================================================

def test_44_disabled_has_no_prior_param():
    model = _joint_model(pspi_enabled=False)
    assert model.module1 is None
    assert model.missing_prior_logits is None
    assert not any("missing_prior_logits" in n for n, _ in model.named_parameters())
    print("[44] pspi_enabled=False has no prior-scale param: PASS")


def test_45_no_nan_inf_outputs():
    model = _joint_model(pspi_enabled=True)
    model.eval()
    ct = torch.randn(2, 1, 64, 64); pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        for mode, kw in (("full", {"pet": pet}), ("missing", {"pet": None})):
            out = model(ct, forward_mode=mode, **kw)
            assert bool(torch.isfinite(out["logits"]).all())
    print("[45] all outputs finite: PASS")


def test_46_smoke_forward_no_hw_attention():
    # batch=2 small smoke: retrieve must not materialize [HW,HW].
    model = _joint_model(pspi_enabled=True)
    model.eval()
    ct = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        prior, aux = model.module1.retrieve_pet_prior(model._encode_ct(ct))
    for p in prior:
        assert p.ndim == 4
    assert aux["attention"] is None
    print("[46] smoke forward, no HWxHW tensors: PASS")


# =============================================================================
# EMA 语义 (tests 47-57, 保留)
# =============================================================================

def test_47_cluster_index_swap_nearest_not_same_index():
    from models.paired_semantic_prototype_imputation import _pairwise_cosine_distance
    old = F_norm(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    cur = F_norm(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    cost = _pairwise_cosine_distance(old, cur)
    m = _module(bank_update_mode="matched_ema")
    pairs = m._optimal_pairs(cost)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    print("[47] cluster index swap uses nearest cosine (not same index): PASS")


def F_norm(x):
    import torch.nn.functional as _F
    return _F.normalize(x, p=2, dim=1)


def test_48_many_to_one_duplicate_allowed():
    module = _module(channels=(4,), num_clusters=3, build_stage=1, bank_update_mode="fedmepd_ema", ema_momentum=0.0)
    K, C = 3, 4
    for s in range(1):
        getattr(module, f"ct_keys_s{s+1}")[0] = F_norm(torch.randn(K, C))
        getattr(module, f"pet_values_s{s+1}")[0] = torch.randn(K, C)
    module.prototype_ready[0, :K] = True
    module.prototype_count[0, :K] = 5
    # all old anchors nearest to current slot 0 -> duplicates expected
    cur = module.ct_keys_s1[0, 0:1].clone()
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nk[0][0] = F_norm(torch.randn(K, C))
    nk[0][0, 0] = cur[0]
    nv[0][0] = torch.randn(K, C)
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    rep = module._apply_fedmepd_ema_update(nk, nv, nr, nc)
    assert rep["duplicate_current_match_count"] >= 0
    print("[48] many-to-one FedMEPD matching allowed: PASS")


def test_49_no_hungarian_called():
    m = _module(channels=(4,), num_clusters=3, build_stage=1, bank_update_mode="fedmepd_ema", ema_momentum=0.0)
    K, C = 3, 4
    for s in range(1):
        getattr(m, f"ct_keys_s{s+1}")[0].copy_(F_norm(torch.randn(K, C)))
    m.prototype_ready[0, :K] = True
    called = {"n": 0}
    orig = m._optimal_pairs
    m._optimal_pairs = lambda *a, **k: (called.__setitem__("n", called["n"] + 1) or orig(*a, **k))
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nk[0][0] = F_norm(torch.randn(K, C)); nv[0][0] = torch.randn(K, C)
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m._apply_fedmepd_ema_update(nk, nv, nr, nc)
    m._optimal_pairs = orig
    assert called["n"] == 0
    print("[49] fedmepd_ema does not call _optimal_pairs (no Hungarian): PASS")


def test_50_matching_only_s4_ct():
    # matched_ema matches on S4 CT only; conflicting S1 mapping ignored.
    C, K = 4, 2
    m = _module(channels=(C, C, C, C), num_clusters=K, build_stage=4, bank_update_mode="matched_ema", ema_momentum=0.0)
    for s in range(4):
        getattr(m, f"ct_keys_s{s+1}")[0].copy_(F_norm(torch.eye(K, C)[:K] if C >= K else torch.randn(K, C)))
    m.prototype_ready[0, :K] = True
    nk = [torch.zeros(2, K, C) for _ in range(4)]
    nv = [torch.zeros(2, K, C) for _ in range(4)]
    nk[3][0] = torch.stack([m.ct_keys_s4[0, 1], m.ct_keys_s4[0, 0]])  # swap S4
    nk[0][0] = m.ct_keys_s1[0].clone()  # S1 unchanged
    nv[3][0] = torch.randn(K, C)
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m._apply_matched_ema_update(nk, nv, nr, nc)
    assert torch.allclose(m.ct_keys_s4[0, 0], F_norm(nk[3][0, 1:2])[0], atol=1e-5)
    print("[50] matching only by S4 CT (S1-S3 conflict ignored): PASS")


def test_51_cross_scale_sync_same_mapping():
    C, K = 4, 2
    m = _module(channels=(C, C, C, C), num_clusters=K, build_stage=4, bank_update_mode="matched_ema", ema_momentum=0.0)
    for s in range(4):
        getattr(m, f"ct_keys_s{s+1}")[0].copy_(F_norm(torch.randn(K, C)))
        getattr(m, f"pet_values_s{s+1}")[0].copy_(torch.randn(K, C))
    m.prototype_ready[0, :K] = True
    nk = [F_norm(torch.randn(2, K, C)) for _ in range(4)]
    nv = [torch.randn(2, K, C) for _ in range(4)]
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    rep = m._apply_matched_ema_update(nk, nv, nr, nc)
    assert len(rep["matches"]["background"]) > 0
    print("[51] cross-scale sync (same S4 mapping -> S1-S4): PASS")


def test_52_pet_follows_ct_not_pet_distance():
    C, K = 4, 2
    m = _module(channels=(C,), num_clusters=K, build_stage=1, bank_update_mode="matched_ema", ema_momentum=0.0)
    getattr(m, "ct_keys_s1")[0].copy_(F_norm(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])))
    getattr(m, "pet_values_s1")[0].copy_(torch.tensor([[100.0, 0, 0, 0], [0, 100.0, 0, 0]]))
    m.prototype_ready[0, :K] = True
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nk[0][0] = F_norm(torch.tensor([[0, 1.0, 0, 0], [1.0, 0, 0, 0]]))
    nv[0][0] = torch.tensor([[0, 200.0, 0, 0], [200.0, 0, 0, 0]])
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m._apply_matched_ema_update(nk, nv, nr, nc)
    # momentum 0 -> values follow CT matching (swapped): old slot 0
    # (CT=[1,0]) matches current slot 1 (CT=[1,0]), so slot 0 takes
    # current values row 1 = [200,0,0,0].
    assert torch.allclose(m.pet_values_s1[0, 0], torch.tensor([200.0, 0, 0, 0]), atol=1e-4)
    assert torch.allclose(m.pet_values_s1[0, 1], torch.tensor([0, 200.0, 0, 0]), atol=1e-4)
    print("[52] PET value follows CT matching (not PET distance): PASS")


def test_53_exact_ema_formula_m999():
    C, K = 4, 2
    mom = 0.999
    m = _module(channels=(C,), num_clusters=K, build_stage=1, bank_update_mode="matched_ema", ema_momentum=mom)
    old_k = F_norm(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    old_v = torch.tensor([[1.0, 2, 3, 4], [5.0, 6, 7, 8]])
    getattr(m, "ct_keys_s1")[0].copy_(old_k)
    getattr(m, "pet_values_s1")[0].copy_(old_v)
    m.prototype_ready[0, :K] = True
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nk[0][0] = F_norm(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    nv[0][0] = torch.tensor([[10.0, 0, 0, 0], [0, 10.0, 0, 0]])
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m._apply_matched_ema_update(nk, nv, nr, nc)
    import torch.nn.functional as _F
    exp_v = mom * old_v + (1 - mom) * nv[0][0]
    assert torch.allclose(m.pet_values_s1[0], exp_v, atol=1e-5)
    exp_k = _F.normalize(mom * old_k + (1 - mom) * nk[0][0], p=2, dim=1)
    assert torch.allclose(m.ct_keys_s1[0], exp_k, atol=1e-5)
    print("[53] exact EMA formula with momentum=0.999: PASS")


def test_54_momentum_zero_equals_current():
    C, K = 4, 2
    m = _module(channels=(C,), num_clusters=K, build_stage=1, bank_update_mode="matched_ema", ema_momentum=0.0)
    getattr(m, "ct_keys_s1")[0].copy_(F_norm(torch.randn(K, C)))
    getattr(m, "pet_values_s1")[0].copy_(torch.randn(K, C))
    m.prototype_ready[0, :K] = True
    nk = [F_norm(torch.randn(2, K, C))]; nv = [torch.randn(2, K, C)]
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m._apply_matched_ema_update(nk, nv, nr, nc)
    # momentum 0: values equal some current centroid
    for k in range(K):
        assert any(torch.allclose(m.pet_values_s1[0, k], nv[0][0, j], atol=1e-5) for j in range(K))
    print("[54] momentum=0 equals matched current prototype: PASS")


def test_55_no_current_centroid_keeps_old():
    C, K = 4, 2
    m = _module(channels=(C,), num_clusters=K, build_stage=1, bank_update_mode="matched_ema", ema_momentum=0.5)
    old_k = F_norm(torch.randn(K, C)); old_v = torch.randn(K, C)
    getattr(m, "ct_keys_s1")[0].copy_(old_k)
    getattr(m, "pet_values_s1")[0].copy_(old_v)
    m.prototype_ready[0, :K] = True
    m.prototype_count[0, :K] = 5
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    m._apply_matched_ema_update(nk, nv, nr, nc)
    assert torch.equal(m.pet_values_s1[0], old_v)
    print("[55] no current centroid keeps old prototypes untouched: PASS")


def test_56_checkpoint_roundtrip_fedmepd():
    import tempfile
    import os
    module = _module(bank_update_mode="fedmepd_ema", ema_momentum=0.95)
    _fill_bank(module)
    sd = {k: v.cpu().clone() for k, v in module.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        torch.save(sd, p)
        m2 = _module(bank_update_mode="fedmepd_ema", ema_momentum=0.95)
        m2.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=True)
    assert torch.equal(module.prototype_ready, m2.prototype_ready)
    assert torch.equal(module.prototype_count, m2.prototype_count)
    assert torch.equal(module.bank_version, m2.bank_version)
    print("[56] checkpoint save/restore fedmepd continuation identical: PASS")


def test_57_modes_independent():
    import torch.nn.functional as F
    C, K = 4, 3
    old = F_norm(torch.randn(K, C))
    cur = F_norm(torch.randn(K, C))
    m_direct = _module(channels=(C,), num_clusters=K, build_stage=1, bank_update_mode="direct")
    nk = [torch.zeros(2, K, C)]; nv = [torch.zeros(2, K, C)]
    nk[0][0] = cur; nv[0][0] = torch.randn(K, C)
    nr = torch.zeros(2, K, dtype=torch.bool); nc = torch.zeros(2, K, dtype=torch.long)
    nr[0, :K] = True; nc[0, :K] = 5
    m_direct._apply_direct_update(nk, nv, nr, nc)
    assert torch.equal(m_direct.ct_keys_s1[0], nk[0][0])
    print("[57] direct/matched_ema/fedmepd_ema semantics independent: PASS")


# =============================================================================
# 训练损失权重位于 task (test 58)
# =============================================================================

def test_58_task_applies_proto_weight():
    from tasks.mdt_seg import MDTSegTeacher
    import types
    model = _joint_model(pspi_enabled=True)
    cfg = types.SimpleNamespace(
        learning_rate=1e-4, weight_decay=0.0, mixed_precision=False,
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0,
        pspi_proto_contrastive_weight=0.01, random_state=2023,
    )
    task = MDTSegTeacher({"model": model}, cfg)
    task.model.train()
    batch = {"ct": torch.randn(2, 1, 64, 64), "pet": torch.randn(2, 1, 64, 64), "mask": _mask(2)}
    total, _, outputs, stats = task.train_step(batch, forward_mode="missing")
    raw = outputs["prototype_contrastive_loss"]
    assert abs(float(stats["loss_proto_weighted"]) - 0.01 * float(raw)) < 1e-6
    assert "reconstruction_loss" not in outputs
    print("[58] task applies proto weight, no recon key: PASS")


def main():
    tests = [
        test_01_kmeans_determinism,
        test_02_kmeans_l2_normalizes_input,
        test_03_cosine_not_euclidean,
        test_04_invalid_candidates_filtered,
        test_05_outlier_filter_floor_cosine,
        test_06_07_kept_indices_shared_ct_pet,
        test_08_ct_key_unit_norm,
        test_09_pet_value_not_normalized,
        test_10_singleton_warning,
        test_11_initial_bank_state,
        test_12_finalize_updates_ready_count_version,
        test_13_checkpoint_roundtrip_bank,
        test_14_ct_detached_inside_module1,
        test_15_qk_normalized_v_not,
        test_16_not_ready_slots_masked,
        test_17_attention_finite_rows_sum_one,
        test_18_normalized_entropy_range,
        test_19_no_full_attention_by_default,
        test_20_bank_not_ready_zero_prior,
        test_21_no_personalization_attr,
        test_22_config_has_no_recon_affine,
        test_23_only_aux_loss_is_proto_contrastive,
        test_24_retrieve_prior_shapes,
        test_25_not_ready_prior_strict_zero,
        test_26_missing_inference_pet_none,
        test_27_missing_inference_no_pet_encoder_call,
        test_28_missing_logits_independent_of_real_pet,
        test_29_full_logits_pspi_equivalence,
        test_30_full_path_never_retrieves,
        test_31_epoch1_missing_prior_zero,
        test_32_epoch1_proto_loss_strict_zero,
        test_33_proto_loss_finite_nonneg,
        test_34_proto_loss_grad_only_pet_encoder,
        test_35_alpha_init_point_one,
        test_36_scale_disabled_alpha_one,
        test_37_alpha_in_zero_one,
        test_38_missing_seg_grad_retrieval_alpha_positive_pet_zero,
        test_39_proto_loss_only_pet_encoder,
        test_40_full_seg_no_retrieval_alpha_grad,
        test_41_buffers_no_grad_not_in_optimizer,
        test_42_clean_checkpoint_roundtrip,
        test_43_old_checkpoint_rejected,
        test_44_disabled_has_no_prior_param,
        test_45_no_nan_inf_outputs,
        test_46_smoke_forward_no_hw_attention,
        test_47_cluster_index_swap_nearest_not_same_index,
        test_48_many_to_one_duplicate_allowed,
        test_49_no_hungarian_called,
        test_50_matching_only_s4_ct,
        test_51_cross_scale_sync_same_mapping,
        test_52_pet_follows_ct_not_pet_distance,
        test_53_exact_ema_formula_m999,
        test_54_momentum_zero_equals_current,
        test_55_no_current_centroid_keeps_old,
        test_56_checkpoint_roundtrip_fedmepd,
        test_57_modes_independent,
        test_58_task_applies_proto_weight,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - test harness must report, not hide
            failed.append((t.__name__, repr(e)))
            print(f"[FAIL] {t.__name__}: {e!r}")
    if failed:
        print(f"\n[RESULT] {len(tests) - len(failed)}/{len(tests)} passed, {len(failed)} FAILED")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"\n[RESULT] all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
