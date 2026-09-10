# -*- coding: utf-8 -*-
"""API-style PSPI Module-1 design tests.

Covers: clustering/bank determinism and invariants, cosine soft retrieval,
API-style spatial affine personalization, Full/Missing boundaries, and the
PET prototype contrastive / balanced reconstruction losses with their exact
gradient contracts.

Run:  python tests/test_pspi_missing_only_design.py
      python -m pytest -q tests/test_pspi_missing_only_design.py
"""

import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.baseline_blocks import UNetStyleDecoder
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.paired_semantic_prototype_imputation import (
    PairedSemanticPrototypeImputation,
    PrototypeCrossAttention,
    SpatialPrototypePersonalization,
    cosine_cluster_outlier_filter,
    deterministic_spherical_kmeans,
)
from tasks.mdt_seg import MDTSegTeacher
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
    return PairedSemanticPrototypeImputation(channels=CHANNELS, **kwargs)


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
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=pspi_enabled, pspi_num_clusters=3, **kwargs,
    )
    if pspi_enabled:
        # Call the model-level API so aligned (real) channel sizes are used.
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
    x = torch.randn(20, 6) * 5.0  # unnormalized magnitudes
    labels, centers, _ = deterministic_spherical_kmeans(x, 3, 25)
    assert torch.allclose(centers.norm(dim=1), torch.ones(3), atol=1e-5)
    # Scale invariance == cosine geometry (normalized before clustering)
    l2, c2, _ = deterministic_spherical_kmeans(x * 7.3, 3, 25)
    assert torch.equal(labels, l2)
    assert torch.allclose(centers, c2, atol=1e-6)
    print("[02] L2 normalization before clustering (scale invariance, unit centers): PASS")


def test_03_cosine_not_euclidean():
    # An outlier far in euclidean norm but same direction must stay in its cluster.
    torch.manual_seed(2)
    base = torch.randn(10, 5)
    scaled = torch.cat([base, base[:1] * 100.0])  # last row same direction, huge norm
    labels, _, _ = deterministic_spherical_kmeans(scaled, 2, 25)
    assert labels[0].item() == labels[-1].item(), "euclidean distance would split direction-equal points"
    print("[03] cosine (not euclidean) geometry: PASS")


def test_04_invalid_candidates_filtered():
    module = _module()
    module.train()
    torch.manual_seed(3)
    ct = [torch.randn(6, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet = [torch.randn(6, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    module.collect_candidates(ct, pet, _mask(6))
    # Corrupt S4 CT candidates: one NaN, one Inf, one zero vector (after concat)
    cache = module._epoch_cache
    cache[1]["ct"][3][0][0] = float("nan")
    cache[1]["ct"][3][0][1] = float("inf")
    cache[1]["ct"][3][0][2] = 0.0
    report = module.finalize_epoch(epoch=1)
    cls = report["classes"]["foreground"]
    assert cls["num_candidates"] == 6
    assert cls["prefilter_discarded"] == 3
    assert cls["clustering"]["num_candidates"] == 3  # only valid rows clustered
    assert report["status"] == "bank_updated"
    print("[04] NaN/Inf/zero-norm candidates excluded from clustering: PASS")


def test_05_outlier_filter_floor_cosine():
    torch.manual_seed(4)
    x = torch.randn(21, 6)
    labels = torch.zeros(21, dtype=torch.long)
    kept, report = cosine_cluster_outlier_filter(x, labels, 1, 0.05)
    # floor(0.05*21)=1 -> keep 20
    assert report["0"]["before_count"] == 21
    assert report["0"]["discarded_count"] == 1
    assert report["0"]["after_count"] == 20
    # floor(0.05*10)=0 -> keep all 10
    x10 = torch.randn(10, 6)
    labels10 = torch.zeros(10, dtype=torch.long)
    _, report10 = cosine_cluster_outlier_filter(x10, labels10, 1, 0.05)
    assert report10["0"]["discarded_count"] == 0
    assert report10["0"]["after_count"] == 10
    # singleton kept
    x1 = torch.randn(1, 6)
    _, report1 = cosine_cluster_outlier_filter(x1, torch.zeros(1, dtype=torch.long), 1, 0.05)
    assert report1["0"]["after_count"] == 1
    print("[05] 5% filter: cosine distance + floor(): PASS")


def _paired_reference_test():
    """Collect a fixed candidate set and snapshot the per-scale caches."""
    module = _module()
    module.train()
    torch.manual_seed(5)
    ct = [torch.randn(8, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet = [torch.randn(8, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    module.collect_candidates(ct, pet, _mask(8))
    snapshot = {}
    for class_idx in range(2):
        snapshot[class_idx] = {
            "ct": [module._concat_cache(class_idx, "ct", s).clone() for s in range(4)],
            "pet": [module._concat_cache(class_idx, "pet", s).clone() for s in range(4)],
        }
    report = module.finalize_epoch(epoch=1)
    return module, snapshot, report


def test_06_07_kept_indices_shared_ct_pet():
    import torch.nn.functional as F
    module, snapshot, report = _paired_reference_test()
    for class_idx in range(2):
        filtering = report["classes"][CLASS_KEY(class_idx)]["filtering"]
        for cluster_key, entry in filtering.items():
            if cluster_key == "singleton_cluster_warning":
                continue
            kept = torch.tensor(entry["kept_indices"], dtype=torch.long)
            for s in range(4):
                ct_all = snapshot[class_idx]["ct"][s]
                pet_all = snapshot[class_idx]["pet"][s]
                expected_key = F.normalize(ct_all[kept].mean(dim=0), dim=0, eps=1e-8)
                expected_val = pet_all[kept].mean(dim=0)
                actual_key = getattr(module, f"ct_keys_s{s+1}")[class_idx, int(cluster_key)]
                actual_val = getattr(module, f"pet_values_s{s+1}")[class_idx, int(cluster_key)]
                assert torch.allclose(actual_key, expected_key, atol=1e-6), f"key mismatch s{s+1}"
                assert torch.allclose(actual_val, expected_val, atol=1e-6), f"value mismatch s{s+1}"
    print("[06/07] S4 kept indices shared by S1-S4, identical CT/PET members: PASS")


def CLASS_KEY(idx):
    return "background" if idx == 0 else "foreground"


def test_08_ct_key_unit_norm():
    module = _banked_module()
    for s in range(1, 5):
        keys = getattr(module, f"ct_keys_s{s}")
        ready = module.prototype_ready
        norms = keys[ready].norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("[08] CT keys L2 normalized (norm ~= 1): PASS")


def test_09_pet_value_not_normalized():
    module = _banked_module()
    values = module.pet_values_s4[module.prototype_ready]
    norms = values.norm(dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3), \
        "PET values must keep raw descriptor magnitude"
    print("[09] PET values NOT L2 normalized: PASS")


def test_10_singleton_warning():
    module = _module(num_clusters=4)
    module.train()
    torch.manual_seed(6)
    ct = [torch.randn(4, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet = [torch.randn(4, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    module.collect_candidates(ct, pet, _mask(4))
    report = module.finalize_epoch(epoch=1)
    # 4 candidates, 4 clusters -> singletons
    found = False
    for cls in report["classes"].values():
        if cls.get("singleton_cluster_warning"):
            found = True
            assert any(e["before_count"] == 1 for k, e in cls["filtering"].items() if k != "singleton_cluster_warning")
    assert found
    print("[10] singleton cluster warning recorded: PASS")


def test_11_initial_bank_state():
    module = _module()
    assert module.bank_ready is False
    assert int(module.bank_version.item()) == 0
    assert int(module.prototype_ready.sum().item()) == 0
    assert int(module.prototype_count.sum().item()) == 0
    print("[11] initial bank not ready / version=0: PASS")


def test_12_finalize_updates_ready_count_version():
    module = _module()
    report = _fill_bank(module)
    assert int(module.bank_version.item()) == 1
    assert report["bank_version_after"] == 1
    assert int(module.prototype_ready.sum().item()) == report["ready_count"]
    assert report["ready_count"] > 0
    assert report["total_slots"] == 2 * module.num_clusters
    for class_idx in range(2):
        counts = module.prototype_count[class_idx]
        ready = module.prototype_ready[class_idx]
        assert torch.equal(counts[ready] > 0, ready)
    print("[12] finalize: ready/count/version consistent: PASS")


def test_13_checkpoint_roundtrip_bank():
    module = _banked_module()
    state = module.state_dict()
    fresh = _module()
    fresh.load_state_dict(state, strict=True)
    for name in [f"ct_keys_s{i}" for i in range(1, 5)] + [f"pet_values_s{i}" for i in range(1, 5)]:
        assert torch.equal(state[name], fresh.state_dict()[name]), name
    assert torch.equal(state["prototype_ready"], fresh.state_dict()["prototype_ready"])
    assert torch.equal(state["prototype_count"], fresh.state_dict()["prototype_count"])
    assert torch.equal(state["bank_version"], fresh.state_dict()["bank_version"])
    # also identical retrieval output after reload
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    module.eval(); fresh.eval()
    p1, _ = module.recover_missing(ct)
    p2, _ = fresh.recover_missing(ct)
    for a, b in zip(p1, p2):
        assert torch.allclose(a, b, atol=0)
    print("[13] checkpoint round-trip: bank element-wise identical: PASS")


# =============================================================================
# 检索 (tests 14-20)
# =============================================================================

def test_14_ct_detached_inside_module1():
    module = _banked_module()
    module.train()
    ct = [torch.randn(2, c, h, w, requires_grad=True) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, _ = module.recover_missing(ct)
    (sum(x.float().pow(2).mean() for x in pet_comp)).backward()
    for c in ct:
        assert c.grad is None, "CT must be detached inside Module-1"
    print("[14] query detached inside Module-1: PASS")


def test_15_qk_normalized_v_not():
    torch.manual_seed(7)
    attn = PrototypeCrossAttention(8, retrieval_temperature=0.1).eval()
    q = torch.randn(1, 8, 4, 4)
    keys = torch.randn(6, 8)
    values = torch.randn(6, 8)
    ready = torch.ones(6, dtype=torch.bool)
    with torch.no_grad():
        out1, a1 = attn(q, keys, values, ready)
        # scaling keys does not change attention (k normalized)
        _, a2 = attn(q, keys * 5.0, values, ready)
        assert torch.allclose(a1, a2, atol=1e-6), "keys must be normalized (cosine)"
        # scaling query does not change attention
        _, a3 = attn(q * 3.0, keys, values, ready)
        assert torch.allclose(a1, a3, atol=1e-6), "query must be normalized (cosine)"
        # scaling values scales retrieved output linearly (v NOT normalized)
        out2, _ = attn(q, keys, values * 2.0, ready)
        assert torch.allclose(out2, 2.0 * out1, atol=1e-5), "values must keep magnitude"
    print("[15] q/k normalized, v not normalized: PASS")


def test_16_not_ready_slots_masked():
    torch.manual_seed(8)
    module = _banked_module(num_clusters=4)
    module.eval()
    # mark 2 of 8 slots not-ready
    module.prototype_ready[0, 2] = False
    module.prototype_ready[1, 3] = False
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    retrieval = module.retrieve(ct, return_attention=True)
    ready_flat = module.prototype_ready.flatten()
    for s, a in enumerate(retrieval["attention"]):
        not_ready_attn = a[0][..., ~ready_flat]
        assert float(not_ready_attn.abs().max()) < 1e-9, "not-ready slots must get no attention"
    print("[16] not-ready slots receive zero attention: PASS")


def test_17_attention_finite_rows_sum_one():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    retrieval = module.retrieve(ct, return_attention=True)
    for a in retrieval["attention"]:
        assert bool(torch.isfinite(a).all())
        sums = a.float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print("[17] attention finite, each token sums to 1: PASS")


def test_18_normalized_entropy_range():
    module = _banked_module(num_clusters=6)
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    aux = module.retrieve(ct)
    for ent, nent in zip(aux["attention_entropy"], aux["normalized_attention_entropy"]):
        assert nent >= 0.0 and nent <= 1.0
        assert ent <= math.log(12) + 1e-4
    print("[18] normalized attention entropy in [0,1]: PASS")


def test_19_no_full_attention_by_default():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)  # default return_attention=False
    assert aux["attention"] is None
    sig = inspect.signature(module.retrieve)
    assert sig.parameters["return_attention"].default is False
    model = _joint_model(pspi_enabled=True)
    out = model(torch.randn(1, 1, 64, 64), pet=torch.randn(1, 1, 64, 64),
                forward_mode="full", mask=_mask(1))
    for key in out:
        assert "attention_map" not in key
        assert not (torch.is_tensor(out[key]) and out[key].ndim == 3), \
            "no [B,N,M] attention tensor may leak into model outputs"
    print("[19] full attention maps not returned by default: PASS")


def test_20_bank_not_ready_zero_comp():
    module = _module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)
    assert aux["bank_ready"] is False
    for p in pet_comp:
        assert bool((p == 0).all()), "compensated PET must be strictly zero before bank exists"
    print("[20] bank not ready -> pet_comp strictly zero: PASS")


# =============================================================================
# API式空间仿射 (tests 21-29)
# =============================================================================

def test_21_gamma_beta_spatial_shapes():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    for s in range(4):
        cond = torch.nn.functional.normalize(ct[s].detach().float(), p=2, dim=1, eps=1e-8).to(ct[s].dtype)
        feat = module.personalization.trunks[s](cond)
        gamma = module.personalization.gamma_heads[s](feat)
        beta = module.personalization.beta_heads[s](feat)
        b, c, h, w = ct[s].shape
        assert tuple(gamma.shape) == (b, c, h, w)
        assert tuple(beta.shape) == (b, c, h, w)
        pet_comp = gamma * pet_proxy[s] + beta
        assert pet_comp.shape == pet_proxy[s].shape
    print("[21] gamma/beta spatial shapes [B,C,H,W]: PASS")


def test_22_no_sigmoid_tanh_on_gamma_beta():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    # amplify head outputs: if tanh/sigmoid were applied, |gamma| would clamp at 1
    with torch.no_grad():
        for head in module.personalization.gamma_heads:
            head.weight.mul_(50.0)
            head.bias.mul_(50.0)
        for head in module.personalization.beta_heads:
            head.weight.mul_(50.0)
            head.bias.mul_(50.0)
    pet_comp, _ = module.recover_missing(ct)
    for s in range(4):
        cond = torch.nn.functional.normalize(ct[s].detach().float(), p=2, dim=1, eps=1e-8).to(ct[s].dtype)
        feat = module.personalization.trunks[s](cond)
        gamma = module.personalization.gamma_heads[s](feat)
        beta = module.personalization.beta_heads[s](feat)
        assert float(gamma.abs().max()) > 1.5, "gamma must be unbounded (no tanh/sigmoid)"
        assert torch.allclose(pet_comp[s], gamma * pet_proxy[s] + beta, atol=1e-4), \
            "affine must be exactly gamma*proto+beta (no clamping)"
    print("[22] gamma/beta unbounded (no sigmoid/tanh): PASS")


def test_23_xavier_init_zero_bias():
    torch.manual_seed(9)
    pers = SpatialPrototypePersonalization(CHANNELS)
    for m in pers.modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.bias is not None
            assert float(m.bias.abs().max()) == 0.0, "bias must be zero-init"
            assert float(m.weight.abs().max()) > 0.0, "weights must not be all-zero"
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            assert float(m.weight.abs().max()) <= bound * 1.001, \
                f"xavier uniform bound violated: {float(m.weight.abs().max())} > {bound}"
    print("[23] Conv2d Xavier uniform weights, zero bias, non-zero last layer: PASS")


def test_24_not_identity_at_init():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    pet_comp, _ = module.recover_missing(ct)
    differs = any(not torch.allclose(pet_comp[s], pet_proxy[s], atol=1e-4) for s in range(4))
    assert differs, "xavier init must NOT reproduce pet_proto (no identity requirement)"
    print("[24] pet_comp != pet_proto at random init: PASS")


def test_25_26_exact_affine_formula():
    module = _banked_module()
    module.eval()
    torch.manual_seed(10)
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    pet_comp, _ = module.recover_missing(ct)
    for s in range(4):
        cond = torch.nn.functional.normalize(ct[s].detach().float(), p=2, dim=1, eps=1e-8).to(ct[s].dtype)
        feat = module.personalization.trunks[s](cond)
        gamma = module.personalization.gamma_heads[s](feat)
        beta = module.personalization.beta_heads[s](feat)
        expected = gamma * pet_proxy[s] + beta
        assert torch.allclose(pet_comp[s], expected, atol=1e-5), \
            "P_comp must equal gamma*P_proto+beta exactly"
        # no extra pet_proto term: P_comp - beta must be gamma*P_proto
        residual = (pet_comp[s] - beta) - gamma * pet_proxy[s]
        assert float(residual.abs().max()) < 1e-5
    print("[25/26] exact gamma*proto+beta, no extra proto term: PASS")


def test_27_no_ct_reference_mean_std():
    module = _banked_module()
    module.eval()
    assert not hasattr(module.personalization, "heads"), "old residual head must be removed"
    assert isinstance(module.personalization, SpatialPrototypePersonalization)
    assert not hasattr(module.personalization, "ct_reference")
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)
    assert aux["ct_reference"] is None
    sig = inspect.signature(module.personalization.forward)
    assert "ct_reference_feats" not in sig.parameters
    assert "reference_valid" not in sig.parameters
    # mean/std-free: shifting the prototype by a constant must scale through gamma only
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    with torch.no_grad():
        cond = torch.nn.functional.normalize(ct[0].detach().float(), p=2, dim=1, eps=1e-8).to(ct[0].dtype)
        feat = module.personalization.trunks[0](cond)
        gamma = module.personalization.gamma_heads[0](feat)
        beta = module.personalization.beta_heads[0](feat)
        shifted = pet_proxy[0] + 3.7
        manual_shifted = gamma * shifted + beta
        # if old formula (proto + gamma*(proto-mu) + beta*sigma) were used,
        # the shift would cancel through mu and change differently.
        pet_comp_shift, _ = module.personalization([ct[0]], [shifted])
        assert torch.allclose(pet_comp_shift[0], manual_shifted, atol=1e-4), \
            "personalization must not use prototype mean/std statistics"
    print("[27] no ct_reference / prototype mean / std: PASS")


def test_28_spatial_affine_false_passthrough():
    module = _banked_module(spatial_affine=False)
    module.eval()
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_proxy = module.retrieve(ct)["pet_proxy"]
    pet_comp, _ = module.recover_missing(ct)
    for s in range(4):
        assert torch.allclose(pet_comp[s], pet_proxy[s], atol=1e-6), \
            "spatial_affine=False must bypass personalization"
    print("[28] pspi_spatial_affine=False -> pet_comp == pet_proto: PASS")


def test_29_finite_and_shapes():
    module = _banked_module()
    module.eval()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, aux = module.recover_missing(ct)
    for s in range(4):
        assert pet_comp[s].shape == ct[s].shape
        assert bool(torch.isfinite(pet_comp[s]).all())
    assert math.isfinite(aux["gamma_abs_mean"]) and math.isfinite(aux["beta_abs_mean"])
    print("[29] pet_comp finite with correct shapes: PASS")


# =============================================================================
# Full/Missing 边界 (tests 30-36)
# =============================================================================

def test_30_full_logits_pspi_equivalence():
    torch.manual_seed(2023)
    m_on = _joint_model(pspi_enabled=True)
    m_off = _joint_model(pspi_enabled=False)
    missing_keys, unexpected = m_off.load_state_dict(
        {k: v for k, v in m_on.state_dict().items() if not k.startswith("module1.")},
        strict=False,
    )
    assert all(k.startswith("module1.") for k in missing_keys)
    assert not unexpected
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask(2)
    with torch.no_grad():
        out_on = m_on(ct, pet=pet, forward_mode="full", mask=mask)
        out_off = m_off(ct, pet=pet, forward_mode="full", mask=mask)
    assert torch.allclose(out_on["logits"], out_off["logits"], rtol=1e-6, atol=1e-6)
    print("[30] Full logits identical with PSPI on/off (rtol/atol 1e-6): PASS")


def test_31_full_logits_independent_of_bank():
    torch.manual_seed(11)
    model = _joint_model(pspi_enabled=True)
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    with torch.no_grad():
        out1 = model(ct, pet=pet, forward_mode="full", mask=mask)
        for s in range(1, 5):
            getattr(model.module1, f"ct_keys_s{s}").normal_()
            getattr(model.module1, f"pet_values_s{s}").normal_()
        model.module1.prototype_ready.fill_(True)
        out2 = model(ct, pet=pet, forward_mode="full", mask=mask)
        model.module1.prototype_ready.fill_(False)
        out3 = model(ct, pet=pet, forward_mode="full", mask=mask)
    assert torch.allclose(out1["logits"], out2["logits"], rtol=1e-6, atol=1e-6)
    assert torch.allclose(out1["logits"], out3["logits"], rtol=1e-6, atol=1e-6)
    print("[31] prototype bank content cannot change Full logits: PASS")


def test_32_missing_inference_pet_none():
    torch.manual_seed(12)
    model = _joint_model(pspi_enabled=True)
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=None, pet_available=None, forward_mode="missing")
    assert out["logits"].shape == (1, 1, 64, 64)
    assert bool(torch.isfinite(out["logits"]).all())
    print("[32] Missing inference with pet=None works: PASS")


def test_33_missing_inference_no_pet_encoder_call():
    torch.manual_seed(13)
    model = _joint_model(pspi_enabled=True)
    calls = {"n": 0}
    orig = model.enc_pet.forward

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    model.enc_pet.forward = spy
    try:
        with torch.no_grad():
            model(torch.randn(1, 1, 64, 64), pet=None, forward_mode="missing")
    finally:
        model.enc_pet.forward = orig
    assert calls["n"] == 0, f"PET encoder called {calls['n']} times in Missing inference"
    print("[33] Missing inference: PET encoder calls == 0: PASS")


def test_34_missing_logits_independent_of_real_pet():
    torch.manual_seed(14)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet_a = torch.randn(1, 1, 64, 64)
    pet_b = torch.randn(1, 1, 64, 64) * 4.0 + 1.0
    mask = _mask(1)
    torch.manual_seed(777)
    out_a = model(ct, pet=pet_a, forward_mode="missing", mask=mask)
    torch.manual_seed(777)
    out_b = model(ct, pet=pet_b, forward_mode="missing", mask=mask)
    assert torch.allclose(out_a["logits"], out_b["logits"], rtol=1e-6, atol=1e-6), \
        "real PET must not influence Missing logits"
    print("[34] same CT + different real PET -> identical Missing logits: PASS")


def test_35_epoch1_missing_zero_comp():
    torch.manual_seed(15)
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=True, pspi_num_clusters=3,
    )
    model.train()
    assert model.module1.bank_ready is False
    assert int(model.module1.bank_version.item()) == 0
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    pet_comp, aux = model.module1.recover_missing(model._encode_ct(ct))
    for p in pet_comp:
        assert bool((p == 0).all()), "epoch-1 compensated PET must be strictly zero"
    assert aux["bank_ready"] is False
    out = model(ct, pet=pet, forward_mode="missing", mask=mask)
    assert float(out["prototype_contrastive_loss"]) == 0.0
    assert float(out["reconstruction_loss"]) == 0.0
    assert float(out["prototype_contrastive_loss_weighted"]) == 0.0
    assert float(out["reconstruction_loss_weighted"]) == 0.0
    print("[35] epoch-1 Missing compensated PET strictly zero: PASS")


def test_36_epoch1_losses_strict_zero():
    torch.manual_seed(16)
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=True, pspi_num_clusters=3,
    )
    model.train()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    out_full = model(ct, pet=pet, forward_mode="full", mask=mask)
    out_missing = model(ct, pet=pet, forward_mode="missing", mask=mask)
    for out, name in ((out_full, "full"), (out_missing, "missing")):
        assert float(out["prototype_contrastive_loss"]) == 0.0, name
        assert float(out["prototype_contrastive_loss_weighted"]) == 0.0, name
        assert float(out["reconstruction_loss"]) == 0.0, name
        assert float(out["reconstruction_loss_weighted"]) == 0.0, name
        assert out["prototype_contrastive_num_terms"] == 0, name
        assert out["reconstruction_num_terms"] == 0, name
    print("[36] epoch-1 proto/reconstruction losses strictly zero (both routes): PASS")


# =============================================================================
# 损失与梯度 (tests 37-45)
# =============================================================================

def test_37_proto_loss_finite_nonneg():
    module = _banked_module()
    module.train()
    pet = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    result = module.compute_pet_prototype_contrastive_loss(pet, _mask(2))
    assert result["num_terms"] > 0
    assert bool(torch.isfinite(result["loss"]).all())
    assert float(result["loss"].item()) >= 0.0
    print("[37] PET prototype contrastive loss finite and non-negative: PASS")


def test_38_proto_loss_grad_only_pet_encoder():
    torch.manual_seed(17)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)

    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    result = model.module1.compute_pet_prototype_contrastive_loss(pet_feats, mask)
    assert result["num_terms"] > 0
    model.zero_grad(set_to_none=True)
    result["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.enc_pet.parameters()), \
        "PET encoder must receive gradient from prototype loss"
    for module_ref, name in ((model.enc_ct, "enc_ct"), (model.decoder, "decoder")):
        assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in module_ref.parameters()), name
    for attn in model.module1.attention:
        assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in attn.parameters()), "retrieval"
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.module1.personalization.parameters()), "personalization"
    print("[38] prototype loss backward: only PET encoder gets gradient: PASS")


def test_39_recon_loss_finite_nonneg():
    module = _banked_module()
    module.train()
    ct = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    pet_comp, _ = module.recover_missing(ct)
    pet_real = [torch.randn(2, c, h, w) for c, (h, w) in zip(CHANNELS, SHAPES)]
    result = module.compute_balanced_reconstruction_loss(pet_comp, pet_real, _mask(2))
    assert result["num_terms"] == 4
    assert bool(torch.isfinite(result["loss"]).all())
    assert float(result["loss"].item()) >= 0.0
    print("[39] reconstruction loss finite and non-negative: PASS")


def test_40_recon_loss_grad_boundary():
    torch.manual_seed(18)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)

    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    pet_comp, _ = model.module1.recover_missing(ct_feats)
    result = model.module1.compute_balanced_reconstruction_loss(pet_comp, pet_feats, mask)
    assert result["num_terms"] > 0
    model.zero_grad(set_to_none=True)
    result["loss"].backward()

    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.enc_ct.parameters()), "CT encoder must get 0 grad"
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.enc_pet.parameters()), "PET encoder must get 0 grad"
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.decoder.parameters()), "decoder must get 0 grad"
    retrieval_grad = sum(
        float(p.grad.abs().sum()) for attn in model.module1.attention for p in attn.parameters() if p.grad is not None
    )
    assert retrieval_grad > 0, "retrieval projections must get gradient"
    pers_grad = sum(
        float(p.grad.abs().sum()) for p in model.module1.personalization.parameters() if p.grad is not None
    )
    assert pers_grad > 0, "spatial personalization must get gradient"
    print("[40] reconstruction loss backward: retrieval+personalization > 0, encoders/decoder == 0: PASS")


def test_41_missing_seg_loss_no_pet_grad():
    torch.manual_seed(19)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    out = model(ct, pet=pet, forward_mode="missing", mask=mask)
    criterion = BCEDiceLoss()
    seg_loss, _ = criterion(out["logits"], mask)
    model.zero_grad(set_to_none=True)
    seg_loss.backward()
    pet_grads = [p.grad for p in model.enc_pet.parameters() if p.grad is not None]
    assert sum(float(g.abs().sum()) for g in pet_grads) == 0.0, \
        "Missing segmentation loss must not reach PET encoder"
    print("[41] Missing seg loss: PET encoder gradient == 0: PASS")


def test_42_missing_total_loss_pet_grad_positive():
    torch.manual_seed(20)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = _mask(1)
    out = model(ct, pet=pet, forward_mode="missing", mask=mask)
    criterion = BCEDiceLoss()
    seg_loss, _ = criterion(out["logits"], mask)
    total = seg_loss + out["prototype_contrastive_loss_weighted"].reshape(()) + out["reconstruction_loss_weighted"].reshape(())
    model.zero_grad(set_to_none=True)
    total.backward()
    pet_grad = sum(
        float(p.grad.abs().sum()) for p in model.enc_pet.parameters() if p.grad is not None
    )
    assert pet_grad > 0.0, "bank-ready Missing total loss must update PET encoder (via proto loss)"
    print("[42] bank-ready Missing total loss: PET encoder gradient > 0: PASS")


def test_43_buffers_no_grad_not_in_optimizer():
    model = _joint_model(pspi_enabled=True)
    cfg = type("C", (), {
        "learning_rate": 1e-4, "weight_decay": 1e-4, "mixed_precision": False,
        "loss_smooth": 1.0, "bce_weight": 1.0, "dice_weight": 1.0, "random_state": 2023,
    })()
    task = MDTSegTeacher({"model": model}, cfg)
    opt_params = {id(p) for group in task.optimizer.param_groups for p in group["params"]}
    model_params = {id(p) for p in model.parameters()}
    for name, buf in model.module1.named_buffers():
        assert id(buf) not in opt_params, f"buffer {name} must not be in optimizer"
        assert id(buf) not in model_params, f"buffer {name} must not be a parameter"
    for name in [f"ct_keys_s{i}" for i in range(1, 5)] + [f"pet_values_s{i}" for i in range(1, 5)]:
        assert not getattr(model.module1, name).requires_grad
    print("[43] prototype buffers: no grad, not parameters, not in optimizer: PASS")


def test_44_module1_trainable_in_optimizer():
    model = _joint_model(pspi_enabled=True)
    cfg = type("C", (), {
        "learning_rate": 1e-4, "weight_decay": 1e-4, "mixed_precision": False,
        "loss_smooth": 1.0, "bce_weight": 1.0, "dice_weight": 1.0, "random_state": 2023,
    })()
    task = MDTSegTeacher({"model": model}, cfg)
    opt_params = {id(p) for group in task.optimizer.param_groups for p in group["params"]}
    trainable = [p for p in model.module1.parameters() if p.requires_grad]
    assert len(trainable) > 0
    for p in trainable:
        assert id(p) in opt_params, "all Module-1 trainable params must be in optimizer"
    # 16 retrieval (4 scales x 4 Linear), 32 personalization
    n_retrieval = sum(1 for attn in model.module1.attention for _ in attn.parameters())
    n_pers = sum(1 for _ in model.module1.personalization.parameters())
    assert n_retrieval == 16, n_retrieval
    assert n_pers == 32, n_pers  # trunks(8) + gamma(8) + beta(8) but counted per Conv2d weight+bias = 32
    print("[44] Module-1 trainable params all in unified optimizer: PASS")


def test_45_no_nan_inf_outputs():
    torch.manual_seed(21)
    model = _joint_model(pspi_enabled=True)
    model.train()
    model.module1.config.collect_candidates_during_training = False
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask(2)
    with torch.no_grad():
        out_full = model(ct, pet=pet, forward_mode="full", mask=mask)
        out_missing = model(ct, pet=pet, forward_mode="missing", mask=mask)
    for out, name in ((out_full, "full"), (out_missing, "missing")):
        assert bool(torch.isfinite(out["logits"]).all()), name
    ct_feats = model._encode_ct(ct)
    for s, f in enumerate(ct_feats):
        assert bool(torch.isfinite(f).all()), f"ct_feats s{s+1}"
        pet_comp, _ = model.module1.recover_missing([x[s:s+1] for x in ct_feats])
        assert bool(torch.isfinite(pet_comp[0]).all()), f"pet_comp s{s+1}"
    print("[45] Full/Missing logits and 4-scale features finite: PASS")


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
        test_20_bank_not_ready_zero_comp,
        test_21_gamma_beta_spatial_shapes,
        test_22_no_sigmoid_tanh_on_gamma_beta,
        test_23_xavier_init_zero_bias,
        test_24_not_identity_at_init,
        test_25_26_exact_affine_formula,
        test_27_no_ct_reference_mean_std,
        test_28_spatial_affine_false_passthrough,
        test_29_finite_and_shapes,
        test_30_full_logits_pspi_equivalence,
        test_31_full_logits_independent_of_bank,
        test_32_missing_inference_pet_none,
        test_33_missing_inference_no_pet_encoder_call,
        test_34_missing_logits_independent_of_real_pet,
        test_35_epoch1_missing_zero_comp,
        test_36_epoch1_losses_strict_zero,
        test_37_proto_loss_finite_nonneg,
        test_38_proto_loss_grad_only_pet_encoder,
        test_39_recon_loss_finite_nonneg,
        test_40_recon_loss_grad_boundary,
        test_41_missing_seg_loss_no_pet_grad,
        test_42_missing_total_loss_pet_grad_positive,
        test_43_buffers_no_grad_not_in_optimizer,
        test_44_module1_trainable_in_optimizer,
        test_45_no_nan_inf_outputs,
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
