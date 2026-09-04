# -*- coding: utf-8 -*-
"""Stable snapshot-consistent CPPI bank construction tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import (
    PCLT20KSegDataset,
    get_pclt20k_loaders_cipa_aligned,
    get_pclt20k_prototype_loader,
)
from models.ct_pet_prototype_imputation import (
    EPS,
    CrossScaleCTPETPrototypeMemory,
    match_cluster_slots,
    spherical_kmeans,
)
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


CHANNELS = (8, 16, 24, 32)
SPATIAL = (16, 8, 4, 2)
NUM_CLUSTERS = 6


def _tmp_mem(tmp_path, channels=CHANNELS, num_clusters=NUM_CLUSTERS, build_stage=4):
    return CrossScaleCTPETPrototypeMemory(
        channels=channels,
        num_clusters=num_clusters,
        build_stage=build_stage,
        output_dir=str(tmp_path / "cppi"),
    )


def _synthetic_batch(batch_size=8, image_size=32, seed=0, lesion=True):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch_size, 1, image_size, image_size)
    if lesion:
        mask[:, :, 8:24, 8:24] = 1.0
    ct_feats, pet_feats = [], []
    for c, s in zip(CHANNELS, SPATIAL):
        ct = torch.randn(batch_size, c, s, s, generator=g)
        pet = torch.randn(batch_size, c, s, s, generator=g)
        lesion_s = F.adaptive_avg_pool2d(mask, (s, s))
        ct = ct + lesion_s * 0.8
        pet = pet + lesion_s * 1.5
        ct_feats.append(ct)
        pet_feats.append(pet)
    return ct_feats, pet_feats, mask


def _collect_n(memory, n=4, seed=0):
    for i in range(n):
        ct_feats, pet_feats, mask = _synthetic_batch(seed=seed + i)
        memory.collect(ct_feats, pet_feats, mask)


def _tiny_png_dataset(root: Path, ids, size=32):
    for image_id in ids:
        case_id = image_id.split("_")[0]
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((size, size), dtype=np.uint8)
        arr[8:20, 8:20] = 255
        Image.fromarray(arr).save(case_dir / f"{image_id}_CT.png")
        Image.fromarray((arr // 2)).save(case_dir / f"{image_id}_PET.png")
        Image.fromarray(arr).save(case_dir / f"{image_id}_mask.png")


# ---------------------------------------------------------------------------
# Test 1: FP32 cache
# ---------------------------------------------------------------------------

def test_fp32_cache(tmp_path):
    memory = _tmp_mem(tmp_path)
    ct_feats, pet_feats, mask = _synthetic_batch()
    memory.collect(ct_feats, pet_feats, mask)
    assert memory.epoch_cache_dtype() == torch.float32
    for class_idx in range(2):
        for modality in ("ct", "pet"):
            for scale_idx in range(memory.num_scales):
                for chunk in memory._epoch_cache[class_idx][modality][scale_idx]:
                    assert chunk.dtype == torch.float32
                    assert chunk.device.type == "cpu"


# ---------------------------------------------------------------------------
# Test 2: K-means reproducibility
# ---------------------------------------------------------------------------

def test_kmeans_reproducibility():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 16, generator=g)
    labels1, centers1 = spherical_kmeans(x, 6)
    labels2, centers2 = spherical_kmeans(x, 6)
    labels3, centers3 = spherical_kmeans(x.clone(), 6)
    assert torch.equal(labels1, labels2)
    assert torch.equal(labels1, labels3)
    assert torch.equal(centers1, centers2)
    assert torch.equal(centers1, centers3)


# ---------------------------------------------------------------------------
# Test 3: K-means first center is farthest from mean direction
# ---------------------------------------------------------------------------

def test_kmeans_first_center_from_mean_direction():
    pts = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0],
            [0.7, 0.0, 0.7],
        ],
        dtype=torch.float32,
    )
    pts = F.normalize(pts, p=2, dim=-1, eps=EPS)
    assert torch.allclose(pts.norm(dim=1), torch.ones(6), atol=1e-6)

    # Old bug: argmax(norm) after normalize would pick index 0.
    naive_idx = int(torch.argmax(pts.norm(dim=1)).item())
    assert naive_idx == 0

    mean_vec = pts.mean(dim=0)
    mean_dir = F.normalize(mean_vec, dim=0, eps=EPS)
    distance_from_mean = 1.0 - torch.matmul(pts, mean_dir)
    expected_first = int(torch.argmax(distance_from_mean).item())
    assert expected_first == 3

    labels, centers = spherical_kmeans(pts, num_clusters=6)
    first_center = F.normalize(centers[0], dim=0, eps=EPS)
    expected_center = pts[expected_first]
    assert torch.allclose(first_center, expected_center, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(first_center, pts[0], atol=1e-3)


# ---------------------------------------------------------------------------
# Test 4: cluster matching recovers permutation
# ---------------------------------------------------------------------------

def test_cluster_matching_recovers_permutation():
    g = torch.Generator().manual_seed(11)
    old = F.normalize(torch.randn(6, 8, generator=g), dim=-1)
    order = [3, 0, 5, 2, 1, 4]  # new = [D,A,F,C,B,E]
    noise = 1e-4 * torch.randn(6, 8, generator=g)
    new = F.normalize(old[order] + noise, dim=-1)
    perm, mean_cos, min_cos = match_cluster_slots(old, new)
    # perm[i] = new index aligned to old i  => new[perm] ~= old
    inverse = [order.index(i) for i in range(6)]
    assert perm == inverse
    assert mean_cos > 0.99
    assert min_cos > 0.99


def test_cluster_matching_tie_break_lexicographic():
    old = F.normalize(torch.eye(3), dim=-1)
    new = old.clone()
    perm, _, _ = match_cluster_slots(old, new)
    assert perm == [0, 1, 2]


def test_cluster_matching_rejects_n_gt_8():
    old = torch.randn(9, 4)
    new = torch.randn(9, 4)
    with pytest.raises(ValueError, match="N<=8"):
        match_cluster_slots(old, new)


# ---------------------------------------------------------------------------
# Test 5: PET pairing uses the same CT permutation
# ---------------------------------------------------------------------------

def test_pet_pairing_follows_ct_permutation(tmp_path):
    memory = _tmp_mem(tmp_path, num_clusters=6)
    k = memory.num_clusters
    perm_src = [3, 0, 5, 2, 1, 4]
    inverse = [perm_src.index(i) for i in range(k)]

    old_keys = []
    old_values = []
    cand_keys = []
    cand_values = []
    for scale_idx, c in enumerate(CHANNELS):
        g = torch.Generator().manual_seed(100 + scale_idx)
        ok = F.normalize(torch.randn(2, k, c, generator=g), dim=-1)
        ov = torch.arange(2 * k, dtype=torch.float32).view(2, k, 1).repeat(1, 1, c) + scale_idx * 100
        nk = torch.zeros_like(ok)
        nv = torch.zeros_like(ov)
        for class_idx in range(2):
            nk[class_idx] = ok[class_idx][perm_src]
            nv[class_idx] = ov[class_idx][perm_src]
        old_keys.append(ok)
        old_values.append(ov)
        cand_keys.append(nk)
        cand_values.append(nv)
        getattr(memory, f"ct_keys_s{scale_idx + 1}").copy_(ok)
        getattr(memory, f"pet_values_s{scale_idx + 1}").copy_(ov)

    memory.prototype_ready.fill_(True)
    memory.prototype_count.fill_(7)
    memory.bank_version.fill_(1)

    cand_ready = torch.ones(2, k, dtype=torch.bool)
    cand_count = torch.full((2, k), 7, dtype=torch.long)
    matched_k, matched_v, _, _, match_report = memory._match_candidate_to_existing_bank(
        cand_keys, cand_values, cand_ready, cand_count
    )
    for class_name in ("background", "foreground"):
        assert match_report[class_name]["matching_permutation"] == inverse
        assert match_report[class_name]["used_identity"] is False

    for scale_idx in range(memory.num_scales):
        assert torch.allclose(matched_k[scale_idx], old_keys[scale_idx], atol=1e-6)
        assert torch.allclose(matched_v[scale_idx], old_values[scale_idx], atol=1e-6)


def test_pet_not_matched_independently(tmp_path):
    """PET must follow the CT permutation even if a PET-only match would differ."""
    memory = _tmp_mem(tmp_path, num_clusters=6)
    k = memory.num_clusters
    ct_src = [3, 0, 5, 2, 1, 4]
    pet_src = [1, 2, 3, 4, 5, 0]
    assert ct_src != pet_src
    ct_inverse = [ct_src.index(i) for i in range(k)]

    cand_keys, cand_values = [], []
    old_values = []
    for scale_idx, c in enumerate(CHANNELS):
        g = torch.Generator().manual_seed(200 + scale_idx)
        ok = F.normalize(torch.randn(2, k, c, generator=g), dim=-1)
        ov = torch.arange(2 * k, dtype=torch.float32).view(2, k, 1).repeat(1, 1, c) + scale_idx * 100
        nk = torch.zeros_like(ok)
        nv = torch.zeros_like(ov)
        for class_idx in range(2):
            nk[class_idx] = ok[class_idx][ct_src]
            nv[class_idx] = ov[class_idx][pet_src]
        getattr(memory, f"ct_keys_s{scale_idx + 1}").copy_(ok)
        getattr(memory, f"pet_values_s{scale_idx + 1}").copy_(ov)
        old_values.append(ov)
        cand_keys.append(nk)
        cand_values.append(nv)

    memory.prototype_ready.fill_(True)
    memory.prototype_count.fill_(7)
    memory.bank_version.fill_(1)
    cand_ready = torch.ones(2, k, dtype=torch.bool)
    cand_count = torch.full((2, k), 7, dtype=torch.long)
    _, matched_v, _, _, match_report = memory._match_candidate_to_existing_bank(
        cand_keys, cand_values, cand_ready, cand_count
    )
    for class_name in ("background", "foreground"):
        assert match_report[class_name]["matching_permutation"] == ct_inverse

    for scale_idx in range(memory.num_scales):
        expected_pet = torch.stack(
            [cand_values[scale_idx][:, src] for src in ct_inverse], dim=1
        )
        assert torch.allclose(matched_v[scale_idx], expected_pet, atol=1e-6)
        # Independent PET matching would restore old PET values; that must not happen.
        assert not torch.allclose(matched_v[scale_idx], old_values[scale_idx], atol=1e-4)

def test_first_bank_bypasses_ema(tmp_path):
    memory = _tmp_mem(tmp_path)
    assert int(memory.bank_version.item()) == 0
    assert not memory.bank_ready
    _collect_n(memory, n=4, seed=3)
    cand_keys, cand_values, cand_ready, cand_count, _ = memory._build_candidate_bank_from_cache()
    report = memory.finalize_epoch(epoch=1, save_json=False, save_visualizations=False, print_info=False, ema_momentum=0.9)
    assert report["status"] == "bank_updated"
    assert report["update_mode"] == "snapshot_matched_ema"
    assert int(memory.bank_version.item()) == 1
    for scale_idx in range(memory.num_scales):
        got_k = getattr(memory, f"ct_keys_s{scale_idx + 1}").cpu()
        got_v = getattr(memory, f"pet_values_s{scale_idx + 1}").cpu()
        assert torch.allclose(got_k, cand_keys[scale_idx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(got_v, cand_values[scale_idx], atol=1e-6, rtol=1e-6)
        # Wrong first-bank EMA would scale PET by 0.1.
        if cand_ready.any():
            assert not torch.allclose(got_v, 0.1 * cand_values[scale_idx], atol=1e-4)


# ---------------------------------------------------------------------------
# Test 7: EMA after matching
# ---------------------------------------------------------------------------

def test_ema_after_matching(tmp_path):
    memory = _tmp_mem(tmp_path, num_clusters=2, channels=(4, 8, 12, 16))
    m = 0.9
    k = 2
    old_keys, old_values = [], []
    matched_keys, matched_values = [], []
    for scale_idx, c in enumerate(memory.channels):
        old_raw = torch.zeros(2, k, c)
        new_raw = torch.zeros(2, k, c)
        old_raw[..., 0] = 1.0
        new_raw[..., 1] = 1.0
        ok = F.normalize(old_raw, dim=-1)
        nk = F.normalize(new_raw, dim=-1)
        ov = torch.ones(2, k, c) * (10.0 + scale_idx)
        nv = torch.ones(2, k, c) * (20.0 + scale_idx)
        getattr(memory, f"ct_keys_s{scale_idx + 1}").copy_(ok)
        getattr(memory, f"pet_values_s{scale_idx + 1}").copy_(ov)
        old_keys.append(ok)
        old_values.append(ov)
        matched_keys.append(nk)
        matched_values.append(nv)
    memory.prototype_ready.fill_(True)
    memory.prototype_count.fill_(5)
    memory.bank_version.fill_(1)

    matched_ready = torch.ones(2, k, dtype=torch.bool)
    matched_count = torch.full((2, k), 9, dtype=torch.long)
    final_k, final_v, final_ready, final_count, merge_report = memory._merge_candidate_bank_with_ema(
        matched_keys, matched_values, matched_ready, matched_count, ema_momentum=m
    )
    assert bool(final_ready.all())
    assert merge_report["ema_momentum"] == m
    assert merge_report["cppi_bank_drift_after_ema"] < merge_report["cppi_bank_drift_before_ema"]

    for scale_idx in range(memory.num_scales):
        expected_k = F.normalize(m * old_keys[scale_idx] + (1.0 - m) * matched_keys[scale_idx], dim=-1, eps=EPS)
        expected_v = m * old_values[scale_idx] + (1.0 - m) * matched_values[scale_idx]
        assert torch.allclose(final_k[scale_idx], expected_k, atol=1e-6, rtol=1e-6)
        assert torch.allclose(final_v[scale_idx], expected_v, atol=1e-6, rtol=1e-6)
        # PET values must not be L2-normalized.
        pet_norm = final_v[scale_idx].norm(dim=-1)
        assert (pet_norm > 1.5).all()


def test_keep_old_when_new_slot_missing(tmp_path):
    memory = _tmp_mem(tmp_path, num_clusters=2, channels=(4, 8, 12, 16))
    for scale_idx, c in enumerate(memory.channels):
        ok = F.normalize(torch.ones(2, 2, c), dim=-1)
        ov = torch.ones(2, 2, c) * 3.0
        getattr(memory, f"ct_keys_s{scale_idx + 1}").copy_(ok)
        getattr(memory, f"pet_values_s{scale_idx + 1}").copy_(ov)
    memory.prototype_ready.fill_(True)
    memory.prototype_count.fill_(11)
    memory.bank_version.fill_(1)

    matched_keys = [torch.zeros_like(getattr(memory, f"ct_keys_s{s + 1}")) for s in range(4)]
    matched_values = [torch.zeros_like(getattr(memory, f"pet_values_s{s + 1}")) for s in range(4)]
    matched_ready = torch.zeros(2, 2, dtype=torch.bool)
    matched_count = torch.zeros(2, 2, dtype=torch.long)
    final_k, final_v, final_ready, final_count, _ = memory._merge_candidate_bank_with_ema(
        matched_keys, matched_values, matched_ready, matched_count, ema_momentum=0.9
    )
    assert bool(final_ready.all())
    assert torch.equal(final_count, torch.full((2, 2), 11, dtype=torch.long))
    for scale_idx in range(4):
        assert torch.allclose(final_k[scale_idx], getattr(memory, f"ct_keys_s{scale_idx + 1}").cpu())
        assert torch.allclose(final_v[scale_idx], getattr(memory, f"pet_values_s{scale_idx + 1}").cpu())


# ---------------------------------------------------------------------------
# Test 8 / 9: no online collection; snapshot collection grows cache
# ---------------------------------------------------------------------------

def test_no_online_collection_and_snapshot_collects():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        cppi_output_dir=tempfile.mkdtemp(prefix="cppi_"),
    )
    model.train()
    model.prototype_memory.reset_epoch_cache()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    model(ct, pet, mask=mask, forward_mode="full")
    model(ct, pet, mask=mask, forward_mode="missing")
    counts = model.prototype_memory.epoch_cache_num_candidates()
    assert counts["background"] == 0
    assert counts["foreground"] == 0

    model.eval()
    with torch.no_grad():
        model.collect_cppi_snapshot(ct=ct, pet=pet, mask=mask)
    snap_counts = model.prototype_memory.epoch_cache_num_candidates()
    assert snap_counts["background"] > 0
    assert snap_counts["foreground"] > 0


# ---------------------------------------------------------------------------
# Test 10: snapshot candidate bank is deterministic
# ---------------------------------------------------------------------------

def test_snapshot_candidate_bank_deterministic():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        cppi_output_dir=tempfile.mkdtemp(prefix="cppi_"),
    )
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ct = torch.randn(4, 1, 64, 64)
    pet = torch.randn(4, 1, 64, 64)
    mask = torch.zeros(4, 1, 64, 64)
    mask[:, :, 8:40, 8:40] = 1.0
    mask[0].zero_()

    def _candidate_once():
        model.load_state_dict(state)
        model.eval()
        model.prototype_memory.reset_epoch_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                model.collect_cppi_snapshot(ct=ct, pet=pet, mask=mask)
        return model.prototype_memory._build_candidate_bank_from_cache()

    k1, v1, r1, c1, _ = _candidate_once()
    k2, v2, r2, c2, _ = _candidate_once()
    assert torch.equal(r1, r2)
    assert torch.equal(c1, c2)
    for scale_idx in range(4):
        assert torch.allclose(k1[scale_idx], k2[scale_idx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(v1[scale_idx], v2[scale_idx], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 11: prediction path regression with frozen bank
# ---------------------------------------------------------------------------

def test_prediction_path_unchanged_by_snapshot_collect():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        cppi_output_dir=tempfile.mkdtemp(prefix="cppi_"),
    )
    model.eval()
    k = model.prototype_memory.num_clusters
    for scale_idx, c in enumerate(model.prototype_memory.channels):
        keys = F.normalize(torch.randn(2, k, c), dim=-1)
        values = torch.randn(2, k, c)
        getattr(model.prototype_memory, f"ct_keys_s{scale_idx + 1}").copy_(keys)
        getattr(model.prototype_memory, f"pet_values_s{scale_idx + 1}").copy_(values)
    model.prototype_memory.prototype_ready.fill_(True)
    model.prototype_memory.prototype_count.fill_(4)
    model.prototype_memory.bank_version.fill_(3)

    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 10:40, 10:40] = 1.0

    with torch.no_grad():
        full_a = model(ct, pet, mask=mask, forward_mode="full")["logits"].clone()
        miss_a = model(ct, pet, mask=mask, forward_mode="missing")["logits"].clone()
        model.collect_cppi_snapshot(ct=ct, pet=pet, mask=mask)
        full_b = model(ct, pet, mask=mask, forward_mode="full")["logits"].clone()
        miss_b = model(ct, pet, mask=mask, forward_mode="missing")["logits"].clone()

    assert torch.allclose(full_a, full_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(miss_a, miss_b, atol=1e-6, rtol=1e-6)


def test_bank_tensors_are_buffers_not_parameters():
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        cppi_output_dir=tempfile.mkdtemp(prefix="cppi_"),
    )
    buffer_names = set(dict(model.prototype_memory.named_buffers()).keys())
    param_names = set(dict(model.prototype_memory.named_parameters()).keys())
    for scale in range(1, 5):
        assert f"ct_keys_s{scale}" in buffer_names
        assert f"pet_values_s{scale}" in buffer_names
        assert f"ct_keys_s{scale}" not in param_names
        assert f"pet_values_s{scale}" not in param_names
    assert "prototype_ready" in buffer_names
    assert "prototype_count" in buffer_names
    assert "bank_version" in buffer_names
    assert any("q_proj" in n for n in param_names)
    assert any("k_proj" in n for n in param_names)
    assert any("v_proj" in n for n in param_names)
    assert any("out_proj" in n for n in param_names)


def test_old_checkpoint_buffer_keys_still_load(tmp_path):
    src = _tmp_mem(tmp_path / "src")
    _collect_n(src, n=3, seed=1)
    src.finalize_epoch(epoch=1, save_json=False, save_visualizations=False, print_info=False)
    state = src.state_dict()
    dst = _tmp_mem(tmp_path / "dst")
    missing, unexpected = dst.load_state_dict(state, strict=True)
    assert missing == []
    assert unexpected == []
    assert int(dst.bank_version.item()) == int(src.bank_version.item())
    assert torch.equal(dst.prototype_ready, src.prototype_ready)


# ---------------------------------------------------------------------------
# Test 12: build_stage default is 4 / idx 3
# ---------------------------------------------------------------------------

def test_build_stage_default_is_four():
    parser = SegMDTConfig.model_parser()
    assert parser.get_default("cppi_build_stage") == 4
    assert parser.get_default("cppi_bank_ema") == 0.9
    model = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    assert model.prototype_memory.config.build_stage == 4
    assert model.prototype_memory.build_stage_idx == 3


# ---------------------------------------------------------------------------
# Test 13: prototype loader is clean train-only deterministic
# ---------------------------------------------------------------------------

def test_prototype_loader_train_only_no_aug(tmp_path):
    root = tmp_path / "data"
    train_ids = ["0001_0", "0001_1", "0002_0"]
    val_ids = ["0003_0"]
    test_ids = ["0004_0"]
    _tiny_png_dataset(root, train_ids + val_ids + test_ids, size=32)
    (root / "train.txt").write_text("\n".join(train_ids) + "\n")
    (root / "val.txt").write_text("\n".join(val_ids) + "\n")
    (root / "test.txt").write_text("\n".join(test_ids) + "\n")

    loader = get_pclt20k_prototype_loader(
        str(root),
        image_size=32,
        batch_size=2,
        num_workers=0,
        random_state=2023,
        pin_memory=False,
        norm_mode="cipa",
        train_split_file="train.txt",
    )
    ds = loader.dataset
    assert isinstance(ds, PCLT20KSegDataset)
    assert ds.train is False
    assert ds.aug_mode == "none"
    assert loader.drop_last is False
    assert getattr(loader, "shuffle", False) in (False, None)
    # DataLoader stores shuffle on sampler.
    assert not isinstance(loader.sampler, torch.utils.data.RandomSampler)
    ids = [r["image_id"] for r in ds.records]
    assert ids == train_ids
    assert "0003_0" not in ids
    assert "0004_0" not in ids

    batches = list(loader)
    assert sum(b["ct"].shape[0] for b in batches) == len(train_ids)
    # Deterministic order across two loader passes.
    first = torch.cat([b["idx"] for b in batches])
    second = torch.cat([b["idx"] for b in get_pclt20k_prototype_loader(
        str(root), image_size=32, batch_size=2, num_workers=0, pin_memory=False, train_split_file="train.txt"
    )])
    assert torch.equal(first, second)

    train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
        str(root),
        image_size=32,
        batch_size=2,
        num_workers=0,
        random_state=2023,
        pin_memory=False,
        aug_mode="cipa",
        norm_mode="cipa",
        train_split_file="train.txt",
        val_split_file="val.txt",
        test_split_file="test.txt",
        checkpoint_dir=str(tmp_path / "ck"),
    )
    assert train_loader.dataset.train is True
    assert train_loader.dataset.aug_mode == "cipa"
    assert val_loader.dataset.train is False
    assert test_loader.dataset.train is False


# ---------------------------------------------------------------------------
# Smoke: train step + snapshot + first bank + matching EMA + eval
# ---------------------------------------------------------------------------

def test_smoke_snapshot_bank_lifecycle(tmp_path):
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        cppi_output_dir=str(tmp_path / "cppi"),
    )
    cfg = type(
        "C",
        (),
        {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "mixed_precision": False,
            "loss_smooth": 1.0,
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "cppi_bank_ema": 0.9,
            "epochs": 2,
        },
    )()
    task = MDTSegTeacher({"model": model}, cfg)
    opt = task.optimizer

    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 12:50, 12:50] = 1.0
    batch = {"ct": ct, "pet": pet, "mask": mask}

    task.model.train()
    opt.zero_grad(set_to_none=True)
    loss_full, _, _, _ = task.train_step(batch, forward_mode="full")
    loss_full.backward()
    opt.step()

    opt.zero_grad(set_to_none=True)
    loss_miss, _, _, _ = task.train_step(batch, forward_mode="missing")
    loss_miss.backward()
    opt.step()
    assert task.model.prototype_memory.epoch_cache_num_candidates()["background"] == 0

    class _OneBatchLoader:
        def __iter__(self):
            yield {
                "ct": ct.clone(),
                "pet": pet.clone(),
                "mask": mask.clone(),
            }

    from run_mdt_seg import rebuild_cppi_bank_from_snapshot

    report1 = rebuild_cppi_bank_from_snapshot(task, _OneBatchLoader(), epoch=1, cfg=cfg)
    assert report1["status"] == "bank_updated"
    assert int(task.model.prototype_memory.bank_version.item()) == 1
    assert task.model.training is True
    first_keys = getattr(task.model.prototype_memory, "ct_keys_s4").clone()
    first_vals = getattr(task.model.prototype_memory, "pet_values_s4").clone()

    report2 = rebuild_cppi_bank_from_snapshot(task, _OneBatchLoader(), epoch=2, cfg=cfg)
    assert report2["status"] == "bank_updated"
    assert int(task.model.prototype_memory.bank_version.item()) == 2
    assert report2["cppi_bg_match_mean_cos"] is not None
    assert report2["cppi_fg_match_mean_cos"] is not None
    if report2.get("cppi_bank_drift_before_ema") is not None and report2.get("cppi_bank_drift_after_ema") is not None:
        assert report2["cppi_bank_drift_after_ema"] <= report2["cppi_bank_drift_before_ema"] + 1e-6

    second_keys = getattr(task.model.prototype_memory, "ct_keys_s4")
    second_vals = getattr(task.model.prototype_memory, "pet_values_s4")
    # EMA should move keys, but not replace them wholesale with a disjoint bank.
    ready = task.model.prototype_memory.prototype_ready.flatten()
    if bool(ready.any()):
        cos = F.cosine_similarity(
            first_keys.reshape(-1, first_keys.shape[-1])[ready],
            second_keys.reshape(-1, second_keys.shape[-1])[ready],
            dim=-1,
        )
        assert float(cos.mean()) > 0.5

    task.model.eval()
    with torch.no_grad():
        full_logits = task.model(ct.to(task.device), pet.to(task.device), mask=mask.to(task.device), forward_mode="full")["logits"]
        miss_logits = task.model(ct.to(task.device), pet.to(task.device), mask=mask.to(task.device), forward_mode="missing")["logits"]
    assert full_logits.shape == miss_logits.shape == (2, 1, 64, 64)
    assert torch.isfinite(full_logits).all()
    assert torch.isfinite(miss_logits).all()
    del first_vals, second_vals
