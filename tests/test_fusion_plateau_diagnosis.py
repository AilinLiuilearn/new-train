"""Tests for the standalone diagnostic; execution still required on the server."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "fusion_plateau_diag",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_fusion_plateau.py")
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def test_synthetic_end_to_end_self_test():
    diag.self_test()


def test_shape_mismatch_is_not_silently_broadcast():
    with pytest.raises(ValueError, match="shapes differ"):
        diag.counts(np.ones((2, 2), bool), np.ones((1, 2), bool))


def test_protocol_mismatch_is_rejected():
    cfg = dict(ct_backbone="convnextv2_nano", pet_backbone="mit_b1",
               decoder_channels=[512, 256, 128, 64], image_size_2d=512,
               norm_mode="imagenet", hd95_mode="surface", train_split_file="train.txt",
               val_split_file="val.txt", test_split_file="test.txt")
    other = dict(cfg, norm_mode="cipa")
    with pytest.raises(ValueError, match="norm_mode"):
        diag.validate_protocol(cfg, other)
    diag.validate_protocol(cfg, dict(cfg))


def test_bootstrap_requires_identical_patient_sets():
    with pytest.raises(ValueError, match="identical paired patients"):
        diag.paired_bootstrap({"a": np.zeros(4, np.int64)},
                              {"b": np.zeros(4, np.int64)}, 10, 1)


def test_component_coverage_is_explicit_not_any_overlap():
    gt = np.ones((10, 10), bool)
    pred = np.zeros_like(gt)
    pred[0, 0] = True
    row, = diag.lesion_info(gt, pred, "x", "full", 256, .1)
    assert row["predicted_coverage"] == .01
    assert not row["detected"] and row["small"]


def test_raw_feature_scaling_does_not_count_as_shape_change():
    import torch
    ct = [torch.ones(2, c, 4, 4) for c in (2, 4, 8, 16)]
    pet = [torch.full_like(c, 3) for c in ct]
    native = [(c + p) * .5 for c, p in zip(ct, pet)]
    rows = diag.feature_rows(ct, pet, native, torch.ones(2, 1, 8, 8), ["a", "b"], {})
    assert len(rows) == 8
    for row in rows:
        assert row["native_vs_add_relative_l2"] == pytest.approx(.5)
        assert row["native_vs_scaled_add_relative_l2"] == pytest.approx(0)
        assert row["fitted_native_add_scale"] == pytest.approx(.5)
