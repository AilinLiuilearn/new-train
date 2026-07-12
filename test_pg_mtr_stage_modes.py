# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from models.pg_mtr import PETGroundedMetabolicTokenRetrieval


def _make_feats(batch_size=2):
    return [
        torch.randn(batch_size, 64, 32, 32),
        torch.randn(batch_size, 128, 16, 16),
        torch.randn(batch_size, 320, 8, 8),
        torch.randn(batch_size, 512, 4, 4),
    ]


def _make_module(stage_mode: str):
    return PETGroundedMetabolicTokenRetrieval(
        [64, 128, 320, 512],
        num_tokens=8,
        temperature=0.07,
        residual_scale_init=0.1,
        stage_mode=stage_mode,
    )


def test_stage_creation():
    m = _make_module("s4")
    assert set(m.stage_modules.keys()) == {"4"}

    m = _make_module("deep")
    assert set(m.stage_modules.keys()) == {"3", "4"}

    m = _make_module("all")
    assert set(m.stage_modules.keys()) == {"1", "2", "3", "4"}


def test_s4_only_missing_path():
    torch.manual_seed(0)
    m = _make_module("s4")
    m.eval()
    feats = _make_feats(batch_size=2)
    missing_feats, aux, diag = m(feats, pet_feats=None, mode="missing")

    assert len(missing_feats) == 4
    for idx in range(3):
        assert torch.equal(missing_feats[idx], feats[idx])
    assert missing_feats[3].shape == feats[3].shape
    assert torch.max(torch.abs(missing_feats[3] - feats[3])).item() < 1e-6
    assert aux == {}
    assert any(k.startswith("pg_mtr_s4_") for k in diag.keys())
    assert not any(k.startswith("pg_mtr_s1_") for k in diag.keys())
    assert not any(k.startswith("pg_mtr_s2_") for k in diag.keys())
    assert not any(k.startswith("pg_mtr_s3_") for k in diag.keys())


def test_full_path_returns_aux_losses():
    torch.manual_seed(0)
    m = _make_module("s4")
    m.eval()
    ct_feats = _make_feats(batch_size=2)
    pet_feats = _make_feats(batch_size=2)
    missing_feats, aux, diag = m(ct_feats, pet_feats=pet_feats, mode="full")

    assert missing_feats is None
    assert set(aux.keys()) == {"pg_mtr_route_loss", "pg_mtr_mem_loss"}
    assert any(k.startswith("pg_mtr_s4_") for k in diag.keys())


def test_invalid_stage_mode_raises():
    try:
        _make_module("invalid")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid stage_mode")


if __name__ == "__main__":
    test_stage_creation()
    test_s4_only_missing_path()
    test_full_path_returns_aux_losses()
    test_invalid_stage_mode_raises()
    print("test_pg_mtr_stage_modes: all checks passed")
