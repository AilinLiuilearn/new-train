# -*- coding: utf-8 -*-
"""Stage-1 unimodal pretraining / Stage-2 initialization tests.

Run:  python tests/test_unimodal_stage1_init.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.build_mdt_seg import load_stage1_unimodal_initialization
from models.baseline_blocks import UNetStyleDecoder
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.unimodal_pretrain import CTOnlySegmentationModel, PETOnlySegmentationModel


def _ct_model():
    return CTOnlySegmentationModel(
        ct_backbone="convnextv2_nano",
        ct_pretrained_path=None,
        out_channels=1,
    )


def _pet_model():
    return PETOnlySegmentationModel(
        pet_backbone="mit_b1",
        pet_pretrained_path=None,
        out_channels=1,
    )


def test_ct_only_structure():
    model = _ct_model()
    ct = torch.randn(2, 1, 64, 64)
    out = model(ct, pet=None, forward_mode="missing", target_size=(64, 64))
    assert out["logits"].shape == (2, 1, 64, 64)
    assert "pred" in out and "aux" in out
    assert hasattr(model, "enc_ct")
    assert hasattr(model, "ct_align")
    assert isinstance(model.decoder, UNetStyleDecoder)
    assert not hasattr(model, "enc_pet")
    assert not hasattr(model, "module1")
    assert not hasattr(model, "fusion")
    print("[TEST A] CT-only structure passed")


def test_pet_only_structure():
    model = _pet_model()
    pet = torch.randn(2, 1, 64, 64)
    out = model(pet=pet, target_size=(64, 64))
    assert out["logits"].shape == (2, 1, 64, 64)
    assert "pred" in out and "aux" in out
    assert hasattr(model, "enc_pet")
    assert isinstance(model.decoder, UNetStyleDecoder)
    assert not hasattr(model, "enc_ct")
    assert not hasattr(model, "ct_align")
    assert not hasattr(model, "module1")
    try:
        model(pet=None)
    except ValueError:
        print("[TEST B] PET-only structure passed (pet=None rejected)")
    else:
        raise AssertionError("PET-only forward must reject pet=None")


def test_checkpoint_roundtrip():
    ct_model = _ct_model()
    pet_model = _pet_model()
    with tempfile.TemporaryDirectory() as tmp:
        ct_path = os.path.join(tmp, "ckpt.best_ct.pth.tar")
        pet_path = os.path.join(tmp, "ckpt.best_pet.pth.tar")
        torch.save(
            {
                "stage": "unimodal_pretrain",
                "modality": "ct",
                "model": ct_model.state_dict(),
                "encoder": ct_model.enc_ct.state_dict(),
                "ct_align": ct_model.ct_align.state_dict(),
                "decoder": ct_model.decoder.state_dict(),
            },
            ct_path,
        )
        torch.save(
            {
                "stage": "unimodal_pretrain",
                "modality": "pet",
                "model": pet_model.state_dict(),
                "encoder": pet_model.enc_pet.state_dict(),
                "decoder": pet_model.decoder.state_dict(),
            },
            pet_path,
        )

        joint = DualSharedAddPETCTBaseline(
            ct_pretrained_path=None,
            pet_pretrained_path=None,
            pspi_enabled=True,
            pspi_num_clusters=3,
        )
        decoder_before = {
            k: v.clone() for k, v in joint.decoder.state_dict().items()
        }
        report = load_stage1_unimodal_initialization(joint, ct_path, pet_path)
        assert report["ct_encoder"] and report["ct_align"] and report["pet_encoder"]
        for key, value in joint.decoder.state_dict().items():
            assert torch.equal(decoder_before[key], value), (
                "Stage-2 decoder changed during Stage-1 init"
            )
        assert all(p.requires_grad for p in joint.enc_ct.parameters())
        assert all(p.requires_grad for p in joint.enc_pet.parameters())
        assert all(p.requires_grad for p in joint.ct_align.parameters())
    print("[TEST C/D/E] checkpoint load, decoder untouched, trainability passed")


def test_stage2_strict_missing_inference():
    """Stage-2 joint model: strict Missing inference with pet=None.

    Stage-1.5 bootstrap was removed; the bank starts empty (epoch-1 cold
    start) and compensated PET is strictly zero until epoch-1 finalize.
    """
    joint = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        pspi_enabled=True,
        pspi_num_clusters=3,
    )
    joint.eval()
    assert joint.module1.bank_ready is False
    assert int(joint.module1.bank_version.item()) == 0
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = joint(ct, pet=None, pet_available=None, forward_mode="missing")
    assert out["logits"].shape == (1, 1, 64, 64)
    assert out["module1_bank_ready"] is False
    print("[TEST F/G] epoch-1 cold start + strict missing pet=None passed")


def main():
    torch.manual_seed(2023)
    test_ct_only_structure()
    test_pet_only_structure()
    test_checkpoint_roundtrip()
    test_stage2_strict_missing_inference()
    print("[SELF-CHECK] passed")


if __name__ == "__main__":
    main()
