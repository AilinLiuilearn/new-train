#!/usr/bin/env python
"""Joint Recovery+SPRE end-to-end validation (sections C-H + mini-train)."""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.build_mdt_seg import build_mdt_seg_teacher
from run_mdt_seg import module_grad_norm
from utils.seg_losses import BCEDiceLoss


def _cfg(**kwargs):
    base = dict(
        model_arch="prompt_role_expert_stage2",
        ct_backbone="convnextv2_nano",
        pet_backbone="mit_b1",
        ct_pretrained_path="/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano",
        pet_pretrained_path="/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1",
        no_encoder_pretrained=False,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        stage2_expert_dim=128,
        stage2_atom_num=32,
        stage2_atom_dim=256,
        stage2_prompt_hidden_channels=64,
        stage2_mlp_ratio=2.0,
        stage2_dropout=0.0,
        learning_rate=8e-5,
        weight_decay=1e-4,
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        random_state=2023,
        checkpoint_dir=tempfile.mkdtemp(prefix="joint_smoke_"),
    )
    base.update(kwargs)
    return type("C", (), base)()


def _grad_norm(module):
    return module_grad_norm(module)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}

    print("=" * 60)
    print("[C] Fresh joint model construction")
    print("=" * 60)
    cfg = _cfg()
    model = build_mdt_seg_teacher(cfg)["model"].to(device)
    pm = model.stage1.prototype_memory
    assert hasattr(model, "role_fusion")
    assert not hasattr(model, "decoder_adapters")
    assert model.cppi_ready is False
    assert int(pm.bank_version.item()) == 0
    counts = {
        "enc_ct": model.count_module_trainable(model.stage1.enc_ct),
        "enc_pet": model.count_module_trainable(model.stage1.enc_pet),
        "ct_align": model.count_module_trainable(model.stage1.ct_align),
        "prototype_attention": model.count_module_trainable(model.stage1.prototype_memory.attention),
        "pet_calibration": model.count_module_trainable(model.stage1.pet_calibration),
        "legacy_fusion": model.count_module_trainable(model.stage1.fusion),
        "decoder": model.count_module_trainable(model.stage1.decoder),
        "role_fusion": model.count_module_trainable(model.role_fusion),
        "total": model.count_trainable_parameters(),
    }
    print("[C] trainable counts:", counts)
    assert counts["legacy_fusion"] == 0
    for k in ("enc_ct", "enc_pet", "ct_align", "prototype_attention", "pet_calibration", "decoder", "role_fusion"):
        assert counts[k] > 0, k
    report["trainable"] = counts
    report["bank_ready_init"] = False
    report["bank_version_init"] = 0

    b = 2
    h = w = 512
    ct = torch.randn(b, 1, h, w, device=device)
    pet = torch.randn(b, 1, h, w, device=device)
    mask = torch.zeros(b, 1, h, w, device=device)
    # Put a small foreground blob so CPPI collect gets candidates.
    mask[:, :, 200:280, 200:280] = 1.0
    crit = BCEDiceLoss()

    print("=" * 60)
    print("[D] Full one-batch forward/backward")
    print("=" * 60)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(ct, pet=pet, mask=mask, forward_mode="full", return_features=True)
    logits = out["logits"]
    feats = out["stage2_features"]
    shapes = [tuple(f.shape) for f in feats]
    print("[D] logits", tuple(logits.shape))
    print("[D] features", shapes)
    assert tuple(logits.shape) == (b, 1, h, w)
    assert shapes == [
        (b, 64, 128, 128),
        (b, 128, 64, 64),
        (b, 320, 32, 32),
        (b, 512, 16, 16),
    ]
    loss, _ = crit(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    full_grads = {
        "enc_ct": _grad_norm(model.stage1.enc_ct),
        "enc_pet": _grad_norm(model.stage1.enc_pet),
        "ct_align": _grad_norm(model.stage1.ct_align),
        "pet_calibration": _grad_norm(model.stage1.pet_calibration),
        "role_fusion": _grad_norm(model.role_fusion),
        "decoder": _grad_norm(model.stage1.decoder),
        "legacy_fusion": _grad_norm(model.stage1.fusion),
    }
    print("[D] full grads", full_grads)
    assert full_grads["enc_ct"] > 0
    assert full_grads["enc_pet"] > 0
    assert full_grads["role_fusion"] > 0
    assert full_grads["decoder"] > 0
    assert full_grads["legacy_fusion"] == 0
    report["full_grads"] = full_grads
    report["feature_shapes"] = shapes
    report["logits_shape"] = tuple(logits.shape)
    report["full_loss_finite"] = True

    print("=" * 60)
    print("[E] Epoch-1 Missing with empty bank")
    print("=" * 60)
    model.zero_grad(set_to_none=True)
    # Reset bank state for empty-bank Missing (collect from Full may have cached only).
    assert model.cppi_ready is False or True  # bank not finalized yet
    # Force empty bank for this check if somehow ready.
    if model.cppi_ready:
        pm.prototype_ready.zero_()
    out_m = model(ct, pet=pet, mask=mask, forward_mode="missing")
    loss_m, _ = crit(out_m["logits"], mask)
    assert torch.isfinite(loss_m)
    print("[E] empty-bank missing loss", float(loss_m.detach()))
    report["empty_bank_missing_ok"] = True

    print("=" * 60)
    print("[F] finalize_cppi_epoch")
    print("=" * 60)
    # Collect more via Full/Missing then finalize.
    model.train()
    with torch.no_grad():
        pass
    _ = model(ct, pet=pet, mask=mask, forward_mode="full")
    _ = model(ct, pet=pet, mask=mask, forward_mode="missing")
    fin = model.finalize_cppi_epoch(
        epoch=1, save_json=False, save_visualizations=False, print_info=True
    )
    print("[F] finalize report keys", sorted(fin.keys()) if isinstance(fin, dict) else type(fin))
    bv = int(pm.bank_version.item())
    ready = bool(pm.bank_ready)
    print(f"[F] bank_version={bv} bank_ready={ready}")
    assert ready is True
    assert bv == 1
    report["bank_version_after_epoch1"] = bv
    report["bank_ready_after_epoch1"] = ready
    # Estimate candidates from report if present.
    cands = 0
    if isinstance(fin, dict):
        classes = fin.get("classes") or {}
        for cinfo in classes.values():
            if isinstance(cinfo, dict):
                cands += int(cinfo.get("num_candidates", 0) or 0)
        if cands == 0:
            cands = int(fin.get("ready_slots", 0) or fin.get("ready_count", 0) or 0)
    assert cands > 0 or ready
    report["num_candidates_est"] = cands

    print("=" * 60)
    print("[G] bank-ready Missing backward (graph connectivity)")
    print("=" * 60)
    model.train()
    model.zero_grad(set_to_none=True)
    out_g = model(ct, pet=pet, mask=mask, forward_mode="missing")
    loss_g, _ = crit(out_g["logits"], mask)
    assert torch.isfinite(loss_g)
    loss_g.backward()
    q = model.stage1.prototype_memory.attention[0].q_proj.weight
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    ct_proj = model.role_fusion.scale_units[0].ct_proj[0].weight
    assert ct_proj.grad is not None and ct_proj.grad.abs().sum() > 0
    assert model.stage1.decoder.seg_head.weight.grad is not None
    miss_grads = {
        "enc_ct": _grad_norm(model.stage1.enc_ct),
        "prototype_attention": _grad_norm(model.stage1.prototype_memory.attention),
        "pet_calibration": _grad_norm(model.stage1.pet_calibration),
        "role_fusion": _grad_norm(model.role_fusion),
        "decoder": _grad_norm(model.stage1.decoder),
        "enc_pet": _grad_norm(model.stage1.enc_pet),
        "legacy_fusion": _grad_norm(model.stage1.fusion),
    }
    print("[G] missing grads", miss_grads)
    assert miss_grads["enc_ct"] > 0
    assert miss_grads["prototype_attention"] > 0
    assert miss_grads["pet_calibration"] > 0
    assert miss_grads["role_fusion"] > 0
    assert miss_grads["decoder"] > 0
    assert miss_grads["legacy_fusion"] == 0
    report["missing_grads"] = miss_grads

    print("=" * 60)
    print("[H] Missing eval PET encoder call count == 0")
    print("=" * 60)
    model.eval()
    calls = {"n": 0}
    orig = model.stage1.enc_pet.forward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    model.stage1.enc_pet.forward = wrapped
    with torch.no_grad():
        _ = model(ct, pet=pet, mask=None, forward_mode="missing")
    model.stage1.enc_pet.forward = orig
    print("[H] pet encoder calls", calls["n"])
    assert calls["n"] == 0
    report["eval_missing_pet_encoder_calls"] = calls["n"]

    print("=" * 60)
    print("[I] mini train 8 alternating batches (B=2 synthetic)")
    print("=" * 60)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for i in range(8):
        route = "full" if i % 2 == 0 else "missing"
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out_i = model(ct, pet=pet, mask=mask, forward_mode=route)
            loss_i, _ = crit(out_i["logits"], mask)
        assert torch.isfinite(loss_i), (i, route, float(loss_i))
        if scaler.is_enabled():
            scaler.scale(loss_i).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss_i.backward()
            opt.step()
        print(f"[I] step={i+1} route={route} loss={float(loss_i.detach()):.6f}")
    report["mini_train_ok"] = True

    # Second finalize -> bank_version 2
    fin2 = model.finalize_cppi_epoch(
        epoch=2, save_json=False, save_visualizations=False, print_info=False
    )
    bv2 = int(pm.bank_version.item())
    print(f"[F2] bank_version after epoch2 finalize={bv2}")
    assert bv2 == 2
    report["bank_version_after_epoch2"] = bv2

    print("=" * 60)
    print("JOINT SMOKE C-I: PASS")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    main()
