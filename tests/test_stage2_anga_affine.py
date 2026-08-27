"""Stage-2 ANGA / paired-affine contract tests (lightweight, no real dataset)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import PrototypeReferencedPETAffineCalibration
from models.stage2_prompt_role_expert_fusion import PromptRoleExpertStage2Seg
from utils.affine_gradient_alignment import (
    EPS,
    clear_all_affine_grads,
    get_affine_head_param_groups,
    merge_affine_grads_per_scale,
    project_missing_onto_full_cone,
    snapshot_all_affine_grads,
    write_param_grads,
)


class _DummyEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.channels = list(channels)
        self.feature_info = type("FI", (), {"channels": lambda self=None: list(channels)})()
        self.probe = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        b = x.shape[0]
        outs = []
        spatial = [(32, 32), (24, 24), (18, 18), (16, 16)]
        for c, (h, w) in zip(self.channels, spatial):
            outs.append(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype) + 0.01 * self.probe)
        return outs


class _DummyAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        return [f + 0.0 * self.bias for f in feats]


class _DummyDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512), decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        self.use_deep_supervision = False
        c1, c2, c3, c4 = channels
        d4, d3, d2, d1 = decoder_channels
        self.proj4 = nn.Conv2d(c4, d4, 1)
        self.proj3 = nn.Conv2d(c3, d3, 1)
        self.proj2 = nn.Conv2d(c2, d2, 1)
        self.proj1 = nn.Conv2d(c1, d1, 1)
        self.fuse3 = nn.Conv2d(d4 + d3, d3, 1)
        self.fuse2 = nn.Conv2d(d3 + d2, d2, 1)
        self.fuse1 = nn.Conv2d(d2 + d1, d1, 1)
        self.seg_head = nn.Conv2d(d1, 1, 1)

    def forward(self, features, target_size):
        # Unused by Stage2 wrapper (adapters call submodule pieces), kept for completeness.
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], 1))
        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        logits = F.interpolate(self.seg_head(d1), size=target_size, mode="bilinear", align_corners=False)
        return {"logits": logits}


class _DummyProto(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.channels = list(channels)
        self.register_buffer("bank_version", torch.tensor(7, dtype=torch.long))
        self.register_buffer("prototype_ready", torch.ones(2, 6, dtype=torch.bool))
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def bank_ready(self):
        return bool(self.prototype_ready.any())

    def retrieve(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        pet_proxy = [torch.randn_like(x) for x in ct_feats]
        # Keep reference distinct from CT so affine delta is nonzero in tests.
        ct_ref = [x.clone() + 0.25 for x in ct_feats]
        report = {}
        if return_ct_reference:
            return pet_proxy, ct_ref, report
        return pet_proxy, report


class DummyStage1(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.enc_ct = _DummyEncoder(channels)
        self.enc_pet = _DummyEncoder(channels)
        self.ct_align = _DummyAlign()
        self.pet_calibration = PrototypeReferencedPETAffineCalibration(channels)
        self.fusion = nn.Identity()
        self.decoder = _DummyDecoder(channels)
        self.prototype_memory = _DummyProto(channels)

    def _to_3ch(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        return self.ct_align(self.enc_ct(self._to_3ch(ct)))

    def _encode_pet(self, pet):
        return self.enc_pet(self._to_3ch(pet))

    def _retrieve_cppi(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        return self.prototype_memory.retrieve(
            ct_feats,
            compute_report=compute_report,
            save_diagnostics=save_diagnostics,
            print_info=print_info,
            return_ct_reference=return_ct_reference,
        )


def _make_stage2(allow_affine=False, seed=0):
    torch.manual_seed(seed)
    stage1 = DummyStage1()
    model = PromptRoleExpertStage2Seg(
        stage1_model=stage1,
        channels=(64, 128, 320, 512),
        decoder_channels=(512, 256, 128, 64),
        allow_affine_trainable=allow_affine,
        require_ready_cppi_for_missing=True,
    )
    return model


def test_stage2_full_missing_shapes():
    model = _make_stage2(allow_affine=False)
    model.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out_f = model(ct, pet=pet, forward_mode="full")
    out_m = model(ct, pet=pet, forward_mode="missing")
    assert out_f["logits"].shape == out_m["logits"].shape == (2, 1, 64, 64)


def test_stage2_missing_pet_encoder_not_called():
    model = _make_stage2(allow_affine=False)
    model.train()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    model.stage1.enc_pet.calls = 0
    _ = model(ct, pet=pet, forward_mode="missing")
    assert model.stage1.enc_pet.calls == 0


def test_stage2_full_pet_encoder_called_once():
    model = _make_stage2(allow_affine=False)
    model.train()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    model.stage1.enc_pet.calls = 0
    _ = model(ct, pet=pet, forward_mode="full")
    assert model.stage1.enc_pet.calls == 1


def test_stage2_boundary_frozen_and_affine():
    m0 = _make_stage2(allow_affine=False)
    m0.assert_stage1_boundary_contract(allow_affine=False)
    for n, p in m0.stage1.named_parameters():
        assert p.requires_grad is False, n

    m1 = _make_stage2(allow_affine=True)
    m1.assert_stage1_boundary_contract(allow_affine=True)
    for n, p in m1.stage1.named_parameters():
        if n.startswith("pet_calibration."):
            assert p.requires_grad is True, n
        else:
            assert p.requires_grad is False, n


def test_stage2_affine_grads_do_not_leak_to_encoders():
    model = _make_stage2(allow_affine=True)
    model.train()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    model.zero_grad(set_to_none=True)
    out = model(ct, pet=pet, forward_mode="full")
    loss = F.binary_cross_entropy_with_logits(out["logits"], mask)
    loss.backward()

    for p in model.stage1.pet_calibration.parameters():
        assert p.grad is not None
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.stage1.pet_calibration.parameters()
    )
    for module in (
        model.stage1.enc_ct,
        model.stage1.enc_pet,
        model.stage1.ct_align,
        model.stage1.decoder,
        model.stage1.prototype_memory,
    ):
        for p in module.parameters():
            assert p.grad is None


def test_alternating_frozen_forward_equivalence():
    """Same weights/eval: Full/Missing logits stable under evidence refactor."""
    torch.manual_seed(123)
    stage1 = DummyStage1()
    # Force deterministic encoder outputs by zeroing noise path via probe and seeded randn
    model_a = PromptRoleExpertStage2Seg(
        stage1_model=stage1,
        allow_affine_trainable=False,
        require_ready_cppi_for_missing=True,
    )
    model_b = PromptRoleExpertStage2Seg(
        stage1_model=copy.deepcopy(stage1),
        allow_affine_trainable=False,
        require_ready_cppi_for_missing=True,
    )
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval()
    model_b.eval()

    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)

    # Freeze encoder randomness by patching encode to fixed tensors
    with torch.no_grad():
        fixed_ct = [torch.randn(1, c, h, w) for c, (h, w) in zip((64, 128, 320, 512), ((32, 32), (24, 24), (18, 18), (16, 16)))]
        fixed_pet = [torch.randn_like(x) for x in fixed_ct]
        fixed_proxy = [torch.randn_like(x) for x in fixed_ct]
        fixed_ref = [x.clone() for x in fixed_ct]

    def encode_ct_fixed(_ct):
        return [x.clone() for x in fixed_ct]

    def encode_pet_fixed(_pet):
        return [x.clone() for x in fixed_pet]

    def retrieve_fixed(ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        if return_ct_reference:
            return [x.clone() for x in fixed_proxy], [x.clone() for x in fixed_ref], {}
        return [x.clone() for x in fixed_proxy], {}

    for m in (model_a, model_b):
        m.stage1._encode_ct = encode_ct_fixed
        m.stage1._encode_pet = encode_pet_fixed
        m.stage1._retrieve_cppi = retrieve_fixed

    with torch.no_grad():
        af = model_a(ct, pet=pet, forward_mode="full")["logits"]
        bf = model_b(ct, pet=pet, forward_mode="full")["logits"]
        am = model_a(ct, pet=pet, forward_mode="missing")["logits"]
        bm = model_b(ct, pet=pet, forward_mode="missing")["logits"]
    assert (af - bf).abs().max().item() <= 1e-6
    assert (am - bm).abs().max().item() <= 1e-6


def test_anga_synthetic_branches():
    # Build tiny param group
    p = nn.Parameter(torch.zeros(4))
    # Full anchor
    gF = [torch.tensor([1.0, 0.0, 0.0, 0.0])]
    # 1) cos <= 0
    gM = [torch.tensor([-1.0, 0.0, 0.0, 0.0])]
    aligned, stats = project_missing_onto_full_cone(gF, gM, tau=0.7)
    assert stats["zero"] == 1.0
    assert aligned[0].abs().sum().item() == 0.0

    # 2) cos >= tau (aligned with Full)
    gM = [torch.tensor([2.0, 0.0, 0.0, 0.0])]
    aligned, stats = project_missing_onto_full_cone(gF, gM, tau=0.7)
    assert stats["inside"] == 1.0
    assert torch.allclose(aligned[0], gM[0])

    # 3) 0 < cos < tau -> projected cosine ~= tau
    gM = [torch.tensor([1.0, 3.0, 0.0, 0.0])]
    aligned, stats = project_missing_onto_full_cone(gF, gM, tau=0.7)
    assert stats["project"] == 1.0
    a = gF[0] / (gF[0].norm() + EPS)
    cos = torch.dot(aligned[0].float(), a.float()) / (aligned[0].float().norm() + EPS)
    assert abs(float(cos.item()) - 0.7) < 1e-4

    # 4) Full ~ 0 => Missing zeroed
    gF0 = [torch.zeros(4)]
    gM = [torch.tensor([0.0, 1.0, 0.0, 0.0])]
    aligned, stats = project_missing_onto_full_cone(gF0, gM, tau=0.7)
    assert stats["zero"] == 1.0
    assert aligned[0].abs().sum().item() == 0.0
    del p


def test_anga_scales_independent():
    calib = PrototypeReferencedPETAffineCalibration((64, 128, 320, 512))
    for p in calib.parameters():
        p.requires_grad = True
    groups = get_affine_head_param_groups(calib)

    # Fabricate grads: scale0 conflicting, others aligned
    gF = []
    gM = []
    for gi, group in enumerate(groups):
        gf = []
        gm = []
        for p in group:
            gf.append(torch.ones_like(p))
            if gi == 0:
                gm.append(-torch.ones_like(p))  # conflict
            else:
                gm.append(torch.ones_like(p))  # inside
        gF.append(gf)
        gM.append(gm)

    stats = merge_affine_grads_per_scale(groups, gF, gM, mode="anga", tau=0.7)
    assert stats["affine_s1_zero_ratio"] == 1.0
    assert stats["affine_s2_inside_ratio"] == 1.0
    assert stats["affine_s3_inside_ratio"] == 1.0
    assert stats["affine_s4_inside_ratio"] == 1.0

    # Scale1 grad should equal gF only (missing zeroed); others gF+gM
    for p, gf in zip(groups[0], gF[0]):
        assert torch.allclose(p.grad, gf)
    for scale in range(1, 4):
        for p, gf, gm in zip(groups[scale], gF[scale], gM[scale]):
            assert torch.allclose(p.grad, gf + gm)


def test_warmup_keeps_affine_fixed_updates_spre():
    model = _make_stage2(allow_affine=True, seed=7)
    model.train()
    stage2_params = list(model.role_fusion.parameters()) + list(model.decoder_adapters.parameters())
    affine_params = list(model.stage1.pet_calibration.parameters())
    opt = torch.optim.AdamW(
        [
            {"name": "stage2", "params": stage2_params, "lr": 1e-3},
            {"name": "stage1_affine", "params": affine_params, "lr": 1e-3},
        ]
    )
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)

    affine_before = [p.detach().clone() for p in affine_params]
    spre_before = [p.detach().clone() for p in stage2_params]

    groups = get_affine_head_param_groups(model.stage1.pet_calibration)
    opt.zero_grad(set_to_none=True)
    loss_f = F.binary_cross_entropy_with_logits(model(ct, pet=pet, forward_mode="full")["logits"], mask)
    (0.5 * loss_f).backward()
    gF = snapshot_all_affine_grads(groups)
    clear_all_affine_grads(groups)
    loss_m = F.binary_cross_entropy_with_logits(model(ct, pet=pet, forward_mode="missing")["logits"], mask)
    (0.5 * loss_m).backward()
    # Warmup: drop affine grads before step
    clear_all_affine_grads(groups)
    del gF
    opt.step()

    for a, b in zip(affine_params, affine_before):
        assert torch.equal(a.detach().cpu(), b.cpu())
    changed = False
    for a, b in zip(stage2_params, spre_before):
        if not torch.equal(a.detach().cpu(), b.cpu()):
            changed = True
            break
    assert changed


def test_cppi_bank_unchanged_after_paired_step():
    model = _make_stage2(allow_affine=True, seed=3)
    model.train()
    pm = model.stage1.prototype_memory
    ver_before = int(pm.bank_version.item())
    ready_before = pm.prototype_ready.detach().clone()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    opt.zero_grad(set_to_none=True)
    (0.5 * F.binary_cross_entropy_with_logits(model(ct, pet=pet, forward_mode="full")["logits"], mask)).backward()
    (0.5 * F.binary_cross_entropy_with_logits(model(ct, pet=pet, forward_mode="missing")["logits"], mask)).backward()
    opt.step()
    report = model.finalize_cppi_epoch(epoch=1, save_json=False, save_visualizations=False, print_info=False)
    assert int(pm.bank_version.item()) == ver_before == int(report["bank_version_after"])
    assert torch.equal(ready_before, pm.prototype_ready.cpu())


def test_checkpoint_contains_affine_keys(tmp_path):
    model = _make_stage2(allow_affine=True, seed=5)
    path = tmp_path / "s2.pth"
    torch.save({"model": model.state_dict()}, path)
    payload = torch.load(path, map_location="cpu")
    keys = list(payload["model"].keys())
    assert any(k.startswith("stage1.pet_calibration.heads.0") for k in keys)
    model2 = _make_stage2(allow_affine=True, seed=9)
    msg = model2.load_state_dict(payload["model"], strict=True)
    assert msg.missing_keys == []
    assert msg.unexpected_keys == []


def test_optimizer_param_groups_exclusive():
    from tasks.mdt_seg import MDTSegTeacher

    model = _make_stage2(allow_affine=True, seed=11)
    cfg = type(
        "C",
        (),
        {
            "learning_rate": 8e-5,
            "weight_decay": 1e-4,
            "mixed_precision": False,
            "loss_smooth": 1.0,
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "random_state": 2023,
            "stage2_train_strategy": "paired_anga_affine",
            "stage2_affine_learning_rate": 8e-6,
            "stage2_affine_warmup_epochs": 1,
            "stage2_anga_tau": 0.7,
        },
    )()
    task = MDTSegTeacher({"model": model}, cfg)
    names = [g.get("name") for g in task.optimizer.param_groups]
    assert names == ["stage2", "stage1_affine"]
    assert abs(task.optimizer.param_groups[0]["lr"] - 8e-5) < 1e-12
    assert abs(task.optimizer.param_groups[1]["lr"] - 8e-6) < 1e-12
    ids = []
    for g in task.optimizer.param_groups:
        for p in g["params"]:
            ids.append(id(p))
    assert len(ids) == len(set(ids))
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(ids) == trainable
