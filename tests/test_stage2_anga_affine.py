"""Joint Recovery+SPRE end-to-end contract tests (lightweight, no real dataset)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import PrototypeReferencedPETAffineCalibration
from models.stage2_prompt_role_expert_fusion import PromptRoleExpertStage2Seg


class _DummyEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.channels = list(channels)
        self.feature_info = type("FI", (), {"channels": lambda self=None: list(channels)})()
        self.probe = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        b = x.shape[0]
        outs = []
        spatial = [(32, 32), (24, 24), (18, 18), (16, 16)]
        for c, (h, w) in zip(self.channels, spatial):
            # Deterministic + probe so grads reach the encoder parameter.
            base = torch.ones(b, c, h, w, device=x.device, dtype=x.dtype) * (0.01 * x.mean())
            outs.append(base + self.probe.view(1, 1, 1, 1))
        return outs


class _DummyAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        return [f + self.bias.view(1, 1, 1, 1) for f in feats]


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
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(
            torch.cat(
                [F.interpolate(d4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3],
                1,
            )
        )
        s2 = self.proj2(x2)
        d2 = self.fuse2(
            torch.cat(
                [F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2],
                1,
            )
        )
        s1 = self.proj1(x1)
        d1 = self.fuse1(
            torch.cat(
                [F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1],
                1,
            )
        )
        logits = F.interpolate(self.seg_head(d1), size=target_size, mode="bilinear", align_corners=False)
        return {"logits": logits}


class _DummyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.register_buffer("ready_mask", torch.tensor(False))

    def forward(self, ct, keys, values, ready):
        # Minimal differentiable retrieval surrogate.
        b, c, h, w = ct.shape
        q = self.q_proj(ct.flatten(2).transpose(1, 2).mean(dim=1))
        if keys is None or (torch.is_tensor(ready) and not bool(ready.any())):
            proxy = torch.zeros_like(ct)
            attn = torch.zeros(b, 1, device=ct.device, dtype=ct.dtype)
            return proxy + 0.0 * q.view(b, c, 1, 1), attn
        k = self.k_proj(keys.float())
        v = self.v_proj(values.float())
        wgt = torch.softmax((q.unsqueeze(1) * k).sum(-1), dim=-1)
        retrieved = (wgt.unsqueeze(-1) * v).sum(1)
        proxy = retrieved.view(b, c, 1, 1).expand_as(ct)
        return self.out_proj(proxy.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(ct), wgt


class _DummyProto(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.channels = list(channels)
        self.register_buffer("bank_version", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prototype_ready", torch.zeros(2, 6, dtype=torch.bool))
        self.attention = nn.ModuleList([_DummyAttention(c) for c in channels])
        self._candidates = 0
        self._force_ready = False

    @property
    def bank_ready(self):
        return bool(self.prototype_ready.any()) or bool(self._force_ready)

    def collect(self, ct_feats, pet_feats, mask, print_info=False, compute_report=False):
        self._candidates += int(mask.numel())
        return {"num_candidates": self._candidates}

    def finalize_epoch(self, epoch, save_json=True, save_visualizations=False, print_info=True):
        before = int(self.bank_version.item())
        if self._candidates > 0:
            self.bank_version.add_(1)
            self.prototype_ready.fill_(True)
            self._force_ready = True
            ready = int(self.prototype_ready.sum().item())
        else:
            ready = int(self.prototype_ready.sum().item())
        after = int(self.bank_version.item())
        report = {
            "bank_version_before": before,
            "bank_version_after": after,
            "ready_count": ready,
            "ready_slots": ready,
            "classes": {
                "background": {"num_candidates": max(self._candidates // 2, 0)},
                "foreground": {"num_candidates": max(self._candidates - self._candidates // 2, 0)},
            },
        }
        self._candidates = 0
        return report

    def retrieve(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        pet_proxy = []
        ct_ref = []
        report = {}
        for scale_idx, ct in enumerate(ct_feats):
            c = ct.shape[1]
            if self.bank_ready:
                keys = torch.randn(ct.shape[0], 4, c, device=ct.device, dtype=ct.dtype)
                values = torch.randn_like(keys)
                ready = torch.ones(4, dtype=torch.bool, device=ct.device)
                proxy, _ = self.attention[scale_idx](ct, keys, values, ready)
            else:
                proxy, _ = self.attention[scale_idx](ct, None, None, torch.tensor(False, device=ct.device))
            pet_proxy.append(proxy)
            ct_ref.append(ct.detach() + 0.1)
        if return_ct_reference:
            return pet_proxy, ct_ref, report
        return pet_proxy, report


class _DummyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_alpha_full = nn.Parameter(torch.zeros(4))
        self.raw_alpha_missing = nn.Parameter(torch.zeros(4))

    def forward(self, *args, **kwargs):
        raise RuntimeError("legacy fusion must not be called")


class DummyStage1(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.enc_ct = _DummyEncoder(channels)
        self.enc_pet = _DummyEncoder(channels)
        self.ct_align = _DummyAlign()
        self.pet_calibration = PrototypeReferencedPETAffineCalibration(channels)
        self.fusion = _DummyFusion()
        self.decoder = _DummyDecoder(channels)
        self.prototype_memory = _DummyProto(channels)

    def _to_3ch(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        return self.ct_align(self.enc_ct(self._to_3ch(ct)))

    def _encode_pet(self, pet):
        return self.enc_pet(self._to_3ch(pet))

    def _collect_cppi(self, ct_feats, pet_feats_real, mask):
        if self.training and mask is not None:
            return self.prototype_memory.collect(ct_feats, pet_feats_real, mask)
        return None

    def _retrieve_cppi(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        return self.prototype_memory.retrieve(
            ct_feats,
            compute_report=compute_report,
            save_diagnostics=save_diagnostics,
            print_info=print_info,
            return_ct_reference=return_ct_reference,
        )


def _make_joint(seed=0):
    torch.manual_seed(seed)
    stage1 = DummyStage1()
    model = PromptRoleExpertStage2Seg(stage1_model=stage1, channels=(64, 128, 320, 512))
    return model


def test_joint_trainable_modules_and_no_adapters():
    model = _make_joint()
    assert not hasattr(model, "decoder_adapters")
    assert model.count_module_trainable(model.stage1.enc_ct) > 0
    assert model.count_module_trainable(model.stage1.enc_pet) > 0
    assert model.count_module_trainable(model.stage1.ct_align) > 0
    assert model.count_module_trainable(model.stage1.prototype_memory.attention) > 0
    assert model.count_module_trainable(model.stage1.pet_calibration) > 0
    assert model.count_module_trainable(model.stage1.decoder) > 0
    assert model.count_module_trainable(model.role_fusion) > 0
    assert model.count_module_trainable(model.stage1.fusion) == 0
    assert model.cppi_ready is False
    assert int(model.stage1.prototype_memory.bank_version.item()) == 0


def test_joint_full_shapes_and_grads():
    model = _make_joint(seed=1)
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    out = model(ct, pet=pet, mask=mask, forward_mode="full", return_features=True)
    assert out["logits"].shape == (2, 1, 64, 64)
    feats = out["stage2_features"]
    assert [tuple(f.shape) for f in feats] == [
        (2, 64, 32, 32),
        (2, 128, 24, 24),
        (2, 320, 18, 18),
        (2, 512, 16, 16),
    ]
    loss = F.binary_cross_entropy_with_logits(out["logits"], mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.stage1.enc_ct.probe.grad is not None
    assert model.stage1.enc_pet.probe.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.role_fusion.parameters())
    assert model.stage1.decoder.seg_head.weight.grad is not None
    for p in model.stage1.fusion.parameters():
        assert p.grad is None


def test_joint_missing_empty_bank_ok_and_eval_no_pet_encoder():
    model = _make_joint(seed=2)
    model.train()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    assert model.cppi_ready is False
    out = model(ct, pet=pet, mask=mask, forward_mode="missing")
    assert torch.isfinite(out["logits"]).all()

    model.eval()
    model.stage1.enc_pet.calls = 0
    _ = model(ct, pet=None, mask=None, forward_mode="missing")
    assert model.stage1.enc_pet.calls == 0


def test_joint_finalize_and_missing_graph_connectivity():
    model = _make_joint(seed=3)
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)

    # Collect via Full/Missing, then finalize.
    _ = model(ct, pet=pet, mask=mask, forward_mode="full")
    _ = model(ct, pet=pet, mask=mask, forward_mode="missing")
    report = model.finalize_cppi_epoch(epoch=1, save_json=False, save_visualizations=False, print_info=False)
    assert int(report["bank_version_after"]) == 1
    assert model.cppi_ready is True

    model.zero_grad(set_to_none=True)
    model.stage1.enc_pet.calls = 0
    out = model(ct, pet=pet, mask=mask, forward_mode="missing")
    assert model.stage1.enc_pet.calls > 0  # collect-only under no_grad
    loss = F.binary_cross_entropy_with_logits(out["logits"], mask)
    loss.backward()

    q = model.stage1.prototype_memory.attention[0].q_proj.weight
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    ct_proj = model.role_fusion.scale_units[0].ct_proj[0].weight
    assert ct_proj.grad is not None and ct_proj.grad.abs().sum() > 0
    assert model.stage1.decoder.seg_head.weight.grad is not None
    assert model.stage1.enc_ct.probe.grad is not None
    # Collect-only PET encoder forward must not create Missing seg grads.
    assert model.stage1.enc_pet.probe.grad is None
    for p in model.stage1.enc_pet.parameters():
        assert p.grad is None



def test_joint_optimizer_excludes_legacy_fusion():
    from tasks.mdt_seg import MDTSegTeacher

    model = _make_joint(seed=4)
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
        },
    )()
    task = MDTSegTeacher({"model": model}, cfg)
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g["params"]}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert opt_ids == trainable_ids
    for p in model.stage1.fusion.parameters():
        assert id(p) not in opt_ids
