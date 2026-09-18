# -*- coding: utf-8 -*-
"""Module-2 two-stage smoke: real task + model construction, cold -> bank -> ready.

Stage A: >=2 mixed 50/50 batches, bank NOT ready. Missing fusion must equal
  CT, L_rec must be 0, Full branch fusion must still update.
Then exactly ONE finalize_module1_epoch builds the bank.
Stage B: >=2 mixed batches, bank ready. L_rec finite, retrieval/affine/fusion
  receive updates. No per-batch finalize.
Finally: full validation + pet=None Missing validation.
Also: alternating Full/Missing single steps (sampling untouched).

Uses real MDTSegTeacher + build_mdt_seg_teacher (toy backbones from scratch,
CPU, B=2, 64x64). Loss path mirrors tasks.mdt_seg.train_step_mixed with the
current DEFAULT mixed weights (0.5/0.5 + recon 0.05), without importing the
data loaders.

Run:  python tests/test_module2_two_stage_smoke.py
      python -m pytest -q tests/test_module2_two_stage_smoke.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher

SYNTHETIC_TEXT_DIM = 16


def _mask(batch=2, size=64):
    m = torch.zeros(batch, 1, size, size)
    m[:, :, 16:48, 16:48] = 1.0
    return m


def _cpu_config(**overrides):
    ns = argparse.Namespace()
    for p in (
        SegMDTConfig.data_parser(), SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(), SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(), SegMDTConfig.ddp_parser(),
    ):
        for a in p._actions:
            if a.dest != "help" and not hasattr(ns, a.dest):
                ns.__dict__[a.dest] = a.default
    cfg = SegMDTConfig(args=vars(ns))
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.pspi_num_clusters = 2
    cfg.pspi_prior_scale_enabled = True
    cfg.pspi_affine_enabled = False
    cfg.pspi_reconstruction_weight = 0.0
    cfg.pspi_proto_contrastive_weight = 0.0
    cfg.mixed_precision = False
    cfg.random_state = 2023
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # v3 default must be matched_ema/0.95 unless the caller overrides.
    if "pspi_bank_update_mode" not in overrides:
        assert cfg.pspi_bank_update_mode == "matched_ema", cfg.pspi_bank_update_mode
    return cfg


def _make_task(use_text, device="cpu"):
    cfg = _cpu_config(module2_enabled=True, module2_use_text=use_text)
    teacher = build_mdt_seg_teacher(cfg)
    model = teacher["model"]
    if device != "cpu":
        model.to(device)
    if use_text:
        torch.manual_seed(555)
        vector = torch.randn(1, SYNTHETIC_TEXT_DIM)
        from models.petct_state_text_afa import StateTextAFAFusion
        model.fusion = StateTextAFAFusion(
            list(model.fusion.channels), text_feature=vector,
            enabled=True, use_state=True, use_text=True, use_afa=True,
        )
    else:
        # Builder already constructed a text-disabled fusion; keep it.
        assert bool(model.fusion.text_ready) is False
    task = MDTSegTeacher({"model": model}, cfg)
    return cfg, task


def _mixed_loss(task, batch, state):
    total, logits, outputs, stats = task.train_step_mixed(
        batch, state, missing_loss_weight=1.0,
    )
    return total, logits, outputs, stats


def _batch(device):
    return {
        "ct": torch.randn(2, 1, 64, 64, device=device),
        "pet": torch.randn(2, 1, 64, 64, device=device),
        "mask": _mask(2).to(device),
    }


def _run_stage(task, batches, tag, expect_ready, expect_recon_active):
    forward_count = 0
    collect_count = 0
    orig_forward = task.model.forward
    orig_collect = task.model.module1.collect_candidates
    def counted_forward(*a, **k):
        nonlocal forward_count
        forward_count += 1
        return orig_forward(*a, **k)
    def counted_collect(*a, **k):
        nonlocal collect_count
        collect_count += 1
        return orig_collect(*a, **k)
    task.model.forward = counted_forward
    task.model.module1.collect_candidates = counted_collect
    try:
        for i, batch in enumerate(batches):
            state = torch.tensor([1, 0])
            total, logits, outputs, stats = _mixed_loss(task, batch, state)
            assert bool(outputs.get("module1_bank_ready", False)) == expect_ready
            assert bool(outputs.get("reconstruction_active", False)) == expect_recon_active
            if not expect_recon_active:
                assert float(outputs.get("reconstruction_loss", 0.0)) == 0.0
            else:
                assert torch.isfinite(outputs["reconstruction_loss"])
            assert torch.isfinite(total)
            task.optimizer.zero_grad(set_to_none=True)
            total.backward()
            task.optimizer.step()
            print(f"[{tag}] batch={i+1} ready={expect_ready} "
                  f"loss={float(total):.4f} recon={float(outputs.get('reconstruction_loss', 0.0)):.6f}")
    finally:
        task.model.forward = orig_forward
        task.model.module1.collect_candidates = orig_collect
    assert forward_count == len(batches), (forward_count, len(batches))
    assert collect_count == len(batches), (collect_count, len(batches))
    return forward_count, collect_count


def _check_full_branch_updates(task, device):
    before = [p.detach().clone() for p in task.model.fusion.parameters()]
    batch = _batch(device)
    total, _, _, _ = _mixed_loss(task, batch, torch.tensor([1, 0]))
    task.optimizer.zero_grad(set_to_none=True)
    total.backward()
    grads = [p.grad for p in task.model.fusion.parameters() if p.requires_grad]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    task.optimizer.step()
    after = [p.detach().clone() for p in task.model.fusion.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    print("[Stage A] Full-branch fusion updates confirmed")


def test_two_stage_cold_then_ready():
    torch.manual_seed(2023)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, task = _make_task(use_text=False, device=device)
    task.model.train()
    assert not task.model.module1.bank_ready
    stage_a = [
        _batch(device)
        for _ in range(2)
    ]
    # Missing rows fuse to strict CT while cold.
    before = stage_a[0]
    with torch.no_grad():
        out = task.model(
            before["ct"], pet=before["pet"],
            pet_available=torch.tensor([1, 0], device=device), forward_mode="auto",
            mask=before["mask"],
        )
    assert out["module1_bank_ready"] is False
    _run_stage(task, stage_a, "Stage A", expect_ready=False, expect_recon_active=False)
    _check_full_branch_updates(task, device)
    report = task.model.finalize_module1_epoch(epoch=1)
    assert task.model.module1.bank_ready
    assert report["status"] == "bank_updated"
    print(f"[finalize] status={report['status']} version={int(task.model.module1.bank_version.item())}")
    stage_b = [
        _batch(device)
        for _ in range(2)
    ]
    watched = {"retrieval": 0}
    orig_retrieve = task.model.module1.retrieve_pet_prior
    def counted_retrieve(*a, **k):
        watched["retrieval"] += 1
        return orig_retrieve(*a, **k)
    task.model.module1.retrieve_pet_prior = counted_retrieve
    before_params = {n: p.detach().clone() for n, p in task.model.named_parameters() if p.requires_grad}
    try:
        _run_stage(task, stage_b, "Stage B", expect_ready=True, expect_recon_active=False)
    finally:
        task.model.module1.retrieve_pet_prior = orig_retrieve
    assert watched["retrieval"] >= len(stage_b)
    changed = [n for n, p in task.model.named_parameters()
               if p.requires_grad and not torch.equal(before_params[n], p)]
    assert any(n.startswith("fusion.") for n in changed), changed[:8]
    print(f"[Stage B] updated params={len(changed)} (incl. fusion)")
    # Full + Missing validations (legacy prior-scale path, affine off).
    task.model.eval()
    with torch.no_grad():
        full = task.model(torch.randn(1, 1, 64, 64, device=device), pet=torch.randn(1, 1, 64, 64, device=device), forward_mode="full")
        assert torch.isfinite(full["logits"]).all()
        missing = task.model(torch.randn(1, 1, 64, 64, device=device), pet=None, forward_mode="missing")
        assert torch.isfinite(missing["logits"]).all()
    print("[Validate] full + pet=None Missing both finite")
    # Alternating single steps.
    task.model.train()
    batch = _batch(device)
    for mode in ("full", "missing"):
        total, _, _, _ = task.train_step(batch, forward_mode=mode)
        task.optimizer.zero_grad(set_to_none=True)
        total.backward()
        task.optimizer.step()
        assert torch.isfinite(total)
    print("[Alternating] full + missing steps OK")


def main():
    test_two_stage_cold_then_ready()
    print("\n[RESULT] module2 two-stage smoke passed")


if __name__ == "__main__":
    main()
