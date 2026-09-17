# -*- coding: utf-8 -*-
"""Module-2 (StateTextAFAFusion) integration tests.

Sections mirror the acceptance spec:
  A. disabled equivalence + legacy config compat + zero CLIP I/O when off
  B. enabled forward/state routing (Full/Missing/auto, cold/ready, PSPI-off)
  C. forbidden calls + real-PET isolation + bank immutability
  D. gradients, freezing, optimizer membership
  F. checkpoints + ablations + mismatch errors

Run:  python -m pytest -q tests/test_module2_integration.py
"""

import argparse
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline

SYNTHETIC_TEXT_DIM = 16


def _mask(batch=2, size=64):
    m = torch.zeros(batch, 1, size, size)
    m[:, :, 16:48, 16:48] = 1.0
    return m


def _synthetic_text(dim=SYNTHETIC_TEXT_DIM):
    torch.manual_seed(1234)
    return torch.randn(1, dim)


def _plain_model(**kwargs):
    kwargs.setdefault("pspi_num_clusters", 2)
    kwargs.setdefault("pspi_prior_scale_enabled", True)
    kwargs.setdefault("pspi_affine_enabled", False)
    return DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None, **kwargs
    )


def _module2_model(**kwargs):
    kwargs.setdefault("pspi_num_clusters", 2)
    kwargs.setdefault("pspi_prior_scale_enabled", True)
    kwargs.setdefault("pspi_affine_enabled", False)
    kwargs.setdefault("module2_enabled", True)
    if kwargs.get("module2_use_text", True):
        kwargs.setdefault("module2_text_feature", _synthetic_text())
    return DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None, **kwargs
    )


def _fill_bank(model, batches=3, samples=4):
    model.train()
    torch.manual_seed(11)
    for _ in range(batches):
        ct_img = torch.randn(samples, 1, 64, 64)
        pet_img = torch.randn(samples, 1, 64, 64)
        model.module1.collect_candidates(
            model._encode_ct(ct_img), model._encode_pet(pet_img), _mask(samples)
        )
    report = model.module1.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    return report


def _banked_plain():
    model = _plain_model()
    _fill_bank(model)
    assert model.module1.bank_ready
    return model


def _banked_module2(**kwargs):
    model = _module2_model(**kwargs)
    _fill_bank(model)
    assert model.module1.bank_ready
    return model


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
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# =============================================================================
# A. Disabled equivalence + legacy compat
# =============================================================================

def test_a01_disabled_is_addfusion_no_new_state():
    model = _plain_model()
    from models.baseline_blocks import AddFusion
    assert isinstance(model.fusion, AddFusion)
    assert model.module2_enabled is False
    assert not any(n.startswith("fusion.") for n, _ in model.named_parameters())
    assert not any(n.startswith("fusion.") for n, _ in model.named_buffers())
    print("[A01] disabled instantiates AddFusion with no fusion state: PASS")


def test_a02_disabled_eval_matches_base_logits():
    torch.manual_seed(7)
    m_new = _plain_model()
    m_new.eval()
    torch.manual_seed(7)
    m_ref = _plain_model()
    m_ref.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        for mode, kw in (
            ("full", {"pet": pet}),
            ("missing", {"pet": None}),
        ):
            a = m_new(ct, forward_mode=mode, **kw)["logits"]
            b = m_ref(ct, forward_mode=mode, **kw)["logits"]
            assert torch.equal(a, b)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    print("[A02] disabled logits bitwise match same-seed construction: PASS")


def test_a03_disabled_train_loss_grad_match():
    from utils.seg_losses import BCEDiceLoss
    torch.manual_seed(9)
    m_new = _plain_model()
    torch.manual_seed(9)
    m_ref = _plain_model()
    m_new.train()
    m_ref.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask(2)
    criterion = BCEDiceLoss()
    for mode in ("full", "missing"):
        m_new.zero_grad(set_to_none=True)
        m_ref.zero_grad(set_to_none=True)
        torch.manual_seed(100)
        out_n = m_new(ct, pet=pet, forward_mode=mode, mask=mask)
        torch.manual_seed(100)
        out_r = m_ref(ct, pet=pet, forward_mode=mode, mask=mask)
        loss_n, _ = criterion(out_n["logits"], mask)
        loss_r, _ = criterion(out_r["logits"], mask)
        assert torch.equal(loss_n, loss_r)
        loss_n.backward()
        loss_r.backward()
        for (nn_, pn), (nr, pr) in zip(
            m_new.named_parameters(), m_ref.named_parameters()
        ):
            assert nn_ == nr
            if pn.grad is None:
                assert pr.grad is None
            else:
                assert torch.equal(pn.grad, pr.grad)
    print("[A03] disabled train loss/gradients match same-seed: PASS")


def test_a04_legacy_config_builds_addfusion():
    cfg = _cpu_config()
    for attr in (
        "module2_enabled", "module2_use_text", "module2_use_state",
        "module2_use_afa", "module2_text_cache", "module2_text_model_path",
        "module2_text_prompt",
    ):
        assert hasattr(cfg, attr)
        if hasattr(cfg, attr):
            delattr(cfg, attr)
    teacher = build_mdt_seg_teacher(cfg)
    from models.baseline_blocks import AddFusion
    assert isinstance(teacher["model"].fusion, AddFusion)
    print("[A04] legacy config without module2 fields builds AddFusion: PASS")


def test_a05_existing_layers_init_unchanged_by_wiring():
    torch.manual_seed(21)
    m_off = _plain_model()
    torch.manual_seed(21)
    m_on = _module2_model(module2_use_text=False)
    for group in ("enc_ct", "enc_pet", "ct_align", "decoder"):
        off = dict(getattr(m_off, group).named_parameters())
        on = dict(getattr(m_on, group).named_parameters())
        assert set(off) == set(on)
        for k in off:
            assert torch.equal(off[k], on[k]), f"{group}.{k} shifted by fusion wiring"
    print("[A05] fusion wiring does not shift existing init (fork_rng): PASS")


def test_a06_disabled_no_clip_io_even_if_missing():
    import models.build_mdt_seg as builder
    calls = {"n": 0}
    orig = builder.encode_fixed_text
    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    builder.encode_fixed_text = spy
    try:
        cfg = _cpu_config()
        cfg.module2_text_model_path = "/nonexistent/clip-dir"
        teacher = build_mdt_seg_teacher(cfg)
        from models.baseline_blocks import AddFusion
        assert isinstance(teacher["model"].fusion, AddFusion)
    finally:
        builder.encode_fixed_text = orig
    assert calls["n"] == 0
    print("[A06] disabled performs zero CLIP loads (bad path harmless): PASS")


# =============================================================================
# B. Enabled forward + state routing
# =============================================================================

def test_b01_enabled_shapes_finite_all_modes():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        cases = [
            ("full", {"pet": pet}, None),
            ("missing", {"pet": None}, None),
            ("auto", {"pet": pet, "pet_available": torch.tensor([1, 1])}, None),
            ("auto", {"pet": pet, "pet_available": torch.tensor([0, 0])}, None),
            ("auto", {"pet": pet, "pet_available": torch.tensor([1, 0])}, None),
        ]
        for mode, kw, _ in cases:
            out = model(ct, forward_mode=mode, mask=_mask(2), **kw)
            assert out["logits"].shape == (2, 1, 64, 64)
            assert torch.isfinite(out["logits"]).all()
    print("[B01] enabled shapes/logits finite across modes: PASS")


def test_b02_state_valid_routing_recorded():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    seen = []
    orig = model.fusion.forward
    def spy(ct_feats, pet_feats, pet_available, *, pet_valid, **kw):
        seen.append((
            torch.as_tensor(pet_available).reshape(-1).tolist(),
            torch.as_tensor(pet_valid).reshape(-1).tolist(),
        ))
        return orig(ct_feats, pet_feats, pet_available, pet_valid=pet_valid, **kw)
    model.fusion.forward = spy
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
              forward_mode="auto", mask=_mask(2))
    model.fusion.forward = orig
    assert seen, "mixed path must call the fusion once"
    assert seen[0][0] == [1, 0] and seen[0][1] == [True, True]
    print("[B02] mixed per-sample state/valid routing correct: PASS")


def test_b03_cold_start_strict_ct_despite_nonzero_bias():
    model = _module2_model(module2_use_text=False)
    model.eval()
    assert not model.module1.bank_ready
    with torch.no_grad():
        for block in model.fusion.scales:
            block.state_prompt.fill_(7.0)
            block.afa_out.bias.fill_(7.0)
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode="missing")
    assert torch.equal(out["logits"], _cold_ct_only(model, ct))
    print("[B03] cold-start Missing strictly CT despite nonzero bias: PASS")


def _cold_ct_only(model, ct):
    with torch.no_grad():
        ct_feats = model._encode_ct(ct)
        return model._decode(ct_feats, ct.shape[-2:])["logits"]


def test_b04_ready_missing_uses_text_afa():
    model = _banked_module2()
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out, infos = model.fusion(
            model._encode_ct(ct),
            [torch.zeros_like(f) for f in model._encode_ct(ct)],
            0, pet_valid=True, return_diagnostics=True,
        )
    assert any("text_gate" in info for info in infos)
    assert any("afa_delta_rms" in info for info in infos)
    print("[B04] ready Missing passes through text/AFA branches: PASS")


def test_b05_pspi_off_validity():
    model = _module2_model(module2_use_text=False, pspi_enabled=False)
    model.eval()
    seen = []
    orig = model.fusion.forward
    def spy(ct_feats, pet_feats, pet_available, *, pet_valid, **kw):
        seen.append(torch.as_tensor(pet_valid).reshape(-1).tolist())
        return orig(ct_feats, pet_feats, pet_available, pet_valid=pet_valid, **kw)
    model.fusion.forward = spy
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
              forward_mode="auto", mask=_mask(2))
    model.fusion.forward = orig
    assert seen[0] == [True, False]
    print("[B05] PSPI-off: Full valid / Missing invalid: PASS")


def test_b06_legacy_affine_off_path_fused():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(ct, pet=pet, forward_mode="missing", mask=_mask(1))
    assert torch.isfinite(out["logits"]).all()
    print("[B06] legacy scaled-prior path fuses through Module-2: PASS")


# =============================================================================
# C. Forbidden calls + isolation
# =============================================================================

def test_c01_missing_inference_no_pet_encoder_collect_finalize():
    for banked in (False, True):
        model = _banked_module2(module2_use_text=False) if banked else _module2_model(module2_use_text=False)
        model.eval()
        calls = {"pet": 0, "collect": 0, "finalize": 0}
        orig_pet = model._encode_pet
        orig_collect = model.module1.collect_candidates
        orig_finalize = model.module1.finalize_epoch
        model._encode_pet = lambda *a, **k: (calls.__setitem__("pet", calls["pet"] + 1), orig_pet(*a, **k))[1]
        model.module1.collect_candidates = lambda *a, **k: (calls.__setitem__("collect", calls["collect"] + 1), orig_collect(*a, **k))[1]
        model.module1.finalize_epoch = lambda *a, **k: (calls.__setitem__("finalize", calls["finalize"] + 1), orig_finalize(*a, **k))[1]
        with torch.no_grad():
            model(torch.randn(1, 1, 64, 64), pet=None, forward_mode="missing")
        model._encode_pet = orig_pet
        model.module1.collect_candidates = orig_collect
        model.module1.finalize_epoch = orig_finalize
        assert calls == {"pet": 0, "collect": 0, "finalize": 0}
    print("[C01] Missing inference never touches PET encoder/collect/finalize: PASS")


def test_c02_full_never_retrieves_or_affine():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    calls = {"retrieve": 0}
    orig = model.module1.retrieve_pet_prior
    def spy(*a, **k):
        calls["retrieve"] += 1
        return orig(*a, **k)
    model.module1.retrieve_pet_prior = spy
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64), pet=torch.randn(1, 1, 64, 64),
              forward_mode="full", mask=_mask(1))
    model.module1.retrieve_pet_prior = orig
    assert calls["retrieve"] == 0
    print("[C02] Full path never retrieves: PASS")


def test_c03_missing_eval_isolated_from_real_pet():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    ct = torch.randn(1, 1, 64, 64)
    pet_a = torch.randn(1, 1, 64, 64)
    pet_b = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        o1 = model(ct, pet=pet_a, forward_mode="missing")["logits"]
        o2 = model(ct, pet=pet_b, forward_mode="missing")["logits"]
    assert torch.equal(o1, o2)
    print("[C03] Missing eval independent of real PET: PASS")


def test_c04_missing_grad_no_real_pet_path():
    # Image-level isolation: Missing-row real PET must not reach the fused
    # output. Encoder outputs are non-leaf (grad is None without retain),
    # so assert on the leaf image tensors instead.
    model = _banked_module2(module2_use_text=False)
    model.train()
    model.zero_grad(set_to_none=True)
    ct = torch.randn(2, 1, 64, 64, requires_grad=True)
    pet = torch.randn(2, 1, 64, 64, requires_grad=True)
    out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
                forward_mode="auto", mask=_mask(2))
    out["logits"].sum().backward()
    assert float(pet.grad[0].abs().sum()) > 0  # Full row keeps a path
    assert float(pet.grad[1].abs().sum()) == 0  # Missing row has none
    print("[C04] Missing fusion output has no path to real PET row: PASS")


def test_c05_bank_buffers_untouched_by_fusion():
    model = _banked_module2(module2_use_text=False)
    model.eval()
    before = {n: b.clone() for n, b in model.module1.named_buffers()}
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
              forward_mode="auto", mask=_mask(2))
    for n, b in model.module1.named_buffers():
        assert torch.equal(before[n], b)
    assert all(not b.requires_grad for b in model.module1.buffers())
    print("[C05] fusion never modifies prototype buffers: PASS")


# =============================================================================
# D. Gradients, freezing, optimizer
# =============================================================================

def test_d01_no_clip_in_model_or_optimizer():
    model = _banked_module2()
    assert not any("clip" in n.lower() or "text_encoder" in n.lower() for n, _ in model.named_modules())
    assert model.fusion.text_feature.requires_grad is False
    print("[D01] CLIP encoder absent from trained model; text is a buffer: PASS")


def test_d02_enabled_params_registered_once():
    from tasks.mdt_seg import MDTSegTeacher
    import types
    model = _banked_module2()
    cfg = types.SimpleNamespace(
        learning_rate=1e-4, weight_decay=0.0, mixed_precision=False,
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0,
        pspi_proto_contrastive_weight=0.0, pspi_reconstruction_weight=0.0,
        random_state=2023,
    )
    task = MDTSegTeacher({"model": model}, cfg)
    opt_names = []
    for group in task.optimizer.param_groups:
        opt_names.extend([n for n, p in model.named_parameters() if any(p is q for q in group["params"])])
    enabled = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(opt_names) == sorted(enabled)
    assert len(opt_names) == len(set(opt_names))
    assert any(n.startswith("fusion.") for n in opt_names)
    print("[D02] every enabled param in optimizer exactly once: PASS")


def test_d03_two_updates_reach_prefixed_layers():
    model = _banked_module2()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask(2)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
                    forward_mode="auto", mask=mask)
        out["logits"].square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    reached = False
    for block in model.fusion.scales:
        if block.afa_ct.weight.grad is not None and float(block.afa_ct.weight.grad.abs().sum()) > 0:
            reached = True
    assert reached
    print("[D03] two optimizer steps reach pre-zero-init AFA layers: PASS")


def test_d04_ablation_branches_frozen():
    model = _module2_model(
        module2_use_text=False, module2_use_state=False, module2_use_afa=False,
    )
    frozen = [n for n, p in model.fusion.named_parameters() if not p.requires_grad]
    assert any(".state_prompt" in n for n in frozen)
    assert any(".text_proj" in n or ".text_gate" in n for n in frozen)
    assert any(".afa_" in n for n in frozen)
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
                forward_mode="auto", mask=_mask(2))
    model.zero_grad(set_to_none=True)
    out["logits"].square().mean().backward()
    for n, p in model.fusion.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n
    print("[D04] disabled ablation branches frozen with no grads: PASS")


def test_d05_cold_missing_only_no_fusion_grad():
    model = _module2_model(module2_use_text=False)
    model.train()
    model.zero_grad(set_to_none=True)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out = model(ct, pet=pet, forward_mode="missing", mask=_mask(2))
    out["logits"].sum().backward()
    assert all(p.grad is None for p in model.fusion.parameters())
    print("[D05] cold-start Missing-only yields no fusion grads (bypassed): PASS")


def test_d06_amp_finite_grads():
    model = _banked_module2(module2_use_text=False)
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = _mask(2)
    with torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16):
        out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
                    forward_mode="auto", mask=mask)
        loss = out["logits"].float().square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad.float()).all() for p in model.parameters() if p.requires_grad)
    print("[D06] AMP forward/loss/grads finite: PASS")


# =============================================================================
# F. Checkpoints + ablations
# =============================================================================

def _roundtrip(model):
    buf = io.BytesIO()
    torch.save({"model": model.state_dict()}, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)["model"]


def test_f01_new_checkpoint_strict_roundtrip():
    model = _banked_module2()
    model.eval()
    sd = _roundtrip(model)
    assert any(k.startswith("fusion.") for k in sd)
    assert "fusion.text_feature" in sd and "fusion.text_ready" in sd
    fresh = _module2_model()
    fresh.load_state_dict(sd, strict=True)
    fresh.eval()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a = fresh(ct, pet=pet, forward_mode="full")["logits"]
        b = model(ct, pet=pet, forward_mode="full")["logits"]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    print("[F01] Module-2 checkpoint strict round-trip identical: PASS")


def test_f02_old_checkpoint_loads_disabled():
    model = _plain_model()
    sd = _roundtrip(model)
    assert not any(k.startswith("fusion.") for k in sd)
    fresh = _plain_model()
    fresh.load_state_dict(sd, strict=True)
    print("[F02] legacy AddFusion checkpoint strict loads when disabled: PASS")


def test_f03_checkpoint_restore_without_text_files():
    import models.build_mdt_seg as builder
    model = _banked_module2()
    model.eval()
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
    calls = {"n": 0}
    orig = builder.encode_fixed_text
    builder.encode_fixed_text = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    try:
        cfg = _cpu_config(module2_enabled=True, module2_text_model_path="/nonexistent/clip")
        fresh = build_mdt_seg_teacher(cfg, module2_checkpoint_state=sd)["model"]
        fresh.load_state_dict(sd, strict=True)
        fresh.eval()
        ct = torch.randn(1, 1, 64, 64)
        pet = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            a = fresh(ct, pet=pet, forward_mode="full")["logits"]
            b = model(ct, pet=pet, forward_mode="full")["logits"]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    finally:
        builder.encode_fixed_text = orig
    assert calls["n"] == 0
    print("[F03] checkpoint restore needs zero text files/loads: PASS")


def test_f04_mismatch_errors():
    import models.build_mdt_seg as builder
    from models.petct_state_text_afa import StateTextAFAFusion
    model = _banked_module2()
    sd = {k: v for k, v in model.state_dict().items()}
    cfg_off = _cpu_config()
    try:
        builder.build_mdt_seg_teacher(cfg_off, module2_checkpoint_state=sd)
    except RuntimeError as e:
        assert "fusion" in str(e)
    else:
        raise AssertionError("fusion keys + disabled config must fail")
    cfg_on = _cpu_config(module2_enabled=True, module2_use_text=False)
    try:
        builder.build_mdt_seg_teacher(cfg_on, module2_checkpoint_state={"enc_ct.foo": torch.zeros(1)})
    except RuntimeError as e:
        assert "fusion" in str(e)
    else:
        raise AssertionError("enabled config + AddFusion state must fail")
    print("[F04] config/state mismatches fail loudly: PASS")


def test_f05_ablations_runnable():
    for flags in (
        {"module2_use_text": False},
        {"module2_use_state": False},
        {"module2_use_afa": False},
    ):
        model = _banked_module2(**flags)
        model.eval()
        ct = torch.randn(2, 1, 64, 64)
        pet = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]),
                        forward_mode="auto", mask=_mask(2))
        assert torch.isfinite(out["logits"]).all()
    off = _plain_model()
    off.eval()
    with torch.no_grad():
        out = off(torch.randn(1, 1, 64, 64), pet=torch.randn(1, 1, 64, 64), forward_mode="full")
    assert torch.isfinite(out["logits"]).all()
    print("[F05] all ablations + disabled runnable: PASS")


def test_f06_extra_state_config_guard():
    model = _banked_module2()
    state = model.fusion.get_extra_state()
    clone = _module2_model()
    clone.fusion.set_extra_state(state)
    bad = dict(state)
    bad["config"] = dict(state["config"])
    bad["config"]["use_afa"] = not bad["config"]["use_afa"]
    try:
        clone.fusion.set_extra_state(bad)
    except ValueError:
        print("[F06] extra-state config mismatch rejected: PASS")
        return
    raise AssertionError("mismatched extra state must fail")


def test_f07_fusion_param_count():
    # 1,159,000 is the reference count for default (64,128,320,512) channels
    # with a 512-D text vector. This suite uses a 16-D synthetic vector to
    # avoid loading CLIP, so assert the exact count for that configuration.
    from models.petct_state_text_afa import StateTextAFAFusion
    ref = StateTextAFAFusion(
        (64, 128, 320, 512), text_feature=torch.randn(1, 512)
    )
    assert sum(p.numel() for p in ref.parameters()) == 1159000
    model = _module2_model()
    total = sum(p.numel() for p in model.fusion.parameters())
    expected = sum(p.numel() for p in StateTextAFAFusion(
        (64, 128, 320, 512), text_feature=torch.randn(1, SYNTHETIC_TEXT_DIM)
    ).parameters())
    assert total == expected, f"fusion params {total} != {expected}"
    print(f"[F07] fusion params total={total} (ref 512-D=1159000): PASS")


def main():
    tests = [
        test_a01_disabled_is_addfusion_no_new_state,
        test_a02_disabled_eval_matches_base_logits,
        test_a03_disabled_train_loss_grad_match,
        test_a04_legacy_config_builds_addfusion,
        test_a05_existing_layers_init_unchanged_by_wiring,
        test_a06_disabled_no_clip_io_even_if_missing,
        test_b01_enabled_shapes_finite_all_modes,
        test_b02_state_valid_routing_recorded,
        test_b03_cold_start_strict_ct_despite_nonzero_bias,
        test_b04_ready_missing_uses_text_afa,
        test_b05_pspi_off_validity,
        test_b06_legacy_affine_off_path_fused,
        test_c01_missing_inference_no_pet_encoder_collect_finalize,
        test_c02_full_never_retrieves_or_affine,
        test_c03_missing_eval_isolated_from_real_pet,
        test_c04_missing_grad_no_real_pet_path,
        test_c05_bank_buffers_untouched_by_fusion,
        test_d01_no_clip_in_model_or_optimizer,
        test_d02_enabled_params_registered_once,
        test_d03_two_updates_reach_prefixed_layers,
        test_d04_ablation_branches_frozen,
        test_d05_cold_missing_only_no_fusion_grad,
        test_d06_amp_finite_grads,
        test_f01_new_checkpoint_strict_roundtrip,
        test_f02_old_checkpoint_loads_disabled,
        test_f03_checkpoint_restore_without_text_files,
        test_f04_mismatch_errors,
        test_f05_ablations_runnable,
        test_f06_extra_state_config_guard,
        test_f07_fusion_param_count,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - harness must report
            failed.append((t.__name__, repr(e)))
            print(f"[FAIL] {t.__name__}: {e!r}")
    if failed:
        print(f"\n[RESULT] {len(tests) - len(failed)}/{len(tests)} passed, {len(failed)} FAILED")
        sys.exit(1)
    print(f"\n[RESULT] all {len(tests)} module2 integration tests passed")


if __name__ == "__main__":
    main()
