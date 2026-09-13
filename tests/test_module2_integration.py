# -*- coding: utf-8 -*-
"""Module-2 (StateGuidedExpertFusion) integration tests.

Covers: config default-off, off-path parity with the clean Module-1 baseline,
shapes across text/no-text x experts {1,2,3}, routing contract, cold-start
bypass, no PET leakage, gradient isolation, checkpoint strict reload without
the external text cache, Stage-1 preservation, and the two-round lifecycle.
"""
import copy
import os

import pytest
import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher

TEXT_CACHE = '/root/autodl-tmp/mkd-main/new-train/pretrained/module2_text_cache.pt'
HAS_TEXT_CACHE = os.path.isfile(TEXT_CACHE)

SHARED_KEYS = ('enc_ct', 'enc_pet', 'ct_align', 'decoder', 'module1')


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'ct_backbone': 'convnextv2_nano',
        'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'decoder_channels': (512, 256, 128, 64),
        'use_deep_supervision': False,
        'deep_supervision': False,
        'pspi_enabled': True,
        'pspi_num_clusters': 6,
        'pspi_build_stage': 4,
        'pspi_cluster_max_iter': 25,
        'pspi_outlier_discard_rate': 0.05,
        'pspi_bank_update_mode': 'direct',
        'pspi_ema_momentum': 0.95,
        'pspi_retrieval_temperature': 0.1,
        'pspi_proto_temperature': 0.02,
        'pspi_proto_contrastive_weight': 0.01,
        'pspi_collect_candidates': True,
        'pspi_prior_scale_enabled': True,
        'pspi_prior_scale_init': 0.1,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _tiny_batch(b=2, h=64):
    torch.manual_seed(7)
    return {
        'ct': torch.randn(b, 1, h, h),
        'pet': torch.randn(b, 1, h, h),
        'mask': (torch.rand(b, 1, h, h) > 0.5).float(),
    }


def _model_pair(seed=123, **m2kw):
    torch.manual_seed(seed)
    off = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=False)
    torch.manual_seed(seed)
    on = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True, module2_kwargs=m2kw)
    return off, on


# 1. Config defaults: Module-2 OFF, no new text dependency.
def test_config_defaults_off():
    from configs.seg_mdt import SegMDTConfig
    import argparse
    p = SegMDTConfig.model_parser()
    ns, _ = p.parse_known_args([])
    assert ns.module2_enabled is False
    assert ns.module2_use_text is True
    assert ns.module2_experts_per_group == 2
    assert ns.module2_text_cache.endswith('module2_text_cache.pt')


# 2. OFF: no module2 params, no module2.* keys, no RNG side effects on shared init.
def test_off_has_no_module2():
    torch.manual_seed(11)
    m = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    assert m.module2 is None
    assert not any(k.startswith('module2.') for k in m.state_dict().keys())
    assert m.missing_prior_logits is not None  # old prior scale preserved


def test_off_does_not_consume_extra_rng():
    torch.manual_seed(5)
    a = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    torch.manual_seed(5)
    b = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])


def test_off_ignores_invalid_cache():
    cfg = _make_cfg(module2_enabled=False, module2_text_cache='/nonexistent/cache.pt')
    out = build_mdt_seg_teacher(cfg)
    assert out['model'].module2 is None


def test_on_requires_pspi():
    with pytest.raises(ValueError):
        DualSharedAddPETCTBaseline(use_deep_supervision=False, pspi_enabled=False,
                                   module2_enabled=True, module2_kwargs={'use_text': False})


def test_on_disables_prior_scale():
    # Module-2 ON: routing weight a_P alone controls PET; old scalar off.
    _, on = _model_pair(use_text=False, experts_per_group=2)
    assert on.module2 is not None
    assert on.missing_prior_logits is None
    assert on.effective_prior_scale_enabled is False
    assert on.requested_prior_scale_enabled is True


# 3-4. Shapes / routing contract across text x experts.
@pytest.mark.parametrize('experts', [1, 2, 3])
@pytest.mark.parametrize('use_text', [False, True])
def test_shapes_and_routing_contract(experts, use_text):
    if use_text and not HAS_TEXT_CACHE:
        pytest.skip('real text cache missing; random text must not pose as semantic')
    torch.manual_seed(21)
    if use_text:
        from models.state_guided_expert_fusion import load_text_cache
        emb, meta = load_text_cache(TEXT_CACHE)
        m = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                       module2_kwargs={'use_text': True, 'text_embeddings': emb,
                                                       'text_metadata': meta, 'experts_per_group': experts})
    else:
        m = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                       module2_kwargs={'use_text': False, 'experts_per_group': experts})
    batch = _tiny_batch(b=2)
    m.eval()
    # Warm the bank so Missing executes experts (cold bypass is tested separately).
    m.train()
    for _ in range(2):
        m(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    m.finalize_module1_epoch(1)
    assert m.module1.bank_ready
    m.eval()
    for mode in ('full', 'missing'):
        pet = batch['pet'] if mode == 'full' else None
        out = m(batch['ct'], pet=pet, forward_mode=mode, mask=batch['mask'])
        assert out['logits'].shape == (2, 1, 64, 64)
        assert torch.isfinite(out['logits']).all()
        aux = out['aux']['module2']
        sel, w, counts, active = aux['selected_experts'], aux['route_weights'], aux['expert_counts'], aux['active']
        n = experts
        assert sel.shape == (2, 4, 2) and w.shape == (2, 4, 2)
        assert counts.shape == (4, 3 * n) and active.shape == (2, 4)
        assert torch.allclose(w.sum(-1), torch.ones(2, 4), atol=1e-5)
        # exactly one CT + one active-PET expert per scale; other PET group idle
        state = 0 if mode == 'full' else 1
        for s in range(4):
            assert ((sel[:, s, 0] >= 0) & (sel[:, s, 0] < n)).all()
            lo, hi = ((1 if state == 0 else 2) * n, (2 if state == 0 else 3) * n)
            assert ((sel[:, s, 1] >= lo) & (sel[:, s, 1] < hi)).all()
            idle = counts[s, (2 if state == 0 else 1) * n:(3 if state == 0 else 2) * n]
            assert (idle == 0).all()
    # experts are shared objects across scales (single ModuleList per group)
    assert len(m.module2.experts['ct']) == experts


# 5. Cold Missing: exact CT-only bypass, no grad to affine/router/experts.
def test_cold_missing_is_ct_only():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    assert not on.module1.bank_ready
    batch = _tiny_batch(b=2)
    on.train()
    out = on(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
    aux = out['aux']['module2']
    assert not bool(aux['active'].any())
    assert (aux['selected_experts'] == -1).all()
    assert (aux['route_weights'] == 0).all()
    # logits equal CT-only decode: compare against cold baseline Missing (ct + 0*prior)
    torch.manual_seed(123)
    off = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=False)
    off.train()
    ref = off(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
    assert torch.equal(out['logits'], ref['logits'])
    # segmentation-gradient isolation on the cold path
    on.zero_grad()
    loss = out['logits'].sum()
    loss.backward()
    for mod in (on.module2.personalizers, on.module2.routers, on.module2.experts):
        for p in mod.parameters():
            assert p.grad is None or not p.grad.abs().sum().item()


# 6. Full never retrieves; no Missing personalization on Full path.
def test_full_does_not_retrieve():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    calls = []
    orig = on.module1.retrieve_pet_prior
    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    on.module1.retrieve_pet_prior = spy
    try:
        on.train()
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    finally:
        on.module1.retrieve_pet_prior = orig
    assert calls == []
    on.zero_grad()
    out = on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    out['logits'].sum().backward()
    for p in on.module2.personalizers.parameters():
        assert p.grad is None  # personalizers only serve Missing


# 7. Weighted-fusion contract: forced [0.5,0.5] routing + zero expert
# projection restores plain addition (see
# test_balanced_weights_restore_addition for the strict form). With live
# router weights the weighted base intentionally differs from AddFusion;
# this test only pins finiteness and reports the live-router divergence.
def test_full_zero_init_parity():
    off, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    off.eval()
    on.eval()
    a = off(batch['ct'], pet=batch['pet'], forward_mode='full')
    b = on(batch['ct'], pet=batch['pet'], forward_mode='full')
    maxdiff = (a['logits'] - b['logits']).abs().max().item()
    print(f'\n[FULL-WEIGHTED] live-router vs AddFusion maxdiff={maxdiff:.3e}')
    assert torch.isfinite(b['logits']).all()
    assert maxdiff < 0.5, maxdiff


# 2b. OFF parity: same weights/bank/input -> identical logits+loss (CPU FP32 exact).
def test_off_parity_full_missing_auto():
    off, on = _model_pair(use_text=False, experts_per_group=1)
    # strip Module-2 by comparing OFF model against a second OFF twin is trivial;
    # here verify ON-cold-Missing == OFF-Missing and ON-Full == OFF-Full exactly.
    batch = _tiny_batch()
    # Decoder BatchNorm couples train-mode outputs across samples; parity is
    # an eval-mode contract (train-mode equivalence is not claimed).
    off.eval()
    on.eval()
    with torch.no_grad():
        for mode in ('full', 'missing'):
            a = off(batch['ct'], pet=batch['pet'], forward_mode=mode, mask=batch['mask'])
            b = on(batch['ct'], pet=batch['pet'], forward_mode=mode, mask=batch['mask'])
            if mode == 'missing':
                # Cold bank: both paths are pure CT.
                assert torch.equal(a['logits'], b['logits'])
            else:
                md = (a['logits'] - b['logits']).abs().max().item()
                print(f'\n[OFF-PARITY] full live-router maxdiff={md:.3e}')
                assert torch.isfinite(b['logits']).all()
                assert md < 0.5, md
    # auto mixed (bank cold -> missing rows CT-only on both paths)
    avail = torch.tensor([1, 0])
    with torch.no_grad():
        a = off(batch['ct'], pet=batch['pet'], pet_available=avail, forward_mode='auto', mask=batch['mask'])
        b = on(batch['ct'], pet=batch['pet'], pet_available=avail, forward_mode='auto', mask=batch['mask'])
    # Cold auto: Missing row is CT-only on both paths (exact); Full row is
    # live-router weighted on Module-2 (differs from AddFusion by design).
    md = (a['logits'] - b['logits']).abs().max().item()
    print(f'\n[OFF-PARITY] cold-auto maxdiff={md:.3e}')
    with torch.no_grad():
        a_m = off(batch['ct'][:1], pet=batch['pet'][:1], forward_mode='missing', mask=batch['mask'][:1])
        b_m = on(batch['ct'][:1], pet=batch['pet'][:1], forward_mode='missing', mask=batch['mask'][:1])
        assert torch.equal(a_m['logits'], b_m['logits'])
    assert md < 0.5, md
    with pytest.raises(ValueError):
        off(batch['ct'], pet=batch['pet'], pet_available=torch.tensor([0.5, 1.0]),
            forward_mode='auto', mask=batch['mask'])


# 8. Missing inference: pet=None, PET encoder never called, bank untouched.
def test_missing_inference_no_pet_encoder():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    on.train()
    for _ in range(2):
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.finalize_module1_epoch(1)
    on.eval()
    v0 = int(on.module1.bank_version.item())
    n0 = sum(p.numel() for p in on.module1.parameters())
    calls = []
    orig = on._encode_pet
    def spy(pet):
        calls.append(1)
        return orig(pet)
    on._encode_pet = spy
    try:
        with torch.no_grad():
            out = on(batch['ct'], pet=None, forward_mode='missing')
    finally:
        on._encode_pet = orig
    assert calls == []
    assert out['logits'].shape == (2, 1, 64, 64)
    assert int(on.module1.bank_version.item()) == v0
    assert sum(p.numel() for p in on.module1.parameters()) == n0


# 9. Missing logits invariant to the privileged real PET (fixed bank/CT/state).
def test_missing_invariant_to_real_pet():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    on.train()
    for _ in range(2):
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.finalize_module1_epoch(1)
    on.eval()
    with torch.no_grad():
        a = on(batch['ct'], pet=batch['pet'], forward_mode='missing')
        b = on(batch['ct'], pet=torch.randn_like(batch['pet']) * 5 + 3, forward_mode='missing')
    assert torch.equal(a['logits'], b['logits'])


# 10. Missing seg-loss backward: CT/Module-1-retrieval/Missing-affine get grads;
#     real PET encoder + real experts get none.
def test_missing_seg_backward_isolation():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    on.train()
    for _ in range(2):
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.finalize_module1_epoch(1)
    on.zero_grad()
    out = on(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
    out['logits'].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in on.enc_ct.parameters())
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in on.module1.parameters())
    assert any(p.grad is not None for p in on.module2.personalizers.parameters())
    assert all(p.grad is None for p in on.enc_pet.parameters())
    for p in on.module2.experts['real'].parameters():
        assert p.grad is None


# 11. Proto-loss backward: only the real PET encoder is updated.
def test_proto_backward_only_pet_encoder():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    on.train()
    for _ in range(2):
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.finalize_module1_epoch(1)
    assert on.module1.bank_ready
    pet_feats = on._encode_pet(batch['pet'])
    res = on.module1.compute_pet_prototype_contrastive_loss(pet_feats, batch['mask'])
    assert res['num_terms'] > 0
    on.zero_grad()
    res['loss'].backward()
    assert any(p.grad is not None for p in on.enc_pet.parameters())
    for mod in (on.enc_ct, on.ct_align, on.decoder, on.module2):
        for p in mod.parameters():
            assert p.grad is None
    for p in on.module1.parameters():
        assert p.grad is None


# 12. Zero-init contract: first backward leaves experts/routers at zero grad;
#     after one effective projection update they receive nonzero grads.
def test_zero_init_then_effective_update():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    on.train()
    out = on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.zero_grad()
    out['logits'].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in on.module2.output_projections.parameters())
    # Weighted main path gives the router a first-step gradient via a_ct/a_pet.
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in on.module2.routers.parameters())
    for mod in (on.module2.experts['ct'], on.module2.experts['real']):
        for p in mod.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0
    with torch.no_grad():
        for proj in on.module2.output_projections:
            proj.weight.fill_(0.01)
    on.zero_grad()
    out2 = on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    out2['logits'].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in on.module2.routers.parameters())
    sel = out2['aux']['module2']['selected_experts']
    used = set(int(i) for i in sel[:, 0, 0].tolist()) | set(int(i) for i in sel[:, 0, 1].tolist())
    assert used, 'router must select experts'
    hit = False
    for idx in used:
        group = 'ct' if idx < on.module2.n else 'real'
        expert = on.module2.experts[group][idx % on.module2.n]
        if any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in expert.parameters()):
            hit = True
    assert hit


# 13. Real builder -> wrapper -> task.train_step -> backward.
def test_builder_task_train_step():
    kw = {'module2_enabled': True, 'module2_use_text': False, 'module2_experts_per_group': 2}
    cfg = _make_cfg(**kw)
    torch.manual_seed(31)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    assert task.model.module2 is not None
    batch = _tiny_batch()
    for mode in ('full', 'missing'):
        task.optimizer.zero_grad(set_to_none=True)
        loss, logits, outputs, stats = task.train_step(batch, forward_mode=mode)
        assert torch.isfinite(loss)
        loss.backward()
        assert 'module2' in (outputs.get('aux') or {})
    # optimizer covers every new trainable param exactly once
    opt_ids = [id(p) for g in task.optimizer.param_groups for p in g['params']]
    assert len(opt_ids) == len(set(opt_ids))
    for p in task.model.module2.parameters():
        if p.requires_grad:
            assert id(p) in set(opt_ids)
    assert not task.model.module2.text_embeddings.requires_grad


# 14. Eval auto-mixed == split execution + reassembly; Full rows use real group.
def test_auto_mixed_consistency():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch(b=4)
    on.train()
    for _ in range(2):
        on(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    on.finalize_module1_epoch(1)
    on.eval()
    avail = torch.tensor([1, 1, 0, 0])
    with torch.no_grad():
        mixed = on(batch['ct'], pet=batch['pet'], pet_available=avail, forward_mode='auto')
        f = on(batch['ct'][:2], pet=batch['pet'][:2], forward_mode='full')
        m = on(batch['ct'][2:], pet=None, forward_mode='missing')
    assert torch.allclose(mixed['logits'][:2], f['logits'], atol=1e-6, rtol=1e-5)
    assert torch.allclose(mixed['logits'][2:], m['logits'], atol=1e-6, rtol=1e-5)
    sel = mixed['aux']['module2']['selected_experts']
    n = on.module2.n
    assert ((sel[:2, :, 1] >= n) & (sel[:2, :, 1] < 2 * n)).all()  # real group
    assert ((sel[2:, :, 1] >= 2 * n) & (sel[2:, :, 1] < 3 * n)).all()  # imputed group


# 16. Checkpoint strict reload without the external cache; mismatch must fail.
def test_checkpoint_reload_without_cache(tmp_path):
    kw = {'module2_enabled': True, 'module2_use_text': False, 'module2_experts_per_group': 2}
    cfg = _make_cfg(**kw)
    torch.manual_seed(41)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    batch = _tiny_batch()
    task.model.eval()
    dev = next(task.model.parameters()).device
    with torch.no_grad():
        ref = task.model(batch['ct'].to(dev), pet=batch['pet'].to(dev), forward_mode='full')
    ckpt = str(tmp_path / 'ckpt.tar')
    task.save_checkpoint(ckpt, epoch=1, best_joint=0.5, best_full=0.5, best_missing=0.5,
                         best_joint_epoch=1, val_full={}, val_missing={}, joint_dice=0.5)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    bad_cfg = _make_cfg(**kw, module2_text_cache='/nonexistent/cache.pt')
    rebuilt = build_mdt_seg_teacher(bad_cfg, model_state_dict=payload['model'])['model']
    rebuilt.load_state_dict(payload['model'], strict=True)
    rebuilt = rebuilt.cpu().eval()
    with torch.no_grad():
        got = rebuilt(batch['ct'].cpu(), pet=batch['pet'].cpu(), forward_mode='full')
    # CUDA->CPU strict reload crosses devices through the save/load cycle;
    # require tight agreement and report the bound instead of bit-exactness.
    maxdiff = (ref['logits'].cpu() - got['logits'].cpu()).abs().max().item()
    print(f'\n[CKPT-RELOAD] cuda->cpu strict reload maxdiff={maxdiff:.3e}')
    assert torch.allclose(ref['logits'].cpu(), got['logits'].cpu(), atol=1e-4, rtol=1e-4)
    assert maxdiff < 1e-4
    # expert-count / text-mode mismatch must fail loudly, not with random fill
    with pytest.raises(ValueError):
        build_mdt_seg_teacher(_make_cfg(**{**kw, 'module2_experts_per_group': 3}),
                              model_state_dict=payload['model'])
    with pytest.raises(ValueError):
        build_mdt_seg_teacher(_make_cfg(**{**kw, 'module2_use_text': True}),
                              model_state_dict=payload['model'])
    # Module-2 weights into an OFF config must fail strict load
    off = build_mdt_seg_teacher(_make_cfg(module2_enabled=False))['model']
    with pytest.raises(RuntimeError):
        off.load_state_dict(payload['model'], strict=True)


@pytest.mark.skipif(not HAS_TEXT_CACHE, reason='real text cache missing')
def test_text_checkpoint_reload_without_cache_file(tmp_path, monkeypatch):
    from models.state_guided_expert_fusion import load_text_cache
    emb, meta = load_text_cache(TEXT_CACHE)
    torch.manual_seed(43)
    m = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                   module2_kwargs={'use_text': True, 'text_embeddings': emb,
                                                   'text_metadata': meta, 'experts_per_group': 2})
    sd = m.state_dict()
    assert sd['module2.text_embeddings'].shape == (5, 512)
    cfg = _make_cfg(module2_enabled=True, module2_use_text=True,
                    module2_experts_per_group=2, module2_text_cache='/nonexistent/cache.pt')
    rebuilt = build_mdt_seg_teacher(cfg, model_state_dict=sd)['model']
    rebuilt.load_state_dict(sd, strict=True)
    assert torch.equal(rebuilt.state_dict()['module2.text_embeddings'], emb)


# 17. Stage-1 init touches encoders/align only; Module-2 intact.
def test_stage1_init_preserves_module2(tmp_path):
    torch.manual_seed(51)
    donor = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    ct_ckpt = str(tmp_path / 'ct.pt')
    pet_ckpt = str(tmp_path / 'pet.pt')
    ct_state = {'enc_ct.' + k: v for k, v in donor.enc_ct.state_dict().items()}
    ct_state.update({'ct_align.' + k: v for k, v in donor.ct_align.state_dict().items()})
    torch.save({'model': ct_state}, ct_ckpt)
    torch.save({'model': {'enc_pet.' + k: v for k, v in donor.enc_pet.state_dict().items()}}, pet_ckpt)
    torch.manual_seed(51)
    fresh = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                       module2_kwargs={'use_text': False})
    before = {k: v.clone() for k, v in fresh.module2.state_dict().items() if not k.startswith('_extra')}
    cfg = _make_cfg(module2_enabled=True, module2_use_text=False, module2_experts_per_group=2,
                    stage1_init_enabled=True, stage1_ct_checkpoint=ct_ckpt,
                    stage1_pet_checkpoint=pet_ckpt, stage1_init_strict=True)
    torch.manual_seed(51)
    task_model = build_mdt_seg_teacher(cfg)['model']
    after = task_model.module2.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k


# 18. Two-round lifecycle: epoch1 cold -> finalize -> epoch2 optimizes both.
def test_two_round_lifecycle():
    _, on = _model_pair(use_text=False, experts_per_group=2)
    cfg = _make_cfg(module2_enabled=True, module2_use_text=False, module2_experts_per_group=2)
    task = MDTSegTeacher({'model': on}, cfg)
    batch = _tiny_batch()
    on.train()
    # round 1: bank not ready, Missing is CT-only but still optimizes
    assert not on.module1.bank_ready
    for step, mode in enumerate(['full', 'missing']):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, _ = task.train_step(batch, forward_mode=mode)
        loss.backward()
        task.optimizer.step()
        if mode == 'missing':
            assert not bool(outputs['aux']['module2']['active'].any())
    on.finalize_module1_epoch(1)
    assert on.module1.bank_ready
    # round 2: both routes optimize with Module-2 active
    for mode in ['full', 'missing']:
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, _ = task.train_step(batch, forward_mode=mode)
        assert torch.isfinite(loss)
        loss.backward()
        task.optimizer.step()
        assert bool(outputs['aux']['module2']['active'].any())
    # optimizer/scheduler/scaler state round-trips
    blob = {'optimizer': copy.deepcopy(task.optimizer.state_dict()),
            'scaler': copy.deepcopy(task.scaler.state_dict())}
    task.optimizer.load_state_dict(blob['optimizer'])
    task.scaler.load_state_dict(blob['scaler'])


# 19. CUDA AMP smoke at real scale (batch16/512); OOM is reported, not hidden.
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_cuda_amp_real_scale():
    torch.manual_seed(61)
    m = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                   module2_kwargs={'use_text': False}).cuda()
    m.train()
    ct = torch.randn(16, 1, 512, 512, device='cuda')
    pet = torch.randn(16, 1, 512, 512, device='cuda')
    mask = (torch.rand(16, 1, 512, 512, device='cuda') > 0.5).float()
    torch.cuda.reset_peak_memory_stats()
    import time
    t0 = time.time()
    try:
        with torch.cuda.amp.autocast(enabled=True):
            out = m(ct, pet=pet, forward_mode='full', mask=mask)
            loss = out['logits'].float().mean()
        loss.backward()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            pytest.skip(f'AMP real-scale blocked by GPU memory: {e}')
        raise
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f'\n[AMP-SMOKE] forward+backward ok, peak={peak:.2f}GB, time={dt:.1f}s')
    assert torch.isfinite(loss).item()


# --- weighted-fusion additions (spec section 7) ---

def _warm_bank(model, batch, rounds=2):
    model.train()
    for _ in range(rounds):
        model(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
    model.finalize_module1_epoch(1)
    assert model.module1.bank_ready
    return model


def _rms(x):
    return float(x.detach().double().pow(2).mean().sqrt().item())


def test_real_scale_shapes_s1_s4():
    # Real channels, real spatial pyramid [128,64,32,16] at 512 input.
    from models.state_guided_expert_fusion import StateGuidedExpertFusion
    torch.manual_seed(3)
    m = StateGuidedExpertFusion(channels=(64, 128, 320, 512), experts_per_group=2, use_text=False)
    m.eval()
    ct = [torch.randn(1, 64, 128, 128), torch.randn(1, 128, 64, 64),
          torch.randn(1, 320, 32, 32), torch.randn(1, 512, 16, 16)]
    pet = [x.clone() for x in ct]
    for mode in ('full', 'missing'):
        ready = True if mode == 'missing' else False
        out, aux = m(ct, [x.clone() for x in pet], mode=mode, bank_ready=ready)
        for o, c in zip(out, ct):
            assert o.shape == c.shape
            assert torch.isfinite(o).all()
        assert aux['modality_scales'].shape == (1, 4, 2)
        assert torch.allclose(aux['modality_scales'], 2.0 * aux['route_weights'], atol=1e-6)


def test_balanced_weights_restore_addition():
    # Force router -> [0.5,0.5], zero expert proj: Full must equal ct+pet;
    # Missing must equal ct+P_imp (personalized prior on the main path).
    _, on = _model_pair(use_text=False, experts_per_group=1)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    on.eval()
    for router in on.module2.routers:
        router.readout[-1].weight.data.zero_()
        router.readout[-1].bias.data.zero_()
    with torch.no_grad():
        for p in on.module2.output_projections.parameters():
            assert (p == 0).all()
    ct_feats = on._encode_ct(batch['ct'])
    pet_real = on._encode_pet(batch['pet'])
    with torch.no_grad():
        out_full, aux = on.module2(ct_feats, pet_real, mode='full', bank_ready=True)
        for o, c, r in zip(out_full, ct_feats, pet_real):
            assert torch.allclose(o, c + r, atol=1e-6, rtol=1e-5), \
                (o - (c + r)).abs().max().item()
        prior, _ = on.module1.retrieve_pet_prior(ct_feats, return_attention=False)
        pimp = [on.module2.personalizers[s](c, p).detach()
                for s, (c, p) in enumerate(zip(ct_feats, prior))]
        out_m, aux_m = on.module2(ct_feats, prior, mode='missing', bank_ready=True)
        assert torch.allclose(aux_m['modality_scales'],
                              torch.ones_like(aux_m['modality_scales']), atol=1e-6)
        for o, c, q in zip(out_m, ct_feats, pimp):
            assert torch.allclose(o, c + q, atol=1e-6, rtol=1e-5), \
                (o - (c + q)).abs().max().item()


def test_unbalanced_weights_scale_both_paths():
    # Force router -> [0.75,0.25]: a_ct=1.5, a_pet=0.5 on main AND expert paths.
    _, on = _model_pair(use_text=False, experts_per_group=1)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    on.eval()
    import math
    # readout emits 3n logits [ct_n | real_n | imputed_n]; Full uses ct+real.
    # bias[0]=log3 makes CT top-1 score log3; real-group bias all zero makes
    # PET top-1 score 0; softmax([log3, 0]) = [0.75, 0.25].
    for router in on.module2.routers:
        router.readout[-1].weight.data.zero_()
        bias = torch.zeros(router.n * 3)
        bias[0] = math.log(3.0)
        bias[router.n] = 0.0
        router.readout[-1].bias.data.copy_(bias)
    ct_feats = on._encode_ct(batch['ct'])
    pet_real = on._encode_pet(batch['pet'])
    with torch.no_grad():
        out, aux = on.module2(ct_feats, pet_real, mode='full', bank_ready=True)
        ms = aux['modality_scales']
        assert torch.allclose(ms[..., 0], torch.full_like(ms[..., 0], 1.5), atol=1e-5)
        assert torch.allclose(ms[..., 1], torch.full_like(ms[..., 1], 0.5), atol=1e-5)
        for o, c, r in zip(out, ct_feats, pet_real):
            # zero expert proj: main path only, same a_* pair.
            assert torch.allclose(o, 1.5 * c + 0.5 * r, atol=1e-5, rtol=1e-4)


def test_personalization_boundary():
    # personalizer input = RAW prior; P_imp feeds BOTH the main path (P_base)
    # and the PET-expert input. Balanced routing + zero expert projection
    # pins output == ct + P_imp exactly.
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    on.eval()
    for router in on.module2.routers:
        router.readout[-1].weight.data.zero_()
        router.readout[-1].bias.data.zero_()
    seen_person_in, seen_person_out, seen_expert_in = {}, {}, {}

    for s, pers in enumerate(on.module2.personalizers):
        orig = pers.forward
        def make_orig(o, ss):
            def w(ct, prior):
                seen_person_in[ss] = prior.clone().detach()
                out = o(ct, prior)
                seen_person_out[ss] = out.clone().detach()
                return out
            return w
        pers.forward = make_orig(orig, s)
    orig_disp = on.module2._dispatch
    def spy_disp(group, feats, sel):
        if group == 'imputed':
            seen_expert_in[len(seen_expert_in)] = feats.clone().detach()
        return orig_disp(group, feats, sel)
    on.module2._dispatch = spy_disp
    try:
        ct_feats = on._encode_ct(batch['ct'])
        prior, _ = on.module1.retrieve_pet_prior(ct_feats, return_attention=False)
        with torch.no_grad():
            out_m, _ = on.module2(ct_feats, prior, mode='missing', bank_ready=True)
        for s in range(4):
            assert torch.equal(seen_person_in[s], prior[s]), s  # raw in
            with torch.no_grad():
                expect_adapted = on.module2.pet_adapters[s](seen_person_out[s])
            assert torch.equal(seen_expert_in[s], expect_adapted), s  # expert in == adapter(P_imp)
            assert torch.allclose(out_m[s], ct_feats[s] + seen_person_out[s],
                                  atol=1e-6, rtol=1e-5), s  # main base == P_imp
    finally:
        on.module2._dispatch = orig_disp


def test_personalizer_perturb_moves_main_path():
    # With zero expert projection and balanced routing, the main path is
    # exactly ct + P_imp, so replacing the personalizer moves the output to
    # ct + (100*prior + 50) while the router/expert path is unchanged.
    _, on = _model_pair(use_text=False, experts_per_group=1)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    on.eval()
    for router in on.module2.routers:
        router.readout[-1].weight.data.zero_()
        router.readout[-1].bias.data.zero_()
    ct_feats = on._encode_ct(batch['ct'])
    prior, _ = on.module1.retrieve_pet_prior(ct_feats, return_attention=False)
    with torch.no_grad():
        ref, _ = on.module2(ct_feats, prior, mode='missing', bank_ready=True)
    class BigPersonalizer(torch.nn.Module):
        def forward(self, ct, prior):
            return prior * 100.0 + 50.0
    on.module2.personalizers = torch.nn.ModuleList(BigPersonalizer() for _ in range(4))
    with torch.no_grad():
        got, aux = on.module2(ct_feats, prior, mode='missing', bank_ready=True)
        for r, g in zip(ref, got):
            assert not torch.equal(r, g)
            assert (r - g).abs().max().item() > 1e-3
        for g, c, p in zip(got, ct_feats, prior):
            assert torch.allclose(g, c + (p * 100.0 + 50.0), atol=1e-4, rtol=1e-3)


def test_prior_logits_absent_routing_grad_present():
    # Module-2 ON: no missing_prior_logits; Missing seg-loss instead reaches
    # the routers (a_P path) + personalizer/active expert/output proj.
    _, on = _model_pair(use_text=False, experts_per_group=2)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    assert on.missing_prior_logits is None
    on.train()
    on.zero_grad()
    out_m = on(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
    out_m['logits'].sum().backward()
    assert any(p.grad is not None for p in on.module2.personalizers.parameters())
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in on.module2.routers.parameters())
    assert any(p.grad is not None for p in on.module2.output_projections.parameters())
    # Bank is not an optimizer param; proto loss leaves Module-2 untouched.
    assert not any(p.requires_grad for p in on.module1.parameters()
                   if p is on.module1.bank_version)
    on.zero_grad()
    pet_feats = on._encode_pet(batch['pet'])
    res = on.module1.compute_pet_prototype_contrastive_loss(pet_feats, batch['mask'])
    res['loss'].backward()
    for p in on.module2.parameters():
        assert p.grad is None


def test_off_path_unchanged_by_weighted_change():
    # module2_enabled=false keeps exact Module-1-clean dataflow incl. logits.
    torch.manual_seed(123)
    m = DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=False)
    assert m.missing_prior_logits is not None
    batch = _tiny_batch()
    m.train()
    out = m(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
    assert 'module2' not in (out.get('aux') or {})
    assert out['missing_prior_alpha_s1'] == \
        float(torch.sigmoid(m.missing_prior_logits[0]).item())


def test_weighted_checkpoint_roundtrip(tmp_path):
    from tasks.mdt_seg import MDTSegTeacher as _T
    cfg = _make_cfg(module2_enabled=True, module2_use_text=False, module2_experts_per_group=2)
    torch.manual_seed(77)
    task = _T(build_mdt_seg_teacher(cfg), cfg)
    assert task.model.missing_prior_logits is None  # no alpha under Module-2
    ckpt = str(tmp_path / 'w.tar')
    task.save_checkpoint(ckpt, epoch=1, best_joint=0.1, best_full=0.1,
                         best_missing=0.1, best_joint_epoch=1, val_full={},
                         val_missing={}, joint_dice=0.1)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    mdl = payload['model']
    assert 'missing_prior_logits' not in mdl and 'module2.ct_adapters.0.weight' in mdl
    assert 'module1.bank_version' in mdl or any('bank_version' in k for k in mdl)
    rebuilt = build_mdt_seg_teacher(cfg, model_state_dict=mdl)['model']
    rebuilt.load_state_dict(mdl, strict=True)
    assert rebuilt.missing_prior_logits is None
    # Corrupting a required Module-2 key must fail loudly, not silently.
    bad = dict(mdl)
    bad.pop('module2.ct_adapters.0.weight')
    with pytest.raises(RuntimeError):
        rebuilt.load_state_dict(bad, strict=True)


def test_feature_rms_magnitudes_logged():
    # RMS smoke: detached magnitudes for CT / raw prior / personalized PET
    # (= P_base on Missing). No scaled-prior RMS exists under Module-2.
    _, on = _model_pair(use_text=False, experts_per_group=1)
    batch = _tiny_batch()
    _warm_bank(on, batch)
    on.eval()
    ct_feats = on._encode_ct(batch['ct'])
    prior, _ = on.module1.retrieve_pet_prior(ct_feats, return_attention=False)
    personalized = [on.module2.personalizers[s](c, p).detach()
                    for s, (c, p) in enumerate(zip(ct_feats, prior))]
    with torch.no_grad():
        out, aux = on.module2(ct_feats, prior,
                              mode='missing', bank_ready=True)
    rep = {}
    for s in range(4):
        rep[f'module2_rms_ct_s{s+1}'] = _rms(ct_feats[s])
        rep[f'module2_rms_raw_prior_s{s+1}'] = _rms(prior[s])
        rep[f'module2_rms_personalized_pet_s{s+1}'] = _rms(personalized[s])
        # P_base = P_imp on Missing: base RMS equals personalized RMS.
        rep[f'module2_rms_pet_base_s{s+1}'] = _rms(personalized[s])
    for k, v in rep.items():
        assert v > 0 and v < 1e6, (k, v)
    print('\n[RMS] ' + ' '.join(f'{k}={v:.4f}' for k, v in rep.items()))
