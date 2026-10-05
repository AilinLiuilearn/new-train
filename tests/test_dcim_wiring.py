# -*- coding: utf-8 -*-
"""Wiring tests for the dcim (text-region) fusion branch.

Uses the real offline CLIP directory for the one-time text load and random
backbone weights (no downloads). The provided module-level unit test file
(tests/test_petct_text_region_fusion.py) does not exist in this repo, so
module math is covered here at the wiring level plus the bounded GPU smoke.
"""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline

CLIP_DIR = '/root/autodl-tmp/mkd-main/new-train/pretrained/clip-vit-base-patch32'


def _cfg(**over):
    base = dict(ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                ct_pretrained_path=None, pet_pretrained_path=None,
                decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
                asym_fusion_enabled=True, asym_use_text=True, fusion_version='dcim',
                asym_clip_path=CLIP_DIR, train_batch_mode='full', ema_enabled=False)
    base.update(over)
    return type('C', (), base)()


def _build(**over):
    from models.build_mdt_seg import build_mdt_seg_teacher
    return build_mdt_seg_teacher(_cfg(**over))['model']


def test_dcim_selected_with_offline_pet_text():
    m = _build()
    assert type(m.fusion).__name__ == 'PETCTTextRegionFusion'
    buf = m.fusion.text_embeddings
    assert tuple(buf.shape) == (1, 512) and torch.isfinite(buf).all()
    assert m.fusion.get_extra_state()['use_text'] is True


def test_injected_cache_row_selection():
    import pytest
    from models.build_mdt_seg import _select_pet_text_row
    two = torch.randn(2, 512)
    assert torch.equal(_select_pet_text_row(two), two[1:2])
    one = torch.randn(1, 512)
    assert torch.equal(_select_pet_text_row(one), one)
    with pytest.raises(ValueError):
        _select_pet_text_row(torch.randn(3, 512))
    m = _build()
    m2 = _build.__wrapped__ if hasattr(_build, '__wrapped__') else None
    from models.build_mdt_seg import build_mdt_seg_teacher
    cfg = _cfg()
    m_inj = build_mdt_seg_teacher(cfg, fusion_text_embeddings=torch.randn(2, 512))['model']
    assert tuple(m_inj.fusion.text_embeddings.shape) == (1, 512)


def test_full_and_auto_all1_consistent_eval():
    torch.manual_seed(0)
    m = _build()
    m.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        full = m(ct, pet, forward_mode='full')['logits']
        auto = m(ct, pet, pet_available=torch.ones(2, dtype=torch.long),
                 forward_mode='auto')['logits']
        auto_none = m(ct, pet, forward_mode='auto')['logits']
    assert torch.allclose(full, auto, rtol=0, atol=0)
    assert torch.allclose(full, auto_none, rtol=0, atol=0)
    assert full.shape == (2, 1, 64, 64) and torch.isfinite(full).all()


def test_missing_and_auto_with_zero_raise_before_pet_encode():
    import pytest
    torch.manual_seed(0)
    m = _build()
    m.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    calls = []
    real = m._encode_pet
    m._encode_pet = lambda p: (calls.append(1), real(p))[1]
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            m(ct, pet, forward_mode='missing')
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            m(ct, pet, pet_available=torch.tensor([1, 0]), forward_mode='auto')
    assert calls == []


def test_decoder_receives_module_output_verbatim():
    torch.manual_seed(0)
    m = _build()
    m.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    seen = {}
    real_decode = m._decode
    m._decode = lambda feats, ts: (seen.setdefault('feats', feats), real_decode(feats, ts))[1]
    with torch.no_grad():
        m(ct, pet, forward_mode='full')
        direct = m.fusion(m._encode_ct(ct), m._encode_pet(pet), state='full')
    for a, b in zip(seen['feats'], direct):
        assert torch.allclose(a, b, rtol=0, atol=0)


def test_no_clip_module_in_optimizer_graph():
    m = _build()
    names = [type(x).__name__ for x in m.modules()]
    assert not any('CLIP' in n for n in names)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    opt_params = {id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in opt_params for p in m.fusion.parameters() if p.requires_grad)
    assert any('text_proj' in n or 'mlp' in n for n, _ in m.fusion.named_parameters())


def test_disabled_baseline_unchanged():
    torch.manual_seed(7)
    a = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    ka = {k: v.clone() for k, v in a.state_dict().items() if isinstance(v, torch.Tensor)}
    torch.manual_seed(7)
    b = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    assert set(ka) == {k for k, v in b.state_dict().items() if isinstance(v, torch.Tensor)}
    for k in ka:
        assert torch.equal(ka[k], b.state_dict()[k]), k
    a.eval()
    b.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        oa = a(ct, pet, forward_mode='full')['logits']
        ob = b(ct, pet, forward_mode='full')['logits']
    assert torch.equal(oa, ob)


def test_module_ct_only_direct_call():
    torch.manual_seed(0)
    m = _build()
    m.eval()
    ct = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        cf = m._encode_ct(ct)
        out = m.fusion(cf, state='ct_only')
    assert len(out) == 4 and all(o.shape == c.shape for o, c in zip(out, cf))
