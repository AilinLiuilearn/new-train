"""Detail-region Module-2 smoke: init/bypass/switches/updates/ckpt/integration (synthetic)."""
import io
import sys

import torch

sys.path.insert(0, '.')
from models.petct_state_text_detail_region import StateTextDetailRegionFusion


def _fusion(**kw):
    args = dict(channels=(8, 16, 24, 32), ct_text_feature=torch.randn(1, 12),
                pet_text_feature=torch.randn(1, 12))
    args.update(kw)
    return StateTextDetailRegionFusion(**args)


def _feats(seed=42, channels=(8, 16, 24, 32), sizes=(8, 4, 3, 2)):
    torch.manual_seed(seed)
    ct = [torch.randn(2, c, s, s) for c, s in zip(channels, sizes)]
    pet = [torch.randn_like(x) for x in ct]
    return ct, pet


def test_init_addition_and_cold_bypass():
    m = _fusion()
    m.float()
    ct, pet = _feats()
    out, infos = m(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert [y.shape for y in out] == [x.shape for x in ct]
    assert all(torch.isfinite(y).all() for y in out)
    for y, c, p in zip(out, ct, pet):
        torch.testing.assert_close(y, (c + p).float(), rtol=1e-4, atol=1e-5)
    out = m(ct, pet, 0, pet_valid=False)
    assert all(torch.equal(y, c) for y, c in zip(out, ct))
    # Cold bypass must hold even with nonzero output bias.
    with torch.no_grad():
        for b in m.scales:
            b.out_proj.bias.fill_(0.7)
    out = m(ct, pet, 0, pet_valid=False)
    assert all(torch.equal(y, c) for y, c in zip(out, ct))
    print('[D1] init C+P, cold rows C (bias-proof): PASS')


def test_text_state_switches():
    m = _fusion(use_text=False)
    ct, pet = _feats()
    out = m(ct, pet, (1, 0), pet_valid=True)
    assert all(torch.isfinite(y).all() for y in out)
    # state-off: perturbing missing_prompt must not change PET gate (same instance).
    m2 = _fusion(use_state=False)
    with torch.no_grad():
        for b in m2.scales:
            b.missing_prompt.fill_(3.0)
    _, info_prompt = m2(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    with torch.no_grad():
        for b in m2.scales:
            b.missing_prompt.zero_()
    _, info_zero = m2(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert all(torch.equal(a['a_pet'], b['a_pet'])
               for a, b in zip(info_prompt, info_zero))
    # state-on: same perturbation changes Missing-row PET gate (same instance).
    # NOTE: final output is still C+P at init because out_proj is zero-init,
    # so the state effect is checked at A_pet, not at F.
    m3 = _fusion(use_state=True)
    with torch.no_grad():
        for b in m3.scales:
            b.missing_prompt.fill_(3.0)
    _, info3_prompt = m3(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    with torch.no_grad():
        for b in m3.scales:
            b.missing_prompt.zero_()
    _, info3_zero = m3(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert any(not torch.equal(a['a_pet'], b['a_pet'])
               for a, b in zip(info3_prompt, info3_zero))
    print('[D2] text-off works w/o IO, state-off ignores prompt: PASS')


def test_few_updates_reach_branches():
    m = _fusion()
    m.train()
    ct, pet = _feats(seed=7)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = sum(y.square().mean() for y in m(ct, pet, (1, 0), pet_valid=True))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert any(float(b.out_proj.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.detail_dw.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.q_proj.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.proj_pet[0].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.missing_prompt.grad.abs().sum()) > 0 for b in m.scales)
    # H/W < pool_size path
    small = _fusion(channels=(8,), region_pool_size=4)
    c = [torch.randn(2, 8, 3, 2)]
    p = [torch.randn(2, 8, 3, 2)]
    assert all(torch.isfinite(y).all() for y in small(c, p, (1, 0), pet_valid=True))
    print('[D3] 3 updates reach detail/region/text/state + small HW: PASS')


def test_checkpoint_roundtrip_and_reject():
    m = _fusion()
    sd = m.state_dict()
    buf = io.BytesIO()
    torch.save({'model': {'fusion.' + k: v for k, v in sd.items()},
                'extra': m.get_extra_state()}, buf)
    buf.seek(0)
    payload = torch.load(buf, map_location='cpu', weights_only=False)
    m2 = _fusion()
    m2.load_state_dict({k[len('fusion.'):]: v for k, v in payload['model'].items()},
                       strict=True)
    m2.set_extra_state(payload['extra'])
    assert torch.equal(m2.ct_text_feature, m.ct_text_feature)
    assert torch.equal(m2.pet_text_feature, m.pet_text_feature)
    try:
        m2.set_extra_state({'version': 4, 'architecture': 'state_text_competitive_v1',
                            'text_metadata': {}, 'config': {}})
    except ValueError as e:
        assert 'competitive' in str(e)
        print('[D4] strict roundtrip + competitive rejection: PASS')
        return
    raise AssertionError('old metadata must fail')


def test_builder_checkpoint_text_buffers():
    import argparse
    from unittest import mock
    from configs.seg_mdt import SegMDTConfig
    from models import build_mdt_seg as B
    ns = argparse.Namespace()
    for p in (SegMDTConfig.data_parser(), SegMDTConfig.model_parser(), SegMDTConfig.train_parser(),
              SegMDTConfig.logging_parser(), SegMDTConfig.task_specific_parser(), SegMDTConfig.ddp_parser()):
        for a in p._actions:
            if a.dest != 'help' and not hasattr(ns, a.dest):
                ns.__dict__[a.dest] = a.default
    base = dict(vars(ns))
    base.update(ct_pretrained_path=None, pet_pretrained_path=None, pspi_num_clusters=2,
                pspi_prior_scale_enabled=True, pspi_affine_enabled=False,
                pspi_reconstruction_weight=0.0, pspi_proto_contrastive_weight=0.0,
                mixed_precision=False, module2_enabled=True, module2_use_text=True)
    cfg = SegMDTConfig(args=dict(base))
    with mock.patch.object(B, 'create_feature_backbone',
                           side_effect=lambda backbone, in_channels=3: B.FallbackFeatureBackbone(in_channels=in_channels)):
        model = B.build_mdt_seg_teacher(cfg)['model']
    sd = model.state_dict()
    ckpt = {k: v for k, v in sd.items() if k.startswith('fusion.')}
    cfg2 = SegMDTConfig(args=dict(base))
    with mock.patch.object(B, 'create_feature_backbone',
                           side_effect=lambda backbone, in_channels=3: B.FallbackFeatureBackbone(in_channels=in_channels)):
        model2 = B.build_mdt_seg_teacher(cfg2, module2_checkpoint_state=ckpt)['model']
    torch.testing.assert_close(model2.fusion.ct_text_feature, model.fusion.ct_text_feature)
    assert not torch.equal(model2.fusion.ct_text_feature, model2.fusion.pet_text_feature)
    print('[D5] builder restores dual buffers from checkpoint: PASS')


def test_short_model_integration():
    import argparse
    from unittest import mock
    from configs.seg_mdt import SegMDTConfig
    from models import build_mdt_seg as B
    ns = argparse.Namespace()
    for p in (SegMDTConfig.data_parser(), SegMDTConfig.model_parser(), SegMDTConfig.train_parser(),
              SegMDTConfig.logging_parser(), SegMDTConfig.task_specific_parser(), SegMDTConfig.ddp_parser()):
        for a in p._actions:
            if a.dest != 'help' and not hasattr(ns, a.dest):
                ns.__dict__[a.dest] = a.default
    base = dict(vars(ns))
    base.update(ct_pretrained_path=None, pet_pretrained_path=None, pspi_num_clusters=2,
                pspi_affine_enabled=True, pspi_prior_scale_enabled=False,
                pspi_reconstruction_weight=0.05, pspi_proto_contrastive_weight=0.0,
                mixed_precision=False, module2_enabled=True, module2_use_text=False)
    cfg = SegMDTConfig(args=dict(base))
    with mock.patch.object(B, 'create_feature_backbone',
                           side_effect=lambda backbone, in_channels=3: B.FallbackFeatureBackbone(in_channels=in_channels)):
        model = B.build_mdt_seg_teacher(cfg)['model']
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1
    out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]), forward_mode='auto', mask=mask)
    assert torch.isfinite(out['logits']).all()
    model.module1.collect_candidates(model._encode_ct(ct), model._encode_pet(pet), mask)
    model.module1.collect_candidates(model._encode_ct(ct), model._encode_pet(pet), mask)
    assert bool(model.module1.finalize_epoch(epoch=1)['status'] == 'bank_updated')
    out = model(ct, pet=pet, pet_available=torch.tensor([1, 0]), forward_mode='auto', mask=mask)
    assert torch.isfinite(out['logits']).all()
    model.eval()
    with torch.no_grad(), mock.patch.object(model, '_encode_pet',
                                            side_effect=AssertionError('real PET encoder called')):
        missing = model(torch.randn(1, 1, 64, 64), pet=None, forward_mode='missing')
        assert torch.isfinite(missing['logits']).all()
    params = [p for p in model.fusion.parameters() if p.requires_grad]
    ids = {id(p) for n, p in model.named_parameters() if n.startswith('fusion.') and p.requires_grad}
    assert len(ids) == len(params) and len(params) > 0
    from torch import optim
    opt = optim.AdamW(model.parameters(), lr=8e-5)
    assert sum(1 for g in opt.param_groups[0]['params'] if any(p is g for p in params)) == len(params)
    print('[D6] mixed cold/ready + missing eval + optimizer membership: PASS')


def test_amp_default_shapes():
    if not torch.cuda.is_available():
        print('[D7] AMP skipped (no CUDA): NOT RUN')
        return
    m = _fusion(channels=(64, 128, 320, 512)).cuda()
    ct = [torch.randn(1, c, s, s, device='cuda') for c, s in zip((64, 128, 320, 512), (32, 16, 8, 4))]
    pet = [torch.randn_like(x) for x in ct]
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(ct, pet, 1, pet_valid=True)
        loss = sum(y.float().square().mean() for y in out)
    loss.backward()
    assert all(torch.isfinite(y).all() for y in out)
    print('[D7] CUDA AMP fwd/bwd finite: PASS')


def main():
    tests = [test_init_addition_and_cold_bypass, test_text_state_switches,
             test_few_updates_reach_branches, test_checkpoint_roundtrip_and_reject,
             test_builder_checkpoint_text_buffers, test_short_model_integration,
             test_amp_default_shapes]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - harness must report
            failed.append((t.__name__, repr(e)))
            print(f'[FAIL] {t.__name__}: {e!r}')
    if failed:
        print(f'\n[RESULT] {len(tests) - len(failed)}/{len(tests)} passed, {len(failed)} FAILED')
        sys.exit(1)
    print(f'\n[RESULT] all {len(tests)} detail-region smoke tests passed')


if __name__ == '__main__':
    main()
