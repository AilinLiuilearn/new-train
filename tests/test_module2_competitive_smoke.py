"""Competitive Module-2 smoke: shapes/init/updates/cold-ready/ckpt/AMP (synthetic)."""
import io
import sys

import torch

sys.path.insert(0, '.')
from models.petct_state_text_competitive import StateTextCompetitiveFusion


def _fusion(**kw):
    args = dict(channels=(8, 16, 24, 32), ct_text_feature=torch.randn(1, 12),
                pet_text_feature=torch.randn(1, 12))
    args.update(kw)
    return StateTextCompetitiveFusion(**args)


def _feats(batch=2, nondegenerate=False):
    torch.manual_seed(7 if nondegenerate else 42)
    sizes = (8, 4, 2, 1)
    ct = [torch.randn(batch, c, s, s) for c, s in zip((8, 16, 24, 32), sizes)]
    pet = [torch.randn_like(x) for x in ct]
    return ct, pet


def test_shapes_finite_and_weight_sum():
    m = _fusion()
    ct, pet = _feats()
    out, infos = m(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert [y.shape for y in out] == [x.shape for x in ct]
    assert all(torch.isfinite(y).all() for y in out)
    for info, c in zip(infos, (8, 16, 24, 32)):
        assert tuple(info['a_ct'].shape) == (2, c) + tuple(info['a_ct'].shape[2:])
        assert tuple(info['a_pet'].shape) == (2, c) + tuple(info['a_pet'].shape[2:])
        torch.testing.assert_close(info['a_ct'] + info['a_pet'],
                                   torch.full_like(info['a_ct'], 2.0), rtol=1e-4, atol=1e-5)
    print('[S1] shapes finite, A_C+A_P=2: PASS')


def test_init_addition_and_cold_bypass():
    m = _fusion()
    ct, pet = _feats()
    out = m(ct, pet, (1, 0), pet_valid=True)
    assert all(torch.equal(y, c + p) for y, c, p in zip(out, ct, pet))
    out = m(ct, pet, 0, pet_valid=False)
    assert all(torch.equal(y, c) for y, c in zip(out, ct))
    print('[S2] init C+P, invalid rows C: PASS')


def test_few_updates_reach_selectors():
    m = _fusion()
    m.train()
    ct, pet = _feats(nondegenerate=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = sum(y.square().mean() for y in m(ct, pet, (1, 0), pet_valid=True))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert any(float(b.channel_selector[-1].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.spatial_selector.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.proj_pet[0].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.missing_prompt.grad.abs().sum()) > 0 for b in m.scales)
    print('[S3] 5 updates reach selectors + text/state paths: PASS')


def test_checkpoint_roundtrip_uses_buffers():
    m = _fusion()
    ct, pet = _feats()
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
    assert not torch.equal(m2.ct_text_feature, m2.pet_text_feature)
    print('[S4] checkpoint roundtrip reuses dual buffers: PASS')


def test_extra_state_rejects_old():
    m = _fusion()
    try:
        m.set_extra_state({'version': 3, 'architecture': 'state_conditioned_value_v4',
                           'text_metadata': {}, 'config': {}})
    except ValueError as e:
        assert 'v4' in str(e) or 'older' in str(e).lower()
        print('[S5] old v4 metadata rejected: PASS')
        return
    raise AssertionError('old metadata must fail')


def test_amp_small_batch():
    if not torch.cuda.is_available():
        print('[S6] AMP skipped (no CUDA): PASS')
        return
    m = _fusion().cuda()
    ct = [torch.randn(2, c, s, s, device='cuda') for c, s in zip((8, 16, 24, 32), (8, 4, 2, 1))]
    pet = [torch.randn_like(x) for x in ct]
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(ct, pet, (1, 0), pet_valid=True)
        loss = sum(y.float().square().mean() for y in out)
    loss.backward()
    assert all(torch.isfinite(y).all() for y in out)
    print('[S6] CUDA AMP fwd/bwd finite: PASS')


def main():
    tests = [test_shapes_finite_and_weight_sum, test_init_addition_and_cold_bypass,
             test_few_updates_reach_selectors, test_checkpoint_roundtrip_uses_buffers,
             test_extra_state_rejects_old, test_amp_small_batch]
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
    print(f'\n[RESULT] all {len(tests)} competitive smoke tests passed')


if __name__ == '__main__':
    main()
