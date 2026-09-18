"""Detail-frequency Module-2 smoke: bypass/FFT/switches/grads/ckpt/integration."""
import io
import math
import sys

import torch

sys.path.insert(0, '.')
from models.petct_state_text_detail_frequency import (
    StateTextDetailFrequencyFusion, band_masks,
)


def _fusion(**kw):
    args = dict(channels=(8, 16, 24, 32), ct_text_feature=torch.randn(1, 12),
                pet_text_feature=torch.randn(1, 12))
    args.update(kw)
    return StateTextDetailFrequencyFusion(**args)


def _feats(seed=42, channels=(8, 16, 24, 32), sizes=(8, 4, 3, 2)):
    torch.manual_seed(seed)
    ct = [torch.randn(2, c, s, s) for c, s in zip(channels, sizes)]
    pet = [torch.randn_like(x) for x in ct]
    return ct, pet


def test_bypass_init_cold_and_spy():
    m = _fusion()
    m.float()
    ct, pet = _feats()
    out, _ = m(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert [y.shape for y in out] == [x.shape for x in ct]
    assert [y.dtype for y in out] == [x.dtype for x in ct]
    assert all(torch.isfinite(y).all() for y in out)
    for y, c, p in zip(out, ct, pet):
        torch.testing.assert_close(y, (c + p).float(), rtol=1e-4, atol=1e-5)
    out = m(ct, pet, 0, pet_valid=False)
    assert all(torch.equal(y, c) for y, c in zip(out, ct))
    with torch.no_grad():
        for b in m.scales:
            b.out_proj.bias.fill_(0.7)
    out = m(ct, pet, 0, pet_valid=False)
    assert all(torch.equal(y, c) for y, c in zip(out, ct))
    # Spy: invalid rows must not run text/conv/FFT branches.
    calls = []
    real_block = m.scales[0].forward
    def spy(*a, **k):
        calls.append(1)
        return real_block(*a, **k)
    m.scales[0].forward = spy
    m(ct, pet, 0, pet_valid=False)
    assert calls == []
    print('[F1] init C+P, cold C (bias-proof), invalid rows bypass: PASS')


def test_fft_correctness_and_switches():
    torch.manual_seed(0)
    block = _fusion(channels=(4,)).scales[0]
    for shape in [(3, 5), (4, 4), (5, 7), (6, 6)]:
        h, w = shape
        x = torch.randn(2, 4, h, w)
        spec = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
        mask = band_masks(h, w, 0.15, x.device)
        assert tuple(mask.shape) == (1, 1, h, w // 2 + 1)
        low = torch.fft.irfft2(spec * mask, s=(h, w), dim=(-2, -1), norm='ortho')
        high = x.float() - low
        torch.testing.assert_close(low + high, x.float(), rtol=1e-5, atol=1e-6)
    const = torch.ones(1, 2, 8, 8)
    spec = torch.fft.rfft2(const, dim=(-2, -1), norm='ortho')
    low = torch.fft.irfft2(spec * band_masks(8, 8, 0.15, const.device),
                            s=(8, 8), dim=(-2, -1), norm='ortho')
    assert float((const - low).abs().max()) < 1e-5
    yy, xx = torch.meshgrid(torch.arange(8).float(), torch.arange(8).float(), indexing='ij')
    smooth = torch.sin(2 * math.pi * 0.05 * xx).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    rough = torch.sin(2 * math.pi * 0.4 * (xx + yy)).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    def low_ratio(x):
        spec = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        low = torch.fft.irfft2(spec * band_masks(8, 8, 0.15, x.device),
                                s=(8, 8), dim=(-2, -1), norm='ortho')
        return float((low.pow(2).mean() / x.pow(2).mean()).item())
    assert low_ratio(smooth) > low_ratio(rough)
    # Text-off: no text IO, no FFT, P_band=P.
    m = _fusion(use_text=False)
    ct, pet = _feats()
    fft_calls = []
    real_fft = torch.fft.rfft2
    def spy_fft(*a, **k):
        fft_calls.append(1)
        return real_fft(*a, **k)
    torch.fft.rfft2 = spy_fft
    try:
        out = m(ct, pet, (1, 0), pet_valid=True)
    finally:
        torch.fft.rfft2 = real_fft
    assert fft_calls == []
    assert all(torch.isfinite(y).all() for y in out)
    # State-off: prompt perturbation must not change PET band gates.
    m2 = _fusion(use_state=False)
    with torch.no_grad():
        for b in m2.scales:
            b.missing_prompt.fill_(3.0)
    _, i1 = m2(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    with torch.no_grad():
        for b in m2.scales:
            b.missing_prompt.zero_()
    _, i2 = m2(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    assert all(torch.equal(a['a_low'], b['a_low']) and torch.equal(a['a_high'], b['a_high'])
               for a, b in zip(i1, i2))
    # State-on: prompt affects Missing-row gates only, not Full rows or CT gate.
    m3 = _fusion(use_state=True)
    with torch.no_grad():
        for b in m3.scales:
            b.missing_prompt.fill_(3.0)
    _, j1 = m3(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    with torch.no_grad():
        for b in m3.scales:
            b.missing_prompt.zero_()
    _, j2 = m3(ct, pet, (1, 0), pet_valid=True, return_diagnostics=True)
    for a, b in zip(j1, j2):
        rows = a['valid_rows'].tolist()
        full_idx = [k for k, r in enumerate(rows) if r == 0]
        miss_idx = [k for k, r in enumerate(rows) if r == 1]
        assert torch.equal(a['a_low'][full_idx], b['a_low'][full_idx])
        assert torch.equal(a['a_ct'], b['a_ct'])
        assert not torch.equal(a['a_low'][miss_idx], b['a_low'][miss_idx])
    print('[F2] FFT restore/constant/freq-selective + text/state switches: PASS')


def test_short_grads_and_frozen():
    m = _fusion()
    m.train()
    ct, pet = _feats(seed=7)
    pet_in = [p.clone().requires_grad_(True) for p in pet]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        for p in pet_in:
            if p.grad is not None:
                p.grad = None
        loss = sum(y.square().mean() for y in m(ct, pet_in, (1, 0), pet_valid=True))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert any(float(b.out_proj.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.detail_dw.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.pet_pw.weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.gate_low[-1].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.gate_high[-1].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.proj_pet[0].weight.grad.abs().sum()) > 0 for b in m.scales)
    assert any(float(b.missing_prompt.grad.abs().sum()) > 0 for b in m.scales)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pet_in)
    frozen = _fusion(use_text=False)
    assert all(not p.requires_grad for b in frozen.scales for p in b.text_params())
    print('[F3] 3 updates reach detail/pet_pw/gates/text/state + PET-grad via FFT: PASS')


def test_checkpoint_sigma_and_buffers():
    m = _fusion()
    o_before = m(*_feats(), (1, 0), pet_valid=True)
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
    o_after = m2(*_feats(), (1, 0), pet_valid=True)
    assert all(torch.equal(a, b) for a, b in zip(o_before, o_after))
    assert torch.equal(m2.ct_text_feature, m.ct_text_feature)
    try:
        m2.set_extra_state({'version': 5, 'architecture': 'state_text_detail_region_v1',
                            'text_metadata': {}, 'config': {}})
    except ValueError as e:
        assert 'detail-region' in str(e)
    else:
        raise AssertionError('old metadata must fail')
    try:
        _fusion(frequency_sigma=0.3).set_extra_state(m.get_extra_state())
    except ValueError as e:
        assert 'sigma' in str(e).lower() or 'config' in str(e).lower()
    else:
        raise AssertionError('sigma mismatch must fail')
    # Builder restores dual buffers without text IO.
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
    ckpt = {k: v for k, v in model.state_dict().items() if k.startswith('fusion.')}
    cfg2 = SegMDTConfig(args=dict(base))
    with mock.patch.object(B, 'create_feature_backbone',
                           side_effect=lambda backbone, in_channels=3: B.FallbackFeatureBackbone(in_channels=in_channels)), \
         mock.patch.object(B, 'encode_text_pair',
                           side_effect=AssertionError('text encode called')), \
         mock.patch('torch.load', side_effect=AssertionError('cache read called')) if False else mock.patch.object(
               B, '_validate_module2_cache', side_effect=AssertionError('cache read called')):
        # checkpoint path must not reach cache validation; encode is bypassed
        # because buffers come from the checkpoint state.
        model2 = B.build_mdt_seg_teacher(cfg2, module2_checkpoint_state=ckpt)['model']
    torch.testing.assert_close(model2.fusion.ct_text_feature, model.fusion.ct_text_feature)
    print('[F4] strict roundtrip + old/sigma rejection + ckpt buffers w/o IO: PASS')


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
        bundle = B.build_mdt_seg_teacher(cfg)
    from tasks.mdt_seg import MDTSegTeacher
    task = MDTSegTeacher(bundle, cfg)
    task.model.train()
    device = task.device
    ct = torch.randn(2, 1, 64, 64, device=device)
    pet = torch.randn(2, 1, 64, 64, device=device)
    mask = torch.zeros(2, 1, 64, 64, device=device)
    mask[:, :, 16:48, 16:48] = 1
    batch = {'ct': ct, 'pet': pet, 'mask': mask}
    avail = torch.tensor([1, 0])
    cold = task.train_step_mixed(batch, avail)
    cold_loss = cold[0]
    cold_loss.backward()
    assert math.isfinite(cold_loss.item())
    task.model.zero_grad(set_to_none=True)
    task.model.module1.collect_candidates(task.model._encode_ct(ct), task.model._encode_pet(pet), mask)
    task.model.module1.collect_candidates(task.model._encode_ct(ct), task.model._encode_pet(pet), mask)
    assert task.model.module1.finalize_epoch(epoch=1)['status'] == 'bank_updated'
    ready = task.train_step_mixed(batch, avail)
    ready_loss = ready[0]
    ready_loss.backward()
    assert math.isfinite(ready_loss.item())
    task.model.eval()
    with torch.no_grad(), mock.patch.object(task.model, '_encode_pet',
                                            side_effect=AssertionError('real PET encoder called')), \
         mock.patch.object(task.model.module1, 'collect_candidates',
                           side_effect=AssertionError('collect called')), \
         mock.patch.object(task.model.module1, 'retrieve_pet_prior',
                           wraps=task.model.module1.retrieve_pet_prior) as spy_ret:
        missing = task.model(torch.randn(1, 1, 64, 64, device=device), pet=None, forward_mode='missing')
        assert torch.isfinite(missing['logits']).all()
        assert spy_ret.call_count >= 1
    params = [p for p in task.model.fusion.parameters() if p.requires_grad]
    opt_ids = [id(p) for p in task.optimizer.param_groups[0]['params']]
    assert len(params) > 0 and all(opt_ids.count(id(p)) == 1 for p in params)
    print('[F5] mixed cold/ready backward + missing eval (no PET enc/collect): PASS')


def test_amp_default_shapes():
    if not torch.cuda.is_available():
        print('[F6] AMP skipped (no CUDA): NOT RUN')
        return
    m = _fusion(channels=(64, 128, 320, 512)).cuda()
    ct = [torch.randn(1, c, s, s, device='cuda') for c, s in zip((64, 128, 320, 512), (128, 64, 32, 16))]
    pet = [torch.randn_like(x) for x in ct]
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(ct, pet, 1, pet_valid=True)
        loss = sum(y.float().square().mean() for y in out)
    loss.backward()
    assert all(torch.isfinite(y).all() and torch.isfinite(p.grad.float()).all()
               for y, p in zip(out, m.parameters()) if p.grad is not None)
    print('[F6] CUDA AMP fwd/bwd finite: PASS')


def main():
    import math as _math
    globals()['math'] = __import__('math')
    tests = [test_bypass_init_cold_and_spy, test_fft_correctness_and_switches,
             test_short_grads_and_frozen, test_checkpoint_sigma_and_buffers,
             test_short_model_integration, test_amp_default_shapes]
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
    print(f'\n[RESULT] all {len(tests)} detail-frequency smoke tests passed')


if __name__ == '__main__':
    import math
    main()
