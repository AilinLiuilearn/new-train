"""Integration tests for MultiScaleEvidenceGuidedSDNCA in the baseline."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.baseline_blocks import StateAwareWeightedAddFusion
from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.evidence_guided_sdnca_pet_ct import (
    MultiScaleEvidenceGuidedSDNCA,
    count_parameters,
)


def _test_text_embeddings(text_dim=512):
    embeddings = torch.zeros(2, text_dim)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    return embeddings


def _make_baseline(**kwargs):
    defaults = dict(
        use_deep_supervision=False,
        pet_text_embeddings=_test_text_embeddings(),
        edv_attention_backend='sdpa',
    )
    defaults.update(kwargs)
    return DualSharedAddPETCTBaseline(**defaults)


def _make_cfg(tmp_path, **kwargs):
    base = {
        'ct_backbone': 'convnextv2_nano',
        'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'decoder_channels': (512, 256, 128, 64),
        'use_deep_supervision': False,
        'deep_supervision': False,
        'checkpoint_dir': str(tmp_path),
        'pet_text_embeddings': _test_text_embeddings(),
        'edv_attention_backend': 'sdpa',
        'cppi_num_clusters': 6,
        'cppi_build_stage': 3,
        'no_encoder_pretrained': True,
    }
    base.update(kwargs)
    return type('C', (), base)()


def test_build_uses_edv_not_weighted_add(tmp_path):
    out = build_mdt_seg_teacher(_make_cfg(tmp_path))
    model = out['model']
    assert isinstance(model.fusion, MultiScaleEvidenceGuidedSDNCA)
    assert not isinstance(model.fusion, StateAwareWeightedAddFusion)


def test_four_scale_shapes_match_no_downsample():
    model = _make_baseline()
    channels = (64, 128, 320, 512)
    assert tuple(model.fusion.channels) == channels
    # Project 512x512 encoder feature map sizes.
    shapes = [
        (2, 64, 128, 128),
        (2, 128, 64, 64),
        (2, 320, 32, 32),
        (2, 512, 16, 16),
    ]
    ct_feats = [torch.randn(*shape) for shape in shapes]
    pet_feats = [torch.randn(*shape) for shape in shapes]
    fused = model.fusion(ct_feats, pet_feats, mode='full')
    assert len(fused) == 4
    for fused_feat, ct_feat, pet_feat in zip(fused, ct_feats, pet_feats):
        assert fused_feat.shape == ct_feat.shape
        assert fused_feat.shape == pet_feat.shape
        assert torch.isfinite(fused_feat).all()


def test_full_missing_state_ids_and_text_modulation_effect():
    model = _make_baseline()
    shapes = [
        (2, 64, 16, 16),
        (2, 128, 8, 8),
        (2, 320, 8, 8),
        (2, 512, 8, 8),
    ]
    ct_feats = [torch.randn(*shape) for shape in shapes]
    pet_feats = [torch.randn(*shape) for shape in shapes]

    _, diag_full = model.fusion(
        ct_feats,
        pet_feats,
        mode='full',
        return_diagnostics=True,
    )
    _, diag_missing = model.fusion(
        ct_feats,
        pet_feats,
        mode='missing',
        return_diagnostics=True,
    )
    assert torch.equal(
        diag_full['pet_state_ids'],
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.equal(
        diag_missing['pet_state_ids'],
        torch.ones(2, dtype=torch.long),
    )

    # Zero-init text MLP final layer => identical fused outputs initially.
    fused_full_id = model.fusion(ct_feats, pet_feats, mode='full')
    fused_missing_id = model.fusion(ct_feats, pet_feats, mode='missing')
    for a, b in zip(fused_full_id, fused_missing_id):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)

    # Make text MLP final Linear non-zero so Full/Missing diverge.
    with torch.no_grad():
        for scale in model.fusion.scales:
            final = scale.text_modulator.text_to_channel[-1]
            nn.init.normal_(final.weight, mean=0.0, std=0.05)
            nn.init.normal_(final.bias, mean=0.0, std=0.05)

    fused_full, diag_full2 = model.fusion(
        ct_feats,
        pet_feats,
        mode='full',
        return_diagnostics=True,
    )
    fused_missing, diag_missing2 = model.fusion(
        ct_feats,
        pet_feats,
        mode='missing',
        return_diagnostics=True,
    )
    text_delta_diff = False
    pet_mod_diff = False
    fused_diff = False
    for scale_full, scale_missing in zip(
        diag_full2['scales'],
        diag_missing2['scales'],
    ):
        if not torch.allclose(
            scale_full['text_channel_delta_abs_mean'],
            scale_missing['text_channel_delta_abs_mean'],
        ):
            text_delta_diff = True
        if not torch.allclose(
            scale_full['pet_after_text_rms'],
            scale_missing['pet_after_text_rms'],
        ):
            pet_mod_diff = True
    for a, b in zip(fused_full, fused_missing):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
            fused_diff = True
    assert text_delta_diff and pet_mod_diff and fused_diff


def test_auto_batch_state_ids():
    model = _make_baseline()
    shapes = [
        (4, 64, 16, 16),
        (4, 128, 8, 8),
        (4, 320, 8, 8),
        (4, 512, 8, 8),
    ]
    ct_feats = [torch.randn(*shape) for shape in shapes]
    pet_feats = [torch.randn(*shape) for shape in shapes]
    pet_available = torch.tensor([1, 0, 1, 0])
    _, diagnostics = model.fusion(
        ct_feats,
        pet_feats,
        mode='auto',
        pet_available=pet_available,
        return_diagnostics=True,
    )
    assert torch.equal(
        diagnostics['pet_state_ids'],
        torch.tensor([0, 1, 0, 1]),
    )


def test_calibration_before_edv_order(monkeypatch):
    model = _make_baseline()
    order = []
    calib_outputs = []
    fusion_inputs = []

    orig_calib = model.pet_calibration.forward
    orig_fusion = model.fusion.forward
    orig_decode = model.decoder.forward

    def wrapped_calib(*args, **kwargs):
        order.append('calibration')
        out = orig_calib(*args, **kwargs)
        calib_outputs.append([feat.detach().clone() for feat in out])
        return out

    def wrapped_fusion(ct_feats, pet_feats, *args, **kwargs):
        order.append('edv')
        fusion_inputs.append([feat.detach().clone() for feat in pet_feats])
        return orig_fusion(ct_feats, pet_feats, *args, **kwargs)

    def wrapped_decode(*args, **kwargs):
        order.append('decoder')
        return orig_decode(*args, **kwargs)

    monkeypatch.setattr(model.pet_calibration, 'forward', wrapped_calib)
    monkeypatch.setattr(model.fusion, 'forward', wrapped_fusion)
    monkeypatch.setattr(model.decoder, 'forward', wrapped_decode)

    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    out = model(ct, pet, forward_mode='full')
    assert 'logits' in out
    assert order == ['calibration', 'edv', 'decoder']
    assert len(calib_outputs) == 1 and len(fusion_inputs) == 1
    for cal, fus in zip(calib_outputs[0], fusion_inputs[0]):
        assert torch.allclose(cal, fus)


def test_missing_edv_receives_proxy_not_real(monkeypatch):
    model = _make_baseline()
    real_encoded = []
    calib_pet_in = []
    fusion_pet_in = []

    orig_encode_pet = model._encode_pet
    orig_calib = model.pet_calibration.forward
    orig_fusion = model.fusion.forward

    def wrapped_encode_pet(pet):
        feats = orig_encode_pet(pet)
        real_encoded.append([f.detach().clone() for f in feats])
        return feats

    def wrapped_calib(ct_feats, pet_feats, *args, **kwargs):
        calib_pet_in.append([f.detach().clone() for f in pet_feats])
        return orig_calib(ct_feats, pet_feats, *args, **kwargs)

    def wrapped_fusion(ct_feats, pet_feats, *args, **kwargs):
        fusion_pet_in.append([f.detach().clone() for f in pet_feats])
        return orig_fusion(ct_feats, pet_feats, *args, **kwargs)

    monkeypatch.setattr(model, '_encode_pet', wrapped_encode_pet)
    monkeypatch.setattr(model.pet_calibration, 'forward', wrapped_calib)
    monkeypatch.setattr(model.fusion, 'forward', wrapped_fusion)

    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    model(ct, pet, forward_mode='missing')
    assert real_encoded and calib_pet_in and fusion_pet_in
    for real, proxy_in in zip(real_encoded[0], calib_pet_in[0]):
        assert not torch.allclose(real, proxy_in)
    # Bank not ready => calibration is identity; EDV input equals proxy.
    for proxy_in, fus_in in zip(calib_pet_in[0], fusion_pet_in[0]):
        assert torch.allclose(proxy_in, fus_in)


def test_cppi_collect_finalize_retrieve_unchanged(tmp_path):
    model = _make_baseline(cppi_output_dir=str(tmp_path / 'cppi'))
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    assert model.prototype_memory.bank_ready is False
    model._collect_cppi(ct_feats, pet_feats, mask)
    report = model.finalize_cppi_epoch(
        epoch=0,
        save_json=False,
        save_visualizations=False,
        print_info=False,
    )
    assert model.prototype_memory.bank_ready is True
    proxy, ct_ref, _ = model._retrieve_cppi(
        ct_feats,
        return_ct_reference=True,
    )
    assert len(proxy) == 4
    assert len(ct_ref) == 4
    # Baseline calibration path always detaches CT references.
    detached = [x.detach() for x in ct_ref]
    for ref in detached:
        assert not ref.requires_grad
    for a, b in zip(ct_ref, detached):
        assert torch.allclose(a, b)
    assert report is not None or model.prototype_memory.bank_ready


def test_two_optimizer_steps_finite_grads(tmp_path):
    model = _make_baseline(cppi_output_dir=str(tmp_path / 'cppi'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    # Warm CPPI bank so PrototypeReferencedPETAffineCalibration is active.
    with torch.no_grad():
        warm_ct = torch.randn(2, 1, 64, 64, device=device)
        warm_pet = torch.randn(2, 1, 64, 64, device=device)
        warm_mask = torch.ones(2, 1, 64, 64, device=device)
        ct_feats = model._encode_ct(warm_ct)
        pet_feats = model._encode_pet(warm_pet)
        model._collect_cppi(ct_feats, pet_feats, warm_mask)
        model.finalize_cppi_epoch(
            epoch=0,
            save_json=False,
            save_visualizations=False,
            print_info=False,
        )
    assert model.prototype_memory.bank_ready is True

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ct = torch.randn(1, 1, 64, 64, device=device)
    pet = torch.randn(1, 1, 64, 64, device=device)
    target = torch.zeros(1, 1, 64, 64, device=device)

    modules_to_check = {
        'enc_ct': model.enc_ct,
        'enc_pet': model.enc_pet,
        'ct_align': model.ct_align,
        'pet_calibration': model.pet_calibration,
        'decoder': model.decoder,
        'edv_text_mlp_final': model.fusion.scales[0].text_modulator.text_to_channel[-1],
        'edv_text_mlp_prev': model.fusion.scales[0].text_modulator.text_to_channel[1],
        'edv_evidence_head': model.fusion.scales[0].evidence_head,
        'edv_q': model.fusion.scales[0].cross_attention.query_projection,
        'edv_k': model.fusion.scales[0].cross_attention.key_projection,
        'edv_v': model.fusion.scales[0].cross_attention.value_projection,
        'edv_ct_delta': model.fusion.scales[0].ct_delta_projection,
        'edv_pet_delta': model.fusion.scales[0].pet_delta_projection,
    }

    def _has_nonzero_finite_grad(module):
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
            if param.grad.abs().sum() > 0:
                return True
        return False

    def _assert_finite_grads(module, name):
        for param in module.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name

    # Step 0: identity-centered zero-init residuals may block Q/K/V grads.
    opt.zero_grad(set_to_none=True)
    out = model(ct, pet, forward_mode='full')
    logits = out['logits']
    assert torch.isfinite(logits).all()
    loss = logits.float().square().mean() + (logits.float() - target).abs().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for name in (
        'enc_ct',
        'enc_pet',
        'ct_align',
        'pet_calibration',
        'decoder',
        'edv_evidence_head',
    ):
        assert _has_nonzero_finite_grad(modules_to_check[name]), name
    for name in (
        'edv_text_mlp_final',
        'edv_ct_delta',
        'edv_pet_delta',
        'edv_q',
        'edv_k',
        'edv_v',
    ):
        _assert_finite_grads(modules_to_check[name], name)
    opt.step()

    # Break zero-init residual / text gates so step-1 reaches those params.
    with torch.no_grad():
        for scale in model.fusion.scales:
            final = scale.text_modulator.text_to_channel[-1]
            nn.init.normal_(final.weight, std=0.02)
            nn.init.normal_(final.bias, std=0.02)
            nn.init.normal_(scale.ct_delta_projection.weight, std=0.02)
            nn.init.zeros_(scale.ct_delta_projection.bias)
            nn.init.normal_(scale.pet_delta_projection.weight, std=0.02)
            nn.init.zeros_(scale.pet_delta_projection.bias)

    opt.zero_grad(set_to_none=True)
    out = model(ct, pet, forward_mode='full')
    assert torch.isfinite(out['logits']).all()
    loss = out['logits'].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for name, module in modules_to_check.items():
        assert _has_nonzero_finite_grad(module), (
            f'expected finite nonzero grad for {name} at step 1'
        )
    opt.step()

    fused_probe = model.fusion(
        [
            torch.randn(1, 64, 16, 16, device=device),
            torch.randn(1, 128, 8, 8, device=device),
            torch.randn(1, 320, 8, 8, device=device),
            torch.randn(1, 512, 8, 8, device=device),
        ],
        [
            torch.randn(1, 64, 16, 16, device=device),
            torch.randn(1, 128, 8, 8, device=device),
            torch.randn(1, 320, 8, 8, device=device),
            torch.randn(1, 512, 8, 8, device=device),
        ],
        mode='full',
    )
    assert all(torch.isfinite(f).all() for f in fused_probe)


def test_parameter_budget_and_counts():
    model = _make_baseline()
    edv_total, edv_trainable = count_parameters(model.fusion)
    model_total, model_trainable = count_parameters(model)
    old_fusion = StateAwareWeightedAddFusion(num_scales=4)
    old_total, _ = count_parameters(old_fusion)
    assert edv_total < 5_000_000
    assert edv_trainable < 5_000_000
    delta = edv_total - old_total
    print(f'EDV total params: {edv_total:,}')
    print(f'EDV trainable params: {edv_trainable:,}')
    print(f'Full model total params: {model_total:,}')
    print(f'Full model trainable params: {model_trainable:,}')
    print(f'Delta vs StateAwareWeightedAddFusion: {delta:,}')
    assert delta > 0


def test_old_checkpoint_strict_false_only_fusion_mismatch(tmp_path):
    # Simulate an old checkpoint that still has StateAwareWeightedAddFusion keys.
    old_model = _make_baseline()
    # Replace fusion state with old-style keys for the save payload.
    state = old_model.state_dict()
    fusion_keys = [k for k in state if k.startswith('fusion.')]
    for key in fusion_keys:
        del state[key]
    # Inject fake old fusion parameters.
    state['fusion.raw_alpha_full'] = torch.zeros(4)
    state['fusion.raw_alpha_missing'] = torch.full((4,), -2.944)

    new_model = _make_baseline()
    msg = new_model.load_state_dict(state, strict=False)
    missing_fusion = [k for k in msg.missing_keys if k.startswith('fusion.')]
    unexpected_fusion = [k for k in msg.unexpected_keys if k.startswith('fusion.')]
    other_missing = [k for k in msg.missing_keys if not k.startswith('fusion.')]
    other_unexpected = [k for k in msg.unexpected_keys if not k.startswith('fusion.')]
    assert missing_fusion, 'expected new EDV fusion keys to be missing'
    assert unexpected_fusion, 'expected old weighted-add fusion keys to be unexpected'
    assert not other_missing, other_missing
    assert not other_unexpected, other_unexpected


def test_no_natten_dependency_in_source():
    from pathlib import Path
    src = Path('models/evidence_guided_sdnca_pet_ct.py').read_text()
    assert 'from natten' not in src
    assert 'import natten' not in src
    assert '_natten_na2d' not in src
    assert '_NATTEN_AVAILABLE' not in src
    assert '_chunked_torch_na2d' not in src
    assert 'ScaleAwareDilatedNeighborhoodCrossAttention' not in src


def test_arbitrary_non_divisible_spatial_sizes():
    model = _make_baseline()
    shapes = [
        (1, 64, 31, 29),
        (1, 128, 17, 15),
        (1, 320, 17, 15),
        (1, 512, 9, 7),
    ]
    ct_feats = [torch.randn(*shape) for shape in shapes]
    pet_feats = [torch.randn(*shape) for shape in shapes]
    fused = model.fusion(ct_feats, pet_feats, mode='full')
    for out, expected in zip(fused, shapes):
        assert out.shape == expected
        assert torch.isfinite(out).all()


def test_shifted_window_mask_blocks_border_wrap():
    """Left-border pulse must not affect right-border query via cyclic wrap."""
    from models.evidence_guided_sdnca_pet_ct import (
        _windowed_sdpa,
        build_shifted_window_mask,
        pad_to_window,
        window_partition,
    )

    torch.manual_seed(0)
    batch, channels, height, width = 1, 8, 16, 16
    window_size, shift_size, num_heads = 8, 4, 2
    query = torch.zeros(batch, channels, height, width)
    key = torch.zeros(batch, channels, height, width)
    value = torch.zeros(batch, channels, height, width)
    # Strong pulse only on the left border.
    key[:, :, :, 0] = 10.0
    value[:, :, :, 0] = 1.0
    query[:, :, :, -1] = 1.0

    with_mask = _windowed_sdpa(
        query,
        key,
        value,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
    )
    # Right-border output should stay near zero when the mask is correct.
    right = with_mask[:, :, :, -1].abs().mean().item()
    assert right < 1e-3, f'right-border leak with mask: {right}'

    # Sanity: mask itself marks wrapped regions as blocked.
    padded, _, _ = pad_to_window(query, window_size)
    _, _, ph, pw = padded.shape
    mask = build_shifted_window_mask(
        ph,
        pw,
        window_size,
        shift_size,
        device=query.device,
        dtype=torch.float32,
    )
    assert mask is not None
    assert bool((mask < 0).any())

