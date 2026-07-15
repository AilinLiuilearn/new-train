import torch

from models.dual_decoder_hatr_task_residual import DualDecoderHATRTaskResidual


def main():
    torch.manual_seed(0)
    model = DualDecoderHATRTaskResidual(use_deep_supervision=False)
    model.eval()
    ct = torch.randn(2, 3, 128, 128)
    pet = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        ct_feats = model._encode_ct(ct)
        pet_feats = model._encode_pet(pet)
        fused = [c + p for c, p in zip(ct_feats, pet_feats)]
        full_out, full_states = model._decode_with_states(model.full_decoder, fused, ct.shape[-2:])
        ct_cf_out, ct_cf_states = model._decode_with_states(model.full_decoder, [c.detach() for c in ct_feats], ct.shape[-2:])
        pred_residuals = model.hatr_recovery([c.detach() for c in ct_feats])
        missing_out, missing_states, base_states = model._decode_missing_with_residuals(model.missing_decoder, ct_feats, pred_residuals, ct.shape[-2:])
    assert torch.isfinite(full_out['logits']).all()
    assert torch.isfinite(missing_out['logits']).all()
    expected_stage_channels = list(model.decoder_channels[::-1])
    for state, ch in zip(full_states, expected_stage_channels):
        assert state.shape[1] == ch
    for f, c in zip(full_states, ct_cf_states):
        assert f.shape == c.shape
    for pred, st in zip(pred_residuals, full_states):
        assert pred.shape == st.shape
    p_f = torch.sigmoid(full_out['logits'])
    p_c = torch.sigmoid(ct_cf_out['logits'])
    y = torch.zeros_like(p_f)
    e_f = (p_f - y).pow(2)
    e_c = (p_c - y).pow(2)
    advantage = torch.relu(e_c - e_f) / (e_c + e_f + 1e-6)
    assert advantage.min().item() >= 0
    assert advantage.max().item() <= 1 + 1e-6
    for idx, (f, c, pred) in enumerate(zip(full_states, ct_cf_states, pred_residuals), start=1):
        adv_s = torch.nn.functional.interpolate(advantage, size=f.shape[-2:], mode='bilinear', align_corners=False)
        target_s = adv_s * (f.detach() - c.detach())
        assert target_s.shape == pred.shape
        adapter = getattr(model, f'correction_adapter{5-idx}')
        assert torch.allclose(adapter.proj.weight, torch.zeros_like(adapter.proj.weight))
    missing_base, _, _ = model._decode_missing_with_residuals(model.missing_decoder, ct_feats, [torch.zeros_like(x) for x in pred_residuals], ct.shape[-2:])
    baseline = model._forward_missing(ct, ct.shape[-2:])
    max_err = (missing_base['logits'] - baseline['logits']).abs().max().item()
    print({'full_finite': True, 'missing_finite': True, 'max_missing_fallback_error': max_err, 'advantage_min': advantage.min().item(), 'advantage_max': advantage.max().item()})
    assert max_err < 1e-6


if __name__ == '__main__':
    main()
