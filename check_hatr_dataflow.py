import torch
import torch.nn.functional as F

from models.dual_decoder_hatr_task_residual import DualDecoderHATRTaskResidual


def max_abs(a, b):
    return (a - b).abs().max().item()


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
        orig = model.full_decoder(fused, ct.shape[-2:])['logits']
        manual, states = model._decode_with_states(model.full_decoder, fused, ct.shape[-2:])
        ct_orig = model.full_decoder(ct_feats, ct.shape[-2:])['logits']
        ct_manual, ct_states = model._decode_with_states(model.full_decoder, ct_feats, ct.shape[-2:])
        missing_orig = model._forward_missing(ct, ct.shape[-2:])['logits']
        preds, hidden = model.hatr_recovery([c.detach() for c in ct_feats])
        missing_manual, miss_states, miss_base = model._decode_missing_with_residuals(model.missing_decoder, ct_feats, preds, ct.shape[-2:])
    bn_before = {n: (b.clone() if torch.is_tensor(b) else b) for n, b in [(k, v) for k, v in model.full_decoder.named_buffers() if 'running_' in k or 'num_batches_tracked' in k]}
    teacher_out, teacher_states, cf_out, cf_states = model._build_hatr_observation(ct_feats, fused, ct.shape[-2:])
    bn_after = {n: (b.clone() if torch.is_tensor(b) else b) for n, b in [(k, v) for k, v in model.full_decoder.named_buffers() if 'running_' in k or 'num_batches_tracked' in k]}
    p_f = torch.sigmoid(teacher_out['logits'])
    p_c = torch.sigmoid(cf_out['logits'])
    y = torch.zeros_like(p_f)
    e_f = (p_f - y).pow(2)
    e_c = (p_c - y).pow(2)
    adv = F.relu(e_c - e_f) / (e_c + e_f + 1e-6)
    out = {
        'decoder_equivalence_error_full': max_abs(orig, manual['logits']),
        'decoder_equivalence_error_ct': max_abs(ct_orig, ct_manual['logits']),
        'missing_fallback_error': max_abs(missing_orig, missing_manual['logits']),
        'bn_running_mean_change': 0.0,
        'bn_running_var_change': 0.0,
        'bn_num_batches_change': 0.0,
        'advantage_min': adv.min().item(),
        'advantage_max': adv.max().item(),
        'residual_init_max': max(x.abs().max().item() for x in preds),
        'adapter_init_max': max(p.abs().max().item() for p in [model.correction_adapter1.proj.weight, model.correction_adapter2.proj.weight, model.correction_adapter3.proj.weight, model.correction_adapter4.proj.weight]),
    }
    for name in bn_before:
        out['bn_running_mean_change'] = max(out['bn_running_mean_change'], max_abs(bn_before[name].float(), bn_after[name].float())) if 'running_mean' in name else out['bn_running_mean_change']
        out['bn_running_var_change'] = max(out['bn_running_var_change'], max_abs(bn_before[name].float(), bn_after[name].float())) if 'running_var' in name else out['bn_running_var_change']
        out['bn_num_batches_change'] = max(out['bn_num_batches_change'], max_abs(bn_before[name].float(), bn_after[name].float())) if 'num_batches_tracked' in name else out['bn_num_batches_change']
    print(out)
    assert out['decoder_equivalence_error_full'] < 1e-6
    assert out['decoder_equivalence_error_ct'] < 1e-6
    assert out['missing_fallback_error'] < 1e-6
    assert out['bn_running_mean_change'] == 0.0
    assert out['bn_running_var_change'] == 0.0
    assert out['bn_num_batches_change'] == 0.0
    assert out['advantage_min'] >= 0
    assert out['advantage_max'] <= 1 + 1e-6
    assert out['residual_init_max'] == 0.0
    assert out['adapter_init_max'] == 0.0


if __name__ == '__main__':
    main()
