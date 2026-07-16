import torch

from models.pet_contribution_ct_only import CTOnlyConvNeXtUNet
from models.pet_contribution_full import FullPETCTAddUNet


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(0)
    ct = torch.randn(1, 3, 64, 64)
    pet = torch.randn(1, 3, 64, 64)

    ct_only = CTOnlyConvNeXtUNet(ct_backbone='convnext_tiny', ct_pretrained_path=None)
    full = FullPETCTAddUNet(ct_backbone='convnext_tiny', pet_backbone='mit_b0', ct_pretrained_path=None, pet_pretrained_path=None)

    ct_only.eval()
    full.eval()
    with torch.no_grad():
        out_ct = ct_only(ct, pet=pet, target_size=(64, 64), forward_mode='auto')
        out_full = full(ct, pet, target_size=(64, 64), forward_mode='auto')

    ct_only_logits = out_ct['logits']
    full_logits = out_full['logits']

    assert ct_only_logits.shape == (1, 1, 64, 64)
    assert full_logits.shape == (1, 1, 64, 64)
    assert torch.isfinite(ct_only_logits).all()
    assert torch.isfinite(full_logits).all()

    assert ct_only.decoder.__class__ is full.decoder.__class__
    assert count_params(ct_only.decoder) == count_params(full.decoder)
    assert count_params(ct_only.enc_ct) == count_params(full.enc_ct)
    assert count_params(ct_only.ct_align) == count_params(full.ct_align)
    assert not hasattr(ct_only, 'enc_pet')
    assert hasattr(full, 'enc_pet')
    assert not hasattr(ct_only, 'missing_decoder')
    assert not hasattr(ct_only, 'full_decoder')
    assert not hasattr(ct_only, 'hatr_recovery')
    assert not hasattr(ct_only, 'correction_adapter')
    assert not hasattr(full, 'missing_decoder')
    assert not hasattr(full, 'full_decoder')
    assert not hasattr(full, 'hatr_recovery')
    assert not hasattr(full, 'correction_adapter')
    assert out_ct.get('aux', {}) == {}
    assert out_full.get('aux', {}) == {}

    pet_alt = pet + 0.5
    with torch.no_grad():
        out_full_alt = full(ct, pet_alt, target_size=(64, 64), forward_mode='auto')
    full_pet_sensitivity = float((out_full_alt['logits'] - out_full['logits']).abs().mean().item())
    assert full_pet_sensitivity > 0

    result = {
        'ct_only_logits_shape': tuple(ct_only_logits.shape),
        'full_logits_shape': tuple(full_logits.shape),
        'ct_encoder_params_equal': count_params(ct_only.enc_ct) == count_params(full.enc_ct),
        'ct_align_params_equal': count_params(ct_only.ct_align) == count_params(full.ct_align),
        'decoder_params_equal': count_params(ct_only.decoder) == count_params(full.decoder),
        'ct_only_has_pet_encoder': hasattr(ct_only, 'enc_pet'),
        'full_has_pet_encoder': hasattr(full, 'enc_pet'),
        'full_pet_sensitivity': full_pet_sensitivity,
    }
    print(result)


if __name__ == '__main__':
    main()
