"""Real encoder contract checks; no pretrained files required."""
import torch
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def test_ct_only_has_no_pet_and_backpropagates():
    model = DualSharedAddPETCTBaseline(baseline='ct_only', pretrained=False)
    assert not hasattr(model, 'enc_pet')
    assert not hasattr(model, 'pet_align')
    assert not hasattr(model, 'fusion')
    model.eval()
    ct = torch.randn(2, 1, 64, 64)
    out = model(ct)['logits']
    assert out.shape == (2, 1, 64, 64)
    out.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.enc_ct.parameters())


def test_full_add_and_missing_contract():
    model = DualSharedAddPETCTBaseline(baseline='full_add', pretrained=False).eval()
    ct, pet = torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)
    full = model(ct, pet)['logits']
    full.square().mean().backward()
    for module in (model.enc_ct, model.enc_pet, model.ct_align, model.pet_align, model.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    def forbidden(*args):
        raise AssertionError('PET encoder called in missing path')
    handle = model.enc_pet.register_forward_pre_hook(forbidden)
    with torch.no_grad():
        a = model(ct, forward_mode='missing')['logits']
        b = model(ct, pet * 100, forward_mode='missing')['logits']
    handle.remove()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
