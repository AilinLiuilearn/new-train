import torch

from models.dual_decoder_add_baseline import DualDecoderAddPETCTBaseline


def _make_model():
    return DualDecoderAddPETCTBaseline(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(32, 16, 8, 4),
        use_deep_supervision=False,
    )


def _clone_grad_state(module):
    return [None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]


def test_decoder_init_same_but_independent():
    model = _make_model()
    for p_full, p_missing in zip(model.full_decoder.parameters(), model.missing_decoder.parameters()):
        assert torch.equal(p_full, p_missing)
        assert p_full.data_ptr() != p_missing.data_ptr()


def test_full_route_gradient_isolation():
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out = model(ct, pet, forward_mode='full')
    loss = out['pred'].mean()
    loss.backward()
    assert any(p.grad is not None for p in model.full_decoder.parameters())
    assert all(p.grad is None for p in model.missing_decoder.parameters())
    assert any(p.grad is not None for p in model.enc_pet.parameters())


def test_missing_route_gradient_isolation():
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    ct = torch.randn(2, 1, 64, 64)
    out = model(ct, None, forward_mode='missing')
    loss = out['pred'].mean()
    loss.backward()
    assert any(p.grad is not None for p in model.missing_decoder.parameters())
    assert all(p.grad is None for p in model.full_decoder.parameters())
    assert all(p.grad is None for p in model.enc_pet.parameters())


def test_mixed_batch_routing_order_and_validity():
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    ct = torch.randn(4, 1, 64, 64)
    pet = torch.randn(4, 1, 64, 64)
    pet_available = torch.tensor([1, 0, 1, 0])
    out = model(ct, pet, pet_available=pet_available, forward_mode='auto')
    assert out['pred'].shape[0] == 4
    assert out['pred'].shape[-2:] == (64, 64)
    assert torch.isfinite(out['pred']).all()
    assert torch.isfinite(out['logits']).all()
    assert torch.allclose(out['pred'], out['logits'])
