import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.dci_fuse import MultiScaleDCIFuse
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


class _ToyPETCTModel(DualSharedAddPETCTBaseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _make_feats(batch=2):
    channels = (64, 128, 320, 512)
    sizes = (32, 16, 8, 4)
    ct = [torch.randn(batch, c, s, s, requires_grad=True) for c, s in zip(channels, sizes)]
    pet = [torch.randn(batch, c, s, s, requires_grad=True) for c, s in zip(channels, sizes)]
    return ct, pet


def test_multiscale_dci_shapes_and_loss():
    model = MultiScaleDCIFuse(channels=(64, 128, 320, 512))
    model.train()
    ct, pet = _make_feats()
    fused, loss_dci = model(ct, pet)
    assert len(fused) == 4
    for out, ref in zip(fused, ct):
        assert out.shape == ref.shape
    assert loss_dci.ndim == 0
    assert torch.isfinite(loss_dci)


def test_full_missing_forward_and_gradients():
    model = _ToyPETCTModel(use_dci=True, dci_sample_during_training=True)
    model.train()
    ct = torch.randn(2, 3, 128, 128)
    pet = torch.randn(2, 3, 128, 128)
    mask = torch.randint(0, 2, (2, 1, 128, 128)).float()

    out_full = model(ct, pet=pet, forward_mode='full', mask=mask)
    assert out_full['logits'].shape[-2:] == mask.shape[-2:]
    assert 'loss_dci_dist' in out_full and torch.isfinite(out_full['loss_dci_dist'])

    model._encode_pet = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('PET encoder should not run'))
    out_missing = model(ct, pet=None, forward_mode='missing')
    assert out_missing['logits'].shape[-2:] == mask.shape[-2:]
    assert torch.isfinite(out_missing['loss_dci_dist'])

    total_loss = out_full['logits'].mean() + out_full['loss_dci_dist'] + out_missing['logits'].mean() + out_missing['loss_dci_dist']
    total_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dci_fusion.parameters())


def test_eval_determinism_and_baseline_fallback():
    model = _ToyPETCTModel(use_dci=True, dci_sample_during_training=False)
    model.eval()
    ct = torch.randn(1, 3, 64, 64)
    pet = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = model(ct, pet=pet, forward_mode='full')['logits']
        b = model(ct, pet=pet, forward_mode='full')['logits']
    assert torch.equal(a, b)

    baseline = _ToyPETCTModel(use_dci=False)
    assert baseline.dci_fusion is None
    assert hasattr(baseline, 'fusion')
    out = baseline(ct, pet=pet, forward_mode='full', mask=torch.zeros(1, 1, 64, 64))
    assert torch.is_tensor(out['loss_dci_dist'])
    assert out['loss_dci_dist'].item() == 0.0


if __name__ == '__main__':
    test_multiscale_dci_shapes_and_loss()
    test_full_missing_forward_and_gradients()
    test_eval_determinism_and_baseline_fallback()
    print('DCI integration tests passed.')
