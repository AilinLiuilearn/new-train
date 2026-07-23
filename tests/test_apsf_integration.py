import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.apsf_module import APSF
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _pyramid(batch=2, channels=(64, 128, 320, 512), sizes=((128, 128), (64, 64), (32, 32), (16, 16)), requires_grad=True):
    return [torch.randn(batch, c, h, w, requires_grad=requires_grad) for c, (h, w) in zip(channels, sizes)]


def test_apsf_shapes_and_sum_equivalence():
    apsf = APSF(channels=(64, 128, 320, 512))
    ct = _pyramid()
    pet = _pyramid()
    proxy = _pyramid()
    full = apsf.forward_full(ct, pet)
    missing = apsf.forward_missing(ct, proxy)
    assert len(full) == 4 and len(missing) == 4
    for out, a, b in zip(full, ct, pet):
        assert out.shape == a.shape == b.shape
        torch.testing.assert_close(out, a, rtol=1e-6, atol=1e-6)
    for out, a, b in zip(missing, ct, proxy):
        assert out.shape == a.shape == b.shape
        torch.testing.assert_close(out, a, rtol=1e-6, atol=1e-6)


def test_apsf_parameter_count():
    apsf = APSF(channels=(64, 128, 320, 512))
    count = apsf.trainable_parameter_count()
    print('apsf_params', count)
    assert count < 3_000_000


def test_apsf_routing(monkeypatch):
    apsf = APSF(channels=(64, 128, 320, 512))
    calls = {'full': 0, 'missing': 0}
    original_full = apsf.forward_full
    original_missing = apsf.forward_missing

    def full_hook(*args, **kwargs):
        calls['full'] += 1
        return original_full(*args, **kwargs)

    def missing_hook(*args, **kwargs):
        calls['missing'] += 1
        return original_missing(*args, **kwargs)

    monkeypatch.setattr(apsf, 'forward_full', full_hook)
    ct = _pyramid()
    pet = _pyramid()
    apsf.forward_full(ct, pet)
    assert calls['full'] == 1 and calls['missing'] == 0
    monkeypatch.setattr(apsf, 'forward_missing', missing_hook)
    apsf.forward_missing(ct, pet)
    assert calls['full'] == 1 and calls['missing'] == 1


def test_missing_pet_isolated_and_backward():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    model.eval()
    calls = {'pet': 0}
    orig = model.enc_pet.forward

    def wrapped(*args, **kwargs):
        calls['pet'] += 1
        return orig(*args, **kwargs)

    model.enc_pet.forward = wrapped
    ct = torch.randn(1, 1, 64, 64)
    out = model(ct, None, forward_mode='missing')
    loss = out['logits'].mean()
    loss.backward()
    assert calls['pet'] == 0
    assert torch.isfinite(loss)


def test_mppc_apsf_boundary_and_gradients():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    model.eval()
    assert model.mppc is not model.apsf
    assert not hasattr(model.apsf, 'mppc')
    ct = torch.randn(1, 1, 64, 64, requires_grad=False)
    pet = torch.randn(1, 1, 64, 64, requires_grad=False)
    out = model(ct, pet, forward_mode='full')
    loss = out['logits'].mean()
    loss.backward()
    grads = [p.grad for p in model.apsf.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_loss_contract_and_state_dict(tmp_path):
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    task = MDTSegTeacher({'model': model}, type('C', (), {'learning_rate': 1e-4, 'weight_decay': 0.0, 'mixed_precision': False, 'loss_smooth': 1.0, 'bce_weight': 1.0, 'dice_weight': 1.0})())
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.is_tensor(loss) and torch.allclose(loss, stats['loss_seg'])
    assert 'loss_dci_dist' not in stats and 'loss_kl' not in stats and 'loss_aux_apsf' not in stats
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    loaded = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    state = torch.load(path, map_location='cpu')
    loaded.load_state_dict(state, strict=True)
    keys = loaded.state_dict().keys()
    assert any(k.startswith('apsf.') for k in keys)
    assert any(k.startswith('mppc.') for k in keys)
    assert not any('dci_fusion' in k for k in keys)
