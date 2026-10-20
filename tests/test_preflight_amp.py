"""Real-model preflight checks with actual CPU GradScaler overflow bookkeeping."""
import math

import pytest
import torch

from models.staged_petct import StagedPETCT
from scripts.check_staged import preflight_update
from utils.amp import make_grad_scaler
from utils.ema import ModelEMA


torch.set_num_threads(1)


def setup_check():
    model = StagedPETCT(2).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    ema = ModelEMA(model, decay=.999, device='cpu')
    batch = {'ct': torch.randn(2, 3, 64, 64), 'pet': torch.randn(2, 3, 64, 64),
             'mask': (torch.rand(2, 1, 64, 64) > .97).float(),
             'is_full': torch.ones(2, dtype=torch.bool)}
    return model, optimizer, ema, batch


@pytest.mark.skipif(not hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                    reason='CPU GradScaler is unavailable in older torch; CUDA legacy is tested separately')
def test_amp_overflow_retries_until_one_actual_update(monkeypatch):
    model, optimizer, ema, batch = setup_check()
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    parameter = next(model.enc_ct.parameters())
    original = parameter.detach().clone()
    attempts, steps = [], []
    def overflow_once(gradient):
        attempts.append(1)
        return torch.full_like(gradient, float('inf')) if len(attempts) == 1 else gradient
    handle = parameter.register_hook(overflow_once)
    real_step = optimizer.step
    def tracked_step(*args, **kwargs):
        steps.append(1)
        return real_step(*args, **kwargs)
    monkeypatch.setattr(optimizer, 'step', tracked_step)
    _, stats, grads, report = preflight_update(model, batch, 2, torch.device('cpu'),
                                              optimizer, scaler, ema)
    handle.remove()
    assert report['attempts'] == 2 and report['skipped_updates'] == 1
    assert report['amp_scale'] == 32768.
    assert len(steps) == ema.updates == 1
    assert stats['ct_samples'] == stats['full_samples'] == stats['missing_samples'] == 2
    assert all(math.isfinite(v) and v > 0 for v in grads.values())
    assert not torch.equal(original, parameter)


@pytest.mark.skipif(not hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                    reason='CPU GradScaler is unavailable in older torch')
def test_persistent_overflow_never_passes_or_updates_weights():
    model, optimizer, ema, batch = setup_check()
    parameter = next(model.enc_ct.parameters())
    original = parameter.detach().clone()
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    with pytest.raises(RuntimeError, match='No finite optimizer update after 2'):
        preflight_update(model, batch, 2, torch.device('cpu'), optimizer, scaler, ema, max_attempts=2)
    assert ema.updates == 0 and not optimizer.state
    torch.testing.assert_close(original, parameter, atol=0, rtol=0)
    assert scaler.get_scale() == 16384.
    # The same nonfinite gradient under FP32 must fail immediately.
    with pytest.raises(RuntimeError, match='Nonfinite FP32 gradients'):
        preflight_update(model, batch, 2, torch.device('cpu'), optimizer,
                         make_grad_scaler('cpu', enabled=False), ema)
    assert ema.updates == 0 and not optimizer.state
    handle.remove()
    with torch.no_grad(): parameter.flatten()[0] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite model parameters/buffers'):
        preflight_update(model, batch, 2, torch.device('cpu'), optimizer,
                         make_grad_scaler('cpu', enabled=False), ema)


@pytest.mark.skipif(not hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                    reason='CPU GradScaler is unavailable in older torch')
def test_finite_elements_with_overflowing_norm_remain_a_strict_error():
    model, optimizer, ema, batch = setup_check()
    parameter = next(model.enc_ct.parameters())
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, 1e30))
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    with pytest.raises(RuntimeError, match='non-finite'):
        preflight_update(model, batch, 2, torch.device('cpu'), optimizer, scaler, ema)
    assert torch.isfinite(parameter.grad).all()
    assert ema.updates == 0 and not optimizer.state
    assert scaler.get_scale() == 65536.
    handle.remove()
