"""Exercise both AMP API signatures without requiring CUDA or Triton."""
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


SOURCE = Path('models/components/lsconv_amp.py')


@pytest.mark.skipif(not hasattr(torch.amp, 'custom_fwd'), reason='Legacy PyTorch AMP API')
def test_modern_decorators_are_used_directly():
    api = runpy.run_path(str(SOURCE))
    assert api['custom_fwd'] is torch.amp.custom_fwd
    assert api['custom_bwd'] is torch.amp.custom_bwd


def legacy_api(monkeypatch):
    calls = []
    legacy = ModuleType('torch.cuda.amp')

    # These deliberately reject device_type, matching the older CUDA-only API.
    def custom_fwd(fwd=None, *, cast_inputs=None):
        calls.append(('forward', cast_inputs))
        return fwd if fwd is not None else lambda fn: fn

    def custom_bwd(bwd):
        calls.append(('backward',))
        return bwd

    legacy.custom_fwd, legacy.custom_bwd = custom_fwd, custom_bwd
    monkeypatch.setitem(sys.modules, 'torch.amp', ModuleType('torch.amp'))
    monkeypatch.setitem(sys.modules, 'torch.cuda.amp', legacy)
    return runpy.run_path(str(SOURCE)), calls


def test_legacy_decorator_forms_strip_device_type(monkeypatch):
    api, calls = legacy_api(monkeypatch)
    fn = lambda *args: args
    assert api['custom_fwd'](device_type='cuda')(fn) is fn
    assert api['custom_bwd'](device_type='cuda')(fn) is fn
    assert api['custom_fwd'](fn, device_type='cuda', cast_inputs=torch.float32) is fn
    assert api['custom_bwd'](fn, device_type='cuda') is fn
    assert calls == [('forward', None), ('backward',),
                     ('forward', torch.float32), ('backward',)]


@pytest.mark.parametrize('name', ['custom_fwd', 'custom_bwd'])
def test_legacy_path_rejects_non_cuda_device(monkeypatch, name):
    api, _ = legacy_api(monkeypatch)
    with pytest.raises(ValueError, match='CUDA only'):
        api[name](device_type='cpu')


def test_real_amp_decorated_autograd_on_cpu():
    # Actual selected decorators remain a no-op outside CUDA autocast.
    api = runpy.run_path(str(SOURCE))

    class Square(torch.autograd.Function):
        @staticmethod
        @api['custom_fwd'](device_type='cuda')
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        @api['custom_bwd'](device_type='cuda')
        def backward(ctx, grad):
            x, = ctx.saved_tensors
            return 2 * x * grad

    x = torch.tensor([2., -3.], requires_grad=True)
    Square.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
