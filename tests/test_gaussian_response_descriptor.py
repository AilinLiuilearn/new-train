"""Standalone tests; also works after copying files to models/ and tests/."""
import importlib.util
import io
import math
import copy
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_source = _here / "gaussian_response_descriptor.py"
if not _source.exists():
    _source = _here.parent / "models" / "gaussian_response_descriptor.py"
_spec = importlib.util.spec_from_file_location("rasfe_descriptor_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GaussianDepthwiseResponse = _module.GaussianDepthwiseResponse
GaussianResponseDescriptor = _module.GaussianResponseDescriptor
sync_fixed_gaussian_buffers = _module.sync_fixed_gaussian_buffers


class GaussianDescriptorTests(unittest.TestCase):
    def test_exact_gaussian_kernels_and_dilation(self):
        m = GaussianResponseDescriptor(8)
        for branch, sigma, dilation in zip(m.branches, (1.0, 1.5, 2.0), (1, 1, 2)):
            self.assertEqual(branch.dilation, dilation)
            self.assertEqual(tuple(branch.weight.shape), (8, 1, 3, 3))
            coordinates = torch.arange(-1, 2, dtype=torch.float64)
            yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
            reference = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
            reference = (reference / reference.sum()).float()
            torch.testing.assert_close(branch.weight[0, 0], reference, rtol=0, atol=0)
            self.assertTrue(bool((branch.weight >= 0).all()))
            torch.testing.assert_close(branch.weight.sum((1, 2, 3)), torch.ones(8))
            x = torch.randn(2, 8, 9, 11)
            expected = F.conv2d(x, branch.weight, padding=dilation, dilation=dilation, groups=8)
            torch.testing.assert_close(branch(x), expected, rtol=0, atol=0)

    def test_no_channel_mixing_in_gaussian_branch(self):
        m = GaussianDepthwiseResponse(4, sigma=1.0, dilation=2)
        x = torch.zeros(1, 4, 9, 9)
        x[0, 1, 4, 4] = 1
        y = m(x)
        self.assertEqual(int(torch.count_nonzero(y[:, [0, 2, 3]])), 0)
        self.assertEqual(int(torch.count_nonzero(y[:, 1])), 9)

    def test_complete_formula_no_response_subtraction(self):
        m = GaussianResponseDescriptor(8)
        x = torch.randn(2, 8, 13, 15)
        response = torch.cat([branch(x) for branch in m.branches], dim=1)
        expected = x + m.mix(F.gelu(m.norm(response)))
        torch.testing.assert_close(m(x), expected, rtol=0, atol=0)

    def test_fixed_and_learnable_identical_at_initialization(self):
        torch.manual_seed(23)
        fixed = GaussianResponseDescriptor(32, descriptor_type="rasfe_fixed")
        torch.manual_seed(23)
        learned = GaussianResponseDescriptor(32, descriptor_type="rasfe_learnable")
        self.assertEqual(set(fixed.state_dict()), set(learned.state_dict()))
        x = torch.randn(2, 32, 17, 19)
        torch.testing.assert_close(fixed(x), learned(x), rtol=0, atol=0)

    def test_gradients_optimizer_and_fixed_kernel_immutability(self):
        for mode in ("rasfe_fixed", "rasfe_learnable"):
            with self.subTest(mode=mode):
                m = GaussianResponseDescriptor(8, descriptor_type=mode)
                before = [branch.weight.detach().clone() for branch in m.branches]
                optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)
                x = torch.randn(2, 8, 11, 13, requires_grad=True)
                m(x).square().mean().backward()
                self.assertTrue(bool(torch.isfinite(x.grad).all()))
                self.assertGreater(float(x.grad.abs().sum()), 0)
                optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
                for branch in m.branches:
                    if mode == "rasfe_fixed":
                        self.assertNotIn(id(branch.weight), optimizer_ids)
                        self.assertIsNone(branch.weight.grad)
                    else:
                        self.assertIn(id(branch.weight), optimizer_ids)
                        self.assertTrue(bool(torch.isfinite(branch.weight.grad).all()))
                        self.assertGreater(float(branch.weight.grad.abs().sum()), 0)
                optimizer.step()
                for branch, old in zip(m.branches, before):
                    self.assertEqual(torch.equal(branch.weight, old), mode == "rasfe_fixed")

    def test_parameter_counts_and_persistent_buffers(self):
        fixed = GaussianResponseDescriptor(32)
        learned = GaussianResponseDescriptor(32, descriptor_type="rasfe_learnable")
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 3296)
        self.assertEqual(sum(p.numel() for p in learned.parameters()), 4160)
        self.assertEqual(sum(b.numel() for b in fixed.buffers()), 864)
        self.assertEqual(sum(b.numel() for b in learned.buffers()), 0)
        self.assertEqual(len([k for k in fixed.state_dict() if k.endswith("weight") and k.startswith("branches.")]), 3)

    def test_odd_shapes_small_shapes_and_dtype(self):
        for channels, height, width in ((8, 1, 1), (7, 5, 7), (32, 17, 19)):
            m = GaussianResponseDescriptor(channels).double()
            x = torch.randn(2, channels, height, width, dtype=torch.float64)
            y = m(x)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(y.dtype, x.dtype)
            self.assertEqual(y.device, x.device)
            self.assertTrue(bool(torch.isfinite(y).all()))

    def test_serialization_same_mode_strict(self):
        for mode in ("rasfe_fixed", "rasfe_learnable"):
            a = GaussianResponseDescriptor(8, descriptor_type=mode)
            b = GaussianResponseDescriptor(8, descriptor_type=mode)
            memory = io.BytesIO()
            torch.save(a.state_dict(), memory)
            memory.seek(0)
            b.load_state_dict(torch.load(memory, weights_only=True), strict=True)
            x = torch.randn(2, 8, 11, 13)
            torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    def test_cpu_autocast(self):
        m = GaussianResponseDescriptor(8)
        x = torch.randn(2, 8, 9, 11, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = m(x)
            loss = y.float().square().mean()
        loss.backward()
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_ema_sync_changes_only_fixed_kernels(self):
        source = GaussianResponseDescriptor(8)
        target = copy.deepcopy(source)
        with torch.no_grad():
            for branch in target.branches:
                branch.weight.add_(0.1)
            target.mix.weight.add_(1)
        learned_before = target.mix.weight.detach().clone()
        sync_fixed_gaussian_buffers(target, source)
        for a, b in zip(source.branches, target.branches):
            self.assertTrue(torch.equal(a.weight, b.weight))
        self.assertTrue(torch.equal(target.mix.weight, learned_before))
        source = GaussianResponseDescriptor(8, "rasfe_learnable")
        target = copy.deepcopy(source)
        with torch.no_grad():
            target.branches[0].weight.add_(0.1)
        before = target.branches[0].weight.detach().clone()
        sync_fixed_gaussian_buffers(target, source)
        self.assertTrue(torch.equal(target.branches[0].weight, before))

    def test_constructor_and_input_validation(self):
        for kwargs in ({"dim": 0}, {"dim": True}, {"dim": 8, "descriptor_type": "unknown"}, {"dim": 8, "descriptor_type": "contrast"}):
            with self.assertRaises((ValueError, TypeError)):
                GaussianResponseDescriptor(**kwargs)
        for sigma in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                GaussianDepthwiseResponse(8, sigma, 1)
        for dilation in (0, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                GaussianDepthwiseResponse(8, 1.0, dilation)
        m = GaussianResponseDescriptor(8)
        for x in (torch.ones(2, 8, 3), torch.ones(2, 7, 3, 3), torch.empty(0, 8, 3, 3), torch.ones(2, 8, 3, 3, dtype=torch.int64)):
            with self.assertRaises((ValueError, TypeError)):
                m(x)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
