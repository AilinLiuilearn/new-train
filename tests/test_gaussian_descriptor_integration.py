"""Run from repository root after applying the descriptor-switch changes.

No encoder weights or datasets required. Tests actual four-scale fusion code.
"""
import io
import unittest
from unittest.mock import patch

import torch

from models.gaussian_response_descriptor import GaussianResponseDescriptor
from models.local_contrast_bidirectional_fusion import (
    LocalContrastBidirectionalFusion,
    LocalContrastFusionPyramid,
    _ContrastDescriptor,
)

TYPES = ("contrast", "rasfe_fixed", "rasfe_learnable")


def pyramid(descriptor_type="contrast"):
    return LocalContrastFusionPyramid(
        channels=(8, 16, 24, 32), dim=8, heads=2,
        chunk_rows=2, checkpoint_chunks=True,
        position_bias_beta=0.25, descriptor_type=descriptor_type,
    )


def features():
    ct = [torch.randn(2, c, size, size + 1) for c, size in
          zip((8, 16, 24, 32), (7, 5, 3, 2))]
    return ct, [torch.randn_like(x) for x in ct]


class IntegrationTests(unittest.TestCase):
    def test_default_contrast_same_state_keys_and_rng(self):
        torch.manual_seed(11)
        default = LocalContrastFusionPyramid(channels=(8, 16, 24, 32), dim=8, heads=2)
        torch.manual_seed(11)
        explicit = LocalContrastFusionPyramid(channels=(8, 16, 24, 32), dim=8, heads=2,
                                             descriptor_type="contrast")
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key in default.state_dict():
            self.assertTrue(torch.equal(default.state_dict()[key], explicit.state_dict()[key]))
        for block in default.scales:
            self.assertIsInstance(block.ct_contrast, _ContrastDescriptor)

    def test_all_scales_independent_and_same_structure(self):
        for mode in TYPES:
            m = pyramid(mode)
            objects = []
            for block in m.scales:
                self.assertEqual(block.descriptor_type, mode)
                objects.extend((block.ct_contrast, block.pet_contrast))
                expected = _ContrastDescriptor if mode == "contrast" else GaussianResponseDescriptor
                self.assertIsInstance(block.ct_contrast, expected)
                self.assertIsInstance(block.pet_contrast, expected)
            self.assertEqual(len({id(x) for x in objects}), 8)

    def test_full_zero_initialized_raw_sum_and_gradients(self):
        for mode in TYPES:
            m = pyramid(mode)
            ct, pet = features()
            ct = [x.requires_grad_() for x in ct]
            pet = [x.requires_grad_() for x in pet]
            fused, ce, pe = m.forward_features(ct, pet)
            for c, p, f, cp, pp in zip(ct, pet, fused, ce, pe):
                self.assertTrue(torch.equal(f, c + p))
                self.assertTrue(torch.equal(cp, c))
                self.assertTrue(torch.equal(pp, p))
            sum(x.square().mean() for x in fused).backward()
            for x in ct + pet:
                self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_missing_and_all_missing_auto_make_no_calls(self):
        for mode in TYPES:
            m = pyramid(mode)
            ct, _ = features()
            # Mock the entire scale block, stronger than only mocking descriptors.
            with patch.object(LocalContrastBidirectionalFusion, "forward",
                              side_effect=AssertionError("Fusion called on missing rows")):
                missing = m(ct, None, mode="missing")
                auto = m(ct, None, mode="auto", pet_available=[0, 0])
            for c, a, b in zip(ct, missing, auto):
                self.assertIs(a, c)
                self.assertIs(b, c)

    def test_mixed_rows_ignore_missing_pet_and_preserve_order(self):
        for mode in TYPES:
            m = pyramid(mode).eval()
            ct, pet = features()
            pet2 = [p.clone() for p in pet]
            for p in pet2:
                p[1].fill_(float("nan"))
            with torch.no_grad():
                a = m(ct, pet, mode="auto", pet_available=[1, 0])
                b = m(ct, pet2, mode="auto", pet_available=[1, 0])
            for c, p, out, other in zip(ct, pet, a, b):
                self.assertTrue(torch.equal(out, other))
                self.assertTrue(torch.equal(out[0], c[0] + p[0]))
                self.assertTrue(torch.equal(out[1], c[1]))

    def test_two_step_learning_keeps_fixed_kernels_but_trains_learnable(self):
        for mode in ("rasfe_fixed", "rasfe_learnable"):
            m = pyramid(mode)
            descriptors = [d for b in m.scales for d in (b.ct_contrast, b.pet_contrast)]
            kernels = [b.weight for d in descriptors for b in d.branches]
            original = [w.detach().clone() for w in kernels]
            optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)
            ct, pet = features()
            # Step 1 opens the zero-initialized fusion update projections.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                sum(x.square().mean() for x in m(ct, pet)).backward()
                optimizer.step()
            for w, old in zip(kernels, original):
                if mode == "rasfe_fixed":
                    self.assertIsNone(w.grad)
                    self.assertTrue(torch.equal(w, old))
                else:
                    self.assertTrue(bool(torch.isfinite(w.grad).all()))
                    self.assertGreater(float(w.grad.abs().sum()), 0)
                    self.assertFalse(torch.equal(w, old))

    def test_same_mode_strict_roundtrip_and_invalid_selector(self):
        for mode in TYPES:
            a, b = pyramid(mode), pyramid(mode)
            memory = io.BytesIO()
            torch.save(a.state_dict(), memory)
            memory.seek(0)
            b.load_state_dict(torch.load(memory, weights_only=True), strict=True)
            for key in a.state_dict():
                self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
        with self.assertRaises(ValueError):
            pyramid("typo")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
