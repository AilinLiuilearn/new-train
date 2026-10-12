# -*- coding: utf-8 -*-
"""Lightweight weighted-add contracts, fake encoders, no real data/GPU."""
import ast
import copy
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_full_weighted_add as runner
from eval_joint_baseline import _select_builder
from models.build_mdt_seg import build_dual_model
from models.components.add_fusion import AddFusion
from models.full_weighted_add_model import build_full_weighted_add_model
from models.weighted_add_fusion import WeightedAddFusion
from tasks.mdt_seg import MDTSegTeacher

CHANNELS = (64, 128, 320, 512)


class FakeEncoder(nn.Module):
    """Test fixture only; production keeps real pretrained encoders."""
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(3, c, 1) for c in channels])
        self.feature_info = SimpleNamespace(channels=lambda: list(channels))
        self.calls = []

    def forward(self, x):
        self.calls.append(x.shape[0])
        return [m(F.adaptive_avg_pool2d(x, (x.shape[-2] // s, x.shape[-1] // s)))
                for m, s in zip(self.proj, (4, 8, 16, 32))]


def fake_encoder(name, in_channels):
    return FakeEncoder((80, 160, 320, 640) if name == 'convnextv2_nano'
                       else CHANNELS)


def config(enabled=True):
    with patch.object(sys, 'argv', ['test', '--pretrained', 'False',
                                   '--decoder_channels', '32', '16', '8', '4',
                                   '--mixed_precision', 'False',
                                   '--ema_enabled', 'True', '--ema_start_epoch', '0',
                                   '--weighted_fusion_enabled', str(enabled)]):
        cfg = runner.FullWeightedAddConfig.parse_arguments()
    cfg.fusion_type = 'WeightedAddFusion' if enabled else 'AddFusion'
    return cfg


def batch():
    return {'ct': torch.randn(2, 1, 64, 64),
            'pet': torch.randn(2, 1, 64, 64),
            'mask': (torch.rand(2, 1, 64, 64) > .95).float()}


class WeightedAddContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2023)
        self.factory = patch('models.dual_shared_add_baseline.create_feature_backbone',
                             side_effect=fake_encoder)
        self.weights = patch('models.dual_shared_add_baseline.load_local_weights_safe')
        self.factory.start()
        self.weights.start()
        self.addCleanup(self.factory.stop)
        self.addCleanup(self.weights.stop)

    def test_four_scale_one_init_and_parameter_count(self):
        fusion = WeightedAddFusion()
        ct = [torch.randn(1, c, h, h) for c, h in zip(CHANNELS, (128, 64, 32, 16))]
        pet = [torch.randn_like(c) for c in ct]
        out = fusion(ct, pet)
        self.assertEqual(sum(p.numel() for p in fusion.parameters()), 8)
        for o, c, p in zip(out, ct, pet):
            self.assertTrue(torch.equal(o, c + p))
        resized = [p[:, :, ::2, ::2] for p in pet]
        for a, b in zip(fusion(ct, resized), AddFusion()(ct, resized)):
            self.assertTrue(torch.equal(a, b))

    def test_contract_rejects_invalid_shapes(self):
        m = WeightedAddFusion()
        ct = [torch.randn(1, c, 2, 2) for c in CHANNELS]
        with self.assertRaises(ValueError):
            m(ct[:3], ct)
        bad = list(ct)
        bad[1] = torch.randn(1, 64, 2, 2)
        with self.assertRaises(ValueError):
            m(ct, bad)
        bad = [c[:, :, :1, :1] for c in ct]
        self.assertEqual(len(m(bad, bad)), 4)

    def test_full_logits_equal_baseline_at_step_zero_and_disabled(self):
        b = batch()
        torch.manual_seed(2023)
        base = build_dual_model(config())['model'].eval()
        torch.manual_seed(2023)
        candidate = build_full_weighted_add_model(config())['model'].eval()
        with torch.no_grad():
            reference = base(b['ct'], b['pet'], forward_mode='full')['logits']
            actual = candidate(b['ct'], b['pet'], forward_mode='full')['logits']
        self.assertTrue(torch.equal(reference, actual))
        torch.manual_seed(2023)
        disabled = build_full_weighted_add_model(config(False))['model'].eval()
        self.assertEqual(set(base.state_dict()), set(disabled.state_dict()))
        with torch.no_grad():
            self.assertTrue(torch.equal(
                reference, disabled(b['ct'], b['pet'], forward_mode='full')['logits']))

    def test_missing_bypasses_pet_and_fusion(self):
        model = build_full_weighted_add_model(config())['model'].eval()
        b = batch()
        with patch.object(model.enc_pet, 'forward', side_effect=AssertionError('PET called')), \
             patch.object(model.fusion, 'forward', side_effect=AssertionError('WeightedAdd called')):
            a = model(b['ct'], None, forward_mode='missing')['logits']
            z = model(b['ct'], b['pet'] * 100, forward_mode='missing')['logits']
            auto = model(b['ct'], None, pet_available=[0, 0])['logits']
        self.assertTrue(torch.equal(a, z))
        self.assertTrue(torch.equal(a, auto))

    def test_auto_only_full_rows_enter_pet_and_weighted(self):
        model = build_full_weighted_add_model(config())['model'].eval()
        b = batch()
        seen = []
        handle = model.fusion.register_forward_pre_hook(
            lambda module, args: seen.append(args[0][0].shape[0]))
        with torch.no_grad():
            out = model(b['ct'], b['pet'], pet_available=[0, 1])
            ct_only = model(b['ct'][:1], None, forward_mode='missing')['logits']
        handle.remove()
        self.assertEqual(model.enc_pet.calls, [1])
        self.assertEqual(seen, [1])
        torch.testing.assert_close(out['logits'][:1], ct_only)
        self.assertEqual(out['num_full'], 1)

    def test_optimizer_ema_and_checkpoint_rebuild(self):
        cfg = config()
        task = MDTSegTeacher(build_full_weighted_add_model(cfg), cfg)
        ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
        self.assertTrue(all(id(p) in ids for p in task.model.fusion.parameters()))
        fusion_keys = [k for k in task.model.state_dict() if k.startswith('fusion.')]
        self.assertTrue(fusion_keys)
        self.assertTrue(all(k in task.ema.state_dict() for k in fusion_keys))
        loss, _, _ = runner.train_step_full(task, batch())
        loss.backward()
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(task.model, name).parameters()
                     if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        task.optimizer.step()
        task.update_ema()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ckpt.pth')
            task.save_checkpoint(path, 1, .5, 1)
            ckpt = task.load_state_dicts(path)
            rebuilt = _select_builder(SimpleNamespace(**ckpt['config']), False)(cfg)['model']
            rebuilt.load_state_dict(ckpt['model'], strict=True)
            rebuilt.load_state_dict(ckpt['model_ema'], strict=True)
        self.assertIs(_select_builder(config(False), False), build_dual_model)

    def test_entry_defaults_are_full_and_shared_defaults_unchanged(self):
        from configs.seg_mdt import SegMDTConfig
        cfg = config()
        self.assertEqual(cfg.train_batch_mode, 'full')
        self.assertTrue(cfg.weighted_fusion_enabled)
        self.assertEqual(cfg.hd95_mode, 'cipa_official')
        with patch.object(sys, 'argv', ['test']):
            base = SegMDTConfig.parse_arguments()
        self.assertEqual(base.train_batch_mode, 'mixed')
        self.assertFalse(hasattr(base, 'weighted_fusion_enabled'))
        with patch.object(sys, 'argv', ['test', '--train_batch_mode', 'mixed']):
            with self.assertRaises(SystemExit):
                runner.FullWeightedAddConfig.parse_arguments()

    def test_independent_entry_one_epoch_and_final_ema_rebuild(self):
        b = batch()
        evaluations = []
        def evaluate(task, loader, eval_mode, tag, model):
            self.assertEqual(eval_mode, 'full')
            self.assertIsInstance(model.fusion, WeightedAddFusion)
            self.assertFalse(model.training)
            evaluations.append(tag)
            return {'dice': .5, 'iou': 1 / 3, 'acc': .5,
                    'acc_pixel': .9, 'hd95': 12., 'total_loss': 1.}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['test', '--checkpoint_root', tmp, '--hash', 'smoke',
                    '--pretrained', 'False', '--decoder_channels', '32', '16', '8', '4',
                    '--mixed_precision', 'False', '--ema_enabled', 'True',
                    '--ema_start_epoch', '0', '--epochs', '1']
            with patch.object(sys, 'argv', argv), \
                 patch.object(runner, '_loaders', return_value=([b], [b], [b])), \
                 patch.object(MDTSegTeacher, 'evaluate', evaluate):
                runner.main()
            result_path = os.path.join(tmp, 'MDT', 'smoke', 'final_test_full.json')
            with open(result_path) as f:
                result = json.load(f)
            self.assertEqual(result['fusion_type'], 'WeightedAddFusion')
            self.assertEqual(result['weights'], 'ema')
            self.assertEqual(result['hd95_mode'], 'cipa_official')
        self.assertEqual(evaluations, ['val_full', 'test_full'])


    def test_negative_and_above_two_weights_are_not_constrained(self):
        fusion = WeightedAddFusion()
        ct = [torch.rand(1, c, 3, 3) for c in CHANNELS]
        pet = [torch.rand_like(c) for c in ct]
        with torch.no_grad():
            fusion.weights.copy_(torch.tensor([[-1., 3.], [2.5, -.5],
                                               [-2., -3.], [0., 4.]]))
        out = fusion(ct, pet)
        for i, (o, c, p) in enumerate(zip(out, ct, pet)):
            torch.testing.assert_close(o, fusion.weights[i, 0]*c + fusion.weights[i, 1]*p)
        self.assertLess(float(fusion.weights.detach().min()), 0.)
        self.assertGreater(float(fusion.weights.detach().max()), 2.)

    def test_all_eight_coefficients_get_gradients_and_update(self):
        fusion = WeightedAddFusion()
        ct = [torch.ones(1, c, 3, 3) for c in CHANNELS]
        pet = [torch.full_like(c, 2.) for c in ct]
        opt = torch.optim.SGD(fusion.parameters(), lr=.1)
        loss = sum(o.square().mean() for o in fusion(ct, pet))
        loss.backward()
        self.assertEqual(fusion.weights.grad.shape, (4, 2))
        self.assertTrue(torch.isfinite(fusion.weights.grad).all())
        self.assertTrue(fusion.weights.grad.ne(0).all())
        opt.step()
        self.assertTrue(fusion.weights.detach().ne(1).all())

    def test_half_and_bfloat16_zero_init_equivalence(self):
        fusion = WeightedAddFusion()
        for dtype in (torch.float16, torch.bfloat16):
            ct = [torch.randn(1, c, 3, 3).to(dtype) for c in CHANNELS]
            pet = [torch.randn_like(c) for c in ct]
            for o, ref in zip(fusion(ct, pet), AddFusion()(ct, pet)):
                self.assertEqual(o.dtype, dtype)
                self.assertTrue(torch.equal(o, ref))


if __name__ == '__main__':
    unittest.main()


