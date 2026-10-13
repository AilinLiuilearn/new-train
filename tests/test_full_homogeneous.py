"""Bounded checks: native Add, independent parameters, checkpoint and runner."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from configs.full_homogeneous import FullHomogeneousConfig
from models.full_homogeneous import FullHomogeneousModel, NATIVE_CHANNELS


class NativeEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.channels = NATIVE_CHANNELS[name]
        self.feature_info = type('Info', (), {'channels': lambda _: list(self.channels)})()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in self.channels)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [layer(F.adaptive_avg_pool2d(x, (h, h)))
                for layer, h in zip(self.layers, (8, 4, 2, 1))]


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_native_add_no_alignment_no_shared_weights_and_optional_fusion(self):
        for name in NATIVE_CHANNELS:
            with self.subTest(name=name), patch('models.full_homogeneous.create_feature_backbone',
                                               lambda n, **kw: NativeEncoder(n)):
                m = FullHomogeneousModel(name, pretrained=False,
                                         decoder_channels=(8, 8, 8, 8)).eval()
                self.assertEqual(sum(p.numel() for p in m.ct_align.parameters()), 0)
                self.assertEqual(sum(p.numel() for p in m.pet_align.parameters()), 0)
                self.assertEqual(sum(p.numel() for p in m.fusion.parameters()), 0)
                ct = torch.randn(2, 3, 32, 32)
                pet = torch.randn_like(ct)
                c, p = m._encode_ct(ct), m._encode_pet(pet)
                for raw, encoded in zip(m.enc_ct(ct), c):
                    self.assertTrue(torch.equal(raw, encoded))
                expected = m.decoder([x+y for x, y in zip(c, p)], (32, 32))['logits']
                actual = m(ct, pet)['logits']
                self.assertTrue(torch.equal(expected, actual))
                actual.square().mean().backward()
                for left, right in zip(m.enc_ct.parameters(), m.enc_pet.parameters()):
                    self.assertNotEqual(left.data_ptr(), right.data_ptr())
                    self.assertIsNotNone(left.grad)
                    self.assertIsNotNone(right.grad)
                calls = m.enc_ct.calls, m.enc_pet.calls
                with self.assertRaises(ValueError):
                    m(ct, pet, forward_mode='missing')
                self.assertEqual(calls, (m.enc_ct.calls, m.enc_pet.calls))
                same = FullHomogeneousModel(name, pretrained=False,
                                            decoder_channels=(8, 8, 8, 8)).eval()
                same.load_state_dict(m.state_dict(), strict=True)
                self.assertTrue(torch.equal(actual, same(ct, pet)['logits']))
                candidate = FullHomogeneousModel(name, pretrained=False,
                    decoder_channels=(8, 8, 8, 8), fusion_enabled=True).eval()
                candidate.load_state_dict(m.state_dict(), strict=False)
                self.assertEqual(candidate.fusion.channels, NATIVE_CHANNELS[name])
                self.assertTrue(torch.equal(actual, candidate(ct, pet)['logits']))

    def test_config_selects_both_weight_paths_and_rejects_heterogeneous_override(self):
        for name in NATIVE_CHANNELS:
            with patch('sys.argv', ['run', '--encoder_pair', name]):
                cfg = FullHomogeneousConfig.parse_arguments()
            self.assertEqual(cfg.ct_backbone, name)
            self.assertEqual(cfg.pet_backbone, name)
            self.assertEqual(cfg.ct_pretrained_path, cfg.pet_pretrained_path)
            self.assertFalse(cfg.fusion_enabled)
            self.assertEqual(cfg.channel_alignment, 'identity')
        with patch('sys.argv', ['run', '--encoder_pair', 'mit_b1', '--ct_backbone', 'convnextv2_nano']):
            with self.assertRaises(ValueError):
                FullHomogeneousConfig.parse_arguments()

    def test_runner_checkpoint_resume_and_eval_reconstruct_selected_architecture(self):
        import run_full_homogeneous_add as entry
        import run_full_structure_response_fusion as runner
        from tasks.mdt_seg import MDTSegTeacher
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, 8:20, 8:20] = 1
        batch = {'ct': torch.randn(2, 3, 32, 32), 'pet': torch.randn(2, 3, 32, 32),
                 'mask': mask, 'image_id': ['a', 'b']}
        loader = [batch]
        with tempfile.TemporaryDirectory() as tmp:
            args = ['run', '--encoder_pair', 'convnextv2_nano', '--pretrained', 'False',
                    '--epochs', '1', '--batch_size', '2', '--num_workers', '0',
                    '--mixed_precision', 'False', '--allow_cpu', 'True', '--hd95_mode', 'surface',
                    '--decoder_channels', '8', '8', '8', '8', '--checkpoint_root', tmp,
                    '--hash', 'native', '--ema_start_epoch', '0']
            def execute(argv):
                with patch('sys.argv', argv), patch.object(runner, '_loaders', return_value=(loader, loader, loader)), \
                     patch('models.full_homogeneous.create_feature_backbone', lambda n, **kw: NativeEncoder(n)):
                    entry.main()
            execute(args)
            root = os.path.join(tmp, 'MDT', 'native')
            best = os.path.join(root, 'ckpt.best_full.pth.tar')
            last = os.path.join(root, 'ckpt.last.pth.tar')
            ckpt = MDTSegTeacher.load_state_dicts(last)
            self.assertEqual(ckpt['config']['encoder_pair'], 'convnextv2_nano')
            self.assertEqual(ckpt['ema_updates'], 1)
            self.assertFalse(any(k.startswith(('ct_align.', 'pet_align.')) for k in ckpt['model']))
            execute(args + ['--resume', last])
            eval_args = list(args)
            eval_args[eval_args.index('convnextv2_nano')] = 'mit_b1'
            execute(eval_args + ['--eval_checkpoint', best, '--hash', 'inspection'])
            with open(os.path.join(tmp, 'MDT', 'inspection', 'final_test_full.json')) as f:
                result = json.load(f)
            self.assertEqual(result['ct_backbone'], 'convnextv2_nano')
            self.assertEqual(result['channel_alignment'], 'identity')
            self.assertEqual(result['weights'], 'ema')
            wrong_resume = list(args)
            wrong_resume[wrong_resume.index('convnextv2_nano')] = 'mit_b1'
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                execute(wrong_resume + ['--resume', last])


if __name__ == '__main__':
    unittest.main()
