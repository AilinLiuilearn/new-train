"""Check CT-only final evaluation and delayed-EMA model-selection consistency."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

import run_ct_only_seg as runner
from tasks.mdt_seg import MDTSegTeacher


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.feature_info = type('Info', (), {'channels': lambda _: [8]*4})()

    def forward(self, x):
        return [F.adaptive_avg_pool2d(self.conv(x), s) for s in (8, 4, 2, 1)]


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_final_uses_checkpoint_tag_and_accepts_no_pet_batch(self):
        gt = torch.zeros(2, 1, 32, 32)
        gt[:, :, 8:20, 8:20] = 1
        loader = [{'ct': torch.randn(2, 3, 32, 32), 'mask': gt}]
        for start, expected in ((3, 'raw'), (0, 'ema')):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as tmp:
                args = ['run', '--pretrained', 'False', '--epochs', '1', '--batch_size', '2',
                        '--mixed_precision', 'False', '--hd95_mode', 'surface', '--num_workers', '0',
                        '--decoder_channels', '8', '8', '8', '8', '--checkpoint_root', tmp,
                        '--hash', 'ct', '--ema_enabled', 'True', '--ema_start_epoch', str(start)]
                with patch('sys.argv', args), patch.object(runner, '_loaders', return_value=(loader, loader, loader)), \
                     patch('models.ct_only_baseline.create_feature_backbone', lambda *a, **k: Encoder()):
                    runner.main()
                folder = os.path.join(tmp, 'MDT', 'ct')
                with open(os.path.join(folder, 'final_test_ct.json')) as f:
                    result = json.load(f)
                ckpt = MDTSegTeacher.load_state_dicts(os.path.join(folder, 'ckpt.best_ct.pth.tar'))
                self.assertEqual(result['weights'], expected)
                self.assertEqual(ckpt['eval_weights'], expected)
                self.assertAlmostEqual(result['dice'], ckpt['val']['dice'])
                self.assertAlmostEqual(result['hd95'], ckpt['val']['hd95'])
                self.assertNotIn('missing_rate', result)
                self.assertTrue(all(not k.startswith('enc_pet.') for k in ckpt['model']))


if __name__ == '__main__':
    unittest.main()
