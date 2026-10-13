"""Only the changed reference-entry contracts; no real-data training."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_full_missing_add_reference import MixedAddReferenceConfig, load_reference_weights
from utils.run_common import build_balanced_pet_available


class ReferenceContracts(unittest.TestCase):
    def test_fixed_half_and_two_eval_states(self):
        with patch.object(sys,'argv',['reference']):
            cfg=MixedAddReferenceConfig.parse_arguments()
        self.assertEqual(cfg.train_batch_mode,'mixed')
        self.assertEqual(cfg.final_test_missing_rates,[0.,1.])
        for step in (0,1,100):
            state=build_balanced_pet_available(16,step,2023,torch.device('cpu'))
            self.assertEqual(int(state.sum()),8)
        with patch.object(sys,'argv',['reference','--train_batch_mode','full']):
            with self.assertRaises(SystemExit):MixedAddReferenceConfig.parse_arguments()

    def test_recorded_weights_win_over_ema_presence(self):
        m=nn.Linear(1,1,bias=False)
        raw={'weight':torch.tensor([[1.]])};ema={'weight':torch.tensor([[7.]])}
        ckpt={'model':raw,'model_ema':ema,'eval_weights':'raw'}
        self.assertEqual(load_reference_weights(m,ckpt),'raw')
        self.assertEqual(m.weight.item(),1.)
        ckpt['eval_weights']='ema'
        self.assertEqual(load_reference_weights(m,ckpt),'ema')
        self.assertEqual(m.weight.item(),7.)
        ckpt['model_ema']=None
        with self.assertRaises(ValueError):load_reference_weights(m,ckpt)


if __name__=='__main__':unittest.main()
