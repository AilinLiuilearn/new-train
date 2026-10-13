"""Only the changed reference-entry contracts; no real-data training."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_full_missing_add_reference import (MixedAddReferenceConfig, load_reference_weights,
    build_reference_pet_available, validate_ratio_config)
from utils.run_common import build_balanced_pet_available


class ReferenceContracts(unittest.TestCase):
    def test_fixed_half_and_two_eval_states(self):
        with patch.object(sys,'argv',['reference']):
            cfg=MixedAddReferenceConfig.parse_arguments()
        self.assertEqual(cfg.train_batch_mode,'mixed')
        self.assertEqual(cfg.final_test_missing_rates,[0.,1.])
        self.assertEqual((cfg.full_ratio,cfg.full_loss_weight,cfg.missing_loss_weight),(.5,.5,.5))
        for step in (0,1,100):
            state=build_balanced_pet_available(16,step,2023,torch.device('cpu'))
            self.assertEqual(int(state.sum()),8)
            self.assertTrue(torch.equal(state,build_reference_pet_available(16,step,2023,
                                                        torch.device('cpu'),.5)))
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

    def test_seventy_percent_cumulative_counts_and_determinism(self):
        total=0
        for step in range(1088):
            s=build_reference_pet_available(16,step,2023,torch.device('cpu'),.7)
            n=int(s.sum());self.assertIn(n,(11,12));total+=n
            self.assertLess(abs(total-16*.7*(step+1)),1.000001)
        self.assertEqual(total,12185)
        s=build_reference_pet_available(16,100,2023,torch.device('cpu'),.7)
        self.assertTrue(torch.equal(s,build_reference_pet_available(16,100,2023,
                                                                  torch.device('cpu'),.7)))
        with patch.object(sys,'argv',['reference','--full_ratio','.7',
                                     '--full_loss_weight','.7','--missing_loss_weight','.3']):
            cfg=MixedAddReferenceConfig.parse_arguments()
        validate_ratio_config(cfg)
        for value in (0,1,.01,float('nan')):
            with self.assertRaises(ValueError):
                build_reference_pet_available(16,0,2023,torch.device('cpu'),value)
        cfg.full_loss_weight=.8
        with self.assertRaises(ValueError):validate_ratio_config(cfg)

    def test_weighted_loss_and_gradients_use_requested_weights(self):
        from tasks.mdt_seg import MDTSegTeacher
        from utils.seg_losses import BCEDiceLoss
        class ToyDual(nn.Module):
            def __init__(self):
                super().__init__();self.a=nn.Parameter(torch.tensor(.3));self.b=nn.Parameter(torch.tensor(.4))
            def forward(self,ct,pet,pet_available,forward_mode):
                state=pet_available.view(-1,1,1,1)
                return {'logits':self.a*ct+self.b*pet*state}
        task=object.__new__(MDTSegTeacher)
        task.model=ToyDual();task.device=torch.device('cpu');task.criterion=BCEDiceLoss()
        batch={'ct':torch.randn(16,1,4,4),'pet':torch.randn(16,1,4,4),
               'mask':(torch.rand(16,1,4,4)>.7).float()}
        state=build_reference_pet_available(16,0,2023,task.device,.7)
        for wf,wm in ((.5,.5),(.7,.3)):
            loss,logits,_,stats=task.train_step_mixed(batch,state,
                                            full_loss_weight=wf,missing_loss_weight=wm)
            lf,_=task.criterion(logits[state==1],batch['mask'][state==1])
            lm,_=task.criterion(logits[state==0],batch['mask'][state==0])
            expected=wf*lf+wm*lm
            torch.testing.assert_close(loss,expected,rtol=0,atol=0)
            grad=torch.autograd.grad(loss,tuple(task.model.parameters()),retain_graph=True)
            ref=torch.autograd.grad(expected,tuple(task.model.parameters()))
            for a,b in zip(grad,ref):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(stats['num_full'],11)
            self.assertEqual(stats['num_missing'],5)


if __name__=='__main__':unittest.main()
