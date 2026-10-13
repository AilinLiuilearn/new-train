"""Necessary contracts only; synthetic inputs, no real training/pretrained download."""
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.structure_response_fusion import StructureResponseFusion
from models.full_structure_response import FullStructureResponseModel
from utils.structure_response_visualization import visualize_fusion


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,1);self.calls=0
        self.feature_info=type('Info',(),{'channels':lambda _:[8]*4})()
    def forward(self,x):
        self.calls+=1
        return [F.adaptive_avg_pool2d(self.conv(x),size) for size in (16,8,4,2)]


def fake_model(enabled=True):
    with patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
        return FullStructureResponseModel(pretrained=False,decoder_channels=(8,8,8,8),fusion_enabled=enabled)


class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(7)

    def test_ema_copies_fixed_kernels_instead_of_averaging_them(self):
        from tasks.full_structure_response import StructureResponseEMA
        m=fake_model();ema=StructureResponseEMA(m,decay=.9,warmup=False)
        with torch.no_grad():
            ema.model.fusion.stages[0].ct_see.kernel_x.add_(.25)
        ema.update(m)
        self.assertTrue(torch.equal(m.fusion.stages[0].ct_see.kernel_x,
                                    ema.model.fusion.stages[0].ct_see.kernel_x))

    def test_all_scales_exact_add_init_frozen_kernels_and_two_stage_gradients(self):
        m=StructureResponseFusion().train()
        ct=[torch.randn(2,c,h,w,requires_grad=True) for c,(h,w) in
            zip((64,128,320,512),((12,16),(6,8),(3,4),(2,3)))]
        pet=[torch.randn_like(x,requires_grad=True) for x in ct]
        out,info=m(ct,pet,return_diagnostics=True,return_maps=True)
        for a,c,p in zip(out,ct,pet):self.assertTrue(torch.equal(a,c+p))
        self.assertEqual(sum(p.numel() for p in m.parameters()),4637536)
        sum(x.square().mean() for x in out).backward()
        for stage in m.stages:
            self.assertGreater(stage.ct_output.weight.grad.abs().sum(),0)
            self.assertGreater(stage.pet_output.weight.grad.abs().sum(),0)
            self.assertIsNone(stage.ct_see.kernel_x.grad)
            self.assertTrue(torch.equal(stage.ct_see.kernel_x[0,0],torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]])))
            with torch.no_grad():stage.ct_output.weight.add_(.01);stage.pet_output.weight.add_(.01)
        m.zero_grad();out=m(ct,pet);sum(x.square().mean() for x in out).backward()
        for stage in m.stages:
            for name in ('ct_projection','ct_see','pet_lrce','ca_ct','ca_pet','ct_mix','pet_mix','ct_sae','pet_sae'):
                grads=[p.grad for p in getattr(stage,name).parameters() if p.grad is not None]
                self.assertTrue(grads,name);self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads),0,name)
        self.assertFalse(any(v.requires_grad for v in info[0]['maps'].values()))

    def test_diagnostics_do_not_change_forward_and_amp_is_finite(self):
        m=StructureResponseFusion(channels=(64,)).eval()
        c,p=torch.randn(2,64,9,11),torch.randn(2,64,9,11)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            plain=m([c],[p]);measured,info=m([c],[p],return_diagnostics=True)
        self.assertTrue(torch.equal(plain[0],measured[0]))
        self.assertTrue(torch.isfinite(measured[0]).all())
        self.assertEqual(float(info[0]['ct_residual_ratio']),0)
        with self.assertRaises(ValueError):m([c],[p[:,:,:,:10]])

    def test_full_wrapper_and_disabled_add_equivalence_reject_missing_before_encoders(self):
        m=fake_model().eval();ct,pet=torch.randn(2,3,32,32),torch.randn(2,3,32,32)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(ct,pet)['logits'],m(ct,pet,use_add=True)['logits']))
        disabled=fake_model(False).eval()
        state={k:v for k,v in m.state_dict().items() if not k.startswith('fusion.')}
        disabled.load_state_dict(state,strict=True)
        self.assertTrue(torch.equal(disabled(ct,pet)['logits'],m(ct,pet,use_add=True)['logits']))
        calls=m.enc_ct.calls,m.enc_pet.calls
        with self.assertRaises(ValueError):m(ct,pet=None,forward_mode='missing')
        self.assertEqual(calls,(m.enc_ct.calls,m.enc_pet.calls))

    def test_visualization_keeps_models_rng_and_exact_numeric_maps(self):
        m=fake_model().train()
        for stage in m.fusion.stages:
            with torch.no_grad():stage.ct_output.weight.add_(.01);stage.pet_output.weight.add_(.01)
        gt=torch.zeros(1,1,32,32);gt[:,:,12:16,12:16]=1
        batch={'ct':torch.randn(1,3,32,32),'pet':torch.randn(1,3,32,32),'mask':gt,'image_id':['tiny']}
        original=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            cfg=type('Cfg',(),{'fusion_vis_samples':1,'checkpoint_dir':tmp,'norm_mode':'imagenet','fusion_enabled':True})()
            files=visualize_fusion(m,[batch],cfg,torch.device('cpu'),1)
            self.assertEqual(len(files),4)
            maps=np.load(next(p for p in files if p.endswith('.npz')))
            self.assertIn('s1_ct_ca',maps);self.assertIn('s4_delta_pet',maps)
            self.assertEqual(maps['s1_ct_ca'].ndim,3)
            self.assertTrue(np.isfinite(maps['fusion_prob']).all())
            self.assertGreater(maps['s1_delta_ct'].mean(),0)
            import shutil
            shutil.copyfile(next(p for p in files if p.endswith('_features.png')),'/tmp/structure_response_feature_smoke.png')
        self.assertTrue(m.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in original.items():self.assertTrue(torch.equal(value,m.state_dict()[key]),key)

    def test_independent_runner_save_resume_and_checkpoint_visualization(self):
        import run_full_structure_response_fusion as runner
        from tasks.mdt_seg import MDTSegTeacher
        gt=torch.zeros(2,1,32,32);gt[:,:,8:20,8:20]=1
        batch={'ct':torch.randn(2,3,32,32),'pet':torch.randn(2,3,32,32),'mask':gt,'image_id':['a','b']}
        loader=[batch]
        with tempfile.TemporaryDirectory() as tmp:
            args=['x','--pretrained','False','--epochs','1','--batch_size','2','--num_workers','0',
                  '--mixed_precision','False','--allow_cpu','True','--hd95_mode','surface',
                  '--decoder_channels','8','8','8','8','--checkpoint_root',tmp,'--hash','contract','--ema_start_epoch','0']
            with patch('sys.argv',args),patch.object(runner,'_loaders',return_value=(loader,loader,loader)),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                runner.main()
            root=os.path.join(tmp,'MDT','contract');last=os.path.join(root,'ckpt.last.pth.tar');best=os.path.join(root,'ckpt.best_full.pth.tar')
            ckpt=MDTSegTeacher.load_state_dicts(last)
            self.assertEqual(ckpt['epoch'],1);self.assertEqual(ckpt['ema_updates'],1)
            with patch('sys.argv',args+['--resume',last]),patch.object(runner,'_loaders',return_value=(loader,loader,loader)),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                runner.main()
            with patch('sys.argv',args+['--eval_checkpoint',best,'--hash','inspect','--fusion_visualize','True','--fusion_vis_samples','1']),\
                 patch.object(runner,'_loaders',return_value=(loader,loader,loader)),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                runner.main()
            with open(os.path.join(tmp,'MDT','inspect','final_test_full.json')) as f:result=json.load(f)
            self.assertEqual(result['weights'],'ema')
            self.assertTrue(os.path.isdir(os.path.join(tmp,'MDT','inspect','fusion_visualizations','epoch_001')))


if __name__=='__main__':unittest.main()
