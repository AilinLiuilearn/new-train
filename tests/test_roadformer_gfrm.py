"""Bounded source-parity, integration, EMA and visualization checks; CPU only."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.roadformer_gfrm import RoadFormerGFRMFusion, RoadFormerGFRMStage
from models.full_roadformer_gfrm import FullRoadFormerGFRMModel
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,8,1)
        self.feature_info = type('Info',(),{'channels':lambda _:[8]*4})()
        self.calls = 0

    def forward(self,x):
        self.calls += 1
        return [F.adaptive_avg_pool2d(self.conv(x), s) for s in (16,8,4,2)]


def build_model(enabled=True):
    with patch('models.dual_shared_add_baseline.create_feature_backbone',
               lambda *a,**k: FakeEncoder()):
        return FullRoadFormerGFRMModel(gfrm_enabled=enabled, image_size_2d=64,
                    decoder_channels=(8,8,8,8), pretrained=False)


def author_classes():
    # Compile exact author class bodies; shims reproduce their framework wrappers.
    path = Path(__file__).resolve().parents[1]/'third_party/RoadFormer/author_roadformer2neck.py'
    root = ast.parse(path.read_text())
    root.body = [node for node in root.body if isinstance(node,ast.ClassDef)
                 and node.name in ('Scale','GFFM','FFRM')]
    def conv_module(in_chan,out_chan,kernel_size,bias=False,norm_cfg=None):
        layers = [nn.Conv2d(in_chan,out_chan,kernel_size,bias=bias)]
        if norm_cfg:
            layers.append(nn.GroupNorm(norm_cfg['num_groups'],out_chan))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    namespace = dict(torch=torch,nn=nn,F=F,BaseModule=nn.Module,ConvModule=conv_module,
                     caffe2_xavier_init=lambda m,**k:None)
    exec(compile(root, str(path),'exec'),namespace)
    return namespace


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_author_forward_and_gradients_nonzero_bidirectional_scales(self):
        ns=author_classes()
        ref_cross=ns['GFFM']((5,7),64).double()
        ref_ff=ns['FFRM'](128,128,norm={'type':'GN','num_groups':32}).double()
        own=RoadFormerGFRMStage(64,(5,7)).double()
        with torch.no_grad():
            own.cross.gammax.scale.fill_(.23)
            own.cross.gammay.scale.fill_(-.17)
            own.cross.norm.weight.uniform_(.8,1.2)
            own.cross.norm.bias.uniform_(-.1,.1)
        ref_cross.load_state_dict(own.cross.state_dict(),strict=True)
        ref_ff.conv_atten.load_state_dict(own.recalibrate.conv_atten.state_dict(),strict=True)
        c=torch.randn(2,64,5,7,dtype=torch.float64,requires_grad=True)
        p=torch.randn_like(c,requires_grad=True)
        c2=c.detach().clone().requires_grad_(); p2=p.detach().clone().requires_grad_()
        pair=ref_ff(ref_cross(torch.cat((c,p),1)))
        expected=sum(pair.chunk(2,1))
        actual=own(c2,p2)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        expected.square().mean().backward(); actual.square().mean().backward()
        torch.testing.assert_close(c.grad,c2.grad,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(p.grad,p2.grad,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(ref_cross.gammax.scale.grad,
                                   own.cross.gammax.scale.grad,rtol=1e-11,atol=1e-11)

    def test_four_scales_shape_backward_diagnostics_and_amp(self):
        sizes=((16,16),(8,8),(4,4),(2,2))
        model=RoadFormerGFRMFusion(feature_sizes=sizes)
        c=[torch.randn(2,d,*hw,requires_grad=True) for d,hw in zip((64,128,320,512),sizes)]
        p=[torch.randn_like(x,requires_grad=True) for x in c]
        plain=model(c,p)
        out,info=model(c,p,True,True)
        for i,(x,y) in enumerate(zip(out,plain),1):
            self.assertTrue(torch.equal(x,y))
            self.assertEqual(x.shape,c[i-1].shape)
            self.assertFalse(torch.equal(x,c[i-1]+p[i-1]))
            self.assertEqual(info[f's{i}']['delta_ct_ratio'],0)
            self.assertFalse(any(x.requires_grad for x in info[f's{i}']['maps'].values()))
        sum(x.square().mean() for x in out).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        for stage in model.stages:
            self.assertGreater(abs(stage.cross.gammax.scale.grad.item()),0)
            self.assertGreater(abs(stage.cross.gammay.scale.grad.item()),0)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            self.assertTrue(all(torch.isfinite(x).all() for x in model(c,p)))
        with self.assertRaises(ValueError):
            model([x[:,:,:,:-1] for x in c],p)

    def test_disabled_equals_base_and_full_only_rejects_before_encoder(self):
        torch.manual_seed(19); disabled=build_model(False).eval()
        torch.manual_seed(19)
        with patch('models.dual_shared_add_baseline.create_feature_backbone',
                   lambda *a,**k:FakeEncoder()):
            base=DualSharedAddPETCTBaseline(pretrained=False,decoder_channels=(8,8,8,8)).eval()
        c,p=torch.randn(2,3,64,64),torch.randn(2,3,64,64)
        self.assertTrue(torch.equal(disabled(c,p)['logits'],base(c,p,forward_mode='full')['logits']))
        enabled=build_model(True)
        before=enabled.enc_ct.calls,enabled.enc_pet.calls
        for args in ({'forward_mode':'missing'},{'pet_available':[1,1]}, {'pet':None}):
            with self.assertRaises(ValueError):
                enabled(c,**({'pet':p}|args))
        self.assertEqual(before,(enabled.enc_ct.calls,enabled.enc_pet.calls))
        loss=enabled(c,p)['logits'].square().mean()
        loss.backward()
        for name in ('enc_ct','enc_pet','ct_align','pet_align','fusion','decoder'):
            self.assertTrue(all(x.grad is not None for x in getattr(enabled,name).parameters()))
        opt=torch.optim.AdamW(enabled.parameters(),lr=1e-4)
        opt_ids={id(x) for group in opt.param_groups for x in group['params']}
        self.assertTrue(all(id(x) in opt_ids for x in enabled.fusion.parameters()))
        opt.step()
        other=build_model(True); other.load_state_dict(enabled.state_dict(),strict=True)
        enabled.eval();other.eval()
        torch.testing.assert_close(enabled(c,p)['logits'],other(c,p)['logits'],rtol=0,atol=0)

    def test_ema_final_weight_selection_and_visualization_state(self):
        from utils.ema import ModelEMA
        from utils.roadformer_gfrm_visualization import visualize_gfrm
        from run_full_roadformer_gfrm import load_selected_checkpoint
        m=build_model(True).train();ema=ModelEMA(m,decay=.9,warmup=False)
        with torch.no_grad():m.fusion.stages[0].cross.gammax.scale.add_(.1)
        ema.update(m)
        cfg=type('Cfg',(),dict(experiment_type='full_roadformer_gfrm',gfrm_enabled=True,
                              image_size_2d=64,ct_backbone='convnextv2_nano',pet_backbone='mit_b1',
                              decoder_channels=(8,8,8,8),fusion_vis_samples=1,norm_mode='imagenet'))()
        config={name:getattr(cfg,name) for name in ('experiment_type','gfrm_enabled','image_size_2d',
                'ct_backbone','pet_backbone','decoder_channels')}
        ckpt={'config':config,'model':m.state_dict(),'model_ema':ema.state_dict(),'eval_weights':'raw'}
        target=build_model(True)
        self.assertEqual(load_selected_checkpoint(target,ckpt,cfg),'raw')
        self.assertAlmostEqual(target.fusion.stages[0].cross.gammax.scale.item(),.1,places=7)
        ckpt['eval_weights']='ema'
        self.assertEqual(load_selected_checkpoint(target,ckpt,cfg),'ema')
        ckpt['config']=config|{'gfrm_enabled':False}
        with self.assertRaises(ValueError):load_selected_checkpoint(target,ckpt,cfg)
        batch={'ct':torch.randn(1,3,64,64),'pet':torch.randn(1,3,64,64),
               'mask':torch.zeros(1,1,64,64)}
        states=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            cfg.checkpoint_dir=folder
            files=visualize_gfrm(m,[batch],cfg,torch.device('cpu'),1,'raw')
            self.assertTrue(any(x.endswith('.png') for x in files))
            data=np.load(next(x for x in files if x.endswith('.npz')))
            self.assertEqual(data['s1_ct'].shape,(16,16))
            self.assertTrue(np.isfinite(data['s1_fused']).all())
        self.assertTrue(m.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in states.items():self.assertTrue(torch.equal(value,m.state_dict()[name]))

    def test_runner_one_epoch_enabled_raw_and_disabled_ema(self):
        import csv
        import json
        import sys
        import run_full_roadformer_gfrm as runner
        gt=torch.zeros(2,1,64,64);gt[:,:,22:36,24:38]=1
        batch={'ct':torch.randn(2,3,64,64),'pet':torch.randn(2,3,64,64),'mask':gt}
        for enabled, start, expected in ((True,3,'raw'),(False,0,'ema')):
            with tempfile.TemporaryDirectory() as folder:
                args=['runner','--pretrained','False','--epochs','1',
                      '--image_size_2d','64','--batch_size','2','--num_workers','0',
                      '--mixed_precision','False','--allow_cpu','True','--hd95_mode','surface',
                      '--decoder_channels','8','8','8','8','--checkpoint_root',folder,
                      '--hash','contract','--ema_enabled','True','--ema_start_epoch',str(start),
                      '--gfrm_enabled',str(enabled),'--fusion_visualize','False']
                with patch.object(sys,'argv',args), \
                     patch.object(runner,'_loaders',lambda cfg:([batch],[batch],[batch])), \
                     patch.object(runner,'build_gfrm_model',lambda cfg:{'model':build_model(cfg.gfrm_enabled)}):
                    runner.main()
                result_path=next(Path(folder).rglob('final_test_full.json'))
                result=json.loads(result_path.read_text())
                self.assertEqual(result['weights'],expected)
                rows=list(csv.DictReader((result_path.parent/'train_log.csv').open()))
                self.assertEqual(rows[0]['eval_weights'],expected)
                self.assertAlmostEqual(float(rows[0]['val_full_dice']),result['dice'],places=5)
                if not enabled:
                    self.assertEqual(rows[0]['gfrm_s1_gamma_ct'],'disabled')


if __name__ == '__main__':
    unittest.main()
