"""Joint role experts, followed by a frozen base and selective residual correction."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from variants.tsbci.model import SHCIPETCT
from models.components.group_unet_decoder import UNetStyleDecoder


def decode_with_feature(decoder,f,size):
    x1,x2,x3,x4=f
    up=lambda x,y:F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
    d4=decoder.proj4(x4);s3=decoder.proj3(x3)
    d3=decoder.fuse3(torch.cat([up(d4,s3),s3],1));s2=decoder.proj2(x2)
    d2=decoder.fuse2(torch.cat([up(d3,s2),s2],1));s1=decoder.proj1(x1)
    feature=decoder.fuse1(torch.cat([up(d2,s1),s1],1))
    return F.interpolate(decoder.seg_head(feature),size=size,mode='bilinear',align_corners=False),feature


class JointModel(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=dict(cfg)
        self.base=SHCIPETCT(cfg,arm='m1',load_generic=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.get('expert_seed',2133))
            self.ct_expert=UNetStyleDecoder(norm='GN')
            self.pet_expert=UNetStyleDecoder(norm='GN')

    def import_m1(self,bundle):
        if bundle.get('eval_weights')!='ema' or 'model_ema' not in bundle:
            raise ValueError('Expected the selected historical M1 EMA bundle')
        self.base.load_state_dict(bundle['model_ema'],strict=True)

    def forward(self,ct,pet,with_feature=False,capture=False):
        c=self.base.encode_ct(ct)
        p=self.base.pet_align(self.base.enc_pet(self.base._input(pet)))
        fused=self.base.fusion(c,p)
        if with_feature:z,feature=decode_with_feature(self.base.decoder_full,fused,ct.shape[-2:])
        else:z=self.base.decoder_full(fused,ct.shape[-2:])['logits'];feature=None
        def decode(dec,features):
            fn=lambda *xs:dec(list(xs),ct.shape[-2:])['logits']
            if self.training and self.cfg.get('checkpoint_experts',True):
                return checkpoint(fn,*features,use_reentrant=False)
            return fn(*features)
        out={'logits':z,'ct_distance':decode(self.ct_expert,c).float().tanh(),
             'pet_logits':decode(self.pet_expert,p),'feature':feature}
        if capture:out['scales']={'ct':[v.detach() for v in c],'pet':[v.detach() for v in p],
                                 'fused':[v.detach() for v in fused]}
        return out


def block(a,b,k=3):
    return nn.Sequential(nn.Conv2d(a,b,k,padding=k//2),nn.GroupNorm(8 if b>=32 else 4,b),nn.SiLU())


class Corrector(nn.Module):
    def __init__(self):
        super().__init__();self.fusion_project=block(64,32,1)
        self.pet_project=block(1,16,1);self.ct_project=block(1,16,1)
        self.consensus=block(32,32)
        self.gate=nn.Sequential(block(65,32),nn.Conv2d(32,1,1))
        self.residual=nn.Sequential(block(65,32),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self,feature,z,pet_logits,distance):
        size=feature.shape[-2:]
        pool=lambda x:F.adaptive_avg_pool2d(x.float(),size)
        h=self.consensus(torch.cat([self.pet_project(pool(pet_logits.float().sigmoid())),
                                    self.ct_project(pool(distance))],1))
        f=self.fusion_project(feature);zsmall=pool(z)
        gate_small=self.gate(torch.cat([f,zsmall,h.detach()],1))
        delta_small=self.residual(torch.cat([f,zsmall,h],1))
        up=lambda x:F.interpolate(x.float(),size=z.shape[-2:],mode='bilinear',align_corners=False)
        gate_logit=up(gate_small);a=gate_logit.sigmoid();delta=up(delta_small)
        return {'logits':z.float()+a*delta,'base_logits':z.float(),'gate_logits':gate_logit,
                'gate':a,'delta':delta,'change':a*delta,'consensus':h}


class CorrectedModel(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base.requires_grad_(False).eval();self.corrector=Corrector()

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,ct,pet,corrector=None,capture=False):
        # Frozen reference and error supervision use FP32, matching original validation.
        with torch.no_grad(),torch.autocast(device_type=ct.device.type,enabled=False):
            o=self.base(ct.float(),pet.float(),with_feature=True,capture=capture)
        r=(self.corrector if corrector is None else corrector)(o['feature'],o['logits'],o['pet_logits'],o['ct_distance'])
        r.update(ct_distance=o['ct_distance'],pet_logits=o['pet_logits'])
        if capture:r['scales']=o['scales']
        return r
