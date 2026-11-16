"""Exact normalized pixel projection and GT region pooling, in bounded chunks."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SemanticRegularizer(nn.Module):
    def __init__(self, anchors, margin=.2, chunk_size=1024):
        super().__init__()
        self.register_buffer('anchors',F.normalize(anchors.detach().float(),dim=-1))
        self.projections=nn.ModuleList(nn.Linear(64,512) for _ in range(2))
        self.margin=margin;self.chunk_size=chunk_size

    @staticmethod
    def region_sum(x,occupancy,weight,bias):
        with torch.autocast(device_type=x.device.type,enabled=False):
            z=F.normalize(F.linear(x.float(),weight.float(),bias.float()),dim=-1,eps=1e-6)
            w=occupancy.float().unsqueeze(-1)
            return (z*w).sum(1),(z*(1-w)).sum(1)

    def region_vectors(self,features,target,index):
        with torch.autocast(device_type=features.device.type,enabled=False):
            g=F.interpolate(target.float(),size=features.shape[-2:],mode='area').flatten(1)
        x=features.flatten(2).transpose(1,2)
        fg=torch.zeros(len(x),512,device=x.device,dtype=torch.float32);bg=torch.zeros_like(fg)
        layer=self.projections[index]
        for start in range(0,x.shape[1],self.chunk_size):
            inputs=(x[:,start:start+self.chunk_size],g[:,start:start+self.chunk_size],layer.weight,layer.bias)
            if self.training and torch.is_grad_enabled():
                f,b=checkpoint(self.region_sum,*inputs,use_reentrant=False,preserve_rng_state=False)
            else:f,b=self.region_sum(*inputs)
            fg=fg+f;bg=bg+b
        with torch.autocast(device_type=x.device.type,enabled=False):
            fmass=g.sum(1);bmass=(1-g).sum(1)
            valid=(fmass>1e-6)&(bmass>1e-6)
            fg=F.normalize(fg/fmass.clamp_min(1e-6)[:,None],dim=-1,eps=1e-6)
            bg=F.normalize(bg/bmass.clamp_min(1e-6)[:,None],dim=-1,eps=1e-6)
        return fg,bg,valid

    def forward(self,features,target):
        terms=[];stats={}
        for index,name in enumerate(['ct','pet']):
            fg,bg,valid=self.region_vectors(features[index],target,index)
            sf=(fg*self.anchors[index]).sum(-1);sb=(bg*self.anchors[index]).sum(-1)
            hinge=(self.margin-sf+sb).clamp_min(0)
            denom=valid.float().sum().clamp_min(1)
            loss=(hinge*valid).sum()/denom;terms.append(loss)
            stats[name+'_sem_fg_cos']=((sf.detach()*valid).sum()/denom)
            stats[name+'_sem_bg_cos']=((sb.detach()*valid).sum()/denom)
            stats[name+'_sem_active_fraction']=((hinge.detach().gt(0)*valid).sum()/denom)
            stats[name+'_sem_valid_fraction']=valid.float().mean()
            stats[name+'_sem_loss']=loss.detach()
        return sum(terms),stats
