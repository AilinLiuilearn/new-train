"""Fixed-direction target-scale spatial distribution regularizer.

One-pixel mass-conserving linear bins; no random sampling or mask thresholding
of predictions. This is a finite, discretized slicedW1 regularizer, not an
instance matcher. FP32 normalization uses log-probabilities.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TSSDA(nn.Module):
    def __init__(self,directions=16,scale_mode='directional'):
        super().__init__()
        if directions!=16:raise ValueError('Protocol fixes16 directions')
        if scale_mode not in ('directional','radius','none'):raise ValueError(scale_mode)
        self.directions=directions;self.scale_mode=scale_mode;self.geometry={};self.bin_plans={}

    def _geometry(self,h,w,device):
        key=(h,w,str(device))
        if key not in self.geometry:
            yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float64),
                torch.arange(w,device=device,dtype=torch.float64),indexing='ij')
            geometry=[]
            for k in range(self.directions):
                theta=k*math.pi/self.directions
                co,si=math.cos(theta),math.sin(theta)
                if abs(co)<1e-12:co=0.
                if abs(si)<1e-12:si=0.
                position=(xx*co+yy*si).flatten()
                shifted=position-position.min()
                rounded=shifted.round()
                shifted=torch.where((shifted-rounded).abs()<1e-10,rounded,shifted)
                low=shifted.floor().long()
                weight=(shifted-low).float()
                bins=int(low.max())+2
                joined=torch.cat([low,low+1])
                order=joined.argsort(stable=True)
                lengths=torch.bincount(joined,minlength=bins)
                self.bin_plans[id(low)]=(order,lengths)
                geometry.append((position.float(),low,weight,bins))
            self.geometry[key]=geometry
        return self.geometry[key]

    def project(self,mass,index,weight):
        # Fixed-order segmented sums avoid CUDA atomic scatter accumulation
        # in this loss. Backbone determinism remains a separate property.
        order,lengths=self.bin_plans[id(index)]
        values=torch.cat([mass*(1-weight),mass*weight],1)[:,order]
        return torch.segment_reduce(values,'sum',
            lengths=lengths[None].expand(len(mass),-1).contiguous(),axis=1)

    def forward(self,logits,target):
        if logits.shape!=target.shape or logits.ndim!=4 or logits.shape[1]!=1:
            raise ValueError('Expected matching B1HW logits/GT')
        with torch.autocast(device_type=logits.device.type,enabled=False):
            z=logits.float().flatten(1);y=target.detach().float().flatten(1)
            valid=y.sum(1)>0
            zero=logits.float().sum()*0
            if not valid.any():
                return zero,{'valid':0.,'scale_min':0.,'scale_mean':0.,'scale_max':0.,'w1_mean':0.}
            z=z[valid];y=y[valid];area=y.sum(1)
            # Exact unit normalization: adding epsilon to the denominator
            # alone would leave different total masses and corrupt W1.
            lp=F.logsigmoid(z);q=(lp-torch.logsumexp(lp,1,keepdim=True)).exp()
            gt=y/area[:,None]
            losses=[];scales=[];distances=[]
            for position,index,weight,bins in self._geometry(*target.shape[-2:],logits.device):
                qp=self.project(q,index,weight)
                qy=self.project(gt,index,weight)
                # One-pixel intervals; omit the endpoint, which has no
                # interval to its right. No image-width normalization.
                distance=(qp.cumsum(1)-qy.cumsum(1)).abs()[:,:-1].sum(1)
                mean=(gt*position).sum(1)
                variance=(gt*(position[None]-mean[:,None]).square()).sum(1)
                if self.scale_mode=='directional':scale=variance.sqrt().clamp_min(1.)
                elif self.scale_mode=='radius':scale=(area/math.pi).sqrt().clamp_min(1.)
                else:scale=torch.ones_like(area)
                losses.append(distance/scale);scales.append(scale);distances.append(distance)
            scale=torch.stack(scales)
            loss=torch.stack(losses).mean()
            return loss,{'valid':float(valid.sum()),'scale_min':float(scale.min()),
                'scale_mean':float(scale.mean()),'scale_max':float(scale.max()),
                'w1_mean':float(torch.stack(distances).mean())}
