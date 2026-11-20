"""Text-conditioned selective residual refinement with a frozen text anchor."""
import torch
from torch import nn
from torch.nn import functional as F


def projection(a,b,k=1):
    return nn.Sequential(nn.Conv2d(a,b,k,padding=k//2,bias=False),
                         nn.GroupNorm(8,b),nn.GELU())


class ScaleRefiner(nn.Module):
    def __init__(self,channels,context_channels,width=32):
        super().__init__()
        self.local_dw=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.local=projection(2*channels,width)
        self.context=projection(context_channels,width)
        self.support=nn.Conv2d(width,1,1)
        self.value=projection(2*width,width,3)
        self.accept=nn.Sequential(nn.Conv2d(2,8,3,padding=1),nn.GELU(),nn.Conv2d(8,1,1))
        self.text_projection=nn.Linear(512,width,bias=False)
        self.channel=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        self.exit=nn.Conv2d(width,channels,1)
        nn.init.zeros_(self.exit.weight);nn.init.zeros_(self.exit.bias)

    def forward(self,feature,context_native,anchor,capture=False):
        local=self.local(torch.cat([feature,self.local_dw(feature)],dim=1))
        context0=self.context(context_native)
        context=F.interpolate(context0,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        ql_logit=self.support(local);qc_logit=self.support(context0)
        ql=ql_logit.float().sigmoid()
        qc=F.interpolate(qc_logit.float().sigmoid(),size=feature.shape[-2:],mode='bilinear',align_corners=False)
        gate=self.accept(torch.cat([ql,qc],1).to(local.dtype)).sigmoid()
        value=self.value(torch.cat([local,context],1))
        e=self.text_projection(anchor).unsqueeze(0).expand(len(feature),-1)
        channel=2*self.channel(torch.cat([value.mean((-2,-1)),e],1)).sigmoid()
        delta=self.exit(value*gate*channel[:,:,None,None])
        out={'feature':feature+delta,'local_logit':ql_logit,'context_logit':qc_logit}
        if capture:
            out['capture']={'local':local,'context':context,'local_score':ql,'context_score':qc,
                'score_difference':qc-ql,'acceptance':gate,'channel':channel,'value':value,'delta':delta}
        return out


class MultiScaleRefiner(nn.Module):
    def __init__(self,anchor,width=32,pool_size=4):
        super().__init__()
        anchor=torch.as_tensor(anchor,dtype=torch.float32).reshape(512)
        if not torch.isfinite(anchor).all() or abs(float(anchor.norm())-1)>1e-4:
            raise ValueError('Expected a finite, normalized512 BioMedCLIP anchor')
        self.register_buffer('text_anchor',anchor)
        self.pool_size=int(pool_size)
        channels=[64,128,320,512]
        self.blocks=nn.ModuleList([ScaleRefiner(c,channels[min(s+1,3)],width) for s,c in enumerate(channels)])

    def forward(self,features,capture=False):
        if len(features)!=4:raise ValueError('Four M1 fusion scales required')
        refined=[];scores=[];details={}
        for s,block in enumerate(self.blocks):
            native=features[s+1] if s<3 else F.adaptive_avg_pool2d(features[3],(self.pool_size,self.pool_size))
            out=block(features[s],native,self.text_anchor,capture)
            refined.append(out['feature']);scores.append((out['local_logit'],out['context_logit']))
            if capture:details[str(s+1)]=out['capture']
        return refined,scores,details
