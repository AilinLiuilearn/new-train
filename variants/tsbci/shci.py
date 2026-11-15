"""Text-conditioned channel selection + ST SoftSort routing + receiver Null.

SoftSort p=1 relaxation adapted from https://github.com/sprillo/softsort,
ICML2020. Actual forward uses exact unique Top-K indices; backward uses dense
soft selection matrices through BOTH gathering and reinsertion. Biased ST
gradients are surrogates, not derivatives of the hard ranking operator.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selection_matrix(scores,k,tau=.1,straight_through=True):
    values,index=scores.topk(k,dim=-1,sorted=True)
    hard=F.one_hot(index,num_classes=scores.shape[-1]).to(scores.dtype)
    if straight_through and scores.requires_grad:
        soft=(-(values.unsqueeze(-1)-scores.unsqueeze(1)).abs()/tau).softmax(-1)
        matrix=hard+(soft-soft.detach())
    else:
        soft=hard;matrix=hard
    return matrix,index,soft


class ChannelBlock(nn.Module):
    def __init__(self,channels,dim=32,ratio=.5,selection_tau=.1):
        super().__init__();self.channels=channels;self.k=math.ceil(channels*ratio);self.selection_tau=selection_tau
        self.ct_descriptor=nn.Sequential(nn.Linear(16,dim),nn.LayerNorm(dim))
        self.pet_descriptor=nn.Sequential(nn.Linear(16,dim),nn.LayerNorm(dim))
        # ONE scorer for both modalities at this scale; no cosine or semantic alignment claim.
        self.selector=nn.Sequential(nn.Linear(dim+512,64),nn.GELU(),nn.Linear(64,1))
        self.ct_q=nn.Linear(dim,dim,bias=False);self.ct_k=nn.Linear(dim,dim,bias=False)
        self.pet_q=nn.Linear(dim,dim,bias=False);self.pet_k=nn.Linear(dim,dim,bias=False)
        self.ct_v_scale=nn.Parameter(torch.ones(channels));self.ct_v_bias=nn.Parameter(torch.zeros(channels))
        self.pet_v_scale=nn.Parameter(torch.ones(channels));self.pet_v_bias=nn.Parameter(torch.zeros(channels))
        self.ct_null=nn.Linear(dim,1);self.pet_null=nn.Linear(dim,1)
        for layer in [self.ct_null,self.pet_null]:
            nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,math.log(self.k))
        raw=math.log((.2-.05)/(1.-.2))
        self.ct_temperature_raw=nn.Parameter(torch.tensor(raw));self.pet_temperature_raw=nn.Parameter(torch.tensor(raw))
        # Diagonal output maps retain recipient-specific Null behavior. A dense
        # post-scatter1x1 could inject another receiver's message into a rejected channel.
        self.ct_out=nn.Conv2d(channels,channels,1,groups=channels,bias=False)
        self.pet_out=nn.Conv2d(channels,channels,1,groups=channels,bias=False)
        nn.init.zeros_(self.ct_out.weight);nn.init.zeros_(self.pet_out.weight)

    def forward(self,c,p,text_features,capture=False):
        if c.shape!=p.shape or c.shape[1]!=self.channels:raise ValueError('Aligned paired input required')
        b,d,h,w=c.shape
        with torch.autocast(device_type=c.device.type,enabled=False):
            hc=self.ct_descriptor(F.adaptive_avg_pool2d(c.float(),(4,4)).flatten(2))
            hp=self.pet_descriptor(F.adaptive_avg_pool2d(p.float(),(4,4)).flatten(2))
            tc=text_features[0].reshape(1,1,512).expand(b,d,-1)
            tp=text_features[1].reshape(1,1,512).expand(b,d,-1)
            rc=self.selector(torch.cat([hc,tc],-1)).squeeze(-1)
            rp=self.selector(torch.cat([hp,tp],-1)).squeeze(-1)
            if capture and rc.requires_grad:rc.retain_grad();rp.retain_grad()
            st=self.training and torch.is_grad_enabled()
            sc,ic,softc=selection_matrix(rc,self.k,self.selection_tau,st)
            sp,ip,softp=selection_matrix(rp,self.k,self.selection_tau,st)
            dc=sc@hc;dp=sp@hp
            qc=F.normalize(self.ct_q(dc),dim=-1,eps=1e-6);kc=F.normalize(self.ct_k(dc),dim=-1,eps=1e-6)
            qp=F.normalize(self.pet_q(dp),dim=-1,eps=1e-6);kp=F.normalize(self.pet_k(dp),dim=-1,eps=1e-6)
            tempc=.05+.95*self.ct_temperature_raw.sigmoid();tempp=.05+.95*self.pet_temperature_raw.sigmoid()
            realc=qc@kp.transpose(-2,-1)/tempc;realp=qp@kc.transpose(-2,-1)/tempp
            ac=torch.cat([realc,self.ct_null(qc)],-1).softmax(-1)
            ap=torch.cat([realp,self.pet_null(qp)],-1).softmax(-1)
            cv=c.float()*self.ct_v_scale.reshape(1,d,1,1)+self.ct_v_bias.reshape(1,d,1,1)
            pv=p.float()*self.pet_v_scale.reshape(1,d,1,1)+self.pet_v_bias.reshape(1,d,1,1)
            # Null value is exactly zero, excluded from these value products.
            mc=ac[...,:self.k]@(sp@pv.flatten(2))
            mp=ap[...,:self.k]@(sc@cv.flatten(2))
            # ST matrices are used for reinsertion too, not merely hard scatter.
            nativec=(sc.transpose(-2,-1)@mc).reshape(b,d,h,w)
            nativep=(sp.transpose(-2,-1)@mp).reshape(b,d,h,w)
            delta_c=self.ct_out(nativec);delta_p=self.pet_out(nativep)
            co=(c.float()+delta_c).to(c.dtype);po=(p.float()+delta_p).to(p.dtype)
        info={}
        if capture:
            info={k:v.detach() for k,v in dict(ct_indices=ic,pet_indices=ip,ct_relation=ac,pet_relation=ap,
                ct_null_mass=ac[...,-1],pet_null_mass=ap[...,-1],ct_donor_usage=ac[...,:self.k].mean(1),pet_donor_usage=ap[...,:self.k].mean(1),
                ct_delta=delta_c,pet_delta=delta_p,ct_actual_delta=co.float()-c.float(),pet_actual_delta=po.float()-p.float(),
                ct_output_gain=self.ct_out.weight,pet_output_gain=self.pet_out.weight,
                ct_temperature=tempc,pet_temperature=tempp,ct_soft_selection=softc,pet_soft_selection=softp).items()}
            # Keep scores live for AMP-correct gradient coverage diagnostics.
            info['ct_score']=rc;info['pet_score']=rp
        return co,po,info


class TSBCI(nn.Module):
    def __init__(self,text_features,channels=(64,128,320,512)):
        super().__init__()
        if text_features.shape!=(2,512):raise ValueError('Two frozen pooled512 BioMedCLIP role vectors required')
        self.register_buffer('text_features',F.normalize(text_features.detach().float().clone(),dim=-1))
        self.blocks=nn.ModuleList([ChannelBlock(c) for c in channels])

    def forward(self,c,p,capture=False):
        if len(c)!=4 or len(p)!=4:raise ValueError('Four scales required')
        co=[];po=[];diag={}
        for i,(block,a,b) in enumerate(zip(self.blocks,c,p)):
            aa,bb,info=block(a,b,self.text_features,capture);co.append(aa);po.append(bb)
            if capture:diag[str(i+1)]=info
        return co,po,diag
