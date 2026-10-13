"""Full-only structure/response complementary residual fusion, PyTorch only.

Input/output: four aligned BCHW CT/PET pairs, channels 64/128/320/512.
F=(C+delta_C)+(P+delta_P); only the two final projections are zero initialized.

Provenance: Bai-Xuecheng/DRWKV Block.py (blob
c61504d8b410dd33b740f96c9117a2df58844f6d), CrossAttention, IEL and Scharr.
CA channel layout, Q/K spatial L2 normalization, learnable multiplicative
temperature and complete IEL/SAE expansion/split/tanh/product are retained.
Adaptations: fixed Scharr BUFFER kernels (released code did not freeze its
Scharr-initialized parameters), half-width internal features, two independent
CA/SAE directions, PET contrast stem, raw CT/PET direct paths and zero output
projections. No CFEM BCHW spatial matmuls, RWKV/CUDA extension, FFT or new loss.
Diagnostics are detached measurements, not calibrated lesion/confidence maps.
"""
from typing import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """DRWKV channels-first LayerNorm; normalize channels at every pixel."""
    def __init__(self,channels:int,eps:float=1e-6):
        super().__init__();self.eps=eps
        self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))

    def forward(self,x:Tensor)->Tensor:
        # Stable moments in FP32 under AMP, restoring the feature dtype.
        z=x.float();mean=z.mean(1,keepdim=True);var=(z-mean).square().mean(1,keepdim=True)
        return (((z-mean)*torch.rsqrt(var+self.eps))*self.weight[None,:,None,None]
                +self.bias[None,:,None,None]).to(x.dtype)


class FixedScharrSEE(nn.Module):
    def __init__(self,channels:int):
        super().__init__();self.channels=channels
        sx=torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]])
        sy=torch.tensor([[-3.,-10.,-3.],[0.,0.,0.],[3.,10.,3.]])
        self.register_buffer('kernel_x',sx[None,None].repeat(channels,1,1,1))
        self.register_buffer('kernel_y',sy[None,None].repeat(channels,1,1,1))
        self.norm=nn.BatchNorm2d(channels)
        self.conv_extra=nn.Sequential(nn.Conv2d(channels,16,1),nn.SiLU(),nn.Conv2d(16,channels,3,padding=1))

    def forward(self,x:Tensor):
        edge=F.conv2d(x,self.kernel_x,padding=1,groups=self.channels).abs()
        edge=edge+F.conv2d(x,self.kernel_y,padding=1,groups=self.channels).abs()
        return self.conv_extra(x+self.norm(edge)),edge


class PETResponseContrast(nn.Module):
    def __init__(self,input_channels:int,channels:int):
        super().__init__()
        self.mix=nn.Conv2d(4*input_channels,channels,1,bias=False)
        self.local=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.activation=nn.GELU()

    def forward(self,p:Tensor):
        contrasts=[p-F.avg_pool2d(p,k,stride=1,padding=k//2,count_include_pad=False) for k in (3,5,7)]
        hp=self.activation(self.local(self.mix(torch.cat([p,*contrasts],dim=1))))
        return hp,contrasts


class ChannelCrossAttention(nn.Module):
    """Faithful DRWKV CrossAttention, no einops/custom CUDA requirement."""
    def __init__(self,dim:int,num_heads:int=4):
        super().__init__()
        if dim%num_heads:
            raise ValueError('Attention width must be divisible by num_heads')
        self.num_heads=num_heads
        self.temperature=nn.Parameter(torch.ones(num_heads,1,1))
        self.q=nn.Conv2d(dim,dim,1,bias=False)
        self.q_dwconv=nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=False)
        self.kv=nn.Conv2d(dim,dim*2,1,bias=False)
        self.kv_dwconv=nn.Conv2d(dim*2,dim*2,3,padding=1,groups=dim*2,bias=False)
        self.project_out=nn.Conv2d(dim,dim,1,bias=False)

    def forward(self,x:Tensor,y:Tensor,diagnose:bool=False):
        b,c,h,w=x.shape
        q=self.q_dwconv(self.q(x));k,v=self.kv_dwconv(self.kv(y)).chunk(2,1)
        q=q.reshape(b,self.num_heads,c//self.num_heads,h*w)
        k=k.reshape(b,self.num_heads,c//self.num_heads,h*w)
        v=v.reshape(b,self.num_heads,c//self.num_heads,h*w)
        with torch.autocast(device_type=x.device.type,enabled=False):
            q=F.normalize(q.float(),dim=-1,eps=1e-12)
            k=F.normalize(k.float(),dim=-1,eps=1e-12)
            attention=((q@k.transpose(-2,-1))*self.temperature.float()).softmax(-1)
            out=(attention@v.float()).reshape(b,c,h,w).to(x.dtype)
        return self.project_out(out),attention.detach() if diagnose else None


class SpectralAlignmentEnhancer(nn.Module):
    """Complete DRWKV IEL/SAE (including both secondary depthwise convolutions)."""
    def __init__(self,dim:int,expansion:float=2.66):
        super().__init__();hidden=int(dim*expansion)
        if hidden<1:
            raise ValueError('SAE hidden width must be positive')
        self.project_in=nn.Conv2d(dim,hidden*2,1,bias=False)
        self.dwconv=nn.Conv2d(hidden*2,hidden*2,3,padding=1,groups=hidden*2,bias=False)
        self.dwconv1=nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden,bias=False)
        self.dwconv2=nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden,bias=False)
        self.project_out=nn.Conv2d(hidden,dim,1,bias=False)

    def forward(self,x:Tensor)->Tensor:
        x1,x2=self.dwconv(self.project_in(x)).chunk(2,1)
        x1=x1+torch.tanh(self.dwconv1(x1));x2=x2+torch.tanh(self.dwconv2(x2))
        return self.project_out(x1*x2)


def rms(x:Tensor)->Tensor:
    return x.detach().float().square().mean().sqrt()


class StructureResponseStage(nn.Module):
    def __init__(self,channels:int,num_heads:int=4,expansion:float=2.66):
        super().__init__();self.channels=channels;dim=channels//2
        if channels%2 or dim%num_heads:
            raise ValueError('Even channels and half-width divisible by num_heads are required')
        self.ct_projection=nn.Conv2d(channels,dim,1,bias=False)
        self.ct_see=FixedScharrSEE(dim)
        self.pet_lrce=PETResponseContrast(channels,dim)
        self.ca_ct=ChannelCrossAttention(dim,num_heads);self.ca_pet=ChannelCrossAttention(dim,num_heads)
        self.ct_mix=nn.Conv2d(channels+2*dim,dim,1,bias=False)
        self.pet_mix=nn.Conv2d(channels+2*dim,dim,1,bias=False)
        self.ct_norm=ChannelLayerNorm(dim);self.pet_norm=ChannelLayerNorm(dim)
        self.ct_sae=SpectralAlignmentEnhancer(dim,expansion);self.pet_sae=SpectralAlignmentEnhancer(dim,expansion)
        self.ct_output=nn.Conv2d(dim,channels,1,bias=False);self.pet_output=nn.Conv2d(dim,channels,1,bias=False)
        nn.init.zeros_(self.ct_output.weight);nn.init.zeros_(self.pet_output.weight)

    def forward(self,c:Tensor,p:Tensor,diagnose:bool=False,return_maps:bool=False):
        if c.ndim!=4 or c.shape!=p.shape or c.shape[1]!=self.channels:
            raise ValueError(f'Expected matching BCHW pairs with {self.channels} channels')
        if c.device!=p.device or c.dtype!=p.dtype or not c.is_floating_point():
            raise ValueError('CT/PET must share floating dtype and device')
        diagnose=diagnose or return_maps
        hc,edge=self.ct_see(self.ct_projection(c));hp,contrasts=self.pet_lrce(p)
        ic,ac=self.ca_ct(hc,hp,diagnose);ip,ap=self.ca_pet(hp,hc,diagnose)
        uc=self.ct_mix(torch.cat([c,hc,ic],1));up=self.pet_mix(torch.cat([p,hp,ip],1))
        dc=self.ct_output(self.ct_sae(self.ct_norm(uc)))
        dp=self.pet_output(self.pet_sae(self.pet_norm(up)))
        fused=(c+dc)+(p+dp)
        if not diagnose:
            return fused,None
        with torch.no_grad():
            def entropy(a):
                n=a.shape[-1]
                return -(a*a.clamp_min(1e-12).log()).sum(-1).mean()/math.log(n) if n>1 else a.new_zeros(())
            dc_rms,dp_rms=rms(dc),rms(dp)
            cosine=(dc.detach().float()*dp.detach().float()).sum()/(
                dc.detach().float().norm()*dp.detach().float().norm()).clamp_min(1e-12)
            info={'ct_residual_ratio':dc_rms/rms(c).clamp_min(1e-8),
                  'pet_residual_ratio':dp_rms/rms(p).clamp_min(1e-8),
                  'ct_residual_rms':dc_rms,'pet_residual_rms':dp_rms,
                  'edge_rms':rms(edge),'ct_ca_entropy':entropy(ac),'pet_ca_entropy':entropy(ap),
                  'ct_temperature':self.ca_ct.temperature.detach().mean(),
                  'pet_temperature':self.ca_pet.temperature.detach().mean(),
                  'delta_cos':cosine if dc_rms>0 and dp_rms>0 else None}
            info.update({f'pet_contrast{k}_rms':rms(d) for k,d in zip((3,5,7),contrasts)})
            if return_maps:
                def magnitude(x):return x.detach().float().square().mean(1,keepdim=True).sqrt()
                maps={name:magnitude(x) for name,x in (
                    ('ct_input',c),('pet_input',p),('ct_structure',edge),('ct_enhanced',hc),
                    ('pet_enhanced',hp),('delta_ct',dc),('delta_pet',dp),('fused',fused))}
                for k,d in zip((3,5,7),contrasts):
                    maps[f'pet_contrast{k}_rms']=magnitude(d)
                    maps[f'pet_contrast{k}_mean']=d.detach().float().mean(1,keepdim=True)
                maps['ct_ca']=ac;maps['pet_ca']=ap
                maps['delta_ct_mean']=dc.detach().float().mean(1,keepdim=True)
                maps['delta_pet_mean']=dp.detach().float().mean(1,keepdim=True)
                info['maps']=maps
        return fused,info


class StructureResponseFusion(nn.Module):
    def __init__(self,channels:Sequence[int]=(64,128,320,512),num_heads:int=4,expansion:float=2.66):
        super().__init__();self.channels=tuple(channels)
        self.stages=nn.ModuleList([StructureResponseStage(c,num_heads,expansion) for c in channels])

    def forward(self,ct:Sequence[Tensor],pet:Sequence[Tensor],return_diagnostics:bool=False,
                return_maps:bool=False):
        if len(ct)!=len(self.stages) or len(pet)!=len(self.stages):
            raise ValueError('One aligned CT/PET pair is required per scale')
        fused=[];diagnostics={}
        for s,(block,c,p) in enumerate(zip(self.stages,ct,pet)):
            f,info=block(c,p,return_diagnostics,return_maps);fused.append(f)
            if info is not None:diagnostics[s]=info
        return (fused,diagnostics) if return_diagnostics or return_maps else fused
