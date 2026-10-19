"""PET DSSA context + CT edge-guided residual fusion.

Source mechanisms (adapted, NOT an original-paper reproduction):
  XiaZunhui/MedFormer @ 3ede41568bd2e6fffb718da9267cf508e01665f0
    models/dssa.py, models/dsa.py: regional top-k, pixel top-k, local DWConv.
  UARK-AICV/MEGANet @ 8ae378def060071c3bca5a3f85f22f58dd26559e
    MEGANet-ResNet/lib/modules.py: Laplacian pyramid, EGA and CBAM.
See ASYMMETRIC_SOURCE.md for attribution, modifications and source status.

Four aligned BCHW pairs -> four BCHW features + region logits. Full requires
the SAME augmented/normalized CT image used by the encoder. Disabled returns
exact addition without calling any branch. Missing must bypass this module.
Only final residual projections are zero initialized. No teacher, retrieval,
baseline-checkpoint loading or protection loss. Attention/edge core is FP32.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _gauss(x: Tensor, multiplier: float = 1.) -> Tensor:
    row = x.new_tensor([1., 4., 6., 4., 1.])
    kernel = (row[:, None] * row[None, :] / 256. * multiplier)
    kernel = kernel.view(1, 1, 5, 5).expand(x.shape[1], 1, 5, 5)
    mode = 'reflect' if min(x.shape[-2:]) > 2 else 'replicate'
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode=mode), kernel,
                    groups=x.shape[1])


def _pyramid_step(x: Tensor) -> tuple[Tensor, Tensor]:
    low = _gauss(x)[:, :, ::2, ::2]
    # Same zero-insertion/Gaussian reconstruction as MEGANet, device agnostic.
    up = low.new_zeros(*low.shape[:2], low.shape[2]*2, low.shape[3]*2)
    up[:, :, ::2, ::2] = low
    up = _gauss(up, 4.)
    if up.shape[-2:] != x.shape[-2:]:
        up = F.interpolate(up, size=x.shape[-2:], mode='nearest')
    return x - up, low


def laplace(x: Tensor) -> Tensor:
    return _pyramid_step(x)[0]


def ct_to_gray(image: Tensor, norm_mode: str) -> Tensor:
    """Invert only the existing dataset normalization, not its augmentation."""
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError('CT image must be Bx1xHxW or Bx3xHxW')
    x = image.float()
    if norm_mode == 'imagenet':
        mean = x.new_tensor([.485, .456, .406])[:x.shape[1]].view(1,-1,1,1)
        std = x.new_tensor([.229, .224, .225])[:x.shape[1]].view(1,-1,1,1)
        x = x * std + mean
    elif norm_mode == 'cipa':
        x = (x + 1.6) / 3.2
    else:
        raise ValueError(f'Unsupported normalization: {norm_mode}')
    return x.mean(dim=1, keepdim=True)


class SparsePETAttention(nn.Module):
    """MedFormer DSSA adapted to BCHW, safe padding and variable tiny grids.

    Routing indices are computed from detached region-average Q/K, matching
    the official code. Selected token Q/K/V and local convolution keep grads.
    Top-k selects keys for EACH query; it is not a foreground detector.
    """
    def __init__(self, dim: int, heads: int = 2, n_win: int = 4,
                 topk: int = 4, pixel_keep: float = .125):
        super().__init__()
        if dim % heads or min(dim, heads, n_win, topk) < 1 or not 0 < pixel_keep <= 1:
            raise ValueError('Invalid sparse attention configuration')
        self.dim, self.heads, self.n_win, self.topk = dim, heads, n_win, topk
        self.pixel_keep = float(pixel_keep)
        self.scale = dim ** -.5  # Preserve the official DSSA scaling.
        self.qkv = nn.Conv2d(dim, 3*dim, 1)
        self.output = nn.Conv2d(dim, dim, 1)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        b,c,h,w = x.shape
        rh, rw = math.ceil(h/min(h,self.n_win)), math.ceil(w/min(w,self.n_win))
        nh, nw = math.ceil(h/rh), math.ceil(w/rw)
        ph, pw = nh*rh-h, nw*rw-w
        q,k,v = self.qkv(x).chunk(3, dim=1)
        local = self.local(v)
        def sequence(t):
            t = F.pad(t, (0,pw,0,ph))
            return (t.reshape(b,self.heads,c//self.heads,nh,rh,nw,rw)
                    .permute(0,1,3,5,4,6,2).reshape(b,self.heads,nh*nw,rh*rw,-1))
        qs,ks,vs = [sequence(t) for t in (q,k,v)]
        valid = F.pad(x.new_ones(1,1,h,w),(0,pw,0,ph))
        valid = (valid.reshape(1,1,nh,rh,nw,rw).permute(0,1,2,4,3,5)
                 .reshape(1,1,nh*nw,rh*rw).bool())
        # Pool across heads/channels exactly as the source's unheaded router.
        qr = (qs.detach()*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
        kr = (ks.detach()*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
        qr = qr.permute(0,2,1,3).flatten(-2)
        kr = kr.permute(0,2,1,3).flatten(-2)
        idx = (qr @ kr.transpose(-1,-2)).topk(min(self.topk,nh*nw),dim=-1).indices
        idx = idx[:,None].expand(-1,self.heads,-1,-1)
        bi = torch.arange(b,device=x.device)[:,None,None,None]
        hi = torch.arange(self.heads,device=x.device)[None,:,None,None]
        kg,vg = ks[bi,hi,idx].flatten(-3,-2),vs[bi,hi,idx].flatten(-3,-2)
        vm = valid.expand(b,self.heads,-1,-1)[bi,hi,idx].flatten(-2)
        scores = ((qs*self.scale) @ kg.transpose(-1,-2)).masked_fill(~vm.unsqueeze(-2),-torch.inf)
        count = max(1,math.ceil(scores.shape[-1]*self.pixel_keep))
        scores, selected = scores.topk(count,dim=-1)
        # Like official dsa.py, gather selected V per query before matmul.
        vg = vg.unsqueeze(-3).expand(-1,-1,-1,qs.shape[-2],-1,-1)
        chosen = torch.gather(vg,4,selected[...,None].expand(-1,-1,-1,-1,-1,c//self.heads))
        out = (scores.softmax(-1).unsqueeze(-2) @ chosen).squeeze(-2)
        out = (out.reshape(b,self.heads,nh,nw,rh,rw,c//self.heads)
               .permute(0,1,6,2,4,3,5).reshape(b,c,nh*rh,nw*rw))
        return self.output(out[:,:,:h,:w]+local)


class _CBAM(nn.Module):
    """MEGANet channel/spatial gates; keep at least one bottleneck channel."""
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels,max(1,channels//16)),nn.ReLU(),
                                 nn.Linear(max(1,channels//16),channels))
        self.spatial = nn.Conv2d(2,1,7,padding=3)

    def forward(self, x: Tensor) -> Tensor:
        channel = (self.mlp(x.mean((2,3)))+self.mlp(x.amax((2,3)))).sigmoid()
        x = x * channel[:,:,None,None]
        return x * self.spatial(torch.cat((x.amax(1,keepdim=True),
                                           x.mean(1,keepdim=True)),1)).sigmoid()


class EdgeGuidedAttention(nn.Module):
    """MEGANet EGA: reverse + prediction-edge + image-detail branches."""
    def __init__(self, channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(nn.Conv2d(3*channels,channels,3,padding=1),
                                         nn.BatchNorm2d(channels),nn.ReLU())
        self.attention = nn.Sequential(nn.Conv2d(channels,1,3,padding=1),
                                       nn.BatchNorm2d(1),nn.Sigmoid())
        self.cbam = _CBAM(channels)

    def forward(self, edge: Tensor, x: Tensor, region_logits: Tensor) -> Tensor:
        pred = region_logits.sigmoid()  # Caller passes logits, not probabilities.
        detail = F.interpolate(edge,size=x.shape[-2:],mode='bilinear',align_corners=True)
        branches = torch.cat((x*(1-pred), x*laplace(pred), x*detail),dim=1)
        fused = self.fusion_conv(branches)
        return self.cbam(x+fused*self.attention(fused))


class _Stage(nn.Module):
    def __init__(self, channels: int, rank: int, context_size: int):
        super().__init__()
        self.context_size = context_size
        self.ct_project = nn.Conv2d(channels,rank,1)
        self.pet_project = nn.Conv2d(channels,rank,1)
        self.attention = SparsePETAttention(rank)
        self.region_head = nn.Conv2d(rank*2,1,1)
        self.ega = EdgeGuidedAttention(rank)
        self.up_pet = nn.Conv2d(rank*2,channels,1)
        self.up_ct = nn.Conv2d(rank,channels,1)
        for layer in (self.up_pet,self.up_ct):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, c: Tensor, p: Tensor, edge: Tensor):
        cp,pp = self.ct_project(c),self.pet_project(p)
        size = tuple(min(v,self.context_size) for v in pp.shape[-2:])
        pooled = F.adaptive_avg_pool2d(pp,size) if size != pp.shape[-2:] else pp
        context = self.attention(pooled)
        if context.shape[-2:] != pp.shape[-2:]:
            context = F.interpolate(context,size=pp.shape[-2:],mode='bilinear',align_corners=False)
        joint = torch.cat((pp,context),dim=1)
        region = self.region_head(joint)
        delta_p = self.up_pet(joint)
        delta_c = self.up_ct(self.ega(edge,cp,region)-cp)
        return delta_c,delta_p,region


class AsymmetricFusion(nn.Module):
    def __init__(self, channels: Sequence[int] = (64,128,320,512),
                 ranks: Sequence[int] = (16,16,32,32), context_size: int = 32,
                 enabled: bool = True, norm_mode: str = 'imagenet'):
        super().__init__()
        if len(channels)!=4 or len(ranks)!=4 or context_size<1:
            raise ValueError('Require four channel/rank values and positive context_size')
        if any(r<2 or r%2 for r in ranks):
            raise ValueError('Each rank must be positive and divisible by two heads')
        if norm_mode not in ('imagenet','cipa'):
            raise ValueError('Unsupported norm_mode')
        self.channels = tuple(channels)
        self.enabled,self.norm_mode = bool(enabled),norm_mode
        self.stages = nn.ModuleList(_Stage(c,r,context_size) for c,r in zip(channels,ranks))

    def forward(self, ct: Sequence[Tensor], pet: Sequence[Tensor],
                ct_image: Tensor | None) -> tuple[list[Tensor],list[Tensor]]:
        if len(ct)!=4 or len(pet)!=4:
            raise ValueError('Require four CT/PET scales')
        for i,(c,p,ch) in enumerate(zip(ct,pet,self.channels)):
            if c.ndim!=4 or c.shape!=p.shape or c.shape[1]!=ch:
                raise ValueError(f'Invalid aligned features at S{i+1}')
            if c.device!=p.device or c.dtype!=p.dtype:
                raise ValueError('CT/PET feature device and dtype must match')
        if not self.enabled:
            return [c+p for c,p in zip(ct,pet)],[]
        if ct_image is None or ct_image.device!=ct[0].device or ct_image.shape[0]!=ct[0].shape[0]:
            raise ValueError('Enabled fusion requires matching augmented CT image')
        out,regions=[],[]
        with torch.autocast(device_type=ct[0].device.type,enabled=False):
            image=ct_to_gray(ct_image,self.norm_mode)
            # Compute one shared Laplacian pyramid; S1 uses its 1/4 level.
            edges=[]
            for _ in range(6):
                detail,image=_pyramid_step(image)
                edges.append(detail)
            for i,(stage,c,p) in enumerate(zip(self.stages,ct,pet)):
                dc,dp,region=stage(c.float(),p.float(),edges[i+2])
                out.append((c+p)+(dc+dp).to(c.dtype))
                regions.append(region)
        return out,regions
