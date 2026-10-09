# -*- coding: utf-8 -*-
"""Standalone PET-guided / CT-structure-constrained local feature fusion.

Approved design, version 1 (2026-10-08):
  PET: full-channel identity + near ACC -> semantic transition -> far ACC;
       concatenate near/far, GN/GELU/PW, identity addition.
  CT:  explicit neighbor-minus-center differences, signed learned updates.
  S(i,j) = PET_location(j) + relative_bias(delta)
           - softplus(kappa) * mean(CT(j)-CT(i))**2.
  A = softmax over VALID local candidate positions (center included).
  CT message = sum_j A(i,j) * signed_weight(delta) * (CT(j)-CT(i)).
  PET message = sum_j A(i,j) * PET_value(j).
  C_plus = C + O_C(PET_gate * CT_message); P_plus = P + O_P(PET_message).
  fused = C_plus + P_plus. O_C/O_P are zero-initialized.

Provenance / deliberate adaptations (NOT full paper reproduction):
  HDNet, TGRS 2025, DOI 10.1109/TGRS.2025.3574962:
    https://github.com/iLearn-Lab/TGRS25-HDNet/blob/main/model/HDNet.py
    https://github.com/iLearn-Lab/TGRS25-HDNet/blob/main/model/MAC_Kernel.py
    Borrow center-surround initialization, progressive contrast/residual
    organization. Adapt original grouped MAC to full-channel near/transition/
    far processing and GroupNorm. Do NOT copy DHPF, CUDA globals, or backbone.
    Original ordinary Conv2d kernels initialized by weight.data remain
    trainable; here initialization uses no_grad/copy_ and is also trainable.
  RAC-Net, TPAMI 2026, DOI 10.1109/TPAMI.2026.3705213, eqs. 11--16:
    Borrow region-guided neighborhood residual correction responsibility.
    No HU recovery, fixed Gaussian RASFE, or physical PVA-inversion claim.
  PiDiViT, ICCV 2025, "When Pixel Difference Patterns Meet ViT":
    Borrow neighbor/center difference concept, not its detector architecture.
    Our shared attention-weighted multimodal difference operator is a proposed
    adaptation, not source-code-identical PDC or a supervised boundary map.

Tensor contract:
  Single scale: C/P = [B,c_s,H,W], floating tensors on same device, same shape.
  C/P dtypes may differ under AMP; direct addition follows PyTorch promotion.
  No implicit resizing, pooling, threshold, top-k, hard ROI, or NaN replacement.
  Multi-scale default channels=(64,128,320,512), fine-to-coarse, 4 scales.
  MultiScale.forward returns list[Tensor], matching clean AddFusion seam.
  forward_with_features returns (fused_list, CT_plus_list, PET_plus_list).
  Missing returns (C,C,zeros_like(C)); no PET-dependent path is called.
  auto processes ONLY available rows, then restores order (CT dtype, matching
  the clean model's auto assembly). It cannot prevent an upstream encoder from
  encoding missing PET: the model integration must also enforce that rule.

Numerics / memory:
  Exact row stripes + non-reentrant activation checkpointing, no approximation.
  Usual local scores, differences and reductions in FP32 with autocast off;
  FP64 inputs keep FP64 reductions for numerical checks. Original input paths
  retain their dtype. No persistent activation/diagnostic cache.
  Requires PyTorch >=2.1; no timm, transformers, numpy, einops, or local imports.

Usage in models/:
  fusion = MultiScalePETGuidedStructureFusion()
  fused_feats = fusion(aligned_ct_feats, pet_feats)  # Full only
  # Missing model path: decode(aligned_ct_feats), do not call fusion/encoder.
  python models/pet_guided_structure_fusion.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['AdaptedPETMACDescriptor', 'PETGuidedStructureFusion',
           'MultiScalePETGuidedStructureFusion']


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(channels, 8), channels, eps=1e-5)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} contains NaN/Inf; no values were masked')


def _validate_state(state: object, batch: int, device: torch.device) -> Tensor:
    if state is None:
        raise ValueError('auto requires explicit pet_available [B]')
    raw = torch.as_tensor(state, device=device)
    ints = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if raw.ndim != 1 or raw.shape[0] != batch:
        raise ValueError(f'pet_available must have shape [{batch}], got {tuple(raw.shape)}')
    if raw.dtype != torch.bool and raw.dtype not in ints:
        raise ValueError('pet_available must be bool or 0/1 integers')
    if not bool(((raw == 0) | (raw == 1)).all()):
        raise ValueError('pet_available must contain only 0/1')
    return raw.bool()


class AdaptedPETMACDescriptor(nn.Module):
    """Task-adapted MAC, full-channel direct + progressive contrast processing.

    u=PW(GN(x)); n=GELU(GN(ACC_d1(u))); t=PW(GELU(GN(u+n)));
    far=GELU(GN(ACC_d2(t))); out=x+PW(GELU(GN(cat(n,far)))).
    Zero-sum contrast kernels at initialization only; they remain learnable.
    Replicate padding avoids creating contrast from zero padding on uniform
    features. This is explicit adaptation of source zero-padding behavior.
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        channels = _positive_int('channels', channels)
        self.channels = channels
        self.input_norm = _gn(channels)
        self.input_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.near = nn.Conv2d(channels, channels, 3, padding=1,
                              groups=channels, bias=False, padding_mode='replicate')
        self.near_norm = _gn(channels)
        self.transition_norm = _gn(channels)
        self.transition_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.far = nn.Conv2d(channels, channels, 3, padding=2, dilation=2,
                             groups=channels, bias=False, padding_mode='replicate')
        self.far_norm = _gn(channels)
        self.mix_norm = _gn(2 * channels)
        self.mix_conv = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.activation = nn.GELU()
        # Exact 3x3 source center-surround pattern (center=1, eight=-1/8).
        contrast = torch.full((1, 1, 3, 3), -1.0 / 8.0)
        contrast[0, 0, 1, 1] = 1.0
        with torch.no_grad():
            self.near.weight.copy_(contrast.expand_as(self.near.weight))
            self.far.weight.copy_(contrast.expand_as(self.far.weight))

    def forward(self, x: Tensor) -> Tensor:
        u = self.input_conv(self.input_norm(x))
        near = self.activation(self.near_norm(self.near(u)))
        transition = self.transition_conv(self.activation(self.transition_norm(u + near)))
        far = self.activation(self.far_norm(self.far(transition)))
        response = self.mix_conv(self.activation(self.mix_norm(torch.cat((near, far), dim=1))))
        return x + response


class PETGuidedStructureFusion(nn.Module):
    """One-scale module. Default forward returns (fused, CT_plus, PET_plus).

    Ablations: use_pet_guidance=False disables BOTH PET location/gate (gate=1),
    use_structure_constraint=False removes only CT energy penalty, and
    ct_update_type='content' replaces only CT difference message with CT
    feature-content reading. Existing weights/modules remain checkpoint-visible.
    Disabled modules have no gradient by design; do not interpret that as failure.
    """
    def __init__(self, channels: int, inner_channels: int = 32, heads: int = 4,
                 kernel_size: int = 5, chunk_rows: int = 16,
                 checkpoint_chunks: bool = True, beta_init: float = 0.0,
                 structure_strength_init: float = 0.1,
                 use_pet_guidance: bool = True,
                 use_structure_constraint: bool = True,
                 ct_update_type: str = 'difference', check_finite: bool = True):
        super().__init__()
        self.channels = _positive_int('channels', channels)
        self.inner_channels = _positive_int('inner_channels', inner_channels)
        self.heads = _positive_int('heads', heads)
        self.kernel_size = _positive_int('kernel_size', kernel_size)
        self.chunk_rows = _positive_int('chunk_rows', chunk_rows)
        if inner_channels % heads:
            raise ValueError('inner_channels must be divisible by heads')
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd and >=3')
        if ct_update_type not in ('difference', 'content'):
            raise ValueError("ct_update_type must be 'difference' or 'content'")
        if not math.isfinite(beta_init) or beta_init < 0:
            raise ValueError('beta_init must be finite and nonnegative')
        if not math.isfinite(structure_strength_init) or structure_strength_init <= 0:
            raise ValueError('structure_strength_init must be finite and >0')
        for name, flag in [('checkpoint_chunks', checkpoint_chunks),
                           ('use_pet_guidance', use_pet_guidance),
                           ('use_structure_constraint', use_structure_constraint),
                           ('check_finite', check_finite)]:
            if not isinstance(flag, bool):raise ValueError(f'{name} must be bool')
        self.checkpoint_chunks = checkpoint_chunks
        self.use_pet_guidance = use_pet_guidance
        self.use_structure_constraint = use_structure_constraint
        self.ct_update_type = ct_update_type
        self.check_finite = check_finite
        self.head_dim = inner_channels // heads
        self.radius = kernel_size // 2
        candidates = kernel_size ** 2
        self.ct_project = nn.Sequential(_gn(channels), nn.Conv2d(channels,inner_channels,1,bias=True))
        self.pet_project = nn.Sequential(_gn(channels), nn.Conv2d(channels,inner_channels,1,bias=True))
        self.mac = AdaptedPETMACDescriptor(inner_channels)
        self.pet_location = nn.Conv2d(inner_channels, heads, 1, bias=True)
        self.pet_gate = nn.Conv2d(inner_channels, 1, 1, bias=True)
        self.pet_value = nn.Conv2d(inner_channels, inner_channels, 1, bias=True)
        self.difference_weight = nn.Parameter(torch.empty(heads, self.head_dim, candidates))
        nn.init.normal_(self.difference_weight, mean=0.0, std=1.0 / math.sqrt(candidates))
        axis = torch.arange(-self.radius, self.radius + 1)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((yy.reshape(-1), xx.reshape(-1)),dim=1)
        self.register_buffer('offsets', offsets, persistent=True)
        distance = offsets.float().square().sum(1) / float(self.radius ** 2)
        # Learnable AFTER initialization. beta is not a fixed physical penalty.
        self.relative_bias = nn.Parameter((-float(beta_init)*distance).repeat(heads,1))
        inv_softplus = structure_strength_init + math.log(-math.expm1(-structure_strength_init))
        self.structure_log_strength = nn.Parameter(torch.full((heads,),inv_softplus))
        self.ct_out = nn.Conv2d(inner_channels, channels, 1, bias=False)
        self.pet_out = nn.Conv2d(inner_channels, channels, 1, bias=True)
        nn.init.zeros_(self.ct_out.weight)
        nn.init.zeros_(self.pet_out.weight)
        nn.init.zeros_(self.pet_out.bias)
        # Tensor-only signature: EMA compatible; rejects semantic checkpoint mixups.
        self.register_buffer('contract_signature', torch.tensor([
            1, channels, inner_channels, heads, kernel_size, int(use_pet_guidance),
            int(use_structure_constraint), int(ct_update_type=='difference')],dtype=torch.int64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key=prefix+'contract_signature'
        if key in state_dict and not torch.equal(state_dict[key].cpu(),self.contract_signature.cpu()):
            error_msgs.append(f'{prefix}fusion contract differs from checkpoint; rebuild from its config')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def _validate_ct(self, ct: Tensor) -> None:
        if not isinstance(ct,Tensor) or ct.ndim!=4 or not ct.is_floating_point():
            raise ValueError('CT must be floating [B,C,H,W]')
        if ct.shape[1]!=self.channels or min(ct.shape[0],ct.shape[2],ct.shape[3])<1:
            raise ValueError(f'CT expected positive B/H/W and {self.channels} channels, got {tuple(ct.shape)}')
        if self.check_finite:_finite('CT input',ct)

    def _validate_pair(self, ct: Tensor, pet: Optional[Tensor]) -> Tensor:
        if not isinstance(pet,Tensor) or pet.ndim!=4 or not pet.is_floating_point():
            raise ValueError('Full requires floating PET [B,C,H,W], not None')
        if pet.shape!=ct.shape or pet.device!=ct.device:
            raise ValueError(f'Full CT/PET must match shape/device; CT={tuple(ct.shape)}/{ct.device}, '
                             f'PET={tuple(pet.shape)}/{pet.device}; no automatic resize')
        if self.check_finite:_finite('PET input',pet)
        return pet

    def _local_chunk(self, ct: Tensor, value: Tensor, location: Tensor,
                     start: int, end: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Exact valid 2D neighborhood for one output-row stripe, all columns.

        Outputs CT/PET messages [B,heads,head_dim,rows,W], A [B,heads,K2,rows,W].
        Computation is shared: the SAME A is used for both content reads.
        """
        b,_,height,width=ct.shape
        rows=end-start; k2=self.kernel_size**2; r=self.radius
        dtype=torch.float64 if ct.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=ct.device.type,enabled=False):
            ct=ct.to(dtype);value=value.to(dtype);location=location.to(dtype)
            def neighbors(x: Tensor, channels_per_head: int) -> Tensor:
                stripe=F.pad(x,(r,r,r,r))[:,:,start:end+2*r,:]
                out=F.unfold(stripe,kernel_size=self.kernel_size,stride=1)
                return out.reshape(b,self.heads,channels_per_head,k2,rows,width)
            center=ct[:,:,start:end,:].reshape(b,self.heads,self.head_dim,1,rows,width)
            ct_neighbor=neighbors(ct,self.head_dim)
            difference=ct_neighbor-center
            if self.use_pet_guidance:
                score=neighbors(location,1).squeeze(2)
            else:
                score=torch.zeros((b,self.heads,k2,rows,width),device=ct.device,dtype=dtype)
            score=score+self.relative_bias.to(dtype)[None,:,:,None,None]
            if self.use_structure_constraint:
                energy=difference.square().mean(dim=2)
                strength=F.softplus(self.structure_log_strength.to(dtype))
                score=score-strength[None,:,None,None,None]*energy
            oy=self.offsets[:,0].view(1,1,k2,1,1)
            ox=self.offsets[:,1].view(1,1,k2,1,1)
            y=torch.arange(start,end,device=ct.device).view(1,1,1,rows,1)+oy
            x=torch.arange(width,device=ct.device).view(1,1,1,1,width)+ox
            valid=(y>=0)&(y<height)&(x>=0)&(x<width)
            attention=score.masked_fill(~valid,float('-inf')).softmax(dim=2)
            weights=attention.unsqueeze(2)
            if self.ct_update_type=='difference':
                signed=self.difference_weight.to(dtype)[None,:,:,:,None,None]
                ct_message=(weights*signed*difference).sum(dim=3)
            else:
                ct_message=(weights*ct_neighbor).sum(dim=3)
            pet_neighbor=neighbors(value,self.head_dim)
            pet_message=(weights*pet_neighbor).sum(dim=3)
        return ct_message,pet_message,attention

    def _aggregate(self, ct: Tensor, value: Tensor, location: Tensor,
                   diagnostics: bool = False) -> Tuple[Tensor,Tensor,Optional[Tensor]]:
        c_chunks,p_chunks,a_chunks=[],[],[]
        for start in range(0,ct.shape[2],self.chunk_rows):
            end=min(start+self.chunk_rows,ct.shape[2])
            # Bind bounds now: backward recomputation must not see final-loop bounds.
            run=functools.partial(self._local_chunk,start=start,end=end)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                mc,mp,a=checkpoint(run,ct,value,location,use_reentrant=False)
            else:
                mc,mp,a=run(ct,value,location)
            c_chunks.append(mc);p_chunks.append(mp)
            if diagnostics:a_chunks.append(a.detach())
        b,_,h,w=ct.shape
        mc=torch.cat(c_chunks,dim=3).reshape(b,self.inner_channels,h,w)
        mp=torch.cat(p_chunks,dim=3).reshape(b,self.inner_channels,h,w)
        return mc,mp,torch.cat(a_chunks,dim=3) if diagnostics else None

    def _full(self, ct: Tensor, pet: Tensor, diagnostics: bool = False):
        xc=self.ct_project(ct);xp=self.pet_project(pet)
        if self.use_pet_guidance:
            descriptor=self.mac(xp)
            location=self.pet_location(descriptor)
            gate=torch.sigmoid(self.pet_gate(descriptor))
        else:
            location=xp.new_zeros((xp.shape[0],self.heads,*xp.shape[-2:]))
            gate=xp.new_ones((xp.shape[0],1,*xp.shape[-2:]))
        value=self.pet_value(xp)  # Not the contrast descriptor, never multiplied by gate.
        mc,mp,attention=self._aggregate(xc,value,location,diagnostics)
        # Cast only messages to each projection's working dtype, as standard AMP.
        dc=self.ct_out((mc*gate.to(mc.dtype)).to(xc.dtype)).to(ct.dtype)
        dp=self.pet_out(mp.to(value.dtype)).to(pet.dtype)
        ce,pe=ct+dc,pet+dp
        fused=ce+pe
        if self.check_finite:_finite('fusion output',fused)
        outputs=(fused,ce,pe)
        if not diagnostics:return outputs
        return outputs, {'attention':attention,'pet_gate':gate.detach(),
                          'pet_location':location.detach(),
                          'structure_strength':F.softplus(self.structure_log_strength).detach()}

    def forward(self, ct: Tensor, pet: Optional[Tensor] = None,
                forward_mode: str = 'full', pet_available: object = None
                ) -> Tuple[Tensor,Tensor,Tensor]:
        self._validate_ct(ct)
        if forward_mode=='missing':
            return ct,ct,torch.zeros_like(ct)
        if forward_mode=='full':
            if pet_available is not None:raise ValueError('pet_available only applies to auto')
            return self._full(ct,self._validate_pair(ct,pet))
        if forward_mode!='auto':raise ValueError(f'unsupported forward_mode={forward_mode!r}')
        state=_validate_state(pet_available,ct.shape[0],ct.device)
        indices=state.nonzero(as_tuple=True)[0]
        if indices.numel()==0:return ct,ct,torch.zeros_like(ct)
        # Validate container/shape but never inspect unavailable PET values.
        if not isinstance(pet,Tensor) or pet.shape!=ct.shape or pet.device!=ct.device:
            raise ValueError('auto with available rows requires same-shape/device PET batch')
        cfull=ct.index_select(0,indices);pfull=pet.index_select(0,indices)
        outputs=self._full(cfull,self._validate_pair(cfull,pfull))
        bases=(ct,ct,torch.zeros_like(ct))
        return tuple(base.index_copy(0,indices,out.to(base.dtype))
                     for base,out in zip(bases,outputs))

    def forward_with_diagnostics(self, ct: Tensor, pet: Tensor):
        """Full-only explicit opt-in. Detached maps returned, never cached.

        Returning attention increases memory; do not enable in every train step.
        """
        self._validate_ct(ct)
        return self._full(ct,self._validate_pair(ct,pet),diagnostics=True)


class MultiScalePETGuidedStructureFusion(nn.Module):
    """Four identical architectures, independent parameters; AddFusion list API."""
    def __init__(self, channels: Sequence[int] = (64,128,320,512), **fusion_kwargs):
        super().__init__()
        if len(channels)!=4:raise ValueError('exactly four fine-to-coarse channel counts required')
        self.channels=tuple(_positive_int('scale channels',c) for c in channels)
        self.scales=nn.ModuleList([PETGuidedStructureFusion(c,**fusion_kwargs) for c in self.channels])

    def forward_with_features(self, ct_feats: Sequence[Tensor],
                              pet_feats: Optional[Sequence[Tensor]] = None,
                              forward_mode: str = 'full', pet_available: object = None
                              ) -> Tuple[List[Tensor],List[Tensor],List[Tensor]]:
        if not isinstance(ct_feats,(list,tuple)) or len(ct_feats)!=4:
            raise ValueError('ct_feats must be a list/tuple of exactly four scales')
        for layer,c in zip(self.scales,ct_feats):
            if not isinstance(c,Tensor) or c.ndim!=4 or not c.is_floating_point():
                raise ValueError('every CT scale must be floating [B,C,H,W]')
            if c.shape[1]!=layer.channels or min(c.shape[0],c.shape[2],c.shape[3])<1:
                raise ValueError('invalid CT scale channels or empty batch/spatial axis')
        if forward_mode not in ('full','missing','auto'):
            raise ValueError(f'unsupported forward_mode={forward_mode!r}')
        # Missing ignores PET, including an invalid object: no hidden PET checks/calls.
        if forward_mode=='missing':pets=[None]*4
        else:
            if forward_mode=='auto':
                state=_validate_state(pet_available,ct_feats[0].shape[0],ct_feats[0].device)
                if not bool(state.any()):pets=[None]*4
                else:pets=pet_feats
            else:pets=pet_feats
            if pets is not None and (not isinstance(pets,(list,tuple)) or len(pets)!=4):
                raise ValueError('pet_feats must contain exactly four scales')
            if pets is None:pets=[None]*4
        batch=ct_feats[0].shape[0]
        device=ct_feats[0].device
        for c in ct_feats:
            if c.shape[0]!=batch or c.device!=device:
                raise ValueError('CT scales must share batch size/device')
        fused,ce,pe=[],[],[]
        for layer,c,p in zip(self.scales,ct_feats,pets):
            f,a,b=layer(c,p,forward_mode=forward_mode,pet_available=pet_available)
            fused.append(f);ce.append(a);pe.append(b)
        return fused,ce,pe

    def forward(self, ct_feats: Sequence[Tensor],
                pet_feats: Optional[Sequence[Tensor]] = None,
                forward_mode: str = 'full', pet_available: object = None) -> List[Tensor]:
        return self.forward_with_features(ct_feats,pet_feats,forward_mode,pet_available)[0]


def _smoke(device: str) -> None:
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(2023)
    if device=='cpu':torch.set_num_threads(1)
    multi=MultiScalePETGuidedStructureFusion().to(device)
    sizes=(128,64,32,16)
    c=[torch.randn(1,ch,n,n,device=device) for ch,n in zip(multi.channels,sizes)]
    p=[torch.randn_like(t) for t in c]
    with torch.no_grad():
        f=multi(c,p)
        assert all(torch.equal(o,a+b) for o,a,b in zip(f,c,p))
        missing=multi(c,None,forward_mode='missing')
        assert all(a is b for a,b in zip(c,missing))
    one=PETGuidedStructureFusion(64).to(device)
    optimizer=torch.optim.AdamW(one.parameters(),lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out=one(c[0][:,:,:16,:16],p[0][:,:,:16,:16])[0]
        loss=out.square().mean();loss.backward();optimizer.step()
    assert all(torch.isfinite(q.grad).all() for q in one.parameters() if q.grad is not None)
    print({'device':device,'four_scale_shapes':[tuple(t.shape) for t in f],
           'parameters':sum(t.numel() for t in multi.parameters()),
           'step_zero':'exact CT+PET','missing':'CT only',
           'two_step_backward':'finite'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.smoke:_smoke(args.device)
    else:parser.print_help()
