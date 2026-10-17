"""Our bidirectional local-content reassembly; not a reproduction of GeminiFusion.

At each scale CT queries PET's 3x3 neighborhood and PET queries CT's.
Each direction includes a learned key with a FIXED zero value (abstain).
Corrections compare transported content to the original center content.
Only the two output projections start at zero: F = C + P exactly at step 0.
Uses ordinary PyTorch, no Triton, deformable sampling or global attention.
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STAGE_CHANNELS = (64, 128, 320, 512)
STAGE_RANKS = (16, 16, 40, 64)
DIAGNOSTIC_KEYS = tuple(f'local_{kind}_s{i}' for i in range(1, 5)
                        for kind in ('delta_ratio', 'null_mass', 'center_mass'))


class LocalTransport(nn.Module):
    """B,r,H,W queries/keys/values -> content plus B,heads,H,W masses.

    Attention covers nine spatial candidates and a null candidate. Invalid
    padded neighbors are masked before softmax, including singleton maps.
    Shifted views avoid materializing an unfolded B*r*9*H*W value tensor.
    Null mass is an attention statistic, NOT calibrated clinical reliability.
    """
    def __init__(self, rank: int, heads: int = 2):
        super().__init__()
        if type(rank) is not int or type(heads) is not int or heads < 1 or rank < 1 or rank % heads:
            raise ValueError('Positive rank must be divisible by positive heads')
        self.rank, self.heads = rank, heads
        self.null_key = nn.Parameter(torch.empty(heads, rank // heads))
        nn.init.normal_(self.null_key, std=.02)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        b, r, h, w = q.shape
        d = r // self.heads
        qh = q.reshape(b, self.heads, d, h, w)
        kp = F.pad(k, (1, 1, 1, 1)).reshape(b, self.heads, d, h+2, w+2)
        vp = F.pad(v, (1, 1, 1, 1)).reshape(b, self.heads, d, h+2, w+2)
        valid = F.pad(torch.ones(1, 1, h, w, dtype=torch.bool, device=q.device), (1,1,1,1))
        scores = []
        for y in range(3):
            for x in range(3):
                score = (qh * kp[..., y:y+h, x:x+w]).sum(2) * d**-.5
                scores.append(score.masked_fill(~valid[..., y:y+h, x:x+w], -torch.inf))
        scores.append((qh * self.null_key[None, :, :, None, None]).sum(2) * d**-.5)
        attention = torch.stack(scores, dim=2).softmax(dim=2)
        content = torch.zeros_like(qh)
        for index, (y, x) in enumerate((y,x) for y in range(3) for x in range(3)):
            content = content + attention[:, :, index:index+1] * vp[..., y:y+h, x:x+w]
        return content.reshape(b,r,h,w), attention[:,:,9], attention[:,:,4]


class ReassemblyBlock(nn.Module):
    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.ct_project = nn.Conv2d(channels, rank, 1, bias=False)
        self.pet_project = nn.Conv2d(channels, rank, 1, bias=False)
        self.ct_norm, self.pet_norm = nn.LayerNorm(rank), nn.LayerNorm(rank)
        self.ct_query, self.pet_query = nn.Conv2d(rank,rank,1,bias=False), nn.Conv2d(rank,rank,1,bias=False)
        self.ct_key, self.pet_key = nn.Conv2d(rank,rank,1,bias=False), nn.Conv2d(rank,rank,1,bias=False)
        self.pet_to_ct, self.ct_to_pet = LocalTransport(rank), LocalTransport(rank)
        self.up_ct, self.up_pet = nn.Conv2d(rank,channels,1,bias=False), nn.Conv2d(rank,channels,1,bias=False)
        nn.init.zeros_(self.up_ct.weight)
        nn.init.zeros_(self.up_pet.weight)

    def forward(self, ct: Tensor, pet: Tensor):
        with torch.autocast(device_type=ct.device.type, enabled=False):
            c, p = self.ct_project(ct.float()), self.pet_project(pet.float())
            cn = self.ct_norm(c.permute(0,2,3,1)).permute(0,3,1,2)
            pn = self.pet_norm(p.permute(0,2,3,1)).permute(0,3,1,2)
            tp, np, cp = self.pet_to_ct(self.ct_query(cn), self.pet_key(pn), p)
            tc, nc, cc = self.ct_to_pet(self.pet_query(pn), self.ct_key(cn), c)
            delta = self.up_pet(tp-p) + self.up_ct(tc-c)
        return delta, (np+nc)*.5, (cp+cc)*.5


class LocalReassemblyFusion(nn.Module):
    """Four CT/PET pairs -> four CT-shaped tensors; model owns state routing.

    Full: C+P+delta_C+delta_P. Disabled: pure Add with zero parameters.
    Missing: parent model bypasses this module and the PET encoder entirely.
    FP32 master weights required; half/bfloat16 FEATURES and autocast supported.
    """
    def __init__(self, enabled: bool = True):
        super().__init__()
        if type(enabled) is not bool:
            raise ValueError('local_enabled must be boolean')
        self.enabled = enabled
        self.blocks = nn.ModuleList([ReassemblyBlock(c,r) for c,r in zip(STAGE_CHANNELS,STAGE_RANKS)]) if enabled else nn.ModuleList()
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._diagnostic_sums = None
        self._diagnostic_samples = 0

    def pop_diagnostics(self):
        values = ([0.]*len(DIAGNOSTIC_KEYS) if self._diagnostic_sums is None else
                  (self._diagnostic_sums / self._diagnostic_samples).cpu().tolist())
        self.reset_diagnostics()
        return dict(zip(DIAGNOSTIC_KEYS, values))

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('Four CT/PET scales required, ordered S1..S4')
        if self.enabled and self.blocks[0].up_ct.weight.dtype != torch.float32:
            raise ValueError('Use FP32 master weights with autocast, not model.half()/double()')
        outputs, diagnostics = [], []
        for i,(ct,pet,channels) in enumerate(zip(ct_feats,pet_feats,STAGE_CHANNELS)):
            if (not isinstance(ct,Tensor) or not isinstance(pet,Tensor) or ct.ndim != 4 or pet.ndim != 4
                    or ct.shape[:2] != pet.shape[:2] or ct.shape[1] != channels
                    or ct.device != pet.device or ct.dtype != pet.dtype
                    or ct.dtype not in (torch.float32,torch.float16,torch.bfloat16)
                    or ct.shape[0] < 1 or min(*ct.shape[-2:],*pet.shape[-2:]) < 1
                    or ct.shape[0] != ct_feats[0].shape[0]):
                raise ValueError(f'S{i+1}: incompatible floating BCHW inputs, channels={channels}')
            if not torch.isfinite(ct).all() or not torch.isfinite(pet).all():
                raise RuntimeError(f'Non-finite local fusion input S{i+1}')
            if ct.shape[-2:] != pet.shape[-2:]:
                pet = F.interpolate(pet,ct.shape[-2:],mode='bilinear',align_corners=False)
            base = ct+pet
            if self.enabled:
                delta,null_mass,center_mass = self.blocks[i](ct,pet)
                out = base + delta.to(base.dtype)
                if self.training:
                    with torch.no_grad():
                        ratio = delta.detach().flatten(1).norm(dim=1) / base.detach().float().flatten(1).norm(dim=1).clamp_min(1e-6)
                        diagnostics.extend([ratio.sum(),null_mass.detach().flatten(1).mean(1).sum(),
                                            center_mass.detach().flatten(1).mean(1).sum()])
            else:
                out = base
            if not torch.isfinite(out).all():
                raise RuntimeError(f'Non-finite local fusion output S{i+1}')
            outputs.append(out)
        if diagnostics:
            sums = torch.stack(diagnostics)
            self._diagnostic_sums = sums if self._diagnostic_sums is None else self._diagnostic_sums+sums
            self._diagnostic_samples += ct_feats[0].shape[0]
        return outputs
