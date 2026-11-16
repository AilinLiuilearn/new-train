"""BCSI-Adapt: hard channel tokens, differentiable routing surrogate, raw values.

The reference is taozh2017/BCSI commit 3bbba85042cfc2bbe2a6f5ffedc5356c8946803d.
This is an independently written paired-2D adaptation, not copied upstream code.
The original uses history retrieval and training-only interaction. Here paired
CT/PET interact at train and inference, using 8x8 descriptions for Q/K and
full-resolution selected source maps for V. No learned residual gate is added.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

WIDTHS = (64, 128, 320, 512)


class Router(nn.Module):
    def __init__(self, width, fraction=.5):
        super().__init__()
        if fraction != .5:
            raise ValueError('The user-selected first recipe fixes 50% channel selection')
        self.k = width // 2
        self.score = nn.Sequential(nn.Conv2d(width, width // 4, 1), nn.ReLU(),
                                   nn.Conv2d(width // 4, width, 1))

    def forward(self, x, capture=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = self.score(x.float().mean((-2, -1), keepdim=True)).flatten(1)
            indices = scores.argsort(dim=1, descending=True, stable=True)[:, :self.k]
            hard = torch.zeros_like(scores).scatter(1, indices, 1.)
            if self.training:
                # Hard forward. Softmax's denominator supplies gradients to unselected scores.
                # This is a surrogate for the discrete selection, not its exact derivative.
                soft = self.k * scores.softmax(1)
                mask = hard + (soft - soft.detach())
            else:
                mask = hard
            selected_weights = mask.gather(1, indices)
            if capture and scores.requires_grad:
                scores.retain_grad()
        selected = x.gather(1, indices[:, :, None, None].expand(-1, -1, *x.shape[-2:]))
        selected = selected * selected_weights.to(selected.dtype)[:, :, None, None]
        return selected, indices, selected_weights, scores


class Direction(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)

    def forward(self, receiver, source, capture=False):
        with torch.autocast(device_type=receiver.device.type, enabled=False):
            receiver_descriptor = F.adaptive_avg_pool2d(receiver.float(), (8, 8)).flatten(2)
            source_descriptor = F.adaptive_avg_pool2d(source.float(), (8, 8)).flatten(2)
            q = self.q(receiver_descriptor)
            k = self.k(source_descriptor)
            attention = (torch.bmm(q, k.transpose(1, 2)) / math.sqrt(64)).softmax(-1)
            # V keeps H*W; the attention is K_receiver x K_source, not spatial attention.
            message = torch.bmm(attention, source.float().flatten(2)).reshape_as(receiver)
            info = {}
            if capture:
                a = attention.detach()
                info = {'attention': a,
                        'normalized_entropy': (-(a * a.clamp_min(1e-12).log()).sum(-1)
                                               / math.log(a.shape[-1])).mean(),
                        'top1': a.amax(-1).mean(),
                        'uniform_top1': 1. / a.shape[-1]}
        return message.to(receiver.dtype), info


class Scale(nn.Module):
    def __init__(self, width, fraction=.5):
        super().__init__()
        self.ct_router = Router(width, fraction)
        self.pet_router = Router(width, fraction)
        self.to_ct = Direction()
        self.to_pet = Direction()

    def forward(self, ct, pet, capture=False):
        if ct.ndim != 4 or ct.shape != pet.shape:
            raise ValueError('Aligned same-shape paired features required')
        cs, ci, cw, cscore = self.ct_router(ct, capture)
        ps, pi, pw, pscore = self.pet_router(pet, capture)
        mc, ac = self.to_ct(cs, ps, capture)
        mp, ap = self.to_pet(ps, cs, capture)
        mc = mc * cw.to(mc.dtype)[:, :, None, None]
        mp = mp * pw.to(mp.dtype)[:, :, None, None]
        dc = torch.zeros_like(ct).scatter(1, ci[:, :, None, None].expand_as(mc), mc)
        dp = torch.zeros_like(pet).scatter(1, pi[:, :, None, None].expand_as(mp), mp)
        info = {}
        if capture:
            info = {'ct_indices': ci.detach(), 'pet_indices': pi.detach(),
                    'ct_scores': cscore, 'pet_scores': pscore,
                    'ct_delta': dc.detach(), 'pet_delta': dp.detach(),
                    'ct_attention': ac, 'pet_attention': ap}
        # Intentional direct residual per the user's proposal, not zero-initialized.
        return ct + dc, pet + dp, info


class BCSIAdapt(nn.Module):
    def __init__(self, fraction=.5):
        super().__init__()
        self.blocks = nn.ModuleList(Scale(width, fraction) for width in WIDTHS)

    def forward(self, ct, pet, capture=False):
        if len(ct) != 4 or len(pet) != 4:
            raise ValueError('Exactly four CT/PET scales required')
        cs, ps, details = [], [], []
        for block, c, p in zip(self.blocks, ct, pet):
            cc, pp, info = block(c, p, capture)
            cs.append(cc); ps.append(pp); details.append(info)
        return cs, ps, details
