"""LPI-BI V1 equations. Custom 2D soft prior, not ZePT query/masked attention.

All sliding neighborhoods preserve pixel order; outside-image entries are never
candidate tokens. Pooling, spatial priors and attention arithmetic run in FP32.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6
WIDTHS = (64, 128, 320, 512)


def spatial_descriptor(x, response):
    """Peak-response mixture with a total-mass epsilon, not per-pixel eta."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, response = x.float(), response.float()
        global_mean = x.mean((-2, -1))
        spatial_mean = ((x * response).sum((-2, -1)) + EPS * global_mean) / (
            response.sum((-2, -1)) + EPS
        )
        peak = response.amax((-2, -1))
        descriptor = (1 - peak) * global_mean + peak * spatial_mean
    return descriptor, peak


def valid_candidates(height, width, device):
    """1x1x9xN: row-major offsets; index 4 is always a valid center."""
    return F.unfold(torch.ones(1, 1, height, width, device=device),
                    kernel_size=3, padding=1).reshape(1, 1, 9, height * width).bool()


def spatial_bias(response, valid):
    with torch.autocast(device_type=response.device.type, enabled=False):
        neighbors = F.unfold(response.float(), 3, padding=1).reshape(
            response.shape[0], 1, 9, -1)
        logs = (neighbors + EPS).log()
        mean_log = logs.masked_fill(~valid, 0).sum(2, keepdim=True) / valid.sum(2, keepdim=True)
        peak = neighbors.masked_fill(~valid, -torch.inf).amax(2, keepdim=True)
        bias = peak * torch.tanh(logs - mean_log)
        return bias.masked_fill(~valid, 0)


class LocalizationChannel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Conv2d(width, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(width, max(16, width // 4)), nn.GELU(),
                                 nn.Linear(max(16, width // 4), width))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        logits = self.head(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            response = logits.float().sigmoid()
        descriptor, peak = spatial_descriptor(x, response)
        gate = 2 * self.mlp(descriptor).sigmoid()
        return logits, response, gate, peak


class Direction(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q = nn.Conv2d(width, 32, 1)
        self.k = nn.Conv2d(width, 32, 1)
        self.v = nn.Conv2d(width, 32, 1)
        self.out = nn.Conv2d(32, width, 1)
        self.bias_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, receiver, source, receiver_gate, source_gate, response, valid,
                capture=False):
        b, _, h, w = receiver.shape
        q = self.q(receiver * receiver_gate.to(receiver.dtype)[:, :, None, None])
        k = self.k(source * source_gate.to(source.dtype)[:, :, None, None])
        v = self.v(source)  # The content path has neither channel nor spatial masking.
        with torch.autocast(device_type=receiver.device.type, enabled=False):
            q = q.float().reshape(b, 4, 8, h * w)
            kn = F.unfold(k.float(), 3, padding=1).reshape(b, 4, 8, 9, h * w)
            vn = F.unfold(v.float(), 3, padding=1).reshape(b, 4, 8, 9, h * w)
            prior = spatial_bias(response, valid)
            beta = self.bias_logit.float().sigmoid()
            scores = torch.einsum('bhdn,bhdkn->bhkn', q, kn) / math.sqrt(8)
            scores = (scores + beta * prior).masked_fill(~valid, -torch.inf)
            attention = scores.softmax(2)
            message = torch.einsum('bhkn,bhdkn->bhdn', attention, vn).reshape(b, 32, h, w)
            details = {}
            if capture:
                a = attention.detach()
                entropy = -(a * a.clamp_min(1e-12).log()).sum(2)
                denominator = valid.sum(2).float().log().clamp_min(1e-8)
                details = {
                    'beta': beta.detach(),
                    'prior_rms': prior.detach().square().mean().sqrt(),
                    'normalized_attention_entropy': (entropy / denominator).mean(),
                    'attention_top1': a.amax(2).mean(),
                    'uniform_top1': valid.sum(2).float().reciprocal().mean(),
                    'message_rms': message.detach().square().mean().sqrt(),
                }
        delta = self.out(message.to(receiver.dtype))
        return delta, details


class Scale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ct_condition = LocalizationChannel(width)
        self.pet_condition = LocalizationChannel(width)
        self.to_ct = Direction(width)
        self.to_pet = Direction(width)

    def forward(self, ct, pet, capture=False):
        if ct.shape != pet.shape or ct.ndim != 4:
            raise ValueError('LPI-BI requires aligned, same-shape B,C,H,W features')
        zc, sc, gc, ac = self.ct_condition(ct)
        zp, sp, gp, ap = self.pet_condition(pet)
        valid = valid_candidates(*ct.shape[-2:], ct.device)
        dc, info_c = self.to_ct(ct, pet, gc, gp, sp, valid, capture)
        dp, info_p = self.to_pet(pet, ct, gp, gc, sc, valid, capture)
        details = {}
        if capture:
            details = {'ct_response': sc.detach(), 'pet_response': sp.detach(),
                       'ct_gate': gc.detach(), 'pet_gate': gp.detach(),
                       'ct_peak': ac.detach(), 'pet_peak': ap.detach(),
                       'ct_delta': dc.detach(), 'pet_delta': dp.detach(),
                       'to_ct': info_c, 'to_pet': info_p}
        return ct + dc, pet + dp, zc, zp, details


class LPIBI(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Scale(d) for d in WIDTHS)

    def forward(self, ct, pet, capture=False):
        if len(ct) != 4 or len(pet) != 4:
            raise ValueError('Exactly four CT/PET scales required')
        cs, ps, zc, zp, details = [], [], [], [], []
        for block, c, p in zip(self.blocks, ct, pet):
            cc, pp, lc, lp, info = block(c, p, capture)
            cs.append(cc); ps.append(pp); zc.append(lc); zp.append(lp); details.append(info)
        return cs, ps, {'ct': zc, 'pet': zp}, details
