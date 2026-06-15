import torch
import torch.nn as nn


class PlaceholderPriorAdapter(nn.Module):
    """Placeholder anatomical prior adapter.

    It keeps a MedDINOv3-replaceable interface while currently deriving prior
    features from CT features with lightweight convolutions.
    """

    def __init__(self, channels):
        super().__init__()
        self.prior_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
            ) for c in channels
        ])
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, 1, 3, padding=1, bias=True),
                nn.Sigmoid(),
            ) for c in channels
        ])
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in channels])

    def forward(self, ct_feats, prior_gates=None):
        enhanced = []
        prior_feats = []
        for i, (feat, prior_conv, spatial_gate, alpha) in enumerate(zip(ct_feats, self.prior_convs, self.spatial_gates, self.alpha)):
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
            prior = prior_conv(feat)
            prior = torch.nan_to_num(prior, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = spatial_gate(torch.cat([feat, prior], dim=1))
            prior_gate = 1.0 if prior_gates is None else prior_gates[i]
            enhanced_feat = feat + alpha * prior_gate * gate * prior
            enhanced.append(torch.nan_to_num(enhanced_feat, nan=0.0, posinf=1e4, neginf=-1e4))
            prior_feats.append(prior)
        return enhanced, prior_feats


class TextGuidedCorrectionFusion(nn.Module):
    def __init__(self, text_in_full_mode=False, full_text_weight=0.0):
        super().__init__()
        self.text_in_full_mode = bool(text_in_full_mode)
        self.full_text_weight = float(full_text_weight)

    def forward(self, enhanced_ct_feats, pet_feats, text_feats, pet_gates, text_gates, pet_available):
        if pet_available.dim() > 1:
            pet_available = pet_available.view(-1)
        m = pet_available.float().view(-1, 1, 1, 1)
        fused = []
        pet_gate_means = []
        text_gate_means = []
        for fact, dpet, dtxt, gpet, gtxt in zip(enhanced_ct_feats, pet_feats, text_feats, pet_gates, text_gates):
            fact = torch.nan_to_num(fact, nan=0.0, posinf=1e4, neginf=-1e4)
            dpet = torch.nan_to_num(dpet, nan=0.0, posinf=1e4, neginf=-1e4)
            dtxt = torch.nan_to_num(dtxt, nan=0.0, posinf=1e4, neginf=-1e4)
            out = fact + m * gpet * dpet + (1.0 - m) * gtxt * dtxt
            if self.text_in_full_mode:
                out = out + m * self.full_text_weight * gtxt * dtxt
            fused.append(torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4))
            pet_gate_means.append(gpet.mean())
            text_gate_means.append(gtxt.mean())
        aux = {
            'pet_gate_mean': torch.stack(pet_gate_means).mean(),
            'text_gate_mean': torch.stack(text_gate_means).mean(),
        }
        return fused, aux
