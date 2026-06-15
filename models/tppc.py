import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedPseudoPETCorrection(nn.Module):
    def __init__(self, channels, embed_dim=256):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.gamma_beta_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, c * 2),
            ) for c in channels
        ])
        self.dw_pw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for c in channels
        ])

    def forward(self, enhanced_ct_feats, text_embed):
        text_feats = []
        if text_embed is not None and not torch.isfinite(text_embed).all():
            raise RuntimeError('[NaN/Inf] text_embed contains invalid values')
        for idx, (feat, gb_mlp, conv) in enumerate(zip(enhanced_ct_feats, self.gamma_beta_mlps, self.dw_pw_convs)):
            if not torch.isfinite(feat).all():
                raise RuntimeError(f'[NaN/Inf] enhanced_ct_feats[{idx}] contains invalid values')
            gb = gb_mlp(text_embed)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            x = gamma * feat + beta
            out = conv(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4))
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            if not torch.isfinite(out).all():
                raise RuntimeError(f'[NaN/Inf] text_feats[{idx}] contains invalid values')
            text_feats.append(out)
        return text_feats
