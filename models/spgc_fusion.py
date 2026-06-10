import torch
import torch.nn as nn

from models.edl_spmc_stage3_fusion import EDLSPMCStage3


class SPGC(nn.Module):
    def __init__(self, dim, init_gamma=0.01):
        super().__init__()
        hidden = max(dim // 4, 16)
        in_dim = dim * 2 + 1
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Tanh(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, ct_feat, pet_feat):
        pet_energy = pet_feat.abs().mean(dim=1, keepdim=True)
        x = torch.cat([ct_feat, pet_feat, pet_energy], dim=1)
        gate = self.gate(x)
        delta = self.delta(x)
        scale = 1.0 + self.gamma * gate * delta
        out = ct_feat + scale * pet_feat
        aux = {
            'gate': gate.detach(),
            'scale': scale.detach(),
            'gamma': self.gamma.detach(),
        }
        return out, aux


class SPGCFusion(nn.Module):
    def __init__(self, channels, init_gamma=0.01, use_stage3_spmc=False):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f'SPGCFusion expects 4 stages, got {len(channels)}')
        self.stage1 = SPGC(channels[0], init_gamma=init_gamma)
        self.stage2 = SPGC(channels[1], init_gamma=init_gamma)
        self.stage3 = EDLSPMCStage3(channels[2], init_gamma=init_gamma) if use_stage3_spmc else None
        self._last_aux = {}

    def forward(self, ct_feats, pet_feats):
        fused1, aux1 = self.stage1(ct_feats[0], pet_feats[0])
        fused2, aux2 = self.stage2(ct_feats[1], pet_feats[1])
        if self.stage3 is None:
            fused3 = ct_feats[2] + pet_feats[2]
            aux3 = None
        else:
            fused3, aux3 = self.stage3(ct_feats[2], pet_feats[2])
        fused4 = ct_feats[3] + pet_feats[3]
        self._last_aux = {'stage1': aux1, 'stage2': aux2, 'stage3': aux3}
        aux = {'spgc_s1': aux1, 'spgc_s2': aux2}
        if aux3 is not None:
            aux['edl_spmc_s3'] = aux3
        return [fused1, fused2, fused3, fused4], aux

    def get_fusion_visuals(self):
        visuals = {}
        aux1 = self._last_aux.get('stage1') if self._last_aux else None
        aux2 = self._last_aux.get('stage2') if self._last_aux else None
        aux3 = self._last_aux.get('stage3') if self._last_aux else None
        if aux1:
            visuals.update({
                'spgc_s1_gate': aux1['gate'],
                'spgc_s1_scale': aux1['scale'],
            })
        if aux2:
            visuals.update({
                'spgc_s2_gate': aux2['gate'],
                'spgc_s2_scale': aux2['scale'],
            })
        if aux3:
            visuals.update({
                'edl_spmc_s3_prior': aux3['prior'],
                'edl_spmc_s3_uncertainty': aux3['uncertainty'],
                'edl_spmc_s3_consistency': aux3['consistency'],
                'edl_spmc_s3_strength': aux3['strength'],
            })
        return visuals
