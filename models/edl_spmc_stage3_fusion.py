import torch
import torch.nn as nn
import torch.nn.functional as F


class MSPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid = max(dim // 2, 16)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.GroupNorm(1, mid),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.region = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.GroupNorm(1, mid),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.GroupNorm(1, mid),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(mid * 3, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, 2, 1),
        )
        nn.init.constant_(self.head[-1].bias, 0.54)

    def forward(self, pet):
        feat = torch.cat([self.local(pet), self.region(pet), self.context(pet)], dim=1)
        alpha = 1.0 + F.softplus(self.head(feat))
        strength = alpha.sum(dim=1, keepdim=True)
        prior = alpha[:, 1:2] / (strength + 1e-6)
        uncertainty = 2.0 / (strength + 1e-6)
        return prior, uncertainty.clamp(0.0, 1.0)


class CTBoundaryResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid = max(dim // 2, 16)
        self.smooth = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.GroupNorm(1, mid),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
        )
        self.gate = nn.Sequential(nn.Conv2d(dim, 1, 1), nn.Sigmoid())

    def forward(self, ct):
        boundary = self.refine(ct - self.smooth(ct))
        return boundary * self.gate(boundary)


class SpatialConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_proj = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.corr = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, prior, ct):
        edge = torch.abs(ct - F.avg_pool2d(ct, 3, stride=1, padding=1)).mean(dim=1, keepdim=True)
        edge = self.edge_proj(edge)
        return self.corr(torch.cat([prior, edge], dim=1))


class AdaptiveSpatialModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_pet = nn.Parameter(torch.tensor(0.0))
        self.mod_mlp = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, prior, uncertainty, consistency):
        r_pet = torch.sigmoid(self.r_pet)
        certainty = (1.0 - uncertainty).clamp(0.0, 1.0)
        base_conf = r_pet * prior * consistency * certainty
        modulation = self.mod_mlp(torch.cat([prior, uncertainty, consistency], dim=1))
        return base_conf, modulation, r_pet


class EDLSPMCStage3(nn.Module):
    def __init__(self, dim, init_gamma=0.01):
        super().__init__()
        self.mspe = MSPE(dim)
        self.boundary = CTBoundaryResidual(dim)
        self.consistency = SpatialConsistency()
        self.asm = AdaptiveSpatialModulation()
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, ct_feat, pet_feat):
        prior, uncertainty = self.mspe(pet_feat)
        boundary = self.boundary(ct_feat)
        consistency = self.consistency(prior, ct_feat)
        base_conf, modulation, r_pet = self.asm(prior, uncertainty, consistency)
        strength = base_conf * modulation
        base = ct_feat + pet_feat
        out = base + self.gamma * strength * boundary
        aux = {
            'prior': prior.detach(),
            'uncertainty': uncertainty.detach(),
            'consistency': consistency.detach(),
            'strength': strength.detach(),
            'r_pet': r_pet.detach(),
            'gamma': self.gamma.detach(),
        }
        return out, aux


class EDLSPMCStage3Fusion(nn.Module):
    def __init__(self, channels, init_gamma=0.01):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f'EDLSPMCStage3Fusion expects 4 stages, got {len(channels)}')
        self.stage3 = EDLSPMCStage3(channels[2], init_gamma=init_gamma)
        self._last_aux = None

    def forward(self, ct_feats, pet_feats):
        fused = [ct_feats[0] + pet_feats[0], ct_feats[1] + pet_feats[1]]
        fused3, aux3 = self.stage3(ct_feats[2], pet_feats[2])
        fused.extend([fused3, ct_feats[3] + pet_feats[3]])
        self._last_aux = aux3
        return fused, {'edl_spmc_s3': aux3}

    def get_fusion_visuals(self):
        if not self._last_aux:
            return {}
        return {
            'edl_spmc_s3_prior': self._last_aux['prior'],
            'edl_spmc_s3_uncertainty': self._last_aux['uncertainty'],
            'edl_spmc_s3_consistency': self._last_aux['consistency'],
            'edl_spmc_s3_strength': self._last_aux['strength'],
        }
