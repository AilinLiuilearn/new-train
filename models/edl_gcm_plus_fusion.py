import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.weight_gen = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        mu = self.gap(x)
        mx = self.gmp(x)
        mean_sq = (x ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(torch.clamp(mean_sq - mu ** 2, min=0.0) + 1e-6)
        fused = self.fusion(torch.cat([mu, mx, std], dim=1))
        weights = self.weight_gen(fused)
        return weights[:, 0:1] * mu + weights[:, 1:2] * mx + weights[:, 2:3] * std


class CEIG(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(f'dim={dim} must be divisible by num_groups={num_groups}')
        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.q_proj = nn.Linear(self.group_dim, self.group_dim)
        self.kv_proj = nn.Linear(self.group_dim, self.group_dim * 2)
        self.out_proj = nn.Linear(self.group_dim, self.group_dim)
        self.scale = self.group_dim ** -0.5

    def forward(self, ct_stat, pet_stat):
        b, c, _, _ = ct_stat.shape
        ct_tokens = ct_stat.view(b, self.num_groups, self.group_dim)
        pet_tokens = pet_stat.view(b, self.num_groups, self.group_dim)
        q = self.q_proj(ct_tokens)
        key, value = self.kv_proj(pet_tokens).chunk(2, dim=-1)
        attn = torch.matmul(q, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = self.out_proj(torch.matmul(attn, value))
        return (ct_tokens + out).view(b, c, 1, 1)


class EDLGlobalHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.constant_(self.net[-1].bias, 0.54)

    def forward(self, x):
        alpha = 1.0 + F.softplus(self.net(x))
        strength = alpha.sum(dim=1, keepdim=True)
        p_tumor = alpha[:, 1:2] / (strength + 1e-6)
        uncertainty = 2.0 / (strength + 1e-6)
        return p_tumor, uncertainty, strength


class DCG(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(f'dim={dim} must be divisible by num_groups={num_groups}')
        self.num_groups = num_groups
        self.cpg = dim // num_groups
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.cpg, kernel_size=1),
            )
            for _ in range(num_groups)
        ])

    def forward(self, ct_stat, pet_evidence):
        b = ct_stat.shape[0]
        ct_groups = ct_stat.view(b, self.num_groups, self.cpg, 1, 1)
        logits = []
        for idx, mlp in enumerate(self.group_mlps):
            group_stat = ct_groups[:, idx].mean(dim=1, keepdim=True)
            logits.append(mlp(torch.cat([group_stat, pet_evidence], dim=1)))
        return torch.cat(logits, dim=1)


class ATM(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, logits, pet_raw):
        b = pet_raw.shape[0]
        pet_fp32 = pet_raw.float()
        pet_mean = pet_fp32.mean(dim=(1, 2, 3))
        pet_std = pet_fp32.std(dim=(1, 2, 3), unbiased=False)
        stats = torch.stack([pet_mean, pet_std], dim=1).to(dtype=logits.dtype)
        tau = 0.5 + self.temp_mlp(stats) * 1.5
        tau = tau.view(b, 1, 1, 1)
        return torch.sigmoid(logits / tau), tau.view(b)


class EDLGCMPlus(nn.Module):
    def __init__(self, dim, num_groups=8, init_gamma=0.01):
        super().__init__()
        self.mssg_ct = MSSG(dim)
        self.mssg_pet = MSSG(dim)
        self.cei_g = CEIG(dim, num_groups=num_groups)
        self.edl_head = EDLGlobalHead(dim)
        self.r_pet = nn.Parameter(torch.tensor(0.0))
        self.dcg = DCG(dim, num_groups=num_groups)
        self.atm = ATM()
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, ct_skip, pet_skip):
        ct_stat = self.mssg_ct(ct_skip)
        pet_stat = self.mssg_pet(pet_skip)
        ct_enhanced = self.cei_g(ct_stat, pet_stat)
        p_tumor, uncertainty, _ = self.edl_head(pet_stat)
        r_pet = torch.sigmoid(self.r_pet)
        strength = r_pet * (1.0 - uncertainty).clamp(min=0.0, max=1.0) * p_tumor
        logits = self.dcg(ct_enhanced, strength)
        weights, tau = self.atm(logits, pet_skip)
        delta = (weights - 0.5) * 2.0
        modulation = self.gamma * strength * delta
        out = ct_skip * (1.0 + modulation)
        aux = {
            'p_tumor': p_tumor.detach(),
            'uncertainty': uncertainty.detach(),
            'r_pet': r_pet.detach(),
            'strength': strength.detach(),
            'tau': tau.detach(),
            'weights': weights.detach(),
            'modulation': modulation.detach().abs().mean(),
        }
        return out, aux


class EDLGCMPlusBottleneckFusion(nn.Module):
    def __init__(self, channels, num_groups=8, init_gamma=0.01, shallow_mode='sum'):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f'EDLGCMPlusBottleneckFusion expects 4 feature stages, got {len(channels)}')
        if shallow_mode not in ('sum', 'ct'):
            raise ValueError(f"Unsupported shallow_mode={shallow_mode}. Use 'sum' or 'ct'.")
        self.shallow_mode = shallow_mode
        self.stage4 = EDLGCMPlus(channels[-1], num_groups=num_groups, init_gamma=init_gamma)
        self._last_aux = None

    def forward(self, ct_feats, pet_feats):
        if self.shallow_mode == 'ct':
            fused_feats = [ct_feats[0], ct_feats[1], ct_feats[2]]
        else:
            fused_feats = [c + p for c, p in zip(ct_feats[:3], pet_feats[:3])]
        fused4, aux4 = self.stage4(ct_feats[3], pet_feats[3])
        fused_feats.append(fused4)
        self._last_aux = aux4
        return fused_feats, {'edl_gcm_plus_stage4': aux4}

    def get_fusion_visuals(self):
        if not self._last_aux:
            return {}
        return {
            'edl_gcm_plus_p_tumor': self._last_aux['p_tumor'],
            'edl_gcm_plus_uncertainty': self._last_aux['uncertainty'],
            'edl_gcm_plus_strength': self._last_aux['strength'],
        }
