import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        hidden = max(in_dim // 2, num_classes * 4)
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, x):
        alpha = 1.0 + F.softplus(self.net(x))
        evidence = alpha - 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        belief = evidence / (strength + 1e-6)
        uncertainty = self.num_classes / (strength + 1e-6)
        return alpha, evidence, belief, uncertainty


class WindowedEvidenceCrossAttention(nn.Module):
    def __init__(self, num_classes=2, embed_dim=16, window_size=8, num_heads=2):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.ct_proj = nn.Conv2d(num_classes, embed_dim, kernel_size=1)
        self.pet_proj = nn.Conv2d(num_classes, embed_dim, kernel_size=1)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, num_classes)
        self.delta_scale = nn.Parameter(torch.tensor(0.01))
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _pad_to_window(x, window_size):
        h, w = x.shape[-2:]
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), h, w

    def forward(self, e_ct, e_pet):
        b, k, h0, w0 = e_ct.shape
        ws = min(self.window_size, h0, w0)

        e_ct_pad, h, w = self._pad_to_window(e_ct, ws)
        e_pet_pad, _, _ = self._pad_to_window(e_pet, ws)
        hp, wp = e_ct_pad.shape[-2:]

        ct_emb = self.ct_proj(e_ct_pad)
        pet_emb = self.pet_proj(e_pet_pad)

        ct_emb = ct_emb.view(
            b, self.embed_dim, hp // ws, ws, wp // ws, ws
        ).permute(0, 2, 4, 3, 5, 1)

        pet_emb = pet_emb.view(
            b, self.embed_dim, hp // ws, ws, wp // ws, ws
        ).permute(0, 2, 4, 3, 5, 1)

        _, nh, nw, _, _, e = ct_emb.shape
        n = ws * ws

        ct_tokens = ct_emb.reshape(b * nh * nw, n, e)
        pet_tokens = pet_emb.reshape(b * nh * nw, n, e)

        q = self.q_proj(ct_tokens).view(
            -1, n, self.num_heads, self.head_dim
        ).transpose(1, 2)

        kv = self.kv_proj(pet_tokens)
        key, value = kv.chunk(2, dim=-1)

        key = key.view(
            -1, n, self.num_heads, self.head_dim
        ).transpose(1, 2)

        value = value.view(
            -1, n, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = torch.matmul(q, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, value).transpose(1, 2).reshape(-1, n, e)
        delta = self.out_proj(out)

        delta = delta.view(
            b, nh, nw, ws, ws, k
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, k, hp, wp)

        delta = delta[:, :, :h, :w]

        return F.softplus(e_ct + self.delta_scale * delta)


class SubjectiveLogicFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_ct_logit = nn.Parameter(torch.tensor(0.0))
        self.r_pet_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, e_ct, e_pet):
        r_ct = torch.sigmoid(self.r_ct_logit)
        r_pet = torch.sigmoid(self.r_pet_logit)
        alpha_fused = r_ct * e_ct + r_pet * e_pet + 1.0
        return alpha_fused, r_ct, r_pet


class EvidenceToFeature(nn.Module):
    def __init__(self, num_classes, out_dim):
        super().__init__()
        hidden = max(out_dim // 2, num_classes * 4)
        self.net = nn.Sequential(
            nn.Conv2d(num_classes, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, alpha):
        prob = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-6)
        return self.net(prob)


class HECCM(nn.Module):
    def __init__(
        self,
        dim,
        num_classes=2,
        window_size=8,
        embed_dim=16,
        num_heads=2,
        init_gamma=0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ct_evd = EvidentialHead(dim, num_classes)
        self.pet_evd = EvidentialHead(dim, num_classes)
        self.evd_cross = WindowedEvidenceCrossAttention(
            num_classes=num_classes,
            embed_dim=embed_dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.sl_fusion = SubjectiveLogicFusion()
        self.evd2feat = EvidenceToFeature(num_classes, dim)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, ct_feat, pet_feat):
        _, e_ct, _, unc_ct = self.ct_evd(ct_feat)
        _, e_pet, _, unc_pet = self.pet_evd(pet_feat)

        e_ct_refined = self.evd_cross(e_ct, e_pet)
        alpha_fused, r_ct, r_pet = self.sl_fusion(e_ct_refined, e_pet)

        fused_feat = self.evd2feat(alpha_fused)
        out = ct_feat + self.gamma * fused_feat

        strength = alpha_fused.sum(dim=1, keepdim=True)
        unc_fused = self.num_classes / (strength + 1e-6)

        aux = {
            "unc_ct": unc_ct.detach(),
            "unc_pet": unc_pet.detach(),
            "unc_fused": unc_fused.detach(),
            "r_ct": r_ct.detach(),
            "r_pet": r_pet.detach(),
            "gamma": self.gamma.detach(),
        }

        return out, aux


class MultiStageHECCMFusion(nn.Module):
    def __init__(
        self,
        channels,
        num_classes=2,
        window_sizes=(16, 8, 8, 4),
        embed_dim=16,
        num_heads=2,
        init_gamma=0.01,
    ):
        super().__init__()

        if len(window_sizes) < len(channels):
            window_sizes = list(window_sizes) + [window_sizes[-1]] * (
                len(channels) - len(window_sizes)
            )

        self.blocks = nn.ModuleList([
            HECCM(
                dim=ch,
                num_classes=num_classes,
                window_size=window_sizes[i],
                embed_dim=embed_dim,
                num_heads=num_heads,
                init_gamma=init_gamma,
            )
            for i, ch in enumerate(channels)
        ])

        self._last_aux = []

    def forward(self, ct_feats, pet_feats):
        fused_feats = []
        aux_list = []

        for block, ct_feat, pet_feat in zip(self.blocks, ct_feats, pet_feats):
            fused, aux = block(ct_feat, pet_feat)
            fused_feats.append(fused)
            aux_list.append(aux)

        self._last_aux = aux_list
        return fused_feats, aux_list

    def get_fusion_visuals(self):
        visuals = {}

        for idx, aux in enumerate(self._last_aux, start=1):
            visuals[f"heccm_s{idx}_unc_fused"] = aux["unc_fused"]
            visuals[f"heccm_s{idx}_unc_ct"] = aux["unc_ct"]
            visuals[f"heccm_s{idx}_unc_pet"] = aux["unc_pet"]

        return visuals
