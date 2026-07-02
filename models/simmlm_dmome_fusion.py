import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    GroupNorm is more stable than BatchNorm for small-batch medical segmentation.
    """
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


DEFAULT_PRIOR_GATE_STAGES = (1, 2, 3, 4)
VALID_PRIOR_GATE_STAGES = frozenset({1, 2, 3, 4})
DEFAULT_HYBRID_CONCAT_STAGES = (1, 2, 3)
DEFAULT_HYBRID_DMOME_STAGES = (4,)
CHANNEL_PRIOR_ALPHA_MAX = 0.20
CHANNEL_PRIOR_RAW_ALPHA_INIT = -2.197  # sigmoid(-2.197) * 0.20 ≈ 0.02


class TextGuidedChannelPriorGate2D(nn.Module):
    """
    Text-guided channel residual prior for gate inputs only.

        concat(text_to_vision(text), GAP+Conv(visual)) -> MLP -> channel_mask [B,C,1,1]
        f_prior = f * (1 + alpha * channel_mask)
        alpha = alpha_max * sigmoid(raw_alpha)
    """

    def __init__(
        self,
        channels: int,
        text_dim: int = 512,
        hidden_dim: int = 256,
        alpha_max: float = CHANNEL_PRIOR_ALPHA_MAX,
        raw_alpha_init: float = CHANNEL_PRIOR_RAW_ALPHA_INIT,
    ):
        super().__init__()

        self.channels = channels
        self.alpha_max = float(alpha_max)
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha_init)))

        self.text_to_vision = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.Flatten(start_dim=1),
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(self, f_m: torch.Tensor, text_embed_m: torch.Tensor):
        B, C, H, W = f_m.shape

        text_vec = self.text_to_vision(text_embed_m)
        text_vec = text_vec.unsqueeze(0).expand(B, -1)

        visual_vec = self.gap_conv(f_m)
        prior_vec = torch.cat([text_vec, visual_vec], dim=1)

        channel_mask = self.channel_mlp(prior_vec).view(B, C, 1, 1)
        alpha = self.alpha()
        prior_scale = 1.0 + alpha * channel_mask
        f_prior = f_m * prior_scale

        return f_prior, channel_mask, alpha, prior_scale


class ModalityExpert2D(nn.Module):
    """
    Lightweight modality-specific expert.

    CT and PET have separate experts.
    Each expert only processes its corresponding modality feature.

    Structure:
        C -> C/r -> DWConv3x3 -> C

    with residual connection.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_hidden: int = 16,
        norm_groups: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()

        hidden = max(channels // reduction, min_hidden)
        self.use_residual = use_residual

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            make_group_norm(hidden, max_groups=norm_groups),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            make_group_norm(hidden, max_groups=norm_groups),
            nn.GELU(),

            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            make_group_norm(channels, max_groups=norm_groups),
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return self.act(x + self.net(x))
        return self.act(self.net(x))


class SimMLMAlignedGate2D(nn.Module):
    """
    Dynamic modality gate.

    Input:
        CT feature, raw PET feature, PET availability token

    Output:
        alpha_ct, alpha_pet
    """

    def __init__(
        self,
        channels: int,
        token_dim: int = None,
        hidden_channels: int = None,
        use_status_token: bool = True,
        temperature: float = 1.0,
        init_ct_bias: float = 0.0,
        norm_groups: int = 8,
    ):
        super().__init__()

        self.channels = channels
        self.temperature = temperature
        self.use_status_token = use_status_token

        token_dim = token_dim or channels
        hidden_channels = hidden_channels or max(channels // 4, 16)

        self.pet_status_embed = nn.Embedding(2, token_dim)

        self.gate_feature_extractor = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            make_group_norm(hidden_channels, max_groups=norm_groups),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            make_group_norm(hidden_channels, max_groups=norm_groups),
            nn.GELU(),
        )

        gate_input_dim = hidden_channels + token_dim if use_status_token else hidden_channels
        self.gate_fc = nn.Linear(gate_input_dim, 2)

        nn.init.zeros_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

        if init_ct_bias != 0.0:
            with torch.no_grad():
                self.gate_fc.bias[0] = init_ct_bias
                self.gate_fc.bias[1] = -init_ct_bias

    def forward(
        self,
        f_ct: torch.Tensor,
        f_pet_masked: torch.Tensor,
        pet_available: torch.Tensor,
    ):
        if pet_available.dim() == 2:
            pet_available = pet_available.squeeze(1)
        pet_available = pet_available.long()

        gate_feat = self.gate_feature_extractor(
            torch.cat([f_ct, f_pet_masked], dim=1)
        )

        gate_vec = F.adaptive_avg_pool2d(gate_feat, output_size=1).flatten(1)

        if self.use_status_token:
            status_vec = self.pet_status_embed(pet_available)
            gate_vec = torch.cat([gate_vec, status_vec], dim=1)

        gate_logits = self.gate_fc(gate_vec) / self.temperature

        modality_mask = torch.stack(
            [
                torch.ones_like(pet_available, dtype=torch.bool),
                pet_available.bool(),
            ],
            dim=1,
        )

        gate_logits = gate_logits.masked_fill(~modality_mask, -1e4)

        weights = torch.softmax(gate_logits, dim=1)

        return weights, gate_logits


class SimMLMAlignedTwoExpertFusion2D(nn.Module):
    """
    Two-expert DMoME feature fusion for PET-CT segmentation.

    Replaces:
        Conv1x1(Concat(F_ct, F_pet))

    With:
        Z_ct = E_ct(F_ct)
        Z_pet = E_pet(F_pet)
        [alpha_ct, alpha_pet] = Gate(F_ct, F_pet, pet_available)
        F_fuse = alpha_ct * Z_ct + alpha_pet * Z_pet

    When use_channel_prior_gate=True, text-guided channel residual priors
    enhance gate inputs only.
    """

    def __init__(
        self,
        channels: int,
        expert_reduction: int = 4,
        gate_hidden_channels: int = None,
        use_status_token: bool = True,
        temperature: float = 1.0,
        init_ct_bias: float = 0.0,
        output_proj: bool = False,
        norm_groups: int = 8,
        use_channel_prior_gate: bool = False,
        text_dim: int = 512,
        prior_hidden_dim: int = 256,
        prior_alpha_max: float = CHANNEL_PRIOR_ALPHA_MAX,
        prior_raw_alpha_init: float = CHANNEL_PRIOR_RAW_ALPHA_INIT,
    ):
        super().__init__()

        self.channels = channels
        self.use_channel_prior_gate = use_channel_prior_gate

        self.ct_expert = ModalityExpert2D(
            channels=channels,
            reduction=expert_reduction,
            norm_groups=norm_groups,
        )

        self.pet_expert = ModalityExpert2D(
            channels=channels,
            reduction=expert_reduction,
            norm_groups=norm_groups,
        )

        if use_channel_prior_gate:
            self.ct_channel_prior = TextGuidedChannelPriorGate2D(
                channels=channels,
                text_dim=text_dim,
                hidden_dim=prior_hidden_dim,
                alpha_max=prior_alpha_max,
                raw_alpha_init=prior_raw_alpha_init,
            )
            self.pet_channel_prior = TextGuidedChannelPriorGate2D(
                channels=channels,
                text_dim=text_dim,
                hidden_dim=prior_hidden_dim,
                alpha_max=prior_alpha_max,
                raw_alpha_init=prior_raw_alpha_init,
            )
        else:
            self.ct_channel_prior = None
            self.pet_channel_prior = None

        self.gate = SimMLMAlignedGate2D(
            channels=channels,
            hidden_channels=gate_hidden_channels,
            use_status_token=use_status_token,
            temperature=temperature,
            init_ct_bias=init_ct_bias,
            norm_groups=norm_groups,
        )

        if output_proj:
            self.output_proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                make_group_norm(channels, max_groups=norm_groups),
                nn.GELU(),
            )
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        f_ct: torch.Tensor,
        f_pet: torch.Tensor,
        pet_available: torch.Tensor,
        modality_text_embeds: torch.Tensor = None,
    ):
        B = f_ct.shape[0]

        if pet_available.dim() == 2:
            pet_available = pet_available.squeeze(1)
        pet_available = pet_available.long()

        pet_mask = pet_available.float().view(B, 1, 1, 1)

        f_pet_masked = f_pet * pet_mask

        z_ct = self.ct_expert(f_ct)
        z_pet = self.pet_expert(f_pet_masked)

        ct_channel_mask = None
        pet_channel_mask = None
        ct_prior_alpha = None
        pet_prior_alpha = None
        ct_prior_scale = None
        pet_prior_scale = None

        if self.use_channel_prior_gate:
            if modality_text_embeds is None:
                raise ValueError(
                    "modality_text_embeds required when use_channel_prior_gate=True"
                )
            f_ct_for_gate, ct_channel_mask, ct_prior_alpha, ct_prior_scale = (
                self.ct_channel_prior(f_ct, modality_text_embeds[0])
            )
            f_pet_for_gate, pet_channel_mask, pet_prior_alpha, pet_prior_scale = (
                self.pet_channel_prior(f_pet_masked, modality_text_embeds[1])
            )
        else:
            f_ct_for_gate = f_ct
            f_pet_for_gate = f_pet_masked

        weights, gate_logits = self.gate(
            f_ct=f_ct_for_gate,
            f_pet_masked=f_pet_for_gate,
            pet_available=pet_available,
        )

        w_ct = weights[:, 0].view(B, 1, 1, 1)
        w_pet = weights[:, 1].view(B, 1, 1, 1)

        f_fuse = w_ct * z_ct + w_pet * z_pet
        f_fuse = self.output_proj(f_fuse)

        aux = {
            "weights": weights.detach(),
            "gate_logits": gate_logits.detach(),
            "w_ct": weights[:, 0].detach(),
            "w_pet": weights[:, 1].detach(),
        }
        if ct_channel_mask is not None:
            aux["ct_channel_mask"] = ct_channel_mask.detach()
            aux["ct_prior_alpha"] = ct_prior_alpha.detach()
            aux["ct_prior_scale"] = ct_prior_scale.detach()
        if pet_channel_mask is not None:
            aux["pet_channel_mask"] = pet_channel_mask.detach()
            aux["pet_prior_alpha"] = pet_prior_alpha.detach()
            aux["pet_prior_scale"] = pet_prior_scale.detach()

        return f_fuse, aux


class StageWiseSimMLMAlignedDMoME(nn.Module):
    """
    Stage-wise DMoME fusion for all skip connections.

    Input:
        ct_feats  = [F_ct1, F_ct2, F_ct3, F_ct4]
        pet_feats = [F_pet1, F_pet2, F_pet3, F_pet4]

    Output:
        fused_feats = [F_fuse1, F_fuse2, F_fuse3, F_fuse4]
    """

    def __init__(
        self,
        channels_list=(64, 128, 320, 512),
        expert_reduction: int = 4,
        use_status_token: bool = True,
        temperature: float = 1.0,
        init_ct_bias: float = 0.0,
        output_proj: bool = False,
        norm_groups: int = 8,
        use_channel_prior_gate: bool = False,
        prior_gate_stages=DEFAULT_PRIOR_GATE_STAGES,
        text_dim: int = 512,
        prior_hidden_dim: int = 256,
        prior_alpha_max: float = CHANNEL_PRIOR_ALPHA_MAX,
        prior_raw_alpha_init: float = CHANNEL_PRIOR_RAW_ALPHA_INIT,
    ):
        super().__init__()

        self.channels_list = list(channels_list)
        self.use_channel_prior_gate = use_channel_prior_gate
        self.prior_gate_stages = tuple(int(s) for s in prior_gate_stages)
        self.text_dim = text_dim

        self.register_buffer(
            "modality_text_embeds",
            torch.zeros(2, text_dim),
            persistent=False,
        )

        self.fusions = nn.ModuleList(
            [
                SimMLMAlignedTwoExpertFusion2D(
                    channels=c,
                    expert_reduction=expert_reduction,
                    gate_hidden_channels=None,
                    use_status_token=use_status_token,
                    temperature=temperature,
                    init_ct_bias=init_ct_bias,
                    output_proj=output_proj,
                    norm_groups=norm_groups,
                    use_channel_prior_gate=use_channel_prior_gate
                    and ((i + 1) in self.prior_gate_stages),
                    text_dim=text_dim,
                    prior_hidden_dim=prior_hidden_dim,
                    prior_alpha_max=prior_alpha_max,
                    prior_raw_alpha_init=prior_raw_alpha_init,
                )
                for i, c in enumerate(self.channels_list)
            ]
        )

    def set_modality_prior_text_embeds(self, text_embeds: torch.Tensor):
        if text_embeds.shape != self.modality_text_embeds.shape:
            raise ValueError(
                f"Expected text_embeds shape {tuple(self.modality_text_embeds.shape)}, "
                f"got {tuple(text_embeds.shape)}"
            )
        self.modality_text_embeds.copy_(
            text_embeds.detach().to(self.modality_text_embeds.device)
        )

    def forward(
        self,
        ct_feats,
        pet_feats,
        pet_available: torch.Tensor,
    ):
        assert len(ct_feats) == len(pet_feats), (
            f"ct_feats and pet_feats length mismatch: "
            f"{len(ct_feats)} vs {len(pet_feats)}"
        )

        assert len(ct_feats) == len(self.fusions), (
            f"feature stages and fusion modules mismatch: "
            f"{len(ct_feats)} vs {len(self.fusions)}"
        )

        text_embeds = self.modality_text_embeds if self.use_channel_prior_gate else None

        fused_feats = []
        aux_all = []

        for i, fusion in enumerate(self.fusions):
            f_ct = ct_feats[i]
            f_pet = pet_feats[i]

            if f_pet.shape[-2:] != f_ct.shape[-2:]:
                f_pet = F.interpolate(
                    f_pet,
                    size=f_ct.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            fused, aux = fusion(
                f_ct=f_ct,
                f_pet=f_pet,
                pet_available=pet_available,
                modality_text_embeds=text_embeds,
            )

            fused_feats.append(fused)
            aux_all.append(aux)

        return fused_feats, aux_all


class ShallowConcatConvFusion2D(nn.Module):
    """Shallow-stage fusion: Concat(F_ct, F_pet) -> Conv1x1."""

    def __init__(self, ct_channels: int, pet_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ct_channels + pet_channels, out_channels, kernel_size=1, bias=False),
            make_group_norm(out_channels, max_groups=norm_groups),
            nn.GELU(),
        )

    def forward(self, f_ct: torch.Tensor, f_pet_masked: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([f_ct, f_pet_masked], dim=1))


class StageWiseHybridConcatDMoMEFusion(nn.Module):
    """
    Hybrid stage-wise fusion:
        shallow stages (default S1-S3): Concat + Conv1x1
        deep stages   (default S4):     plain DMoME (no text prior)
    """

    def __init__(
        self,
        ct_channels_list=(64, 128, 320, 512),
        pet_channels_list=(64, 128, 320, 512),
        concat_stages=DEFAULT_HYBRID_CONCAT_STAGES,
        dmome_stages=DEFAULT_HYBRID_DMOME_STAGES,
        expert_reduction: int = 4,
        use_status_token: bool = True,
        temperature: float = 1.0,
        init_ct_bias: float = 0.0,
        output_proj: bool = False,
        norm_groups: int = 8,
    ):
        super().__init__()

        self.ct_channels_list = list(ct_channels_list)
        self.pet_channels_list = list(pet_channels_list)
        self.concat_stages = tuple(int(s) for s in concat_stages)
        self.dmome_stages = tuple(int(s) for s in dmome_stages)
        self._validate_stage_split()

        self.shallow_fusions = nn.ModuleList()
        self.dmome_fusions = nn.ModuleList()
        for i, (c_ct, c_pet) in enumerate(zip(self.ct_channels_list, self.pet_channels_list)):
            stage = i + 1
            if stage in self.concat_stages:
                self.shallow_fusions.append(
                    ShallowConcatConvFusion2D(
                        ct_channels=c_ct,
                        pet_channels=c_pet,
                        out_channels=c_ct,
                        norm_groups=norm_groups,
                    )
                )
            else:
                self.shallow_fusions.append(None)

            if stage in self.dmome_stages:
                self.dmome_fusions.append(
                    SimMLMAlignedTwoExpertFusion2D(
                        channels=c_ct,
                        expert_reduction=expert_reduction,
                        gate_hidden_channels=None,
                        use_status_token=use_status_token,
                        temperature=temperature,
                        init_ct_bias=init_ct_bias,
                        output_proj=output_proj,
                        norm_groups=norm_groups,
                        use_channel_prior_gate=False,
                    )
                )
            else:
                self.dmome_fusions.append(None)

    def _validate_stage_split(self):
        concat_set = set(self.concat_stages)
        dmome_set = set(self.dmome_stages)
        if concat_set & dmome_set:
            raise ValueError(
                f'hybrid concat_stages and dmome_stages overlap: '
                f'{self.concat_stages} vs {self.dmome_stages}'
            )
        all_stages = set(range(1, len(self.ct_channels_list) + 1))
        if concat_set | dmome_set != all_stages:
            raise ValueError(
                f'hybrid stages must cover all encoder stages {sorted(all_stages)}, '
                f'got concat={self.concat_stages} dmome={self.dmome_stages}'
            )
        invalid = (concat_set | dmome_set) - VALID_PRIOR_GATE_STAGES
        if invalid:
            raise ValueError(f'Invalid hybrid stage indices: {sorted(invalid)}')

    def forward(
        self,
        ct_feats,
        pet_feats,
        pet_available: torch.Tensor,
    ):
        assert len(ct_feats) == len(pet_feats) == len(self.ct_channels_list)

        if pet_available.dim() == 2:
            pet_available = pet_available.squeeze(1)
        pet_available = pet_available.long()
        B = ct_feats[0].shape[0]
        pet_mask = pet_available.float().view(B, 1, 1, 1)

        fused_feats = []
        aux_all = []

        for i, (f_ct, f_pet) in enumerate(zip(ct_feats, pet_feats)):
            stage = i + 1
            if f_pet.shape[-2:] != f_ct.shape[-2:]:
                f_pet = F.interpolate(
                    f_pet,
                    size=f_ct.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            f_pet_masked = f_pet * pet_mask

            if stage in self.concat_stages:
                fused = self.shallow_fusions[i](f_ct, f_pet_masked)
                aux = {"fusion_mode": "concat_conv"}
            else:
                fused, aux = self.dmome_fusions[i](
                    f_ct=f_ct,
                    f_pet=f_pet,
                    pet_available=pet_available,
                    modality_text_embeds=None,
                )
                aux = dict(aux)
                aux["fusion_mode"] = "dmome"

            fused_feats.append(fused)
            aux_all.append(aux)

        return fused_feats, aux_all


def summarize_dmome_weights(fusion_aux):
    """Helper for logging dynamic modality weights and channel prior stats."""
    summary = {}

    for i, aux in enumerate(fusion_aux):
        stage = i + 1
        if "w_ct" not in aux:
            continue
        summary[f"stage{stage}_w_ct"] = aux["w_ct"].mean().item()
        summary[f"stage{stage}_w_pet"] = aux["w_pet"].mean().item()

        if "ct_prior_alpha" in aux:
            summary[f"stage{stage}_ct_prior_alpha"] = aux["ct_prior_alpha"].item()
        if "pet_prior_alpha" in aux:
            summary[f"stage{stage}_pet_prior_alpha"] = aux["pet_prior_alpha"].item()

        if "ct_channel_mask" in aux:
            mask = aux["ct_channel_mask"]
            summary[f"stage{stage}_ct_channel_mask_mean"] = mask.mean().item()
            summary[f"stage{stage}_ct_channel_mask_std"] = mask.std().item()
        if "pet_channel_mask" in aux:
            mask = aux["pet_channel_mask"]
            summary[f"stage{stage}_pet_channel_mask_mean"] = mask.mean().item()
            summary[f"stage{stage}_pet_channel_mask_std"] = mask.std().item()

        if "ct_prior_scale" in aux:
            summary[f"stage{stage}_ct_prior_scale_mean"] = aux["ct_prior_scale"].mean().item()
        if "pet_prior_scale" in aux:
            summary[f"stage{stage}_pet_prior_scale_mean"] = aux["pet_prior_scale"].mean().item()

    return summary


def make_full_pet_available(batch_size: int, device) -> torch.Tensor:
    return torch.ones(batch_size, device=device, dtype=torch.long)
