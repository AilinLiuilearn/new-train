"""End-to-End Joint Recovery-Fusion-Segmentation model.

Joint architecture (no frozen Stage-1 / no decoder adapters):

    CT/PET Encoders
        -> Dynamic CT-PET Prototype Memory (CPPI)
        -> Missing PET Retrieval / Real PET
        -> PET Affine Calibration
        -> Prompt-Guided Role-Specialized Expert Fusion (SPRE)
        -> Trainable original UNetStyleDecoder
        -> Segmentation

Module-1 components live on ``stage1`` (DualSharedAddPETCTBaseline container).
Module-2 is ``role_fusion`` (PromptGuidedRoleExpertFusion).

Stage-1 StateAwareWeightedAddFusion is bypassed and disabled
(requires_grad=False) because SPRE fully replaces it.

Text OFF by default (external_prompt_dim=None).
No balance / ANGA / adapter / distillation losses in the first joint experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _check_feature_list(
    name: str,
    xs: Sequence[torch.Tensor],
    channels: Sequence[int],
) -> None:
    if len(xs) != len(channels):
        raise ValueError(f"{name}: expected {len(channels)} scales, got {len(xs)}")
    for i, (x, c) in enumerate(zip(xs, channels), start=1):
        if x.ndim != 4:
            raise ValueError(f"{name}[S{i}] must be BCHW, got {tuple(x.shape)}")
        if x.shape[1] != c:
            raise ValueError(
                f"{name}[S{i}] channel mismatch: expected {c}, got {x.shape[1]}"
            )
        if not torch.isfinite(x).all():
            raise RuntimeError(f"{name}[S{i}] contains NaN/Inf")


def _to_float_stats(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()


# -----------------------------------------------------------------------------
# 1. TG-ECNet-style learnable TaskPrompt
# -----------------------------------------------------------------------------


class TaskPromptGenerator(nn.Module):
    """CondNet -> GAP -> softmax prompt atoms -> learnable dictionary.

    This follows the Taskprompt structure used in TG-ECNet, adapted so that the
    input is the learned joint CT/PET representation at one encoder scale.

    Args:
        in_channels: channels of the joint feature (normally expert_dim=128).
        atom_num: number of learnable prompt atoms.
        atom_dim: dimension of the prompt dictionary atoms.
        hidden_channels: CondNet hidden channels.
        out_dim: final prompt dimension consumed by the router.
    """

    def __init__(
        self,
        in_channels: int,
        atom_num: int = 32,
        atom_dim: int = 256,
        hidden_channels: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.atom_num = int(atom_num)
        self.atom_dim = int(atom_dim)
        self.out_dim = int(out_dim)

        # Same high-level structure as TG-ECNet Taskprompt:
        # two 3x3 stride-3 convolutions, then 1x1 conditioning layers.
        self.cond_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 32, kernel_size=1),
        )
        self.atom_logits = nn.Linear(32, self.atom_num)
        self.dictionary = nn.Parameter(torch.randn(self.atom_num, self.atom_dim))
        self.act = nn.GELU()
        self.prompt_proj = nn.Linear(self.atom_dim, self.out_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] < 9 or x.shape[-1] < 9:
            raise ValueError(
                "TaskPromptGenerator requires spatial size >= 9x9 because its "
                "TG-ECNet-style CondNet contains two kernel=3,stride=3 convs; "
                f"got {tuple(x.shape[-2:])}."
            )

        z = self.cond_net(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)  # [B, 32]
        atom_weights = F.softmax(self.atom_logits(z), dim=-1)  # [B, atom_num]
        prompt = atom_weights @ self.dictionary  # [B, atom_dim]
        prompt = self.act(prompt)
        prompt = self.prompt_proj(prompt)  # [B, out_dim]
        return prompt, atom_weights


class PromptConditioner(nn.Module):
    """Reserved insertion point for a future fixed-text/state embedding.

    Text is OFF when ``external_prompt_dim`` is None. In that case this module
    is an exact identity on the learnable visual TaskPrompt.

    If enabled later:
        visual_prompt [B,D]
        external_prompt [B,Dt]
          -> Linear(Dt,D)
          -> concat [visual, external]
          -> Linear(2D,D) + GELU
          -> conditioned prompt [B,D]

    The conditioned prompt is used by the router only; experts still receive
    visual feature maps only.
    """

    def __init__(self, prompt_dim: int, external_prompt_dim: Optional[int] = None) -> None:
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.external_prompt_dim = (
            None if external_prompt_dim is None else int(external_prompt_dim)
        )

        if self.external_prompt_dim is None:
            self.external_proj = None
            self.fuse = None
        else:
            self.external_proj = nn.Linear(self.external_prompt_dim, self.prompt_dim)
            self.fuse = nn.Sequential(
                nn.Linear(2 * self.prompt_dim, self.prompt_dim),
                nn.GELU(),
            )

    def forward(
        self,
        visual_prompt: torch.Tensor,
        external_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if external_prompt is None:
            return visual_prompt

        if self.external_proj is None or self.fuse is None:
            raise ValueError(
                "external_prompt was provided but external_prompt_dim=None at "
                "construction. Rebuild Stage-2 with external_prompt_dim set to "
                "the future fixed-text embedding dimension."
            )
        if external_prompt.ndim != 2:
            raise ValueError(
                f"external_prompt must be [B,D], got {tuple(external_prompt.shape)}"
            )
        if external_prompt.shape[0] != visual_prompt.shape[0]:
            raise ValueError("visual/external prompt batch sizes do not match")

        ext = self.external_proj(external_prompt.float())
        fused = torch.cat([visual_prompt.float(), ext], dim=-1)
        return self.fuse(fused).to(dtype=visual_prompt.dtype)


# -----------------------------------------------------------------------------
# 2. Role-specialized experts
# -----------------------------------------------------------------------------


class ResidualChannelExpert(nn.Module):
    """Token MLP implemented as memory-friendly 1x1 convolutions.

    A channel-wise MLP on each spatial token is equivalent to two Linear layers
    after BCHW<->token rearrangement, but avoids materializing huge S1 token
    matrices for batch size 16.

    E(x) = x + W2(GELU(W1(GN(x))))
    """

    def __init__(
        self,
        dim: int = 128,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.norm = nn.GroupNorm(1, dim)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


# -----------------------------------------------------------------------------
# 3. Prompt-guided role router
# -----------------------------------------------------------------------------


class PromptGuidedRoleRouter(nn.Module):
    """Memory-efficient equivalent of Linear([CT; PET; TaskPrompt]) -> 4 logits.

    TG-ECNet concatenates local feature tokens with a projected TaskPrompt before
    gating. For S1=128x128 and batch=16, explicitly expanding the prompt to all
    tokens is unnecessarily expensive. A linear layer on a concatenation can be
    decomposed exactly as:

        Wc*c + Wp*p + Wq*q + b.

    We therefore compute local CT/PET logits using 1x1 convs and sample-level
    prompt logits using a Linear layer, then broadcast the latter spatially.

    Output role order is always:
        0: CT expert
        1: PET-state expert (RealPET for Full, ProxyPET for Missing)
        2: current-scale expert
        3: all-scale shared expert
    """

    ROLE_NAMES = ("ct", "pet", "scale", "shared")

    def __init__(self, dim: int = 128, num_roles: int = 4) -> None:
        super().__init__()
        if num_roles != 4:
            raise ValueError("Current role design requires exactly four active roles")
        self.num_roles = num_roles

        self.ct_gate = nn.Conv2d(dim, num_roles, kernel_size=1, bias=False)
        self.pet_gate = nn.Conv2d(dim, num_roles, kernel_size=1, bias=False)
        self.prompt_gate = nn.Linear(dim, num_roles, bias=True)

        # Mild initialization prevents an early single-role monopoly while still
        # allowing immediate data-dependent routing.
        nn.init.normal_(self.ct_gate.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.pet_gate.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.prompt_gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.prompt_gate.bias)

    def forward(
        self,
        ct_feat: torch.Tensor,
        pet_feat: torch.Tensor,
        task_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ct_feat.shape != pet_feat.shape:
            raise ValueError(
                f"Router CT/PET shape mismatch: {tuple(ct_feat.shape)} vs {tuple(pet_feat.shape)}"
            )
        b = ct_feat.shape[0]
        if task_prompt.shape != (b, ct_feat.shape[1]):
            raise ValueError(
                "Task prompt must be [B,D] matching expert channels; got "
                f"prompt={tuple(task_prompt.shape)} feature={tuple(ct_feat.shape)}"
            )

        logits = self.ct_gate(ct_feat.float()) + self.pet_gate(pet_feat.float())
        prompt_logits = self.prompt_gate(task_prompt.float()).view(b, self.num_roles, 1, 1)
        logits = logits + prompt_logits
        gates = F.softmax(logits, dim=1)
        return gates, logits


# -----------------------------------------------------------------------------
# 4. One scale: evidence projection -> Prompt -> routing -> expert fusion
# -----------------------------------------------------------------------------


@dataclass
class ScaleFusionOutput:
    feature: torch.Tensor
    stats: Dict[str, torch.Tensor]


class ScaleRoleFusionUnit(nn.Module):
    """Scale-specific frontend/router/fusion head.

    Global CT/RealPET/ProxyPET/Shared experts are passed in by the parent module
    and are therefore truly shared across S1-S4. The scale expert lives here and
    is unique to this scale.
    """

    def __init__(
        self,
        in_channels: int,
        expert_dim: int = 128,
        atom_num: int = 32,
        atom_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        external_prompt_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.expert_dim = int(expert_dim)

        self.ct_proj = nn.Sequential(
            nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, expert_dim),
            nn.GELU(),
        )
        self.pet_proj = nn.Sequential(
            nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, expert_dim),
            nn.GELU(),
        )

        # No hand-crafted |C-P| or C*P terms: relationship is learned from concat.
        self.joint_proj = nn.Sequential(
            nn.Conv2d(2 * expert_dim, expert_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, expert_dim),
            nn.GELU(),
        )

        self.task_prompt = TaskPromptGenerator(
            in_channels=expert_dim,
            atom_num=atom_num,
            atom_dim=atom_dim,
            hidden_channels=prompt_hidden_channels,
            out_dim=expert_dim,
        )
        self.prompt_conditioner = PromptConditioner(
            prompt_dim=expert_dim,
            external_prompt_dim=external_prompt_dim,
        )
        self.router = PromptGuidedRoleRouter(dim=expert_dim, num_roles=4)

        self.scale_expert = ResidualChannelExpert(
            dim=expert_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Router weights decide participation, then a learned fusion head decides
        # how the four role representations should be integrated. This is more
        # expressive than a simple weighted sum while preserving role semantics.
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(4 * expert_dim, 2 * expert_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, 2 * expert_dim),
            nn.GELU(),
            nn.Conv2d(2 * expert_dim, expert_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, expert_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(expert_dim, in_channels, kernel_size=1, bias=True)

    def forward(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
        route: str,
        ct_expert: nn.Module,
        real_pet_expert: nn.Module,
        proxy_pet_expert: nn.Module,
        shared_expert: nn.Module,
        external_prompt: Optional[torch.Tensor] = None,
        pet_available: Optional[torch.Tensor] = None,
    ) -> ScaleFusionOutput:
        if ct.shape != pet.shape:
            raise ValueError(
                f"Scale CT/PET shape mismatch: {tuple(ct.shape)} vs {tuple(pet.shape)}"
            )

        c = self.ct_proj(ct)
        p = self.pet_proj(pet)
        joint = self.joint_proj(torch.cat([c, p], dim=1))

        prompt, atom_weights = self.task_prompt(joint)
        prompt = self.prompt_conditioner(prompt, external_prompt)

        gates, router_logits = self.router(c, p, prompt)  # [B,4,H,W]

        h_ct = ct_expert(c)
        h_scale = self.scale_expert(joint)
        h_shared = shared_expert(joint)

        route = str(route).lower()
        if route == "full":
            h_pet = real_pet_expert(p)
        elif route == "missing":
            h_pet = proxy_pet_expert(p)
        elif route == "auto":
            if pet_available is None:
                raise ValueError("route='auto' requires pet_available [B]")
            state = pet_available.to(device=p.device).float().view(-1, 1, 1, 1)
            if state.shape[0] != p.shape[0]:
                raise ValueError("pet_available must contain one value per sample")
            # Both expert functions are evaluated, but hard state masking ensures
            # Full samples only receive Real-PET expert output and Missing samples
            # only receive Proxy-PET expert output.
            h_real = real_pet_expert(p)
            h_proxy = proxy_pet_expert(p)
            h_pet = state * h_real + (1.0 - state) * h_proxy
        else:
            raise ValueError(f"Unsupported route={route!r}")

        weighted = [
            gates[:, 0:1] * h_ct,
            gates[:, 1:2] * h_pet,
            gates[:, 2:3] * h_scale,
            gates[:, 3:4] * h_shared,
        ]
        fused = self.fusion_proj(torch.cat(weighted, dim=1))
        out = self.out_proj(fused).to(dtype=ct.dtype)

        # Diagnostics only; never used as training losses by this module.
        with torch.no_grad():
            eps = 1e-8
            router_entropy = -(gates.clamp_min(eps) * gates.clamp_min(eps).log()).sum(dim=1).mean()
            atom_entropy = -(
                atom_weights.clamp_min(eps) * atom_weights.clamp_min(eps).log()
            ).sum(dim=-1).mean()
            role_mean = gates.mean(dim=(0, 2, 3))
            top_role = gates.argmax(dim=1)
            top_freq = torch.stack(
                [(top_role == i).float().mean() for i in range(4)], dim=0
            )
            stats = {
                "router_entropy": router_entropy.detach().float(),
                "prompt_atom_entropy": atom_entropy.detach().float(),
                "role_mean": role_mean.detach().float(),
                "top_role_freq": top_freq.detach().float(),
                "router_logit_std": router_logits.detach().float().std(unbiased=False),
                "ct_expert_abs_mean": h_ct.detach().float().abs().mean(),
                "pet_expert_abs_mean": h_pet.detach().float().abs().mean(),
                "scale_expert_abs_mean": h_scale.detach().float().abs().mean(),
                "shared_expert_abs_mean": h_shared.detach().float().abs().mean(),
                "fused_abs_mean": out.detach().float().abs().mean(),
            }

        return ScaleFusionOutput(feature=out, stats=stats)


# -----------------------------------------------------------------------------
# 5. Four-scale role-expert fusion
# -----------------------------------------------------------------------------


@dataclass
class MultiScaleFusionOutput:
    features: List[torch.Tensor]
    stats: Dict[str, torch.Tensor]


class PromptGuidedRoleExpertFusion(nn.Module):
    """Four-scale direct Stage-2 fusion with eight interpretable experts.

    Expert inventory:
        1 x CT expert                 : shared over S1-S4 and Full/Missing
        1 x Real-PET expert           : shared over S1-S4, Full only
        1 x Proxy-PET expert          : shared over S1-S4, Missing only
        4 x Scale experts             : one for each S1..S4
        1 x Shared joint expert       : shared over all scales/states
        -------------------------------------------------------------
        Total = 8 experts

    At a fixed scale/state, exactly four role outputs participate:
        CT + PET-state + current-scale + shared.
    Their contributions are soft-routed by the scale-specific Prompt-guided
    router, then concatenated and fused into a new feature map. There is no
    residual connection to Stage-1 weighted-add fusion.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        expert_dim: int = 128,
        atom_num: int = 32,
        atom_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        external_prompt_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 4:
            raise ValueError("Current decoder integration expects exactly four scales")
        self.expert_dim = int(expert_dim)

        # Cross-scale role experts.
        self.ct_expert = ResidualChannelExpert(expert_dim, mlp_ratio, dropout)
        self.real_pet_expert = ResidualChannelExpert(expert_dim, mlp_ratio, dropout)
        self.proxy_pet_expert = ResidualChannelExpert(expert_dim, mlp_ratio, dropout)
        self.shared_expert = ResidualChannelExpert(expert_dim, mlp_ratio, dropout)

        # Each ScaleRoleFusionUnit owns exactly one unique scale expert plus its
        # scale-specific projections, TaskPrompt, router and fusion head.
        self.scale_units = nn.ModuleList(
            [
                ScaleRoleFusionUnit(
                    in_channels=c,
                    expert_dim=expert_dim,
                    atom_num=atom_num,
                    atom_dim=atom_dim,
                    prompt_hidden_channels=prompt_hidden_channels,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    external_prompt_dim=external_prompt_dim,
                )
                for c in self.channels
            ]
        )

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_cal: Sequence[torch.Tensor],
        route: str,
        external_prompt: Optional[torch.Tensor] = None,
        pet_available: Optional[torch.Tensor] = None,
    ) -> MultiScaleFusionOutput:
        _check_feature_list("ct_feats", ct_feats, self.channels)
        _check_feature_list("pet_feats_cal", pet_feats_cal, self.channels)

        out_features: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {}

        for idx, (ct, pet, unit) in enumerate(
            zip(ct_feats, pet_feats_cal, self.scale_units), start=1
        ):
            result = unit(
                ct=ct,
                pet=pet,
                route=route,
                ct_expert=self.ct_expert,
                real_pet_expert=self.real_pet_expert,
                proxy_pet_expert=self.proxy_pet_expert,
                shared_expert=self.shared_expert,
                external_prompt=external_prompt,
                pet_available=pet_available,
            )
            out_features.append(result.feature)
            for key, value in result.stats.items():
                stats[f"s{idx}_{key}"] = value

        stats["external_prompt_used"] = torch.tensor(
            0.0 if external_prompt is None else 1.0,
            device=out_features[0].device,
        )
        return MultiScaleFusionOutput(features=out_features, stats=stats)


# -----------------------------------------------------------------------------
# 6. Joint end-to-end Recovery + Role Fusion segmentation model
# -----------------------------------------------------------------------------


class PromptRoleExpertStage2Seg(nn.Module):
    """Joint PET recovery + role-specialized fusion segmentation model.

    ``stage1`` is a DualSharedAddPETCTBaseline used as a Module-1 component
    container (encoders / CPPI / calibration / decoder). Its legacy
    ``StateAwareWeightedAddFusion`` is NOT used in the forward path.
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        channels: Sequence[int] = (64, 128, 320, 512),
        expert_dim: int = 128,
        atom_num: int = 32,
        atom_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        external_prompt_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.stage1 = stage1_model
        self.channels = tuple(int(c) for c in channels)

        required_attrs = (
            "_encode_ct",
            "_encode_pet",
            "_retrieve_cppi",
            "_collect_cppi",
            "pet_calibration",
            "prototype_memory",
            "decoder",
        )
        missing = [name for name in required_attrs if not hasattr(self.stage1, name)]
        if missing:
            raise TypeError(
                "stage1_model is not compatible with the target baseline branch; "
                f"missing attributes: {missing}"
            )

        # All active Module-1 networks remain trainable. Only the unused legacy
        # weighted-add fusion is disabled because SPRE replaces it.
        if hasattr(self.stage1, "fusion"):
            for p in self.stage1.fusion.parameters():
                p.requires_grad = False
            print("[JOINT] legacy weighted-add fusion disabled=True", flush=True)

        self.role_fusion = PromptGuidedRoleExpertFusion(
            channels=self.channels,
            expert_dim=expert_dim,
            atom_num=atom_num,
            atom_dim=atom_dim,
            prompt_hidden_channels=prompt_hidden_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            external_prompt_dim=external_prompt_dim,
        )

    @property
    def enc_ct(self) -> nn.Module:
        return self.stage1.enc_ct

    @property
    def enc_pet(self) -> nn.Module:
        return self.stage1.enc_pet

    @property
    def ct_align(self) -> nn.Module:
        return self.stage1.ct_align

    @property
    def decoder(self) -> nn.Module:
        return self.stage1.decoder

    @property
    def prototype_memory(self) -> nn.Module:
        return self.stage1.prototype_memory

    @property
    def cppi_ready(self) -> bool:
        return bool(self.stage1.prototype_memory.bank_ready)

    def _calibrate_real_pet(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Affine-calibrate real PET. Keep encoder/PET grads; detach only CT ref."""
        if self.cppi_ready:
            _, ct_reference_feats, _ = self.stage1._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            # Module-1 calibration design: CT reference is detached.
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            return list(
                self.stage1.pet_calibration(
                    ct_feats,
                    pet_feats_real,
                    ct_reference_feats,
                    reference_valid=True,
                )
            )
        return list(
            self.stage1.pet_calibration(
                ct_feats,
                pet_feats_real,
                None,
                reference_valid=False,
            )
        )

    def _retrieve_and_calibrate_proxy(
        self,
        ct_feats: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Retrieve proxy PET (grad through attention) and affine-calibrate."""
        pet_proxy, ct_reference_feats, _ = self.stage1._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        # Keep pet_proxy in the graph so Missing loss updates PrototypeCrossAttention.
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        return list(
            self.stage1.pet_calibration(
                ct_feats,
                pet_proxy,
                ct_reference_feats,
                reference_valid=self.cppi_ready,
            )
        )

    def _extract_full_evidence(
        self,
        ct: torch.Tensor,
        pet: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if pet is None:
            raise ValueError("Full path requires real PET input")
        ct_feats = list(self.stage1._encode_ct(ct))
        pet_real = list(self.stage1._encode_pet(pet))
        # collect is @torch.no_grad internally; does not pollute the main graph.
        self.stage1._collect_cppi(ct_feats, pet_real, mask)
        pet_cal = self._calibrate_real_pet(ct_feats, pet_real)
        return ct_feats, pet_cal

    def _extract_missing_evidence(
        self,
        ct: torch.Tensor,
        pet: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Missing prediction uses CPPI proxy only; real PET is collect-only."""
        ct_feats = list(self.stage1._encode_ct(ct))

        # Training-time memory construction mirrors Stage-1 Missing:
        # real PET encoder may run for collect, but must NOT enter prediction.
        if self.training and mask is not None:
            if pet is None:
                raise ValueError(
                    "Missing training with mask requires real PET for CPPI collect"
                )
            pet_real_for_memory = list(self.stage1._encode_pet(pet))
            self.stage1._collect_cppi(ct_feats, pet_real_for_memory, mask)

        pet_cal = self._retrieve_and_calibrate_proxy(ct_feats)
        return ct_feats, pet_cal

    def _extract_auto_evidence(
        self,
        ct: torch.Tensor,
        pet: Optional[torch.Tensor],
        pet_available: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-sample Full/Missing. Eval must not update the prototype bank."""
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError("pet_available must contain one state per sample")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0/1")

        ct_feats_all = list(self.stage1._encode_ct(ct))
        pet_cal_all = [torch.empty_like(x) for x in ct_feats_all]

        full_idx = torch.nonzero(availability == 1, as_tuple=False).flatten()
        miss_idx = torch.nonzero(availability == 0, as_tuple=False).flatten()

        # Optional collect only in train with mask; uses full PET if present.
        if self.training and mask is not None and pet is not None:
            pet_real_all = list(self.stage1._encode_pet(pet))
            self.stage1._collect_cppi(ct_feats_all, pet_real_all, mask)

        if full_idx.numel() > 0:
            if pet is None:
                raise ValueError("auto path has Full samples but pet=None")
            ct_full = [x.index_select(0, full_idx) for x in ct_feats_all]
            pet_input_full = pet.index_select(0, full_idx)
            pet_real_full = list(self.stage1._encode_pet(pet_input_full))
            pet_cal_full = self._calibrate_real_pet(ct_full, pet_real_full)
            for dst, src in zip(pet_cal_all, pet_cal_full):
                dst.index_copy_(0, full_idx, src)

        if miss_idx.numel() > 0:
            ct_miss = [x.index_select(0, miss_idx) for x in ct_feats_all]
            pet_cal_miss = self._retrieve_and_calibrate_proxy(ct_miss)
            for dst, src in zip(pet_cal_all, pet_cal_miss):
                dst.index_copy_(0, miss_idx, src)

        return ct_feats_all, pet_cal_all

    def forward(
        self,
        ct: torch.Tensor,
        pet: Optional[torch.Tensor] = None,
        pet_available: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        forward_mode: str = "auto",
        mask: Optional[torch.Tensor] = None,
        external_prompt: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, object]:
        if target_size is None:
            target_size = tuple(ct.shape[-2:])

        mode = str(forward_mode).strip().lower()
        if mode == "full":
            ct_feats, pet_cal = self._extract_full_evidence(ct, pet, mask=mask)
            fusion_route = "full"
            state = None
        elif mode == "missing":
            ct_feats, pet_cal = self._extract_missing_evidence(ct, pet=pet, mask=mask)
            fusion_route = "missing"
            state = None
        elif mode == "auto":
            if pet_available is None:
                pet_available = torch.ones(
                    ct.shape[0], device=ct.device, dtype=torch.long
                )
            ct_feats, pet_cal = self._extract_auto_evidence(
                ct, pet, pet_available, mask=mask
            )
            fusion_route = "auto"
            state = pet_available
        else:
            raise ValueError(
                f"Unsupported forward_mode={forward_mode!r}; use full/missing/auto"
            )

        fusion = self.role_fusion(
            ct_feats=ct_feats,
            pet_feats_cal=pet_cal,
            route=fusion_route,
            external_prompt=external_prompt,
            pet_available=state,
        )

        # Direct trainable original UNet decoder (no adapters).
        dec_out = self.stage1.decoder(fusion.features, target_size)
        logits = dec_out["logits"]

        out: Dict[str, object] = dict(dec_out)
        out["pred"] = logits
        out["aux"] = {
            "stage2_stats": fusion.stats,
            "stage2_aux_loss": logits.new_zeros((), dtype=torch.float32),
        }

        if return_features:
            out["stage2_features"] = fusion.features
            out["stage1_ct_evidence"] = ct_feats
            out["stage1_pet_cal_evidence"] = pet_cal
        return out

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_module_trainable(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    @torch.no_grad()
    def finalize_cppi_epoch(
        self,
        epoch: int,
        save_json: bool = True,
        save_visualizations: bool = False,
        print_info: bool = True,
    ) -> Dict[str, object]:
        """Rebuild the dynamic prototype bank from this epoch's collect cache."""
        return self.stage1.prototype_memory.finalize_epoch(
            epoch=epoch,
            save_json=save_json,
            save_visualizations=save_visualizations,
            print_info=print_info,
        )


# -----------------------------------------------------------------------------
# 7. Optional core smoke test
# -----------------------------------------------------------------------------


def _smoke_test_core() -> None:
    """Tests PromptGuidedRoleExpertFusion without encoders/CPPI data."""
    torch.manual_seed(0)
    channels = (64, 128, 320, 512)
    spatial = ((32, 32), (24, 24), (18, 18), (16, 16))
    ct = [torch.randn(1, c, h, w) for c, (h, w) in zip(channels, spatial)]
    pet = [torch.randn_like(x) for x in ct]

    fusion = PromptGuidedRoleExpertFusion(channels=channels, expert_dim=128)
    fusion.train()
    out_full = fusion(ct, pet, route="full")
    out_missing = fusion(ct, pet, route="missing")

    for source, result in (("full", out_full), ("missing", out_missing)):
        for i, (x, y) in enumerate(zip(ct, result.features), start=1):
            assert x.shape == y.shape, (source, i, x.shape, y.shape)
            assert torch.isfinite(y).all(), (source, i)
        loss = sum(x.float().mean() for x in result.features)
        loss.backward()
        fusion.zero_grad(set_to_none=True)

    print("PromptGuidedRoleExpertFusion smoke test: PASS")
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in fusion.parameters() if p.requires_grad):,}",
    )


if __name__ == "__main__":
    _smoke_test_core()
