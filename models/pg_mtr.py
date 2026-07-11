# -*- coding: utf-8 -*-
"""
PG-MTR: PET-Grounded Metabolic Token Retrieval

Purpose
-------
This module is designed for PET-optional PET-CT segmentation.

- CT is the always-available anatomical anchor.
- PET is the optional metabolic guidance.
- Real PET is used during Full steps to ground a small metabolic token memory.
- When PET is missing, CT retrieves from this PET-grounded memory.
- The retrieved memory is not added to CT directly. It must first pass through
  a CT-supported interaction branch, and only the resulting deep residual is
  injected into stage 3 and stage 4 features.

Important design constraints
----------------------------
1. Only stage 3 and stage 4 are adapted.
2. Full segmentation features are NEVER modified by PG-MTR.
3. Full-step grounding losses use detached CT/PET backbone features so that
   the auxiliary memory objective does not directly distort the encoders.
4. Missing-route output projection is zero-initialized, therefore the initial
   missing route is exactly the E1-Control route.
5. No BatchNorm, no dynamic sigmoid gate, no PET image reconstruction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "StagePETGroundedMetabolicTokenRetrieval",
    "PETGroundedMetabolicTokenRetrieval",
]


# ============================================================
# Utility functions
# ============================================================

def _valid_group_count(
    channels: int,
    max_groups: int = 8,
) -> int:
    """
    Return the largest valid GroupNorm group count
    that divides channels.
    """

    channels = int(channels)

    if channels <= 0:
        raise ValueError(
            f"channels must be positive, got {channels}"
        )

    for groups in range(
        min(int(max_groups), channels),
        0,
        -1,
    ):
        if channels % groups == 0:
            return groups

    return 1


def _trunc_normal_(
    tensor: torch.Tensor,
    std: float = 0.02,
) -> torch.Tensor:
    """
    Compatibility wrapper for truncated-normal initialization.
    """

    try:
        return nn.init.trunc_normal_(
            tensor,
            mean=0.0,
            std=float(std),
            a=-2.0,
            b=2.0,
        )

    except Exception:
        with torch.no_grad():
            return tensor.normal_(
                mean=0.0,
                std=float(std),
            )


def _rms(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Root mean square.
    """

    return (
        x.float()
        .pow(2)
        .mean()
        .add(eps)
        .sqrt()
    )


def _normalized_assignment_entropy(
    assignment: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Parameters
    ----------
    assignment:
        [B, K, H, W]

    Returns
    -------
    Normalized routing entropy.

    Approximately:
        1.0 → almost uniform routing
        0.0 → highly concentrated routing
    """

    probs = (
        assignment
        .float()
        .clamp_min(eps)
    )

    entropy = -(
        probs
        * probs.log()
    ).sum(
        dim=1
    ).mean()

    max_entropy = math.log(
        float(assignment.shape[1])
    )

    return entropy / max(
        max_entropy,
        eps,
    )


def _assignment_peak(
    assignment: torch.Tensor,
) -> torch.Tensor:
    """
    Mean maximum token probability.

    Approximately:
        1/K → uniform routing
        close to 1 → very sharp routing
    """

    return (
        assignment
        .float()
        .max(dim=1)
        .values
        .mean()
    )


def _mean_off_diagonal_token_cosine(
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Diagnostic for token collapse.

    Lower off-diagonal cosine similarity generally
    indicates more diverse tokens.
    """

    num_tokens = int(
        tokens.shape[0]
    )

    if num_tokens <= 1:
        return tokens.new_zeros(
            (),
            dtype=torch.float32,
        )

    norm_tokens = F.normalize(
        tokens.float(),
        dim=-1,
        eps=1e-6,
    )

    similarity = (
        norm_tokens
        @ norm_tokens.transpose(0, 1)
    )

    off_diagonal_sum = (
        similarity.sum()
        - torch.diagonal(
            similarity
        ).sum()
    )

    denominator = float(
        num_tokens
        * (num_tokens - 1)
    )

    return (
        off_diagonal_sum
        / denominator
    )


# ============================================================
# Single-stage PG-MTR
# ============================================================

class StagePETGroundedMetabolicTokenRetrieval(
    nn.Module
):
    """
    One deep-stage PG-MTR block.

    Full mode
    ---------
    Real PET and CT features are projected into a shared
    latent space and assigned to the same learnable
    metabolic token memory.

    The module returns:

        1. Route Alignment Loss
        2. Memory Grounding Loss

    The caller MUST keep the original Full fusion unchanged:

        F_full = C + P

    PG-MTR must not replace or alter the Full features.


    Missing mode
    ------------
    CT retrieves metabolic memory:

        CT
        ↓
        CT token assignment
        ↓
        PET-grounded token memory
        ↓
        Retrieved metabolic memory

    The memory is NOT directly added to CT.

    Instead:

        U = phi(CT) * psi(Memory)

        R = W_out(
                DWConv(
                    GELU(U)
                )
            )

        F_missing = CT + alpha * R

    The output projection is zero-initialized.

    Therefore:

        Initial Missing Route
        =
        E1-Control Missing Route
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int = 8,
        latent_dim: Optional[int] = None,
        temperature: float = 0.07,
        residual_scale_init: float = 0.1,
    ) -> None:

        super().__init__()

        self.in_channels = int(
            in_channels
        )

        self.num_tokens = int(
            num_tokens
        )

        self.temperature = float(
            temperature
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if self.in_channels <= 0:
            raise ValueError(
                "in_channels must be positive, "
                f"got {in_channels}"
            )

        if self.num_tokens <= 1:
            raise ValueError(
                "num_tokens must be > 1, "
                f"got {num_tokens}"
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be > 0, "
                f"got {temperature}"
            )

        # ----------------------------------------------------
        # Latent dimension
        #
        # S3:
        #   C = 320 → d = 80
        #
        # S4:
        #   C = 512 → d = 128
        # ----------------------------------------------------

        if latent_dim is None:

            latent_dim = min(
                max(
                    self.in_channels // 4,
                    32,
                ),
                128,
            )

        self.latent_dim = int(
            latent_dim
        )

        if self.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive, "
                f"got {latent_dim}"
            )

        # ====================================================
        # 1. PET-grounded metabolic token memory
        #
        # Shape:
        #   [K, d]
        # ====================================================

        self.memory_tokens = nn.Parameter(
            torch.empty(
                self.num_tokens,
                self.latent_dim,
            )
        )

        # ----------------------------------------------------
        # Token key:
        #   used for routing
        #
        # Token value:
        #   used as retrieved metabolic content
        # ----------------------------------------------------

        self.token_key = nn.Linear(
            self.latent_dim,
            self.latent_dim,
            bias=False,
        )

        self.token_value = nn.Linear(
            self.latent_dim,
            self.latent_dim,
            bias=False,
        )

        # ====================================================
        # 2. CT / PET query projections
        #
        # CT:
        #   learns how to retrieve PET-grounded memory
        #
        # PET:
        #   defines the true metabolic token assignment
        # ====================================================

        self.ct_query_proj = nn.Sequential(

            nn.GroupNorm(
                _valid_group_count(
                    self.in_channels
                ),
                self.in_channels,
            ),

            nn.Conv2d(
                self.in_channels,
                self.latent_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.pet_query_proj = nn.Sequential(

            nn.GroupNorm(
                _valid_group_count(
                    self.in_channels
                ),
                self.in_channels,
            ),

            nn.Conv2d(
                self.in_channels,
                self.latent_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        # ====================================================
        # 3. CT-supported memory interaction
        #
        # Retrieved memory cannot directly modify CT.
        #
        # CT anatomy must first support the metabolic memory.
        #
        # No sigmoid gate.
        # No softmax gate.
        # ====================================================

        self.ct_support_proj = nn.Sequential(

            nn.GroupNorm(
                _valid_group_count(
                    self.in_channels
                ),
                self.in_channels,
            ),

            nn.Conv2d(
                self.in_channels,
                self.latent_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.memory_support_proj = nn.Sequential(

            nn.GroupNorm(
                _valid_group_count(
                    self.latent_dim
                ),
                self.latent_dim,
            ),

            nn.Conv2d(
                self.latent_dim,
                self.latent_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        # ====================================================
        # 4. Lightweight spatial refinement
        #
        # PET mainly contributes lesion localization.
        #
        # Therefore a lightweight spatial operator is retained.
        # ====================================================

        self.spatial_refine = nn.Sequential(

            nn.Conv2d(
                self.latent_dim,
                self.latent_dim,
                kernel_size=3,
                padding=1,
                groups=self.latent_dim,
                bias=False,
            ),

            nn.GELU(),
        )

        # ====================================================
        # 5. Residual output projection
        #
        # Zero initialization is critical.
        #
        # At initialization:
        #
        #   residual = 0
        #
        # Therefore:
        #
        #   F_missing = CT
        # ====================================================

        self.out_proj = nn.Conv2d(
            self.latent_dim,
            self.in_channels,
            kernel_size=1,
            bias=True,
        )

        # ----------------------------------------------------
        # Learnable residual amplitude.
        #
        # This is NOT a dynamic gate.
        # ----------------------------------------------------

        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(
                    residual_scale_init
                ),
                dtype=torch.float32,
            )
        )

        self.reset_parameters()

    # ========================================================
    # Initialization
    # ========================================================

    def reset_parameters(
        self,
    ) -> None:

        # ----------------------------------------------------
        # Metabolic token memory
        # ----------------------------------------------------

        _trunc_normal_(
            self.memory_tokens,
            std=0.02,
        )

        # Remove shared token bias.
        with torch.no_grad():

            self.memory_tokens.sub_(

                self.memory_tokens.mean(
                    dim=0,
                    keepdim=True,
                )
            )

        # ----------------------------------------------------
        # Token key / value
        # ----------------------------------------------------

        nn.init.xavier_uniform_(
            self.token_key.weight
        )

        nn.init.xavier_uniform_(
            self.token_value.weight
        )

        # ----------------------------------------------------
        # Projection branches
        # ----------------------------------------------------

        for module in (

            self.ct_query_proj,
            self.pet_query_proj,
            self.ct_support_proj,
            self.memory_support_proj,

        ):

            for submodule in module.modules():

                if isinstance(
                    submodule,
                    nn.Conv2d,
                ):

                    nn.init.kaiming_normal_(
                        submodule.weight,
                        mode="fan_out",
                        nonlinearity="linear",
                    )

                    if (
                        submodule.bias
                        is not None
                    ):

                        nn.init.zeros_(
                            submodule.bias
                        )

        # ----------------------------------------------------
        # Spatial refinement
        # ----------------------------------------------------

        for submodule in (
            self.spatial_refine.modules()
        ):

            if isinstance(
                submodule,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    submodule.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )

                if (
                    submodule.bias
                    is not None
                ):

                    nn.init.zeros_(
                        submodule.bias
                    )

        # ----------------------------------------------------
        # Critical:
        # exact E1-Control initialization
        # ----------------------------------------------------

        nn.init.zeros_(
            self.out_proj.weight
        )

        if (
            self.out_proj.bias
            is not None
        ):

            nn.init.zeros_(
                self.out_proj.bias
            )

    # ========================================================
    # Validation
    # ========================================================

    def _validate_ct(
        self,
        ct_feat: torch.Tensor,
    ) -> None:

        if ct_feat.ndim != 4:

            raise ValueError(
                "ct_feat must have shape "
                "[B,C,H,W], "
                f"got {tuple(ct_feat.shape)}"
            )

        if (
            ct_feat.shape[1]
            != self.in_channels
        ):

            raise ValueError(
                "Expected CT channels="
                f"{self.in_channels}, "
                "got "
                f"{ct_feat.shape[1]}"
            )

    def _prepare_pet(
        self,
        pet_feat: torch.Tensor,
        ct_feat: torch.Tensor,
    ) -> torch.Tensor:

        if pet_feat.ndim != 4:

            raise ValueError(
                "pet_feat must have shape "
                "[B,C,H,W], "
                f"got {tuple(pet_feat.shape)}"
            )

        if (
            pet_feat.shape[0]
            != ct_feat.shape[0]
        ):

            raise ValueError(
                "CT/PET batch size mismatch: "
                f"{ct_feat.shape[0]} "
                "vs "
                f"{pet_feat.shape[0]}"
            )

        if (
            pet_feat.shape[1]
            != self.in_channels
        ):

            raise ValueError(
                "Expected PET channels="
                f"{self.in_channels}, "
                "got "
                f"{pet_feat.shape[1]}"
            )

        if (
            pet_feat.shape[-2:]
            != ct_feat.shape[-2:]
        ):

            pet_feat = F.interpolate(
                pet_feat,
                size=ct_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return pet_feat

    # ========================================================
    # Token operations
    # ========================================================

    def _token_key_value(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        token_key = self.token_key(
            self.memory_tokens
        )

        token_value = self.token_value(
            self.memory_tokens
        )

        return (
            token_key,
            token_value,
        )

    def _assignment(
        self,
        query: torch.Tensor,
        token_key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token assignment.

        query:
            [B, d, H, W]

        token_key:
            [K, d]

        output:
            [B, K, H, W]
        """

        # ----------------------------------------------------
        # FP32 routing for AMP stability
        # ----------------------------------------------------

        query_fp32 = F.normalize(
            query.float(),
            dim=1,
            eps=1e-6,
        )

        key_fp32 = F.normalize(
            token_key.float(),
            dim=-1,
            eps=1e-6,
        )

        logits = torch.einsum(
            "bdhw,kd->bkhw",
            query_fp32,
            key_fp32,
        )

        logits = (
            logits
            / self.temperature
        )

        assignment = torch.softmax(
            logits,
            dim=1,
        )

        return assignment

    @staticmethod
    def _read_memory(
        assignment: torch.Tensor,
        token_value: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        assignment:
            [B, K, H, W]

        token_value:
            [K, d]

        output:
            [B, d, H, W]
        """

        memory = torch.einsum(
            "bkhw,kd->bdhw",
            assignment.float(),
            token_value.float(),
        )

        return memory.to(
            dtype=output_dtype
        )

    # ========================================================
    # Losses
    # ========================================================

    @staticmethod
    def _route_alignment_loss(
        pet_assignment: torch.Tensor,
        ct_assignment: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        KL(
            stopgrad(PET assignment)
            ||
            CT assignment
        )

        Real PET acts as the teacher.
        """

        pet_target = (
            pet_assignment
            .detach()
            .float()
            .clamp_min(eps)
        )

        ct_prediction = (
            ct_assignment
            .float()
            .clamp_min(eps)
        )

        kl_map = (

            pet_target

            * (

                pet_target.log()

                -

                ct_prediction.log()
            )

        ).sum(
            dim=1
        )

        return kl_map.mean()

    @staticmethod
    def _memory_grounding_loss(
        pet_memory: torch.Tensor,
        pet_query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ground the retrieved memory to real PET latent features.

        The PET latent target is stop-gradient.
        """

        memory_norm = F.normalize(
            pet_memory.float(),
            dim=1,
            eps=1e-6,
        )

        pet_target = F.normalize(
            pet_query
            .detach()
            .float(),
            dim=1,
            eps=1e-6,
        )

        cosine = (

            memory_norm

            * pet_target

        ).sum(
            dim=1
        )

        return (
            1.0
            - cosine.mean()
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def _common_diagnostics(
        self,
        ct_assignment: torch.Tensor,
        token_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        return {

            "ct_route_entropy":
                _normalized_assignment_entropy(
                    ct_assignment
                ).detach(),

            "ct_route_peak":
                _assignment_peak(
                    ct_assignment
                ).detach(),

            "token_cosine_offdiag":
                _mean_off_diagonal_token_cosine(
                    token_value
                ).detach(),

            "residual_scale":
                self.residual_scale.detach(),
        }

    # ========================================================
    # Full mode
    # ========================================================

    def forward_full(
        self,
        ct_feat: torch.Tensor,
        pet_feat: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Full mode only trains the PET-grounded memory.

        Important:
        CT/PET backbone features are detached.

        Therefore auxiliary PG-MTR losses do not directly
        modify CT Encoder or PET Encoder.

        The original Full segmentation route remains:

            C + P
            ↓
            Decoder
        """

        self._validate_ct(
            ct_feat
        )

        pet_feat = self._prepare_pet(
            pet_feat,
            ct_feat,
        )

        # ----------------------------------------------------
        # Important scientific isolation:
        #
        # PG-MTR auxiliary objectives train the memory module,
        # not the backbone encoders.
        # ----------------------------------------------------

        ct_source = (
            ct_feat.detach()
        )

        pet_source = (
            pet_feat.detach()
        )

        # ----------------------------------------------------
        # CT / PET queries
        # ----------------------------------------------------

        ct_query = self.ct_query_proj(
            ct_source
        )

        pet_query = self.pet_query_proj(
            pet_source
        )

        # ----------------------------------------------------
        # Token key / value
        # ----------------------------------------------------

        (
            token_key,
            token_value,
        ) = self._token_key_value()

        # ----------------------------------------------------
        # PET-grounded routing
        # ----------------------------------------------------

        ct_assignment = self._assignment(
            ct_query,
            token_key,
        )

        pet_assignment = self._assignment(
            pet_query,
            token_key,
        )

        # ----------------------------------------------------
        # Read PET-grounded metabolic memory
        # ----------------------------------------------------

        pet_memory = self._read_memory(
            pet_assignment,
            token_value,
            output_dtype=pet_query.dtype,
        )

        # ----------------------------------------------------
        # Auxiliary losses
        # ----------------------------------------------------

        route_loss = (
            self._route_alignment_loss(
                pet_assignment,
                ct_assignment,
            )
        )

        memory_loss = (
            self._memory_grounding_loss(
                pet_memory,
                pet_query,
            )
        )

        aux_losses = {

            "route_loss":
                route_loss,

            "mem_loss":
                memory_loss,
        }

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        diagnostics = (
            self._common_diagnostics(
                ct_assignment,
                token_value,
            )
        )

        diagnostics.update({

            "pet_route_entropy":
                _normalized_assignment_entropy(
                    pet_assignment
                ).detach(),

            "pet_route_peak":
                _assignment_peak(
                    pet_assignment
                ).detach(),

            "pet_memory_rms":
                _rms(
                    pet_memory
                ).detach(),

            "route_loss":
                route_loss.detach(),

            "mem_loss":
                memory_loss.detach(),
        })

        return (
            aux_losses,
            diagnostics,
        )

    # ========================================================
    # Missing mode
    # ========================================================

    def forward_missing(
        self,
        ct_feat: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Missing mode:

            CT
            ↓
            CT token retrieval
            ↓
            PET-grounded memory
            ↓
            CT-supported interaction
            ↓
            safe residual
        """

        self._validate_ct(
            ct_feat
        )

        # ----------------------------------------------------
        # CT predicts PET token usage
        # ----------------------------------------------------

        ct_query = self.ct_query_proj(
            ct_feat
        )

        (
            token_key,
            token_value,
        ) = self._token_key_value()

        ct_assignment = self._assignment(
            ct_query,
            token_key,
        )

        # ----------------------------------------------------
        # Retrieve metabolic memory
        # ----------------------------------------------------

        retrieved_memory = (
            self._read_memory(
                ct_assignment,
                token_value,
                output_dtype=ct_feat.dtype,
            )
        )

        # ----------------------------------------------------
        # CT-supported interaction
        #
        # Important:
        # No dynamic gate.
        #
        # Only metabolic evidence supported by current
        # CT anatomy can produce a strong response.
        # ----------------------------------------------------

        ct_support = (
            self.ct_support_proj(
                ct_feat
            )
        )

        memory_support = (
            self.memory_support_proj(
                retrieved_memory
            )
        )

        interaction = (

            ct_support

            * memory_support
        )

        interaction = F.gelu(
            interaction
        )

        # ----------------------------------------------------
        # Lightweight spatial correction
        # ----------------------------------------------------

        interaction = (
            self.spatial_refine(
                interaction
            )
        )

        # ----------------------------------------------------
        # Safe residual
        # ----------------------------------------------------

        residual = self.out_proj(
            interaction
        )

        residual = (

            residual

            * self.residual_scale.to(
                dtype=residual.dtype
            )
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        diagnostics = (
            self._common_diagnostics(
                ct_assignment,
                token_value,
            )
        )

        ct_rms = _rms(
            ct_feat
        )

        residual_rms = _rms(
            residual
        )

        diagnostics.update({

            "retrieved_memory_rms":
                _rms(
                    retrieved_memory
                ).detach(),

            "residual_rms":
                residual_rms.detach(),

            "ct_rms":
                ct_rms.detach(),

            "residual_ct_ratio":
                (
                    residual_rms
                    / (
                        ct_rms
                        + 1e-6
                    )
                ).detach(),
        })

        return (
            residual,
            diagnostics,
        )

    # ========================================================
    # Public forward
    # ========================================================

    def forward(
        self,
        ct_feat: torch.Tensor,
        pet_feat: Optional[
            torch.Tensor
        ] = None,
        mode: str = "missing",
    ):

        mode = str(
            mode
        ).lower()

        if mode == "full":

            if pet_feat is None:

                raise ValueError(
                    "mode='full' "
                    "requires pet_feat"
                )

            return self.forward_full(
                ct_feat,
                pet_feat,
            )

        if mode == "missing":

            return self.forward_missing(
                ct_feat
            )

        raise ValueError(
            "Unsupported PG-MTR "
            f"mode={mode!r}"
        )


# ============================================================
# Four-stage wrapper
# ============================================================

class PETGroundedMetabolicTokenRetrieval(
    nn.Module
):
    """
    Hierarchical PG-MTR wrapper.

    Encoder stages:

        S1
        S2
        S3
        S4

    Only:

        S3
        S4

    use PET-grounded missing compensation.


    Full mode
    ---------

    Returns:

        missing_feats = None

        aux_losses = {
            "pg_mtr_route_loss",
            "pg_mtr_mem_loss",
        }

        diagnostics = {...}

    Full segmentation features must remain:

        C + P


    Missing mode
    ------------

    Returns:

        [
            C1,
            C2,
            C3 + R3,
            C4 + R4,
        ]
    """

    def __init__(
        self,
        channels_list: Sequence[int],
        num_tokens: int = 8,
        temperature: float = 0.07,
        residual_scale_init: float = 0.1,
        stage_mode: str = "deep",
    ) -> None:

        super().__init__()

        channels = [
            int(channel)
            for channel
            in channels_list
        ]

        if len(channels) != 4:

            raise ValueError(
                "PG-MTR expects exactly "
                "four encoder stages, "
                f"got {len(channels)}"
            )

        self.channels_list = channels
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        self.stage_mode = str(stage_mode)
        if self.stage_mode == "deep":
            active_stage_numbers = (3, 4)
        elif self.stage_mode == "all":
            active_stage_numbers = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported stage_mode={stage_mode!r}")
        self.active_stage_numbers = active_stage_numbers
        self.active_stage_indices = tuple(stage_number - 1 for stage_number in active_stage_numbers)
        self.stage_modules = nn.ModuleDict()
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            self.stage_modules[str(stage_number)] = StagePETGroundedMetabolicTokenRetrieval(
                in_channels=channels[stage_index],
                num_tokens=self.num_tokens,
                temperature=self.temperature,
                residual_scale_init=residual_scale_init,
            )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_feature_list(
        features: Sequence[
            torch.Tensor
        ],
        name: str,
    ) -> None:

        if len(features) != 4:

            raise ValueError(
                f"{name} must contain "
                "four stage features, "
                f"got {len(features)}"
            )

    @staticmethod
    def _prefix_diagnostics(
        stage_number: int,
        diagnostics: Dict[
            str,
            torch.Tensor,
        ],
    ) -> Dict[
        str,
        torch.Tensor,
    ]:

        return {

            (
                f"pg_mtr_s"
                f"{stage_number}_"
                f"{key}"
            ):
                value

            for key, value
            in diagnostics.items()
        }

    # ========================================================
    # Full mode
    # ========================================================

    def forward_full(
        self,
        aligned_ct_feats: Sequence[
            torch.Tensor
        ],
        pet_feats: Sequence[
            torch.Tensor
        ],
    ) -> Tuple[
        None,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:

        self._validate_feature_list(
            aligned_ct_feats,
            "aligned_ct_feats",
        )

        self._validate_feature_list(
            pet_feats,
            "pet_feats",
        )

        route_losses: List[
            torch.Tensor
        ] = []

        memory_losses: List[
            torch.Tensor
        ] = []

        diagnostics: Dict[
            str,
            torch.Tensor,
        ] = {}

        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):

            stage_module = (
                self.stage_modules[
                    str(stage_number)
                ]
            )

            (
                stage_losses,
                stage_diagnostics,
            ) = stage_module.forward_full(

                aligned_ct_feats[
                    stage_index
                ],

                pet_feats[
                    stage_index
                ],
            )

            route_losses.append(

                stage_losses[
                    "route_loss"
                ]
            )

            memory_losses.append(

                stage_losses[
                    "mem_loss"
                ]
            )

            diagnostics.update(

                self._prefix_diagnostics(

                    stage_number,
                    stage_diagnostics,
                )
            )

        # ----------------------------------------------------
        # Average over active stages
        # ----------------------------------------------------

        route_loss = torch.stack(
            route_losses
        ).mean()

        memory_loss = torch.stack(
            memory_losses
        ).mean()

        aux_losses = {

            "pg_mtr_route_loss":
                route_loss,

            "pg_mtr_mem_loss":
                memory_loss,
        }

        diagnostics.update({

            "pg_mtr_route_loss_mean":
                route_loss.detach(),

            "pg_mtr_mem_loss_mean":
                memory_loss.detach(),
        })

        # ----------------------------------------------------
        # Full features are intentionally NOT returned.
        #
        # Caller must continue to use:
        #
        #   C + P
        # ----------------------------------------------------

        return (
            None,
            aux_losses,
            diagnostics,
        )

    # ========================================================
    # Missing mode
    # ========================================================

    def forward_missing(
        self,
        aligned_ct_feats: Sequence[
            torch.Tensor
        ],
    ) -> Tuple[
        List[torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:

        self._validate_feature_list(
            aligned_ct_feats,
            "aligned_ct_feats",
        )

        # ----------------------------------------------------
        # S1 / S2 stay exactly CT-only.
        # ----------------------------------------------------

        missing_feats = list(
            aligned_ct_feats
        )

        diagnostics: Dict[
            str,
            torch.Tensor,
        ] = {}

        residual_ratios: List[
            torch.Tensor
        ] = []

        residual_scales: List[
            torch.Tensor
        ] = []

        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):

            stage_module = (
                self.stage_modules[
                    str(stage_number)
                ]
            )

            (
                residual,
                stage_diagnostics,
            ) = stage_module.forward_missing(

                aligned_ct_feats[
                    stage_index
                ]
            )

            missing_feats[
                stage_index
            ] = (

                aligned_ct_feats[
                    stage_index
                ]

                + residual
            )

            diagnostics.update(

                self._prefix_diagnostics(

                    stage_number,
                    stage_diagnostics,
                )
            )

            residual_ratios.append(

                stage_diagnostics[
                    "residual_ct_ratio"
                ]
            )

            residual_scales.append(

                stage_diagnostics[
                    "residual_scale"
                ].float()
            )

        diagnostics.update({

            "pg_mtr_residual_ratio_mean":
                torch.stack(
                    residual_ratios
                ).mean().detach(),

            "pg_mtr_residual_scale_mean":
                torch.stack(
                    residual_scales
                ).mean().detach(),
        })

        return (
            missing_feats,
            {},
            diagnostics,
        )

    # ========================================================
    # Public forward
    # ========================================================

    def forward(
        self,
        aligned_ct_feats: Sequence[
            torch.Tensor
        ],
        pet_feats: Optional[
            Sequence[torch.Tensor]
        ] = None,
        mode: str = "missing",
    ) -> Tuple[
        Optional[
            List[torch.Tensor]
        ],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:

        mode = str(
            mode
        ).lower()

        if mode == "full":

            if pet_feats is None:

                raise ValueError(
                    "mode='full' "
                    "requires pet_feats"
                )

            return self.forward_full(
                aligned_ct_feats,
                pet_feats,
            )

        if mode == "missing":

            return self.forward_missing(
                aligned_ct_feats
            )

        raise ValueError(
            "Unsupported PG-MTR "
            f"mode={mode!r}"
        )