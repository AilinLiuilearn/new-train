"""DCI-Fuse: Distribution-Calibrated Common-Innovation Fusion.

This file is a standalone PyTorch implementation for PET-optional PET-CT
segmentation.  CT is always the structural anchor.  ``aux`` can be either a
real PET feature or a compensated PET feature; the block intentionally does
not receive a modality-state flag.

Data flow of one scale
----------------------
1. Model CT and auxiliary features as local Gaussian distributions.
2. Build a cross-modal Gaussian representation and use it only to predict a
   residual calibration of the auxiliary feature.
3. Construct a low-rank common representation from CT and calibrated PET.
4. Define the PET innovation as calibrated PET minus the common evidence.
5. Allocate every location among reject/common/innovation states, then write
   only selected PET evidence into the original CT anchor.

The Gaussian heads and reparameterization follow the core design of UMFNet's
UAM, while the low-rank down-unify-up skeleton is adapted from KTB's MAPA.
The common/innovation/reject decomposition and CT-anchored write-back are the
task-specific DCI-Fuse design.

Reference implementations:
    https://github.com/zitalk/UMFNet/blob/main/models/UMFNet.py
    https://github.com/imcjx/KTB/blob/main/semseg/models/backbones/swin.py

Typical integration
-------------------
    fusion = MultiScaleDCIFuse(channels=(64, 128, 320, 512))
    fused_feats, loss_dist = fusion(ct_feats, pet_or_proxy_feats)
    prediction = decoder(fused_feats)
    loss = loss_seg + 1e-3 * loss_dist

Run ``python dci_fuse.py`` for a minimal shape, gradient, and parameter test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = [
    "ConvMlp",
    "GaussianHead",
    "CrossModalGaussianHead",
    "DCIFuse",
    "MultiScaleDCIFuse",
    "DCIFuseDetails",
    "kl_to_standard_normal",
    "count_trainable_parameters",
]


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _zero_init(module: nn.Module) -> None:
    """Initialize a Conv/Linear layer to output exactly zero."""
    if not hasattr(module, "weight"):
        raise TypeError("_zero_init expects a module with a weight parameter.")
    nn.init.zeros_(module.weight)
    bias = getattr(module, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


class ConvMlp(nn.Module):
    """The point-wise convolutional MLP used by the Gaussian heads."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        _require_positive_int("channels", channels)
        hidden_channels = hidden_channels or channels
        _require_positive_int("hidden_channels", hidden_channels)

        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GaussianHead(nn.Module):
    """Predict a pixel-wise diagonal Gaussian distribution for one modality."""

    def __init__(
        self,
        channels: int,
        logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        _require_positive_int("channels", channels)
        if logvar_clamp[0] >= logvar_clamp[1]:
            raise ValueError("logvar_clamp must satisfy lower < upper.")

        self.logvar_clamp = logvar_clamp
        self.local_context = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
        )
        self.norm_mu = nn.BatchNorm2d(channels)
        self.norm_logvar = nn.BatchNorm2d(channels)
        self.mu_head = ConvMlp(channels)
        self.logvar_head = ConvMlp(channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = x + self.local_context(x)
        mu = self.mu_head(self.norm_mu(h))
        logvar = self.logvar_head(self.norm_logvar(h))
        logvar = torch.clamp(logvar, *self.logvar_clamp)
        std = torch.exp(0.5 * logvar.float()).to(dtype=logvar.dtype)
        return mu, logvar, std


class CrossModalGaussianHead(nn.Module):
    """Predict the local joint Gaussian of CT and auxiliary distributions."""

    def __init__(
        self,
        channels: int,
        logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        _require_positive_int("channels", channels)
        if logvar_clamp[0] >= logvar_clamp[1]:
            raise ValueError("logvar_clamp must satisfy lower < upper.")

        self.logvar_clamp = logvar_clamp
        self.cross_context = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
        )
        self.norm_mu = nn.BatchNorm2d(channels)
        self.norm_logvar = nn.BatchNorm2d(channels)
        self.mu_head = ConvMlp(channels)
        self.logvar_head = ConvMlp(channels)

    def forward(
        self,
        mu_ct: Tensor,
        mu_aux: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if mu_ct.shape != mu_aux.shape:
            raise ValueError(
                "CrossModalGaussianHead expects equal shapes, got "
                f"CT={tuple(mu_ct.shape)} and AUX={tuple(mu_aux.shape)}."
            )
        h = self.cross_context(torch.cat([mu_ct, mu_aux], dim=1))
        mu = self.mu_head(self.norm_mu(h))
        logvar = self.logvar_head(self.norm_logvar(h))
        logvar = torch.clamp(logvar, *self.logvar_clamp)
        std = torch.exp(0.5 * logvar.float()).to(dtype=logvar.dtype)
        return mu, logvar, std


def kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean KL divergence KL[N(mu, var) || N(0, I)] in float32.

    Float32 evaluation avoids overflow or loss of precision under AMP.  The
    returned scalar remains differentiable with respect to the original input.
    """
    mu_f = mu.float()
    logvar_f = logvar.float()
    kl = 0.5 * (torch.exp(logvar_f) + mu_f.square() - 1.0 - logvar_f)
    return kl.mean()


@dataclass
class DCIFuseDetails:
    """Optional diagnostic tensors returned for visualization/analysis."""

    aux_calibrated: Tensor
    common_evidence: Tensor
    innovation_evidence: Tensor
    weight_reject: Tensor
    weight_common: Tensor
    weight_innovation: Tensor
    logvar_ct: Tensor
    logvar_aux: Tensor
    logvar_joint: Tensor

    def detached(self) -> "DCIFuseDetails":
        """Return a graph-free copy suitable for logging."""
        return DCIFuseDetails(
            **{name: value.detach() for name, value in self.__dict__.items()}
        )


class _Projection(nn.Sequential):
    """1x1 projection used by the low-rank common-evidence pathway."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DCIFuse(nn.Module):
    """Single-scale DCI-Fuse block.

    Args:
        channels: Number of CT/output channels.
        aux_channels: Number of auxiliary channels.  If omitted, it is assumed
            equal to ``channels``.  A 1x1 convolution aligns unequal channels.
        latent_dim: Gaussian latent channels.  Default: ``max(32, C // 4)``.
        rank_dim: Low-rank common-evidence channels.  Default:
            ``max(16, C // 4)``.
        logvar_clamp: Numerical range for predicted log-variance.
        sample_during_training: Use Gaussian reparameterization noise in train
            mode.  Evaluation is always deterministic.
        calibration_scale_init: Initial learnable scale for the PET calibration
            residual.  Its output head is also zero-initialized, so the initial
            calibrated feature is exactly the input auxiliary feature.
        residual_scale_init: Initial per-channel layer scale for evidence
            write-back.  A small value makes the initial block close to CT-only.
        reject_bias_init: Initial logit advantage of the reject state.  This is
            a prior only; it remains trainable through the evidence head.

    Input:
        ct:  [B, channels, H, W]
        aux: [B, aux_channels, H, W], real PET or compensated PET

    Returns:
        By default ``(fused, dist_loss)``.  With ``return_details=True``, returns
        ``(fused, dist_loss, details)``.
    """

    def __init__(
        self,
        channels: int,
        aux_channels: Optional[int] = None,
        latent_dim: Optional[int] = None,
        rank_dim: Optional[int] = None,
        logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
        sample_during_training: bool = True,
        calibration_scale_init: float = 0.1,
        residual_scale_init: float = 1e-3,
        reject_bias_init: float = 2.0,
    ) -> None:
        super().__init__()
        _require_positive_int("channels", channels)
        aux_channels = channels if aux_channels is None else aux_channels
        _require_positive_int("aux_channels", aux_channels)

        latent_dim = latent_dim or max(32, channels // 4)
        rank_dim = rank_dim or max(16, channels // 4)
        _require_positive_int("latent_dim", latent_dim)
        _require_positive_int("rank_dim", rank_dim)

        self.channels = channels
        self.aux_channels = aux_channels
        self.latent_dim = latent_dim
        self.rank_dim = rank_dim
        self.sample_during_training = sample_during_training

        # External channel alignment only; this is Identity for the expected
        # [64, 128, 320, 512] already-aligned PET/CT features.
        self.aux_align: nn.Module
        if aux_channels == channels:
            self.aux_align = nn.Identity()
        else:
            self.aux_align = nn.Conv2d(aux_channels, channels, kernel_size=1)

        # Part 1: the retained UAM core -- local unimodal Gaussian heads and a
        # cross-modal Gaussian head.  Unlike full UAM, CT is not enhanced here,
        # and the auxiliary feature is not replaced by a new generated feature.
        self.proj_ct = nn.Conv2d(channels, latent_dim, kernel_size=1)
        self.proj_aux = nn.Conv2d(channels, latent_dim, kernel_size=1)
        self.gaussian_ct = GaussianHead(latent_dim, logvar_clamp)
        self.gaussian_aux = GaussianHead(latent_dim, logvar_clamp)
        self.gaussian_joint = CrossModalGaussianHead(latent_dim, logvar_clamp)

        self.calibration = nn.Sequential(
            nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, channels, kernel_size=1),
        )
        _zero_init(self.calibration[-1])
        self.calibration_scale = nn.Parameter(
            torch.tensor(float(calibration_scale_init))
        )

        # Part 2: low-rank down -> unify -> up path adapted from MAPA's central
        # skeleton.  No modality-specific prompts or symmetric updates remain.
        self.down_ct = _Projection(channels, rank_dim)
        self.down_aux = _Projection(channels, rank_dim)
        self.unify = nn.Sequential(
            nn.Conv2d(2 * rank_dim, rank_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(rank_dim),
            nn.GELU(),
        )
        self.up_common = nn.Conv2d(rank_dim, channels, kernel_size=1)

        # Part 3: three-state allocation.  Five low-rank relations plus three
        # bounded uncertainty summaries are sufficient; no large attention is
        # introduced.
        evidence_in_channels = 5 * rank_dim + 3
        self.evidence_head = nn.Sequential(
            nn.Conv2d(
                evidence_in_channels,
                rank_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(rank_dim),
            nn.GELU(),
            nn.Conv2d(
                rank_dim,
                rank_dim,
                kernel_size=3,
                padding=1,
                groups=rank_dim,
                bias=False,
            ),
            nn.BatchNorm2d(rank_dim),
            nn.GELU(),
            nn.Conv2d(rank_dim, 3, kernel_size=1),
        )
        with torch.no_grad():
            self.evidence_head[-1].bias.copy_(
                torch.tensor([reject_bias_init, 0.0, 0.0])
            )

        # Per-channel CT-anchored write-back scale.  This makes initial training
        # stable without preventing the module from learning stronger PET use.
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_scale_init))
        )

    @staticmethod
    def _reparameterize(
        mu: Tensor,
        std: Tensor,
        use_random_sample: bool,
    ) -> Tensor:
        if use_random_sample:
            return mu + std * torch.randn_like(std)
        return mu

    @staticmethod
    def _uncertainty_summary(logvar: Tensor) -> Tensor:
        # Bound the statistic so the evidence head never receives extreme raw
        # log-variance values under AMP or early training.
        return torch.tanh(0.25 * logvar.mean(dim=1, keepdim=True))

    def _validate_inputs(self, ct: Tensor, aux: Tensor) -> None:
        if ct.ndim != 4 or aux.ndim != 4:
            raise ValueError(
                "DCIFuse expects 4D NCHW tensors, got "
                f"CT.ndim={ct.ndim}, AUX.ndim={aux.ndim}."
            )
        if ct.shape[0] != aux.shape[0] or ct.shape[-2:] != aux.shape[-2:]:
            raise ValueError(
                "CT and AUX must share batch/spatial dimensions, got "
                f"CT={tuple(ct.shape)}, AUX={tuple(aux.shape)}."
            )
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Expected CT channels={self.channels}, got {ct.shape[1]}."
            )
        if aux.shape[1] != self.aux_channels:
            raise ValueError(
                f"Expected AUX channels={self.aux_channels}, got {aux.shape[1]}."
            )

    def forward(
        self,
        ct: Tensor,
        aux: Tensor,
        return_details: bool = False,
        detach_details: bool = True,
    ) -> Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, DCIFuseDetails],
    ]:
        self._validate_inputs(ct, aux)
        aux = self.aux_align(aux)

        # 1) Distribution modeling.
        ct_latent = self.proj_ct(ct)
        aux_latent = self.proj_aux(aux)
        mu_ct, logvar_ct, std_ct = self.gaussian_ct(ct_latent)
        mu_aux, logvar_aux, std_aux = self.gaussian_aux(aux_latent)
        mu_joint, logvar_joint, std_joint = self.gaussian_joint(mu_ct, mu_aux)

        use_random_sample = self.training and self.sample_during_training
        # z_ct is deliberately not sampled: DCI-Fuse never rewrites the CT
        # anchor in the distribution-calibration stage.
        del std_ct
        z_aux = self._reparameterize(mu_aux, std_aux, use_random_sample)
        z_joint = self._reparameterize(mu_joint, std_joint, use_random_sample)

        # 2) Residual-only auxiliary calibration.  Zero initialization makes
        # aux_calibrated == aux at the start of training.
        delta_aux = self.calibration(torch.cat([z_aux, z_joint], dim=1))
        alpha = torch.tanh(self.calibration_scale).to(dtype=delta_aux.dtype)
        aux_calibrated = aux + alpha * delta_aux

        # 3) Common evidence in a low-rank joint space.
        ct_low = self.down_ct(ct)
        aux_low = self.down_aux(aux_calibrated)
        joint_low = self.unify(torch.cat([ct_low, aux_low], dim=1))
        common = self.up_common(joint_low)

        # 4) PET-specific residual not explained by the learned common evidence.
        innovation = aux_calibrated - common
        innovation_low = aux_low - joint_low

        # 5) Reject/common/innovation allocation.  The reject state does not
        # enter the residual directly; Softmax rejection suppresses both writes.
        relation = torch.cat(
            [
                ct_low,
                aux_low,
                joint_low,
                torch.abs(ct_low - aux_low),
                innovation_low,
                self._uncertainty_summary(logvar_ct),
                self._uncertainty_summary(logvar_aux),
                self._uncertainty_summary(logvar_joint),
            ],
            dim=1,
        )
        logits = self.evidence_head(relation)
        weights = F.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        w_reject, w_common, w_innovation = weights.chunk(3, dim=1)

        selected_evidence = w_common * common + w_innovation * innovation
        residual_scale = self.residual_scale.to(dtype=selected_evidence.dtype)
        fused = ct + residual_scale * selected_evidence

        # Match the original UAM core: regularize only the two unimodal
        # distributions.  The joint representation remains task-driven.
        loss_dist = 0.5 * (
            kl_to_standard_normal(mu_ct, logvar_ct)
            + kl_to_standard_normal(mu_aux, logvar_aux)
        )

        if not return_details:
            return fused, loss_dist

        details = DCIFuseDetails(
            aux_calibrated=aux_calibrated,
            common_evidence=common,
            innovation_evidence=innovation,
            weight_reject=w_reject,
            weight_common=w_common,
            weight_innovation=w_innovation,
            logvar_ct=logvar_ct,
            logvar_aux=logvar_aux,
            logvar_joint=logvar_joint,
        )
        if detach_details:
            details = details.detached()
        return fused, loss_dist, details


class MultiScaleDCIFuse(nn.Module):
    """Independent DCI-Fuse blocks for all encoder scales.

    No feature or parameter is shared across scales.  The distribution loss is
    the equal-weight mean across scales, so no per-scale loss hyperparameter is
    introduced.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        aux_channels: Optional[Sequence[int]] = None,
        latent_dims: Optional[Sequence[int]] = None,
        rank_dims: Optional[Sequence[int]] = None,
        **block_kwargs: object,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        if not channels:
            raise ValueError("channels cannot be empty.")

        n_scales = len(channels)
        aux_channels = tuple(aux_channels) if aux_channels is not None else channels
        if len(aux_channels) != n_scales:
            raise ValueError("aux_channels must have the same length as channels.")

        if latent_dims is not None and len(latent_dims) != n_scales:
            raise ValueError("latent_dims must have the same length as channels.")
        if rank_dims is not None and len(rank_dims) != n_scales:
            raise ValueError("rank_dims must have the same length as channels.")

        self.channels = channels
        self.aux_channels = aux_channels
        self.blocks = nn.ModuleList(
            [
                DCIFuse(
                    channels=channels[i],
                    aux_channels=aux_channels[i],
                    latent_dim=None if latent_dims is None else latent_dims[i],
                    rank_dim=None if rank_dims is None else rank_dims[i],
                    **block_kwargs,
                )
                for i in range(n_scales)
            ]
        )

    def forward(
        self,
        ct_features: Sequence[Tensor],
        aux_features: Sequence[Tensor],
        return_details: bool = False,
        detach_details: bool = True,
    ) -> Union[
        Tuple[List[Tensor], Tensor],
        Tuple[List[Tensor], Tensor, List[DCIFuseDetails]],
    ]:
        if len(ct_features) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} CT scales, got {len(ct_features)}."
            )
        if len(aux_features) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} AUX scales, got {len(aux_features)}."
            )

        fused_features: List[Tensor] = []
        scale_losses: List[Tensor] = []
        all_details: List[DCIFuseDetails] = []

        for block, ct, aux in zip(self.blocks, ct_features, aux_features):
            if return_details:
                fused, scale_loss, details = block(
                    ct,
                    aux,
                    return_details=True,
                    detach_details=detach_details,
                )
                all_details.append(details)
            else:
                fused, scale_loss = block(ct, aux)
            fused_features.append(fused)
            scale_losses.append(scale_loss)

        loss_dist = torch.stack(scale_losses).mean()
        if return_details:
            return fused_features, loss_dist, all_details
        return fused_features, loss_dist


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _self_test() -> None:
    """Minimal CPU/CUDA smoke test; not used by the training code."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = (64, 128, 320, 512)
    sizes = (32, 16, 8, 4)
    model = MultiScaleDCIFuse(channels=channels).to(device)
    model.train()

    ct_features = [
        torch.randn(2, c, s, s, device=device, requires_grad=True)
        for c, s in zip(channels, sizes)
    ]
    aux_features = [
        torch.randn(2, c, s, s, device=device, requires_grad=True)
        for c, s in zip(channels, sizes)
    ]

    fused, loss_dist, details = model(
        ct_features,
        aux_features,
        return_details=True,
    )
    assert all(out.shape == ct.shape for out, ct in zip(fused, ct_features))
    assert torch.isfinite(loss_dist), "Distribution loss is not finite."

    loss = sum(x.square().mean() for x in fused) + 1e-3 * loss_dist
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    ), "No finite parameter gradient was produced."

    # The three evidence weights must form a per-pixel probability simplex.
    for d in details:
        weight_sum = d.weight_reject + d.weight_common + d.weight_innovation
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)

    # Evaluation uses distribution means and must therefore be deterministic.
    model.eval()
    with torch.no_grad():
        out_a, _ = model(ct_features, aux_features)
        out_b, _ = model(ct_features, aux_features)
    assert all(torch.equal(a, b) for a, b in zip(out_a, out_b))

    print("DCI-Fuse self-test passed.")
    print(f"Device: {device}")
    print(f"Output shapes: {[tuple(x.shape) for x in fused]}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Distribution loss: {loss_dist.item():.6f}")


if __name__ == "__main__":
    _self_test()