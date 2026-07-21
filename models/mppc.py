"""MPPC: Mono-modalized Paired Prototype Compensation.

Standalone PyTorch implementation for PET-optional CT-PET segmentation.

The module is inserted after the existing CT/PET channel-alignment layers and
before the existing element-wise SUM fusion.  It deliberately keeps the clean
baseline fusion unchanged for complete (Full) batches:

    Full:    fused_s = ct_s + pet_s
    Missing: fused_s = ct_s + mppc_s(ct_s)

For Full training batches, MPPC only writes paired CT-key/PET-value prototypes
using the ground-truth segmentation mask.  For Missing batches, it never needs
the PET image or PET features: local CT queries retrieve PET references through
CT-to-CT similarity, while an EMA PET-to-PET consistency score estimates how
trustworthy each paired prototype is.  Low-confidence reads produce a zero
additive residual, so SUM fusion falls back to the unchanged CT representation.

No reconstruction, distillation, feature-alignment, or auxiliary loss is used.
All prototype banks are FP32 buffers, are saved in ``state_dict``, and receive
no gradient.  Only one near-zero-initialized residual scale per feature level is
learnable.

Minimal integration
-------------------
    mppc = MPPC(channels=(64, 128, 320, 512), num_slots=3)

    # Full batch: real PET features pass through unchanged; the bank is updated.
    pet_for_sum = mppc(
        ct_features, pet_features, target=mask, mode="full"
    )
    fused_features = [c + p for c, p in zip(ct_features, pet_for_sum)]

    # Missing batch: do not load PET and do not run the PET encoder.
    pet_for_sum = mppc(ct_features, pet_features=None, mode="missing")
    fused_features = [c + p for c, p in zip(ct_features, pet_for_sum)]

Important training rules
------------------------
1. Alternate batch-level Full/Missing routes as in the clean baseline.
2. Start an epoch/run with a Full batch so the bank has valid entries.
3. Call ``model.eval()`` for validation/test; the bank is then always frozen.
4. With model EMA, EMA the parameters normally, but copy MPPC banks directly:
       online.mppc.copy_bank_to(ema.mppc)
   Do not EMA the bank buffers a second time.
5. Under DDP, every rank must execute the same Full/Missing route per step.
   MPPC gathers Full-batch prototypes and applies identical bank updates on all
   ranks.  This avoids rank-specific prototype banks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


Mode = Literal["full", "missing"]
Ablation = Literal["normal", "off", "shuffle_values"]


class MPPC(nn.Module):
    """Four-scale paired-prototype compensation for optional PET features.

    Args:
        channels: Aligned CT/PET channel count at every feature scale.
        num_classes: Prototype semantic classes. The current task uses two:
            background and tumor.
        num_slots: Number of paired CT-key/PET-value slots per class and scale.
        momentum: EMA momentum used to update keys, values, and pair consistency.
        temperature: Softmax temperature for local CT-to-CT retrieval.
        gate_init_logit: Initial logit of each learnable residual scale. ``-6``
            gives sigmoid(-6) ~= 0.0025 and therefore starts close to the clean
            CT-only Missing path.
        eps: Numerical stability constant.

    Input features must already be channel- and spatially aligned within every
    CT/PET scale.  The module returns the additive feature used by the original
    SUM fusion: real PET features in Full mode and a confidence-controlled PET
    reference residual in Missing mode.
    """

    _BANK_BUFFER_KINDS: Tuple[str, ...] = (
        "ct_keys",
        "pet_values",
        "pair_consistency",
        "slot_counts",
        "slot_initialized",
    )

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        num_classes: int = 2,
        num_slots: int = 3,
        momentum: float = 0.9,
        temperature: float = 0.1,
        gate_init_logit: float = -6.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not channels or any(int(c) <= 0 for c in channels):
            raise ValueError(f"channels must contain positive integers, got {channels}")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")

        self.channels: Tuple[int, ...] = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.num_classes = int(num_classes)
        self.num_slots = int(num_slots)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # One conservative residual strength per scale.  There is intentionally
        # no learned spatial/channel gate and no CT-to-PET generator.
        self.residual_gate_logits = nn.Parameter(
            torch.full((self.num_scales,), float(gate_init_logit), dtype=torch.float32)
        )

        # Each scale has [semantic class, slot, channel] paired buffers.
        for scale_idx, channel_count in enumerate(self.channels):
            self.register_buffer(
                f"ct_keys_{scale_idx}",
                torch.zeros(self.num_classes, self.num_slots, channel_count),
            )
            self.register_buffer(
                f"pet_values_{scale_idx}",
                torch.zeros(self.num_classes, self.num_slots, channel_count),
            )
            self.register_buffer(
                f"pair_consistency_{scale_idx}",
                torch.zeros(self.num_classes, self.num_slots),
            )
            self.register_buffer(
                f"slot_counts_{scale_idx}",
                torch.zeros(self.num_classes, self.num_slots, dtype=torch.long),
            )
            self.register_buffer(
                f"slot_initialized_{scale_idx}",
                torch.zeros(self.num_classes, self.num_slots, dtype=torch.bool),
            )

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def forward(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Optional[Sequence[Tensor]] = None,
        target: Optional[Tensor] = None,
        mode: Optional[Mode] = None,
        *,
        update_bank: bool = True,
        ablation: Ablation = "normal",
        return_diagnostics: bool = False,
        return_maps: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], Dict[str, Any]]]:
        """Return features to be added to CT by the existing SUM fusion.

        Args:
            ct_features: Multi-scale aligned CT features ``[B, C_s, H_s, W_s]``.
            pet_features: Multi-scale aligned PET features in Full mode only.
                It must be ``None`` in Missing mode.
            target: Binary label map ``[B,H,W]``/``[B,1,H,W]`` or one-hot
                map ``[B,num_classes,H,W]``. Required only when a Full training
                call updates the bank.
            mode: ``"full"`` or ``"missing"``. If omitted, it is inferred from
                whether ``pet_features`` is present.
            update_bank: Allow a Full *training* call to update buffers. The bank
                never updates in eval mode, regardless of this flag.
            ablation:
                - ``"normal"``: standard MPPC read.
                - ``"off"``: zero additive residual in Missing mode.
                - ``"shuffle_values"``: circularly permute active PET values
                  before reading, breaking CT-key/PET-value pairing for a
                  falsification experiment without changing the stored bank.
            return_diagnostics: Also return scalar bank/retrieval diagnostics.
            return_maps: Include detached spatial confidence maps in diagnostics.
                This implies ``return_diagnostics=True``.

        Returns:
            A list with the same shapes as ``ct_features``. In Full mode these
            are the original PET tensors (gradient path preserved). In Missing
            mode these are confidence-controlled additive residuals. If
            diagnostics are requested, returns ``(features, diagnostics)``.
        """

        if mode is None:
            mode = "full" if pet_features is not None else "missing"
        if mode not in ("full", "missing"):
            raise ValueError(f"mode must be 'full' or 'missing', got {mode!r}")
        if ablation not in ("normal", "off", "shuffle_values"):
            raise ValueError(f"unknown ablation: {ablation!r}")

        ct_features = self._validate_ct_features(ct_features)
        diagnostics: Dict[str, Any] = {"mode": mode, "scales": []}

        if mode == "full":
            if pet_features is None:
                raise ValueError("Full mode requires real pet_features")
            pet_features = self._validate_pet_features(ct_features, pet_features)

            if self.training and update_bank:
                if target is None:
                    raise ValueError(
                        "target is required when a Full training batch updates the MPPC bank"
                    )
                self.update_from_full(ct_features, pet_features, target)

            # Critical invariant: MPPC must not replace or modulate real PET on
            # the Full route.  Return the exact tensors to keep the baseline path.
            output = list(pet_features)
            if return_diagnostics or return_maps:
                diagnostics["scales"] = self.bank_summary()
                return output, diagnostics
            return output

        # Missing route: accepting PET here could silently leak unavailable data.
        if pet_features is not None:
            raise ValueError(
                "Missing mode requires pet_features=None; do not load or encode PET"
            )
        if target is not None:
            # A Missing read never needs the current sample's label. Silently
            # accepting it would make accidental train-time leakage harder to spot.
            raise ValueError("Missing mode does not consume target masks")

        output, scale_diagnostics = self._read_missing(
            ct_features,
            ablation=ablation,
            return_maps=return_maps,
        )
        if return_diagnostics or return_maps:
            diagnostics["scales"] = scale_diagnostics
            return output, diagnostics
        return output

    @torch.no_grad()
    def update_from_full(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        target: Tensor,
    ) -> None:
        """Extract paired prototypes from a Full batch and update the FP32 bank.

        The CT prototype alone selects the slot. The PET prototype is always
        written to that same slot. CT and PET never cluster independently.
        """

        ct_features = self._validate_ct_features(ct_features)
        pet_features = self._validate_pet_features(ct_features, pet_features)

        for scale_idx, (ct, pet) in enumerate(zip(ct_features, pet_features)):
            class_masks = self._resize_class_masks(
                target, size=ct.shape[-2:], device=ct.device
            )

            # Parameter-free local smoothing makes the prototype less sensitive
            # to isolated feature pixels while preserving the feature dimension.
            ct_local = F.avg_pool2d(
                ct.detach().float(), 3, stride=1, padding=1, count_include_pad=False
            )
            pet_local = F.avg_pool2d(
                pet.detach().float(), 3, stride=1, padding=1, count_include_pad=False
            )

            ct_proto, pet_proto, class_ids = self._masked_pair_prototypes(
                ct_local, pet_local, class_masks
            )
            ct_proto, pet_proto, class_ids = self._ddp_gather_prototypes(
                ct_proto, pet_proto, class_ids
            )
            self._update_scale_bank(scale_idx, ct_proto, pet_proto, class_ids)

    @torch.no_grad()
    def reset_bank(self) -> None:
        """Clear all learned prototype statistics without changing parameters."""

        for scale_idx in range(self.num_scales):
            for kind in self._BANK_BUFFER_KINDS:
                getattr(self, f"{kind}_{scale_idx}").zero_()

    @torch.no_grad()
    def copy_bank_to(self, other: "MPPC") -> None:
        """Copy online MPPC buffers directly to an EMA/evaluation MPPC instance."""

        if not isinstance(other, MPPC):
            raise TypeError(f"other must be MPPC, got {type(other).__name__}")
        if (
            self.channels != other.channels
            or self.num_classes != other.num_classes
            or self.num_slots != other.num_slots
        ):
            raise ValueError("source and destination MPPC bank shapes do not match")
        for scale_idx in range(self.num_scales):
            for kind in self._BANK_BUFFER_KINDS:
                source = getattr(self, f"{kind}_{scale_idx}")
                destination = getattr(other, f"{kind}_{scale_idx}")
                destination.copy_(source.to(device=destination.device))

    @torch.no_grad()
    def bank_summary(self) -> List[Dict[str, Any]]:
        """Return lightweight per-scale bank statistics for logging."""

        summary: List[Dict[str, Any]] = []
        gates = torch.sigmoid(self.residual_gate_logits.detach())
        for scale_idx in range(self.num_scales):
            initialized = self._bank(scale_idx, "slot_initialized")
            counts = self._bank(scale_idx, "slot_counts")
            consistency = self._bank(scale_idx, "pair_consistency")
            active_consistency = consistency[initialized]
            summary.append(
                {
                    "scale": scale_idx,
                    "channels": self.channels[scale_idx],
                    "active_slots": int(initialized.sum().item()),
                    "total_slots": self.num_classes * self.num_slots,
                    "slot_counts": counts.detach().cpu().clone(),
                    "pair_consistency": consistency.detach().cpu().clone(),
                    "mean_active_pair_consistency": (
                        float(active_consistency.mean().item())
                        if active_consistency.numel() > 0
                        else 0.0
                    ),
                    "residual_scale": float(gates[scale_idx].item()),
                }
            )
        return summary

    def trainable_parameter_count(self) -> int:
        """Number of trainable MPPC parameters (normally one scalar per scale)."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------------------
    # Missing read
    # ---------------------------------------------------------------------
    def _read_missing(
        self,
        ct_features: Sequence[Tensor],
        *,
        ablation: Ablation,
        return_maps: bool,
    ) -> Tuple[List[Tensor], List[Dict[str, Any]]]:
        outputs: List[Tensor] = []
        diagnostics: List[Dict[str, Any]] = []

        for scale_idx, ct in enumerate(ct_features):
            initialized = self._bank(scale_idx, "slot_initialized").reshape(-1)
            active_indices = initialized.nonzero(as_tuple=False).flatten()
            active_count = int(active_indices.numel())
            residual_scale = torch.sigmoid(self.residual_gate_logits[scale_idx])

            base_diag: Dict[str, Any] = {
                "scale": scale_idx,
                "channels": self.channels[scale_idx],
                "active_slots": active_count,
                "total_slots": self.num_classes * self.num_slots,
                "residual_scale": residual_scale.detach(),
            }

            # This is a zero additive residual, not a fake zero PET input. SUM
            # therefore gives exactly the unchanged CT representation.
            if ablation == "off" or active_count == 0:
                residual = torch.zeros_like(ct)
                outputs.append(residual)
                base_diag.update(
                    {
                        "mean_max_similarity": 0.0,
                        "mean_entropy_confidence": 0.0,
                        "mean_pair_confidence": 0.0,
                        "mean_reliability": 0.0,
                        "compensation_to_ct_norm": 0.0,
                    }
                )
                if return_maps:
                    zero_map = ct.new_zeros((ct.shape[0], 1, *ct.shape[-2:]))
                    base_diag.update(
                        {
                            "max_similarity_map": zero_map,
                            "entropy_confidence_map": zero_map.clone(),
                            "pair_confidence_map": zero_map.clone(),
                            "reliability_map": zero_map.clone(),
                        }
                    )
                diagnostics.append(base_diag)
                continue

            # Query and bank similarity are explicitly FP32 under AMP.
            ct_local = F.avg_pool2d(
                ct.float(), 3, stride=1, padding=1, count_include_pad=False
            )
            batch_size, channels, height, width = ct_local.shape
            query = ct_local.flatten(2).transpose(1, 2)  # [B, HW, C]
            query = F.normalize(query, dim=-1, eps=self.eps)

            all_keys = self._bank(scale_idx, "ct_keys").reshape(-1, channels)
            all_values = self._bank(scale_idx, "pet_values").reshape(-1, channels)
            all_consistency = self._bank(
                scale_idx, "pair_consistency"
            ).reshape(-1)

            keys = all_keys.index_select(0, active_indices).float()
            values = all_values.index_select(0, active_indices).float()
            consistency = all_consistency.index_select(0, active_indices).float()

            if ablation == "shuffle_values" and active_count > 1:
                # Deterministic circular permutation: CT addressing is unchanged,
                # but the stored cross-modal association is deliberately broken.
                values = torch.roll(values, shifts=1, dims=0)

            keys = F.normalize(keys, dim=-1, eps=self.eps)
            similarity = torch.matmul(query, keys.transpose(0, 1))
            attention = torch.softmax(similarity / self.temperature, dim=-1)

            max_similarity = similarity.max(dim=-1).values.clamp(min=0.0, max=1.0)
            if active_count > 1:
                entropy = -(
                    attention * attention.clamp_min(self.eps).log()
                ).sum(dim=-1)
                entropy_confidence = (
                    1.0 - entropy / math.log(active_count)
                ).clamp(min=0.0, max=1.0)
            else:
                # A single address cannot establish retrieval selectivity.
                entropy_confidence = torch.zeros_like(max_similarity)

            pair_confidence = torch.matmul(
                attention, consistency.clamp(min=0.0, max=1.0)
            )
            reliability = (
                max_similarity * entropy_confidence * pair_confidence
            ).clamp(min=0.0, max=1.0)

            retrieved_pet = torch.matmul(attention, values)
            residual_tokens = (
                residual_scale.float()
                * reliability.unsqueeze(-1)
                * retrieved_pet
            )
            residual = residual_tokens.transpose(1, 2).reshape(
                batch_size, channels, height, width
            )
            residual = residual.to(dtype=ct.dtype)
            outputs.append(residual)

            ct_norm = ct.detach().float().flatten(1).norm(dim=1).mean()
            residual_norm = residual.detach().float().flatten(1).norm(dim=1).mean()
            norm_ratio = residual_norm / ct_norm.clamp_min(self.eps)
            base_diag.update(
                {
                    "mean_max_similarity": max_similarity.detach().mean(),
                    "mean_entropy_confidence": entropy_confidence.detach().mean(),
                    "mean_pair_confidence": pair_confidence.detach().mean(),
                    "mean_reliability": reliability.detach().mean(),
                    "compensation_to_ct_norm": norm_ratio.detach(),
                }
            )

            if return_maps:
                def as_map(x: Tensor) -> Tensor:
                    return x.reshape(batch_size, 1, height, width).detach()

                base_diag.update(
                    {
                        "max_similarity_map": as_map(max_similarity),
                        "entropy_confidence_map": as_map(entropy_confidence),
                        "pair_confidence_map": as_map(pair_confidence),
                        "reliability_map": as_map(reliability),
                    }
                )
            diagnostics.append(base_diag)

        return outputs, diagnostics

    # ---------------------------------------------------------------------
    # Full write
    # ---------------------------------------------------------------------
    def _masked_pair_prototypes(
        self,
        ct: Tensor,
        pet: Tensor,
        class_masks: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return valid per-sample/per-class paired prototypes."""

        batch_size, channels, _, _ = ct.shape
        weights = class_masks.flatten(2).float()  # [B, classes, HW]
        mass = weights.sum(dim=-1)  # [B, classes]

        ct_flat = ct.flatten(2)
        pet_flat = pet.flatten(2)
        ct_proto = torch.einsum("bkn,bcn->bkc", weights, ct_flat)
        pet_proto = torch.einsum("bkn,bcn->bkc", weights, pet_flat)
        denominator = mass.unsqueeze(-1).clamp_min(self.eps)
        ct_proto = ct_proto / denominator
        pet_proto = pet_proto / denominator

        valid = mass > self.eps
        # A zero CT vector cannot provide a meaningful cosine address.
        valid = valid & (ct_proto.norm(dim=-1) > self.eps)
        class_grid = torch.arange(
            self.num_classes, device=ct.device, dtype=torch.long
        ).view(1, -1).expand(batch_size, -1)

        return (
            ct_proto[valid].reshape(-1, channels).float(),
            pet_proto[valid].reshape(-1, channels).float(),
            class_grid[valid].reshape(-1),
        )

    @torch.no_grad()
    def _update_scale_bank(
        self,
        scale_idx: int,
        ct_prototypes: Tensor,
        pet_prototypes: Tensor,
        class_ids: Tensor,
    ) -> None:
        if ct_prototypes.numel() == 0:
            return

        keys = self._bank(scale_idx, "ct_keys")
        values = self._bank(scale_idx, "pet_values")
        consistency = self._bank(scale_idx, "pair_consistency")
        counts = self._bank(scale_idx, "slot_counts")
        initialized = self._bank(scale_idx, "slot_initialized")
        momentum = self.momentum

        # Every rank receives prototypes in the same gathered order, making
        # sequential online assignment deterministic across DDP workers.
        for ct_proto, pet_proto, class_id_tensor in zip(
            ct_prototypes, pet_prototypes, class_ids
        ):
            class_id = int(class_id_tensor.item())
            ct_proto = F.normalize(ct_proto.float(), dim=0, eps=self.eps)
            pet_proto = pet_proto.float()

            empty = (~initialized[class_id]).nonzero(as_tuple=False).flatten()
            if empty.numel() > 0:
                slot = int(empty[0].item())
                keys[class_id, slot].copy_(ct_proto)
                values[class_id, slot].copy_(pet_proto)
                # The first pair is self-consistent. Later samples test whether
                # similar CT patterns really have stable PET correspondences.
                consistency[class_id, slot].fill_(1.0)
                initialized[class_id, slot].fill_(True)
                counts[class_id, slot].fill_(1)
                continue

            class_keys = F.normalize(
                keys[class_id].float(), dim=-1, eps=self.eps
            )
            slot = int(torch.mv(class_keys, ct_proto).argmax().item())

            previous_pet = values[class_id, slot].float()
            if previous_pet.norm() > self.eps and pet_proto.norm() > self.eps:
                pet_similarity = F.cosine_similarity(
                    pet_proto.unsqueeze(0), previous_pet.unsqueeze(0), dim=-1
                )[0].clamp(min=0.0, max=1.0)
            else:
                pet_similarity = pet_proto.new_tensor(0.0)

            updated_key = F.normalize(
                momentum * keys[class_id, slot].float()
                + (1.0 - momentum) * ct_proto,
                dim=0,
                eps=self.eps,
            )
            updated_value = (
                momentum * previous_pet + (1.0 - momentum) * pet_proto
            )
            updated_consistency = (
                momentum * consistency[class_id, slot].float()
                + (1.0 - momentum) * pet_similarity
            ).clamp(min=0.0, max=1.0)

            keys[class_id, slot].copy_(updated_key)
            values[class_id, slot].copy_(updated_value)
            consistency[class_id, slot].copy_(updated_consistency)
            counts[class_id, slot].add_(1)

    # ---------------------------------------------------------------------
    # Masks, validation, and DDP
    # ---------------------------------------------------------------------
    def _resize_class_masks(
        self,
        target: Tensor,
        *,
        size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        if not isinstance(target, Tensor):
            raise TypeError("target must be a torch.Tensor")
        target = target.detach().to(device=device)

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4:
            raise ValueError(
                f"target must have 3 or 4 dimensions, got shape {tuple(target.shape)}"
            )

        if target.shape[1] == self.num_classes:
            class_masks = target.float()
        elif target.shape[1] == 1:
            if self.num_classes == 2:
                tumor = target.float()
                if tumor.numel() > 0 and (
                    tumor.min().item() < 0.0 or tumor.max().item() > 1.0
                ):
                    raise ValueError(
                        "binary target values must be in [0, 1]; convert 0/255 masks first"
                    )
                tumor = tumor.clamp(0.0, 1.0)
                class_masks = torch.cat((1.0 - tumor, tumor), dim=1)
            else:
                labels = target[:, 0].long()
                if labels.numel() > 0 and (
                    labels.min().item() < 0
                    or labels.max().item() >= self.num_classes
                ):
                    raise ValueError("target contains a class index outside num_classes")
                class_masks = F.one_hot(
                    labels, num_classes=self.num_classes
                ).permute(0, 3, 1, 2).float()
        else:
            raise ValueError(
                "target channel dimension must be 1 or equal to num_classes"
            )

        # Area interpolation retains fractional occupancy of small tumors at
        # deeper scales instead of deleting them with nearest-neighbor sampling.
        if class_masks.shape[-2:] != size:
            class_masks = F.interpolate(class_masks, size=size, mode="area")
        return class_masks.clamp(0.0, 1.0)

    def _validate_ct_features(self, features: Sequence[Tensor]) -> List[Tensor]:
        if not isinstance(features, (list, tuple)):
            raise TypeError("ct_features must be a list or tuple of tensors")
        if len(features) != self.num_scales:
            raise ValueError(
                f"expected {self.num_scales} CT scales, got {len(features)}"
            )

        result = list(features)
        first_batch: Optional[int] = None
        for scale_idx, (feature, expected_channels) in enumerate(
            zip(result, self.channels)
        ):
            if not isinstance(feature, Tensor) or feature.ndim != 4:
                raise ValueError(
                    f"CT scale {scale_idx} must be a 4D tensor, got {type(feature)}"
                )
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"CT scale {scale_idx} has {feature.shape[1]} channels; "
                    f"expected {expected_channels} after alignment"
                )
            if first_batch is None:
                first_batch = feature.shape[0]
            elif feature.shape[0] != first_batch:
                raise ValueError("all CT scales must have the same batch size")
            bank_device = self._bank(scale_idx, "ct_keys").device
            if feature.device != bank_device:
                raise RuntimeError(
                    f"CT scale {scale_idx} is on {feature.device}, but MPPC is on "
                    f"{bank_device}; move the whole model/module to the feature device"
                )
        return result

    def _validate_pet_features(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
    ) -> List[Tensor]:
        if not isinstance(pet_features, (list, tuple)):
            raise TypeError("pet_features must be a list or tuple of tensors")
        if len(pet_features) != self.num_scales:
            raise ValueError(
                f"expected {self.num_scales} PET scales, got {len(pet_features)}"
            )
        result = list(pet_features)
        for scale_idx, (ct, pet) in enumerate(zip(ct_features, result)):
            if not isinstance(pet, Tensor) or pet.ndim != 4:
                raise ValueError(f"PET scale {scale_idx} must be a 4D tensor")
            if pet.shape != ct.shape:
                raise ValueError(
                    f"aligned CT/PET shapes differ at scale {scale_idx}: "
                    f"{tuple(ct.shape)} vs {tuple(pet.shape)}"
                )
            if pet.device != ct.device:
                raise RuntimeError("aligned CT/PET features must use the same device")
        return result

    def _bank(self, scale_idx: int, kind: str) -> Tensor:
        return getattr(self, f"{kind}_{scale_idx}")

    def _ddp_gather_prototypes(
        self,
        ct_prototypes: Tensor,
        pet_prototypes: Tensor,
        class_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """All-gather variable-length prototype rows without object collectives."""

        if not dist.is_available() or not dist.is_initialized():
            return ct_prototypes, pet_prototypes, class_ids

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            [ct_prototypes.shape[0]], device=ct_prototypes.device, dtype=torch.long
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(x.item()) for x in gathered_sizes]
        max_size = max(sizes)
        if max_size == 0:
            return ct_prototypes, pet_prototypes, class_ids

        def gather_rows(tensor: Tensor) -> Tensor:
            pad_shape = (max_size,) + tuple(tensor.shape[1:])
            padded = tensor.new_zeros(pad_shape)
            if tensor.shape[0] > 0:
                padded[: tensor.shape[0]].copy_(tensor)
            gathered = [torch.zeros_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            valid_parts = [part[:size] for part, size in zip(gathered, sizes) if size > 0]
            return torch.cat(valid_parts, dim=0)

        return (
            gather_rows(ct_prototypes),
            gather_rows(pet_prototypes),
            gather_rows(class_ids),
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, num_classes={self.num_classes}, "
            f"num_slots={self.num_slots}, momentum={self.momentum}, "
            f"temperature={self.temperature}, "
            f"trainable_params={self.trainable_parameter_count()}"
        )


def _standalone_self_test() -> None:
    """Small CPU smoke test; run with ``python mppc.py``."""

    torch.manual_seed(7)
    channels = (8, 16, 24, 32)
    sizes = ((32, 32), (16, 16), (8, 8), (4, 4))
    batch_size = 3

    module = MPPC(channels=channels, num_slots=3)
    module.train()
    ct = [
        torch.randn(batch_size, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, sizes)
    ]
    pet = [
        torch.randn(batch_size, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, sizes)
    ]
    target = torch.zeros(batch_size, 1, 128, 128)
    target[0, :, 35:65, 40:72] = 1.0
    target[1, :, 70:92, 60:85] = 1.0
    target[2, :, 48:76, 80:105] = 1.0

    # Three Full writes fill all three slots per class.
    for _ in range(3):
        full_output = module(ct, pet, target=target, mode="full")
        assert all(output is original for output, original in zip(full_output, pet))

    bank_before_eval = {
        name: value.clone()
        for name, value in module.named_buffers()
        if any(kind in name for kind in module._BANK_BUFFER_KINDS)
    }

    missing_output, diagnostics = module(
        ct,
        mode="missing",
        return_diagnostics=True,
        return_maps=True,
    )
    assert all(x.shape == c.shape for x, c in zip(missing_output, ct))
    fused = [c + residual for c, residual in zip(ct, missing_output)]
    loss = sum(x.square().mean() for x in fused)
    loss.backward()
    assert module.residual_gate_logits.grad is not None
    assert all(not buffer.requires_grad for buffer in module.buffers())

    # Off ablation must be exact identity after the caller's SUM.
    off_output = module(ct, mode="missing", ablation="off")
    assert all(torch.count_nonzero(x).item() == 0 for x in off_output)
    assert all(torch.equal(c + x, c) for c, x in zip(ct, off_output))

    # Eval Full calls do not mutate the bank.
    module.eval()
    _ = module(ct, pet, mode="full")
    for name, value in module.named_buffers():
        if name in bank_before_eval:
            assert torch.equal(value, bank_before_eval[name])

    # state_dict preserves every bank buffer.
    clone = MPPC(channels=channels, num_slots=3)
    clone.load_state_dict(module.state_dict())
    for scale_idx in range(module.num_scales):
        assert torch.equal(
            module._bank(scale_idx, "ct_keys"),
            clone._bank(scale_idx, "ct_keys"),
        )

    active = [item["active_slots"] for item in module.bank_summary()]
    reliabilities = [
        float(item["mean_reliability"].item())
        for item in diagnostics["scales"]
    ]
    print("MPPC self-test passed")
    print(f"trainable parameters: {module.trainable_parameter_count()}")
    print(f"active slots per scale: {active}")
    print(f"mean reliability per scale: {reliabilities}")


if __name__ == "__main__":
    _standalone_self_test()
