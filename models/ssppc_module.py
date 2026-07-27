"""
SSPPC: Spatial-Semantic Paired Prototype Compensation

API-style missing-modality protocol:
1) CT/PET are both encoded during training;
2) real pre-fusion CT/PET features are detached and used to build prototypes;
3) the real PET branch is masked before the first CT-PET fusion for Missing mode;
4) CT queries CT prototypes and reads PET tumor-background residuals.

Each scale stores four buffers:
- CT background prototype
- CT tumor prototype
- PET background prototype
- PET tumor prototype

Missing compensation:
    p_tumor = softmax(cos(CT, CT prototypes))[:, tumor]
    delta_pet = p_tumor * (PET_tumor_proto - PET_background_proto)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

TensorList = Sequence[Tensor]


class SpatialSemanticPairedPrototypeCompensation(nn.Module):
    """Spatial-semantic paired prototype compensation for binary PET-CT segmentation."""

    BACKGROUND = 0
    TUMOR = 1
    NUM_CLASSES = 2

    def __init__(
        self,
        channels: Sequence[int],
        outlier_ratio: float = 0.05,
        eps: float = 1e-6,
        cache_on_cpu: bool = True,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels cannot be empty")
        if any(int(c) <= 0 for c in channels):
            raise ValueError(f"all channels must be positive, got {channels}")
        if not 0.0 <= float(outlier_ratio) < 1.0:
            raise ValueError("outlier_ratio must be in [0, 1)")

        self.channels = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.outlier_ratio = float(outlier_ratio)
        self.eps = float(eps)
        self.cache_on_cpu = bool(cache_on_cpu)

        for s, c in enumerate(self.channels):
            self.register_buffer(f"ct_prototypes_{s}", torch.zeros(2, c))
            self.register_buffer(f"pet_prototypes_{s}", torch.zeros(2, c))
            self.register_buffer(f"prototype_ready_{s}", torch.tensor(False))

        self.reset_epoch_cache()

    # ------------------------------------------------------------------
    # Buffer/cache helpers
    # ------------------------------------------------------------------
    def _ct_proto(self, s: int) -> Tensor:
        return getattr(self, f"ct_prototypes_{s}")

    def _pet_proto(self, s: int) -> Tensor:
        return getattr(self, f"pet_prototypes_{s}")

    def _ready(self, s: int) -> Tensor:
        return getattr(self, f"prototype_ready_{s}")

    def is_ready(self, scale_idx: Optional[int] = None) -> bool:
        if scale_idx is not None:
            return bool(self._ready(scale_idx).item())
        return all(bool(self._ready(s).item()) for s in range(self.num_scales))

    def reset_epoch_cache(self) -> None:
        self._cache: Dict[str, List[List[List[Tensor]]]] = {
            "ct": [[[] for _ in range(2)] for _ in range(self.num_scales)],
            "pet": [[[] for _ in range(2)] for _ in range(self.num_scales)],
        }

    @torch.no_grad()
    def reset_prototypes(self) -> None:
        for s in range(self.num_scales):
            self._ct_proto(s).zero_()
            self._pet_proto(s).zero_()
            self._ready(s).fill_(False)
        self.reset_epoch_cache()

    def _validate_features(
        self,
        ct_feats: TensorList,
        pet_feats: Optional[TensorList] = None,
    ) -> None:
        if len(ct_feats) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} CT scales, got {len(ct_feats)}")
        if pet_feats is not None and len(pet_feats) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} PET scales, got {len(pet_feats)}")

        for s, ct in enumerate(ct_feats):
            if ct.ndim != 4:
                raise ValueError(f"ct_feats[{s}] must be [B,C,H,W], got {tuple(ct.shape)}")
            if ct.shape[1] != self.channels[s]:
                raise ValueError(
                    f"ct_feats[{s}] expected C={self.channels[s]}, got C={ct.shape[1]}"
                )
            if pet_feats is not None:
                pet = pet_feats[s]
                if pet.ndim != 4:
                    raise ValueError(f"pet_feats[{s}] must be [B,C,H,W], got {tuple(pet.shape)}")
                if pet.shape != ct.shape:
                    raise ValueError(
                        f"CT/PET scale {s} shapes must match, got {tuple(ct.shape)} vs {tuple(pet.shape)}"
                    )

    @staticmethod
    def _prepare_mask(mask: Tensor) -> Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask must be [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
        return mask.float().clamp(0.0, 1.0)

    def _soft_mask(self, mask: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(mask, size=size, mode="area").clamp(0.0, 1.0)

    def _weighted_pool(self, feat: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor]:
        denominator = weight.sum(dim=(2, 3))  # [B,1]
        valid = denominator[:, 0] > self.eps
        pooled = (feat * weight).sum(dim=(2, 3)) / denominator.clamp_min(self.eps)
        return pooled, valid

    def _append(
        self,
        modality: str,
        scale_idx: int,
        class_idx: int,
        vectors: Tensor,
        valid: Tensor,
    ) -> None:
        selected = vectors[valid].detach().float()
        if selected.numel() == 0:
            return
        if self.cache_on_cpu:
            selected = selected.cpu()
        self._cache[modality][scale_idx][class_idx].extend(
            [row.contiguous() for row in selected]
        )

    # ------------------------------------------------------------------
    # Prototype collection and epoch-end update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        ct_feats: TensorList,
        pet_feats_real: TensorList,
        mask: Tensor,
    ) -> Dict[str, List[int]]:
        """
        Collect prototype candidates before the first fusion.

        Call this for both Full and simulated-Missing training batches.
        In Missing mode, pet_feats_real may only be used here after detach;
        it must not enter the prediction fusion path.
        """
        self._validate_features(ct_feats, pet_feats_real)
        mask = self._prepare_mask(mask)
        if mask.shape[0] != ct_feats[0].shape[0]:
            raise ValueError("mask batch size does not match feature batch size")

        summary = {
            "ct_background": [],
            "ct_tumor": [],
            "pet_background": [],
            "pet_tumor": [],
        }

        for s, (ct, pet) in enumerate(zip(ct_feats, pet_feats_real)):
            ct = ct.detach().float()
            pet = pet.detach().float()
            m = self._soft_mask(mask.to(device=ct.device, dtype=ct.dtype), ct.shape[-2:])
            bg = 1.0 - m

            ct_t, valid_ct_t = self._weighted_pool(ct, m)
            ct_b, valid_ct_b = self._weighted_pool(ct, bg)
            pet_t, valid_pet_t = self._weighted_pool(pet, m)
            pet_b, valid_pet_b = self._weighted_pool(pet, bg)

            self._append("ct", s, self.TUMOR, ct_t, valid_ct_t)
            self._append("ct", s, self.BACKGROUND, ct_b, valid_ct_b)
            self._append("pet", s, self.TUMOR, pet_t, valid_pet_t)
            self._append("pet", s, self.BACKGROUND, pet_b, valid_pet_b)

            summary["ct_tumor"].append(int(valid_ct_t.sum().item()))
            summary["ct_background"].append(int(valid_ct_b.sum().item()))
            summary["pet_tumor"].append(int(valid_pet_t.sum().item()))
            summary["pet_background"].append(int(valid_pet_b.sum().item()))

        return summary

    def _filtered_mean(self, candidates: List[Tensor]) -> Tuple[Optional[Tensor], Dict[str, float]]:
        if not candidates:
            return None, {
                "num_total": 0.0,
                "num_kept": 0.0,
                "num_removed": 0.0,
                "mean_distance": math.nan,
                "max_distance": math.nan,
            }

        x = torch.stack(candidates, dim=0).float()  # [N,C]
        center = x.mean(dim=0, keepdim=True)
        distance = torch.norm(x - center, p=2, dim=1)
        n = x.shape[0]
        keep_n = max(1, min(n, int(math.ceil((1.0 - self.outlier_ratio) * n))))
        keep_idx = torch.topk(distance, k=keep_n, largest=False, sorted=False).indices
        proto = x.index_select(0, keep_idx).mean(dim=0)
        stats = {
            "num_total": float(n),
            "num_kept": float(keep_n),
            "num_removed": float(n - keep_n),
            "mean_distance": float(distance.mean().item()),
            "max_distance": float(distance.max().item()),
        }
        return proto, stats

    @torch.no_grad()
    def finalize_epoch(self, clear_cache: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Build new prototypes from the current epoch cache:
        center -> remove farthest 5% -> mean of remaining 95%.

        The resulting prototypes should be used from the next epoch.
        """
        report: Dict[str, Dict[str, Dict[str, float]]] = {}

        for s in range(self.num_scales):
            scale_name = f"scale_{s + 1}"
            report[scale_name] = {}
            updated = []

            for modality in ("ct", "pet"):
                buffer = self._ct_proto(s) if modality == "ct" else self._pet_proto(s)
                for class_idx, class_name in ((self.BACKGROUND, "background"), (self.TUMOR, "tumor")):
                    proto, stats = self._filtered_mean(self._cache[modality][s][class_idx])
                    report[scale_name][f"{modality}_{class_name}"] = stats
                    if proto is not None:
                        buffer[class_idx].copy_(proto.to(buffer.device, buffer.dtype))
                        updated.append(True)
                    else:
                        updated.append(False)

            if all(updated):
                self._ready(s).fill_(True)

            report[scale_name]["ready"] = {"value": float(self.is_ready(s))}

        if clear_cache:
            self.reset_epoch_cache()
        return report

    # ------------------------------------------------------------------
    # Missing compensation
    # ------------------------------------------------------------------
    def _forward_one(self, ct_feat: Tensor, scale_idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, _, h, w = ct_feat.shape
        if not self.is_ready(scale_idx):
            zeros = torch.zeros_like(ct_feat)
            debug = {
                "similarity": torch.zeros(b, 2, h, w, device=ct_feat.device, dtype=ct_feat.dtype),
                "class_probability": torch.zeros(b, 2, h, w, device=ct_feat.device, dtype=ct_feat.dtype),
                "tumor_probability": torch.zeros(b, 1, h, w, device=ct_feat.device, dtype=ct_feat.dtype),
                "compensation_norm": torch.zeros(b, 1, h, w, device=ct_feat.device, dtype=ct_feat.dtype),
                "ready": torch.tensor(False, device=ct_feat.device),
            }
            return zeros, debug

        ct_proto = self._ct_proto(scale_idx).to(ct_feat.device, ct_feat.dtype)  # [2,C]
        pet_proto = self._pet_proto(scale_idx).to(ct_feat.device, ct_feat.dtype)  # [2,C]

        query = F.normalize(ct_feat, p=2, dim=1, eps=self.eps)
        key = F.normalize(ct_proto, p=2, dim=1, eps=self.eps)
        similarity = torch.einsum("bchw,kc->bkhw", query, key)  # [B,2,H,W]
        probability = F.softmax(similarity, dim=1)
        tumor_probability = probability[:, self.TUMOR:self.TUMOR + 1]

        pet_delta = (pet_proto[self.TUMOR] - pet_proto[self.BACKGROUND]).view(1, -1, 1, 1)
        compensation = tumor_probability * pet_delta
        compensation_norm = torch.linalg.vector_norm(
            compensation.float(), ord=2, dim=1, keepdim=True
        ).to(ct_feat.dtype)

        debug = {
            "similarity": similarity,
            "class_probability": probability,
            "tumor_probability": tumor_probability,
            "compensation_norm": compensation_norm,
            "pet_delta": pet_delta,
            "ready": torch.tensor(True, device=ct_feat.device),
        }
        return compensation, debug

    def forward(
        self,
        ct_feats: TensorList,
        return_debug: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[Dict[str, Tensor]]]]:
        self._validate_features(ct_feats)
        compensation: List[Tensor] = []
        debug_list: List[Dict[str, Tensor]] = []
        for s, ct in enumerate(ct_feats):
            comp, debug = self._forward_one(ct, s)
            compensation.append(comp)
            debug_list.append(debug)
        if return_debug:
            return compensation, debug_list
        return compensation

    # ------------------------------------------------------------------
    # Routing and diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def route_pet_features(
        pet_feats_real: TensorList,
        pet_feats_comp: TensorList,
        pet_missing: Union[bool, Tensor],
    ) -> List[Tensor]:
        """Route real PET for Full mode and compensated PET for Missing mode."""
        if len(pet_feats_real) != len(pet_feats_comp):
            raise ValueError("real and compensated PET must have the same number of scales")

        if isinstance(pet_missing, bool):
            return list(pet_feats_comp if pet_missing else pet_feats_real)

        if not torch.is_tensor(pet_missing):
            raise TypeError("pet_missing must be bool or Tensor")

        output: List[Tensor] = []
        for real, comp in zip(pet_feats_real, pet_feats_comp):
            if real.shape != comp.shape:
                raise ValueError("real PET and compensated PET shapes must match")
            missing = pet_missing.to(device=real.device, dtype=torch.bool)
            if missing.ndim == 1:
                missing = missing[:, None, None, None]
            elif missing.ndim == 2 and missing.shape[1] == 1:
                missing = missing[:, :, None, None]
            if missing.ndim != 4:
                raise ValueError("sample-level pet_missing must be [B] or [B,1]")
            output.append(torch.where(missing, comp, real))
        return output

    def cache_sizes(self) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        for s in range(self.num_scales):
            result[f"scale_{s + 1}"] = {
                "ct_background": len(self._cache["ct"][s][self.BACKGROUND]),
                "ct_tumor": len(self._cache["ct"][s][self.TUMOR]),
                "pet_background": len(self._cache["pet"][s][self.BACKGROUND]),
                "pet_tumor": len(self._cache["pet"][s][self.TUMOR]),
            }
        return result

    @torch.no_grad()
    def prototype_diagnostics(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for s in range(self.num_scales):
            ct = self._ct_proto(s).float()
            pet = self._pet_proto(s).float()
            result[f"scale_{s + 1}"] = {
                "ready": float(self.is_ready(s)),
                "ct_background_norm": float(ct[0].norm().item()),
                "ct_tumor_norm": float(ct[1].norm().item()),
                "pet_background_norm": float(pet[0].norm().item()),
                "pet_tumor_norm": float(pet[1].norm().item()),
                "ct_bg_tumor_cosine": float(F.cosine_similarity(ct[0:1], ct[1:2], dim=1, eps=self.eps).item()),
                "pet_bg_tumor_cosine": float(F.cosine_similarity(pet[0:1], pet[1:2], dim=1, eps=self.eps).item()),
            }
        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    @staticmethod
    def _select_sample(x: Optional[Tensor], sample_index: int) -> Optional[Tensor]:
        if x is None:
            return None
        if x.ndim >= 4:
            return x[sample_index]
        return x

    @staticmethod
    def _display_array(x: Tensor):
        import numpy as np

        x = x.detach().float().cpu()
        if x.ndim == 3:
            x = x[0] if x.shape[0] == 1 else x.mean(dim=0)
        if x.ndim != 2:
            raise ValueError(f"cannot visualize shape {tuple(x.shape)}")
        arr = x.numpy()
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros_like(arr)
        lo, hi = np.percentile(arr[finite], [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(arr[finite].min()), float(arr[finite].max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _resize_map(x: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            x = x[None]
        kwargs = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(x.float(), **kwargs)

    @torch.no_grad()
    def save_visualizations(
        self,
        debug_list: Sequence[Dict[str, Tensor]],
        output_dir: Union[str, Path],
        prefix: str,
        sample_index: int = 0,
        ct_image: Optional[Tensor] = None,
        pet_image: Optional[Tensor] = None,
        gt_mask: Optional[Tensor] = None,
        scales: Optional[Sequence[int]] = None,
        save_tensor_data: bool = True,
    ) -> List[Path]:
        """Save per-scale CT prototype probability and PET compensation heatmaps."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("visualization requires matplotlib") from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_scales = list(range(self.num_scales)) if scales is None else list(scales)

        ct_sample = self._select_sample(ct_image, sample_index)
        pet_sample = self._select_sample(pet_image, sample_index)
        mask_sample = self._select_sample(gt_mask, sample_index)

        if mask_sample is not None:
            target_size = tuple(mask_sample.shape[-2:])
        elif ct_sample is not None:
            target_size = tuple(ct_sample.shape[-2:])
        elif pet_sample is not None:
            target_size = tuple(pet_sample.shape[-2:])
        else:
            target_size = tuple(debug_list[selected_scales[0]]["tumor_probability"].shape[-2:])

        saved: List[Path] = []
        for s in selected_scales:
            debug = debug_list[s]
            tumor = debug["tumor_probability"][sample_index:sample_index + 1]
            bg = debug["class_probability"][sample_index:sample_index + 1, self.BACKGROUND:self.BACKGROUND + 1]
            norm = debug["compensation_norm"][sample_index:sample_index + 1]

            tumor_up = self._resize_map(tumor, target_size)[0, 0]
            bg_up = self._resize_map(bg, target_size)[0, 0]
            norm_up = self._resize_map(norm, target_size)[0, 0]

            panels = []
            if ct_sample is not None:
                panels.append(("CT", ct_sample, "gray"))
            if pet_sample is not None:
                panels.append(("Real PET", pet_sample, "gray"))
            if mask_sample is not None:
                panels.append(("GT Mask", mask_sample, "gray"))
            panels.extend([
                ("CT Tumor Prototype Probability", tumor_up, "magma"),
                ("CT Background Prototype Probability", bg_up, "viridis"),
                ("PET Compensation Norm", norm_up, "inferno"),
            ])

            cols = min(3, len(panels))
            rows = int(math.ceil(len(panels) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, (title, tensor, cmap) in zip(axes.flat, panels):
                im = ax.imshow(self._display_array(tensor), cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                if "Probability" in title or "Norm" in title:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f"{prefix} | Scale {s + 1} | ready={bool(debug['ready'].item())}")
            fig.tight_layout()
            png_path = output_dir / f"{prefix}_scale{s + 1}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(png_path)

            if save_tensor_data:
                pt_path = output_dir / f"{prefix}_scale{s + 1}.pt"
                torch.save({
                    "tumor_probability": tumor.cpu(),
                    "background_probability": bg.cpu(),
                    "compensation_norm": norm.cpu(),
                    "similarity": debug["similarity"][sample_index:sample_index + 1].cpu(),
                    "ready": bool(debug["ready"].item()),
                }, pt_path)
                saved.append(pt_path)

        return saved


def example_baseline_forward(
    module: SpatialSemanticPairedPrototypeCompensation,
    ct_feats: TensorList,
    pet_feats_real: TensorList,
    mask: Optional[Tensor],
    pet_missing: bool,
    training: bool,
    return_debug: bool = False,
):
    """Minimal example showing the correct embedding order inside a baseline."""
    if training:
        if mask is None:
            raise ValueError("mask is required for prototype collection during training")
        module.collect(ct_feats, pet_feats_real, mask)

    if return_debug:
        pet_comp, debug = module(ct_feats, return_debug=True)
    else:
        pet_comp = module(ct_feats)
        debug = None

    pet_aux = module.route_pet_features(pet_feats_real, pet_comp, pet_missing)
    fused_feats = [ct + pet for ct, pet in zip(ct_feats, pet_aux)]
    return (fused_feats, debug) if return_debug else fused_feats


if __name__ == "__main__":
    torch.manual_seed(2023)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [64, 128, 320, 512]
    sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]
    module = SpatialSemanticPairedPrototypeCompensation(channels).to(device)

    last_ct = last_pet = last_mask = None
    for _ in range(2):
        ct = [torch.randn(4, c, h, w, device=device) for c, (h, w) in zip(channels, sizes)]
        pet = [torch.randn(4, c, h, w, device=device) for c, (h, w) in zip(channels, sizes)]
        mask = torch.zeros(4, 1, 512, 512, device=device)
        mask[:, :, 160:330, 180:350] = 1.0
        module.collect(ct, pet, mask)
        last_ct, last_pet, last_mask = ct, pet, mask

    print("cache:", module.cache_sizes())
    print("finalize:", module.finalize_epoch())
    print("diagnostics:", module.prototype_diagnostics())

    assert last_ct is not None and last_pet is not None and last_mask is not None
    comp, debug = module(last_ct, return_debug=True)
    for i, (ct_i, comp_i) in enumerate(zip(last_ct, comp)):
        print(i, tuple(ct_i.shape), tuple(comp_i.shape), tuple(debug[i]["tumor_probability"].shape))
        assert ct_i.shape == comp_i.shape

    fused = example_baseline_forward(
        module, last_ct, last_pet, last_mask,
        pet_missing=True, training=False, return_debug=False,
    )
    assert all(x.shape == c.shape for x, c in zip(fused, last_ct))
    print("SSPPC self-test passed")
