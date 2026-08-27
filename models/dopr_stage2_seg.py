"""Joint end-to-end Recovery + Dual-Origin PET Rectification segmentation.

Architecture:

    CT/PET Encoders
        -> Dynamic CT-PET Prototype Memory (CPPI)
        -> Full: Real PET / Missing: Proxy PET
        -> Prototype-Referenced PET Affine Calibration
        -> Dual-Origin PET Rectification (DOPR)
        -> Shared CT-Anchored Complementary Fusion
        -> Original Trainable UNetStyleDecoder
        -> Segmentation

``stage1`` is DualSharedAddPETCTBaseline used only as a Module-1 container
(encoders / CPPI / calibration / decoder). Its legacy
``StateAwareWeightedAddFusion`` is bypassed and disabled (requires_grad=False)
because DOPR fully replaces it.

No MoE / Prompt / Router / auxiliary DOPR losses in this joint experiment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.dual_origin_pet_rectification_fusion import DualOriginPETRectificationFusion


class DOPRStage2Seg(nn.Module):
    """Joint PET recovery + DOPR fusion segmentation model."""

    def __init__(
        self,
        stage1_model: nn.Module,
        channels: Sequence[int] = (64, 128, 320, 512),
        latent_cap: int = 128,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
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

        # Disable unused legacy weighted-add fusion only. Never freeze Module-1.
        if hasattr(self.stage1, "fusion"):
            for p in self.stage1.fusion.parameters():
                p.requires_grad = False
            print("[DOPR] legacy weighted-add fusion disabled=True", flush=True)

        self.dopr_fusion = DualOriginPETRectificationFusion(
            channels=self.channels,
            latent_cap=int(latent_cap),
            num_heads=int(num_heads),
            ffn_expansion=float(ffn_expansion),
            layer_scale_init=float(layer_scale_init),
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
        """Affine-calibrate real PET. Keep PET encoder grads; detach only CT ref."""
        if self.cppi_ready:
            _, ct_reference_feats, _ = self.stage1._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
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

        if self.training and mask is not None:
            if pet is None:
                raise ValueError(
                    "Missing training with mask requires real PET for CPPI collect"
                )
            with torch.no_grad():
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
        """Per-sample Full/Missing. Collect-only PET encode stays under no_grad."""
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError("pet_available must contain one state per sample")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0/1")

        ct_feats_all = list(self.stage1._encode_ct(ct))
        pet_cal_all = [torch.empty_like(x) for x in ct_feats_all]

        full_idx = torch.nonzero(availability == 1, as_tuple=False).flatten()
        miss_idx = torch.nonzero(availability == 0, as_tuple=False).flatten()

        # Collect-only PET forward must not share the prediction graph.
        if self.training and mask is not None and pet is not None:
            with torch.no_grad():
                pet_real_for_memory = list(self.stage1._encode_pet(pet))
            self.stage1._collect_cppi(ct_feats_all, pet_real_for_memory, mask)

        if full_idx.numel() > 0:
            if pet is None:
                raise ValueError("auto path has Full samples but pet=None")
            ct_full = [x.index_select(0, full_idx) for x in ct_feats_all]
            pet_input_full = pet.index_select(0, full_idx)
            # Prediction PET encode keeps gradient for Full subset.
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
        return_features: bool = False,
        return_rectified_pet: bool = False,
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

        fusion = self.dopr_fusion(
            ct_feats=ct_feats,
            pet_feats_cal=pet_cal,
            route=fusion_route,
            pet_available=state,
            return_rectified_pet=return_rectified_pet,
        )

        dec_out = self.stage1.decoder(fusion.features, target_size)
        logits = dec_out["logits"]

        out: Dict[str, object] = dict(dec_out)
        out["pred"] = logits
        out["aux"] = {
            "dopr_stats": fusion.stats,
            "stage2_aux_loss": logits.new_zeros((), dtype=torch.float32),
        }

        if return_features:
            out["dopr_features"] = fusion.features
            if return_rectified_pet:
                out["dopr_rectified_pet"] = fusion.rectified_pet_features
            out["module1_ct_evidence"] = ct_feats
            out["module1_pet_cal_evidence"] = pet_cal
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
