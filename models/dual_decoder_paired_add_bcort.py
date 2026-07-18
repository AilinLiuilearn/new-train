from __future__ import annotations

from typing import Dict

import torch

from models.baseline_petct_unet import _sanitize
from models.bcort_module import BCORT
from models.dual_decoder_paired_add_baseline import (
    DualDecoderPairedAddPETCTBaseline,
)


class DualDecoderPairedAddBCORT(DualDecoderPairedAddPETCTBaseline):
    def __init__(
        self,
        ct_backbone: str = "convnextv2_nano",
        pet_backbone: str = "mit_b1",
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision: bool = False,
    ) -> None:
        super().__init__(
            ct_backbone=ct_backbone,
            pet_backbone=pet_backbone,
            ct_pretrained_path=ct_pretrained_path,
            pet_pretrained_path=pet_pretrained_path,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_channels=decoder_channels,
            use_deep_supervision=use_deep_supervision,
        )
        self.bcort = BCORT(list(self.enc_pet.feature_info.channels()))

    @staticmethod
    def _prefix_diagnostics(diagnostics: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {f"{prefix}_{k}": v for k, v in diagnostics.items()}

    def _forward_full_only(self, ct: torch.Tensor, pet: torch.Tensor, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused_feats = [_sanitize(ct_feat + pet_feat) for ct_feat, pet_feat in zip(ct_feats, pet_feats)]
        refined_feats, diagnostics = self.bcort(fused_feats, return_diagnostics=True)
        refined_feats = [_sanitize(x) for x in refined_feats]
        outputs = self._decode(self.full_decoder, refined_feats, target_size)
        outputs["diagnostics"] = self._prefix_diagnostics(diagnostics, "full")
        return outputs

    def _forward_missing_only(self, ct: torch.Tensor, target_size):
        ct_feats = self._encode_ct(ct)
        refined_feats, diagnostics = self.bcort(ct_feats, return_diagnostics=True)
        refined_feats = [_sanitize(x) for x in refined_feats]
        outputs = self._decode(self.missing_decoder, refined_feats, target_size)
        outputs["diagnostics"] = self._prefix_diagnostics(diagnostics, "missing")
        return outputs

    def forward(self, ct: torch.Tensor, pet: torch.Tensor | None, pet_available=None, target_size=None, forward_mode: str = "auto"):
        if target_size is None:
            target_size = ct.shape[-2:]
        forward_mode = str(forward_mode)
        if forward_mode == "full":
            if self.training:
                if pet is None:
                    raise ValueError('forward_mode="full" requires PET during training.')
                full_outputs = self._forward_full_only(ct, pet, target_size)
                missing_outputs = self._forward_missing_only(ct, target_size)
                diagnostics = {}
                diagnostics.update(full_outputs.get("diagnostics", {}))
                diagnostics.update(missing_outputs.get("diagnostics", {}))
                return {
                    "logits": missing_outputs["logits"],
                    "paired_joint": True,
                    "paired_full_logits": full_outputs["logits"],
                    "paired_missing_logits": missing_outputs["logits"],
                    "diagnostics": diagnostics,
                }
            if pet is None:
                raise ValueError('forward_mode="full" requires PET input.')
            return self._forward_full_only(ct, pet, target_size)
        if forward_mode == "missing":
            return self._forward_missing_only(ct, target_size)
        raise ValueError('forward_mode must be "full" or "missing" for paired BCORT.')
