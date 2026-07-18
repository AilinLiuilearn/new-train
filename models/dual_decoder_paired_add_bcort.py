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

    def _forward_pair(self, ct: torch.Tensor, pet: torch.Tensor, target_size):
        if pet is None:
            raise ValueError("Paired BCORT training requires PET input.")

        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)

        fused_feats = [_sanitize(c + p) for c, p in zip(ct_feats, pet_feats)]

        full_feats, full_diag = self.bcort(
            fused_feats,
            return_diagnostics=True,
        )
        missing_feats, missing_diag = self.bcort(
            ct_feats,
            return_diagnostics=True,
        )

        full_feats = [_sanitize(x) for x in full_feats]
        missing_feats = [_sanitize(x) for x in missing_feats]

        full_outputs = self._decode(
            self.full_decoder,
            full_feats,
            target_size,
        )

        missing_outputs = self._decode(
            self.missing_decoder,
            missing_feats,
            target_size,
        )

        diagnostics = {}
        diagnostics.update(self._prefix_diagnostics(full_diag, "full"))
        diagnostics.update(self._prefix_diagnostics(missing_diag, "missing"))
        return full_outputs, missing_outputs, diagnostics

    def forward(self, ct: torch.Tensor, pet: torch.Tensor | None, pet_available=None, target_size=None, forward_mode: str = "auto"):
        if target_size is None:
            target_size = ct.shape[-2:]
        forward_mode = str(forward_mode)
        if forward_mode == "full":
            if self.training:
                full_outputs, missing_outputs, diagnostics = self._forward_pair(ct, pet, target_size)
                return {
                    "logits": missing_outputs["logits"],
                    "paired_joint": True,
                    "paired_full_logits": full_outputs["logits"],
                    "paired_missing_logits": missing_outputs["logits"],
                    "diagnostics": diagnostics,
                }
            if pet is None:
                raise ValueError('forward_mode="full" requires PET input.')
            ct_feats = self._encode_ct(ct)
            pet_feats = self._encode_pet(pet)
            fused_feats = [_sanitize(c + p) for c, p in zip(ct_feats, pet_feats)]
            refined_feats, diagnostics = self.bcort(fused_feats, return_diagnostics=True)
            refined_feats = [_sanitize(x) for x in refined_feats]
            outputs = self._decode(self.full_decoder, refined_feats, target_size)
            outputs["diagnostics"] = self._prefix_diagnostics(diagnostics, "full")
            return outputs
        if forward_mode == "missing":
            ct_feats = self._encode_ct(ct)
            refined_feats, diagnostics = self.bcort(ct_feats, return_diagnostics=True)
            refined_feats = [_sanitize(x) for x in refined_feats]
            outputs = self._decode(self.missing_decoder, refined_feats, target_size)
            outputs["diagnostics"] = self._prefix_diagnostics(diagnostics, "missing")
            return outputs
        raise ValueError('forward_mode must be "full" or "missing" for paired BCORT.')
