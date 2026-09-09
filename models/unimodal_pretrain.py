# -*- coding: utf-8 -*-
"""Stage-1 unimodal task-pretraining models.

Shared model definitions for the SimMLM-style two-stage recipe:

    CTOnlySegmentationModel: CT -> ConvNeXtV2-Nano -> StageChannelAlign
                             -> UNetStyleDecoder
    PETOnlySegmentationModel: PET -> MiT-B1 -> UNetStyleDecoder

Both reuse the exact architectures of the joint baseline. The two temporary
decoders share the same architecture but have independent parameters; Stage-1
decoder weights are NOT transferred to Stage-2. run_ct_only_seg.py imports the
CT model from here so there is only one CT-only definition in the repo.
"""

import torch
import torch.nn as nn

from models.baseline_blocks import UNetStyleDecoder, _check_tensor, _check_tensor_list
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_shared_add_baseline import StageChannelAlign

# The joint baseline defines the decoder input channels via MiT-B1's four
# stages. Both unimodal models must produce exactly these channels.
DECODER_INPUT_CHANNELS = (64, 128, 320, 512)


def _check_decoder_input_channels(channels, name):
    channels = [int(c) for c in channels]
    if channels != list(DECODER_INPUT_CHANNELS):
        raise ValueError(
            f"{name} stage channels {channels} do not match the decoder input "
            f"channels {list(DECODER_INPUT_CHANNELS)}; a projection would "
            "change the architecture and is intentionally not added."
        )


class CTOnlySegmentationModel(nn.Module):
    """ConvNeXtV2-Nano CT encoder plus the baseline's shared decoder."""

    def __init__(
        self,
        ct_backbone="convnextv2_nano",
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)

        # CT is the only encoder instantiated in this model.
        self.enc_ct = create_feature_backbone(
            ct_backbone,
            in_channels=in_channels,
        )
        load_local_weights_safe(
            self.enc_ct,
            ct_pretrained_path,
            name="CT_Encoder",
        )

        ct_channels = list(self.enc_ct.feature_info.channels())
        decoder_input_channels = list(DECODER_INPUT_CHANNELS)

        # Same StageChannelAlign used by the joint baseline; its outputs go
        # directly to the decoder skips.
        self.ct_align = StageChannelAlign(
            ct_channels,
            decoder_input_channels,
        )
        self.decoder = UNetStyleDecoder(
            decoder_input_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list("ct_feats", ct_feats)
        aligned_ct = self.ct_align(ct_feats)
        _check_tensor_list("aligned_ct", aligned_ct)
        return aligned_ct

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode="missing",
    ):
        # Retained for MDTSegTeacher compatibility; never affect this forward.
        del pet, pet_available, forward_mode

        if target_size is None:
            target_size = ct.shape[-2:]

        out = self.decoder(self._encode_ct(ct), target_size)
        _check_tensor("logits", out["logits"])
        out["pred"] = out["logits"]
        out["aux"] = {}
        return out


class PETOnlySegmentationModel(nn.Module):
    """MiT-B1 PET encoder feeding the same UNetStyleDecoder architecture.

    MiT-B1's four stages already output (64, 128, 320, 512), so no channel
    alignment is added in this first version. If a different backbone changes
    the stage channels, construction fails loudly instead of silently adding a
    new projection.
    """

    def __init__(
        self,
        pet_backbone="mit_b1",
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)

        # PET is the only encoder instantiated in this model.
        self.enc_pet = create_feature_backbone(
            pet_backbone,
            in_channels=in_channels,
        )
        load_local_weights_safe(
            self.enc_pet,
            pet_pretrained_path,
            name="PET_Encoder",
        )

        pet_channels = list(self.enc_pet.feature_info.channels())
        _check_decoder_input_channels(pet_channels, "PET encoder")
        decoder_input_channels = list(DECODER_INPUT_CHANNELS)

        self.decoder = UNetStyleDecoder(
            decoder_input_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_pet(self, pet):
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list("pet_feats", pet_feats)
        return pet_feats

    def forward(
        self,
        ct=None,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode="full",
    ):
        # Retained for MDTSegTeacher compatibility; never affect this forward.
        del ct, pet_available, forward_mode
        if pet is None:
            raise ValueError("PET-only pretraining requires PET input")

        if target_size is None:
            target_size = pet.shape[-2:]

        out = self.decoder(self._encode_pet(pet), target_size)
        _check_tensor("logits", out["logits"])
        out["pred"] = out["logits"]
        out["aux"] = {}
        return out
