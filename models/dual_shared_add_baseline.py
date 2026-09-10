import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.paired_semantic_prototype_imputation import (
    PairedSemanticPrototypeImputation,
)


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    """AddFusion baseline with a missing-only Module-1.

    Full route is the raw baseline (CT + real PET via AddFusion). Module-1 is
    bypassed entirely for Full prediction; real PET on Full batches is only
    used for detached prototype candidate collection.

    Missing route: CT -> Module-1 (retrieval + personalization) -> P_comp,
    then the same AddFusion boundary: F = CT + P_comp.
    """

    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        pspi_enabled=True,
        pspi_num_clusters=6,
        pspi_build_stage=4,
        pspi_cluster_max_iter=25,
        pspi_outlier_discard_rate=0.05,
        pspi_bank_update_mode='direct',
        pspi_ema_momentum=0.999,
        pspi_semantic_loss_weight=0.01,
        pspi_collect_candidates=True,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(
            pet_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

        self.pspi_enabled = bool(pspi_enabled)
        if self.pspi_enabled:
            self.module1 = PairedSemanticPrototypeImputation(
                channels=pet_channels,
                num_clusters=pspi_num_clusters,
                build_stage=pspi_build_stage,
                cluster_max_iter=pspi_cluster_max_iter,
                outlier_discard_rate=pspi_outlier_discard_rate,
                bank_update_mode=pspi_bank_update_mode,
                ema_momentum=pspi_ema_momentum,
                semantic_loss_weight=pspi_semantic_loss_weight,
                collect_candidates_during_training=pspi_collect_candidates,
            )
        else:
            self.module1 = None

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('PET encoder requires PET input; Missing inference must not call it')
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _attach_pspi_stats(self, out, semantic=None, module1_aux=None, ref_tensor=None):
        if semantic is not None:
            out['semantic_loss'] = semantic['loss']
            out['semantic_loss_weighted'] = semantic['weighted_loss']
            out['semantic_loss_num_terms'] = semantic['num_terms']
        elif ref_tensor is not None:
            zero = ref_tensor.new_zeros(())
            out['semantic_loss'] = zero
            out['semantic_loss_weighted'] = zero
            out['semantic_loss_num_terms'] = 0
        else:
            raise ValueError('semantic or ref_tensor is required')
        if module1_aux is not None:
            out['module1_bank_ready'] = bool(module1_aux.get('bank_ready', False))
            out['module1_bank_version'] = int(module1_aux.get('bank_version', 0))
        else:
            out['module1_bank_ready'] = False
            out['module1_bank_version'] = 0
        # Legacy names kept as aliases so old logging code does not crash.
        out['prototype_loss'] = out['semantic_loss']
        out['prototype_loss_weighted'] = out['semantic_loss_weighted']
        out['prototype_loss_num_terms'] = out['semantic_loss_num_terms']
        return out

    def _maybe_collect(self, ct_feats, pet_feats_real, mask):
        if not self.pspi_enabled or self.module1 is None:
            return None
        if not self.training:
            return None
        if not self.module1.config.collect_candidates_during_training:
            return None
        if mask is None:
            return None
        return self.module1.collect_candidates(ct_feats, pet_feats_real, mask)

    def _forward_full(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)

        # Full candidate collection is detached/no-grad and never affects the
        # Full prediction path below.
        self._maybe_collect(ct_feats, pet_feats_real, mask)

        # Full prediction = raw baseline: CT + real PET via AddFusion.
        # Module-1 is fully bypassed.
        fused_feats = self.fusion(ct_feats, pet_feats_real, None)
        out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, ref_tensor=out['logits'])

    def _forward_missing(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)

        if self.pspi_enabled:
            # Strict Missing EVAL / inference: never encode or use PET.
            if not self.training:
                pet_comp, module1_aux = self.module1.recover_missing(ct_feats)
                fused_feats = self.fusion(ct_feats, pet_comp, None)
                out = self._decode(fused_feats, target_size)
                return self._attach_pspi_stats(
                    out, module1_aux=module1_aux, ref_tensor=out['logits']
                )

            # Missing TRAIN: real PET is privileged teacher / candidate source only.
            if pet is None:
                raise ValueError('Missing training requires real PET as privileged teacher')
            if mask is None:
                raise ValueError('Missing training requires mask for PSPI candidate/teacher path')
            pet_feats_real = self._encode_pet(pet)
            self._maybe_collect(ct_feats, pet_feats_real, mask)

            # Prediction path uses NO real PET.
            pet_comp, module1_aux = self.module1.recover_missing(ct_feats)

            # Semantic supervision (build stage only; teacher detached inside).
            semantic = self.module1.compute_semantic_relation_loss(
                pet_comp, pet_feats_real, mask
            )

            fused_feats = self.fusion(ct_feats, pet_comp, None)
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(
                out, semantic=semantic, module1_aux=module1_aux,
                ref_tensor=out['logits'],
            )
        else:
            module1_aux = None
            if self.training:
                pet_feats_real = self._encode_pet(pet)
                pet_for_fusion = [torch.zeros_like(feat) for feat in pet_feats_real]
            else:
                pet_for_fusion = [torch.zeros_like(feat) for feat in ct_feats]
            fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None):
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')

        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size, mask=mask)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, pet, target_size, mask=mask)

        # Mixed batch: keep API compatibility. Per-sample PET evidence chooses
        # real PET (available) or compensated PET (missing); fusion is always
        # the same AddFusion boundary.
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._maybe_collect(ct_feats, pet_feats_real, mask)

        if self.pspi_enabled:
            pet_comp, module1_aux = self.module1.recover_missing(ct_feats)
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = []
            for real_feat, comp_feat in zip(pet_feats_real, pet_comp):
                pet_for_fusion.append(real_feat * availability + comp_feat * (1.0 - availability))
        else:
            module1_aux = None
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = [feat * availability for feat in pet_feats_real]

        fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
        out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, module1_aux=module1_aux, ref_tensor=out['logits'])

    @torch.no_grad()
    def collect_module1_bootstrap_batch(self, ct, pet, mask):
        """Extract features with the current encoders and feed Module-1's cache.

        Used only for the pre-epoch-1 prototype bank bootstrap. No segmentation
        forward, no optimizer, no backward; encoder parameters are untouched.
        """
        if not self.pspi_enabled:
            raise RuntimeError("bootstrap requires pspi_enabled=True")
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        return self.module1.collect_candidates(ct_feats, pet_feats, mask)

    @torch.no_grad()
    def finalize_module1_epoch(self, epoch):
        if not self.pspi_enabled:
            return None
        return self.module1.finalize_epoch(epoch=epoch)

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode='auto',
        mask=None,
    ):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, mask=mask)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size, mask=mask)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
