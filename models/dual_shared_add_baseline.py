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
    """AddFusion baseline with a missing-only Module-1 (API-style)."""

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
        pspi_retrieval_temperature=0.1,
        pspi_proto_contrastive_weight=0.01,
        pspi_proto_temperature=0.02,
        pspi_reconstruction_weight=0.1,
        pspi_spatial_affine=True,
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
                retrieval_temperature=pspi_retrieval_temperature,
                proto_contrastive_weight=pspi_proto_contrastive_weight,
                proto_temperature=pspi_proto_temperature,
                reconstruction_weight=pspi_reconstruction_weight,
                spatial_affine=pspi_spatial_affine,
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

    def _attach_pspi_stats(self, out, proto_result=None, recon_result=None, module1_aux=None, ref_tensor=None):
        # proto/recon results may be None or dict with loss/weighted_loss/num_terms
        def _zero(ref):
            return ref.new_zeros(()) if ref is not None else torch.tensor(0.0)
        ref = ref_tensor if ref_tensor is not None else out.get('logits')
        if ref is None:
            raise ValueError('ref_tensor or logits required for zero losses')
        if proto_result is not None:
            out['prototype_contrastive_loss'] = proto_result['loss']
            out['prototype_contrastive_loss_weighted'] = proto_result['weighted_loss']
            out['prototype_contrastive_num_terms'] = proto_result['num_terms']
        else:
            z = _zero(ref)
            out['prototype_contrastive_loss'] = z
            out['prototype_contrastive_loss_weighted'] = z
            out['prototype_contrastive_num_terms'] = 0
        if recon_result is not None:
            out['reconstruction_loss'] = recon_result['loss']
            out['reconstruction_loss_weighted'] = recon_result['weighted_loss']
            out['reconstruction_num_terms'] = recon_result['num_terms']
        else:
            z = _zero(ref)
            out['reconstruction_loss'] = z
            out['reconstruction_loss_weighted'] = z
            out['reconstruction_num_terms'] = 0
        if module1_aux is not None:
            out['module1_bank_ready'] = bool(module1_aux.get('bank_ready', False))
            out['module1_bank_version'] = int(module1_aux.get('bank_version', 0))
            # attention entropy
            ent = module1_aux.get('attention_entropy', [0.0]*4)
            nent = module1_aux.get('normalized_attention_entropy', [0.0]*4)
            for i in range(4):
                out[f'attention_entropy_s{i+1}'] = float(ent[i]) if i < len(ent) else 0.0
                out[f'normalized_attention_entropy_s{i+1}'] = float(nent[i]) if i < len(nent) else 0.0
            # aggregated entropy
            out['attention_entropy'] = float(sum(ent)/len(ent)) if ent else 0.0
            out['normalized_attention_entropy'] = float(sum(nent)/len(nent)) if nent else 0.0
            # gamma/beta stats
            for k in ('gamma_mean','gamma_std','gamma_abs_mean','beta_mean','beta_std','beta_abs_mean','pet_proto_norm','pet_comp_norm'):
                out[k] = float(module1_aux.get(k, 0.0))
        else:
            out['module1_bank_ready'] = False
            out['module1_bank_version'] = 0
            for i in range(4):
                out[f'attention_entropy_s{i+1}'] = 0.0
                out[f'normalized_attention_entropy_s{i+1}'] = 0.0
            out['attention_entropy'] = 0.0
            out['normalized_attention_entropy'] = 0.0
            for k in ('gamma_mean','gamma_std','gamma_abs_mean','beta_mean','beta_std','beta_abs_mean','pet_proto_norm','pet_comp_norm'):
                out[k] = 0.0
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
        pet_real_feats = self._encode_pet(pet)
        # Detached candidate collection never affects the Full prediction path.
        self._maybe_collect(ct_feats, pet_real_feats, mask)

        proto_result = None
        recon_result = None
        module1_aux = None
        if self.pspi_enabled:
            if self.training and mask is not None:
                # PET prototype contrastive supervision (grad -> PET encoder).
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
                if self.module1.bank_ready:
                    # Compensation is generated ONLY for reconstruction loss;
                    # it can never reach Full logits.
                    pet_comp, module1_aux = self.module1.recover_missing(ct_feats, return_attention=False)
                    recon_result = self.module1.compute_balanced_reconstruction_loss(pet_comp, pet_real_feats, mask)
            if module1_aux is None:
                module1_aux = {
                    "bank_ready": self.module1.bank_ready,
                    "bank_version": int(self.module1.bank_version.item()),
                    "attention_entropy": [0.0, 0.0, 0.0, 0.0],
                    "normalized_attention_entropy": [0.0, 0.0, 0.0, 0.0],
                }

        # Full prediction = raw baseline: CT + real PET via AddFusion.
        fused_feats = self.fusion(ct_feats, pet_real_feats, None)
        out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, proto_result=proto_result, recon_result=recon_result, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _forward_missing(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        if self.pspi_enabled:
            if not self.training:
                pet_comp, module1_aux = self.module1.recover_missing(ct_feats, return_attention=False)
                fused_feats = self.fusion(ct_feats, pet_comp, None)
                out = self._decode(fused_feats, target_size)
                return self._attach_pspi_stats(out, proto_result=None, recon_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])
            # training
            if pet is None:
                raise ValueError('Missing training requires real PET for privileged supervision')
            if mask is None:
                raise ValueError('Missing training requires mask')
            pet_real_feats = self._encode_pet(pet)
            self._maybe_collect(ct_feats, pet_real_feats, mask)
            proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
            pet_comp, module1_aux = self.module1.recover_missing(ct_feats, return_attention=False)
            recon_result = self.module1.compute_balanced_reconstruction_loss(pet_comp, pet_real_feats, mask)
            fused_feats = self.fusion(ct_feats, pet_comp, None)
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=proto_result, recon_result=recon_result, module1_aux=module1_aux, ref_tensor=out['logits'])
        else:
            module1_aux = None
            if self.training:
                pet_feats_real = self._encode_pet(pet)
                pet_for_fusion = [torch.zeros_like(feat) for feat in pet_feats_real]
            else:
                pet_for_fusion = [torch.zeros_like(feat) for feat in ct_feats]
            fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=None, recon_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])

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
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._maybe_collect(ct_feats, pet_feats_real, mask)
        if self.pspi_enabled:
            pet_comp, module1_aux = self.module1.recover_missing(ct_feats, return_attention=False)
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = []
            for real_feat, comp_feat in zip(pet_feats_real, pet_comp):
                pet_for_fusion.append(real_feat * availability + comp_feat * (1.0 - availability))
            # For mixed batch, proto/recon not well-defined; return zeros but keep bank info
            proto_result = None
            recon_result = None
            if self.training and mask is not None:
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
                recon_result = self.module1.compute_balanced_reconstruction_loss(pet_comp, pet_real_feats, mask)
        else:
            module1_aux = None
            proto_result = None
            recon_result = None
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = [feat * availability for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
        out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, proto_result=proto_result, recon_result=recon_result, module1_aux=module1_aux, ref_tensor=out['logits'])

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
