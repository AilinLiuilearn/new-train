import math

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
    """AddFusion baseline with clean Module-1 (paired PET prior retrieval).

    Missing boundary: F_missing^l = C^l + alpha_l * P_prior^l with a
    per-scale scalar alpha_l = sigmoid(a_l), a_l init at logit(0.1).
    The scalar lives on the model boundary (NOT inside Module-1) and only
    controls the overall strength of the retrieved prior into AddFusion.
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
        pspi_retrieval_temperature=0.1,
        pspi_proto_temperature=0.02,
        pspi_collect_candidates=True,
        pspi_prior_scale_enabled=True,
        pspi_prior_scale_init=0.1,
        module2_enabled=False,
        module2_kwargs=None,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.module2_enabled = bool(module2_enabled)
        self.requested_prior_scale_enabled = bool(pspi_prior_scale_enabled)
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
                proto_temperature=pspi_proto_temperature,
                collect_candidates_during_training=pspi_collect_candidates,
            )
        else:
            self.module1 = None

        # Prior scale contract: when Module-2 is enabled, the old per-scale
        # scalar (missing_prior_logits) is replaced by Module-2 fusion.
        # effective_prior_scale_enabled=False, logits registered as None,
        # and _scaled_prior is never called on the Module-2 path.
        if self.module2_enabled and self.pspi_enabled:
            self.effective_prior_scale_enabled = False
        else:
            self.effective_prior_scale_enabled = bool(pspi_prior_scale_enabled)
        self.pspi_prior_scale_enabled = self.effective_prior_scale_enabled
        self.pspi_prior_scale_init = float(pspi_prior_scale_init)
        if not 0.0 < self.pspi_prior_scale_init < 1.0:
            raise ValueError(
                f"pspi_prior_scale_init must be in (0,1), got {self.pspi_prior_scale_init!r}"
            )
        if self.pspi_enabled and self.effective_prior_scale_enabled:
            initial_logit = math.log(
                self.pspi_prior_scale_init / (1.0 - self.pspi_prior_scale_init)
            )
            self.missing_prior_logits = nn.Parameter(
                torch.full((4,), initial_logit)
            )
        else:
            # No dangling trainable parameter when PSPI is off, scale is off,
            # or Module-2 replaces the old scalar fusion.
            self.register_parameter("missing_prior_logits", None)

        # Module-2 is constructed AFTER all shared components so common
        # initializations (encoders/align/decoder/Module-1) are unchanged.
        self.module2 = None
        self.module2_config = None
        if self.module2_enabled:
            if not self.pspi_enabled:
                raise ValueError(
                    "module2_enabled=True requires pspi_enabled=True"
                )
            from models.state_guided_expert_fusion import StateGuidedExpertFusion
            module2_kwargs = dict(module2_kwargs or {})
            module2_kwargs.setdefault("channels", tuple(int(c) for c in pet_channels))
            self.module2_config = dict(module2_kwargs)
            self.module2 = StateGuidedExpertFusion(**module2_kwargs)

    def missing_prior_alpha_vals(self):
        """Return per-scale alpha_l = sigmoid(a_l) detached tensors."""
        if self.missing_prior_logits is None:
            return [torch.ones(()) for _ in range(4)]
        with torch.no_grad():
            return [
                torch.sigmoid(self.missing_prior_logits[i]).detach().float()
                for i in range(4)
            ]

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

    def _scaled_prior(self, pet_prior):
        """Apply per-scale scalar alpha_l to the retrieved PET prior."""
        pet_for_fusion = []
        for scale_idx, prior_scale in enumerate(pet_prior):
            if self.missing_prior_logits is not None:
                alpha = torch.sigmoid(self.missing_prior_logits[scale_idx])
            else:
                alpha = prior_scale.new_tensor(1.0)
            pet_for_fusion.append(alpha * prior_scale)
        return pet_for_fusion

    def _attach_pspi_stats(self, out, proto_result=None, module1_aux=None, ref_tensor=None):
        def _zero(ref):
            return ref.new_zeros(()) if ref is not None else torch.tensor(0.0)
        ref = ref_tensor if ref_tensor is not None else out.get('logits')
        if ref is None:
            raise ValueError('ref_tensor or logits required for zero losses')
        if proto_result is not None:
            out['prototype_contrastive_loss'] = proto_result['loss']
            out['prototype_contrastive_num_terms'] = proto_result['num_terms']
        else:
            z = _zero(ref)
            out['prototype_contrastive_loss'] = z
            out['prototype_contrastive_num_terms'] = 0
        if module1_aux is not None:
            out['module1_bank_ready'] = bool(module1_aux.get('bank_ready', False))
            out['module1_bank_version'] = int(module1_aux.get('bank_version', 0))
            ent = module1_aux.get('attention_entropy', [0.0]*4)
            nent = module1_aux.get('normalized_attention_entropy', [0.0]*4)
            for i in range(4):
                out[f'attention_entropy_s{i+1}'] = float(ent[i]) if i < len(ent) else 0.0
                out[f'normalized_attention_entropy_s{i+1}'] = float(nent[i]) if i < len(nent) else 0.0
            out['attention_entropy'] = float(sum(ent)/len(ent)) if ent else 0.0
            out['normalized_attention_entropy'] = float(sum(nent)/len(nent)) if nent else 0.0
        else:
            out['module1_bank_ready'] = False
            out['module1_bank_version'] = 0
            for i in range(4):
                out[f'attention_entropy_s{i+1}'] = 0.0
                out[f'normalized_attention_entropy_s{i+1}'] = 0.0
            out['attention_entropy'] = 0.0
            out['normalized_attention_entropy'] = 0.0
        # Prior contribution scale + retrieved prior norm.
        # In Module-2 mode the old scalar is disabled; report it as such and
        # never present route weights as prior alpha.
        self.module1_is_module1_personalization_none = True
        if self.missing_prior_logits is not None:
            with torch.no_grad():
                alphas = [float(torch.sigmoid(v).item()) for v in self.missing_prior_logits]
        elif self.module2_enabled:
            alphas = [float("nan"), float("nan"), float("nan"), float("nan")]
        else:
            alphas = [1.0, 1.0, 1.0, 1.0]
        for i, a in enumerate(alphas):
            out[f'missing_prior_alpha_s{i+1}'] = float(a)
        prior_feats = (module1_aux or {}).get('pet_prior')
        if prior_feats is not None:
            with torch.no_grad():
                out['pet_prior_norm'] = float(
                    sum(f.detach().float().pow(2).mean().sqrt().item() for f in prior_feats) / len(prior_feats)
                )
        else:
            out['pet_prior_norm'] = 0.0
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
        module1_aux = None
        if self.pspi_enabled:
            if self.training and mask is not None:
                # PET prototype contrastive supervision (grad -> PET encoder).
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
            module1_aux = {
                "bank_ready": self.module1.bank_ready,
                "bank_version": int(self.module1.bank_version.item()),
                "attention_entropy": [0.0, 0.0, 0.0, 0.0],
                "normalized_attention_entropy": [0.0, 0.0, 0.0, 0.0],
            }

        # Full prediction: CT + real PET. Full NEVER calls retrieve_pet_prior
        # or missing_prior_logits. With Module-2 the raw AddFusion call is
        # replaced by the state-guided expert fusion (mode='full'); the
        # zero-initialized expert residual keeps the first forward identical
        # to the baseline. _decode rebuilds out['aux'], so module2_aux is
        # attached AFTER _decode.
        if self.module2 is not None:
            fused_feats, module2_aux = self.module2(
                ct_feats,
                pet_real_feats,
                mode='full',
                bank_ready=self.module1.bank_ready,
            )
            out = self._decode(fused_feats, target_size)
            out['aux']['module2'] = module2_aux
        else:
            fused_feats = self.fusion(ct_feats, pet_real_feats, None)
            out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, proto_result=proto_result, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _forward_missing(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        if self.pspi_enabled:
            if not self.training:
                # Missing inference: CT encoder/align + retrieval + fusion + decoder only.
                # Real PET is never encoded; bank/candidates are never updated.
                pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
                module1_aux = dict(module1_aux)
                module1_aux['pet_prior'] = pet_prior
                if self.module2 is not None:
                    # pet_prior is RAW (no _scaled_prior); its retrieval
                    # projection keeps the segmentation gradient. CT keeps its
                    # main-path gradient; only Module-2's internal
                    # personalization detaches CT.
                    fused_feats, module2_aux = self.module2(
                        ct_feats,
                        pet_prior,
                        mode='missing',
                        bank_ready=self.module1.bank_ready,
                    )
                    out = self._decode(fused_feats, target_size)
                    out['aux']['module2'] = module2_aux
                else:
                    fused_feats = self.fusion(ct_feats, self._scaled_prior(pet_prior), None)
                    out = self._decode(fused_feats, target_size)
                return self._attach_pspi_stats(out, proto_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])
            # Missing training: real PET is privileged supervision only, never enters logits.
            if pet is None:
                raise ValueError('Missing training requires real PET for privileged supervision')
            if mask is None:
                raise ValueError('Missing training requires mask')
            pet_real_feats = self._encode_pet(pet)
            self._maybe_collect(ct_feats, pet_real_feats, mask)
            proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
            pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
            module1_aux = dict(module1_aux)
            module1_aux['pet_prior'] = pet_prior
            if self.module2 is not None:
                fused_feats, module2_aux = self.module2(
                    ct_feats,
                    pet_prior,
                    mode='missing',
                    bank_ready=self.module1.bank_ready,
                )
                out = self._decode(fused_feats, target_size)
                out['aux']['module2'] = module2_aux
            else:
                fused_feats = self.fusion(ct_feats, self._scaled_prior(pet_prior), None)
                out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=proto_result, module1_aux=module1_aux, ref_tensor=out['logits'])
        else:
            module1_aux = None
            if self.training:
                pet_feats_real = self._encode_pet(pet)
                pet_for_fusion = [torch.zeros_like(feat) for feat in pet_feats_real]
            else:
                pet_for_fusion = [torch.zeros_like(feat) for feat in ct_feats]
            fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None):
        raw_available = pet_available
        if torch.is_tensor(raw_available) and raw_available.dtype in (torch.float16, torch.float32, torch.float64):
            if not bool(torch.all((raw_available == 0) | (raw_available == 1)).item()):
                raise ValueError('pet_available values must be 0 or 1')
        for v in (raw_available.detach().flatten().tolist() if torch.is_tensor(raw_available) else list(raw_available)):
            if isinstance(v, float):
                if v not in (0.0, 1.0):
                    raise ValueError('pet_available values must be 0 or 1')
            elif int(v) not in (0, 1):
                raise ValueError('pet_available values must be 0 or 1')
        pet_available = torch.as_tensor(raw_available, device=ct.device).long().view(-1)
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
            pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
            module1_aux = dict(module1_aux)
            module1_aux['pet_prior'] = pet_prior
            proto_result = None
            if self.training and mask is not None:
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
            if self.module2 is not None:
                availability = pet_available.bool().view(-1, 1, 1, 1)
                # Full rows use real PET; Missing rows use RAW retrieved prior.
                pet_for_module2 = [
                    torch.where(availability, real_feat, prior_feat)
                    for real_feat, prior_feat in zip(pet_feats_real, pet_prior)
                ]
                fused_feats, module2_aux = self.module2(
                    ct_feats,
                    pet_for_module2,
                    mode='auto',
                    pet_available=pet_available,
                    bank_ready=self.module1.bank_ready,
                )
                out = self._decode(fused_feats, target_size)
                out['aux']['module2'] = module2_aux
            else:
                availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
                # Full samples use real PET; Missing samples use scaled PET prior (no leakage).
                prior_for_fusion = self._scaled_prior(pet_prior)
                pet_for_fusion = [
                    real_feat * availability + prior_feat * (1.0 - availability)
                    for real_feat, prior_feat in zip(pet_feats_real, prior_for_fusion)
                ]
                fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
                out = self._decode(fused_feats, target_size)
        else:
            module1_aux = None
            proto_result = None
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = [feat * availability for feat in pet_feats_real]
            fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
            out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, proto_result=proto_result, module1_aux=module1_aux, ref_tensor=out['logits'])

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
