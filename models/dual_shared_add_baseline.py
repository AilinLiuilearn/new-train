import math

import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.ct_conditioned_pet_affine import CTConditionedPETAffine
from models.petct_state_text_competitive import StateTextCompetitiveFusion
from models.paired_semantic_prototype_imputation import (
    PairedSemanticPrototypeImputation,
)
from utils.pet_feature_reconstruction import (
    balanced_multi_scale_smooth_l1_reconstruction,
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

    Missing boundary (new scheme): F_missing^l = C^l + pet_comp^l with
    pet_comp^l = gamma^l(C.detach()) * P_prior^l + beta^l(C.detach()),
    where P_prior is the existing Module-1 retrieval output. The fusion stays
    the original AddFusion (direct add), unchanged for the Full route:
    F_full^l = C^l + P_real^l.
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
        pspi_bank_update_mode='matched_ema',
        pspi_ema_momentum=0.95,
        pspi_retrieval_temperature=0.1,
        pspi_proto_temperature=0.02,
        pspi_collect_candidates=True,
        pspi_prior_scale_enabled=True,
        pspi_prior_scale_init=0.1,
        pspi_affine_enabled=False,
        pspi_reconstruction_weight=0.0,
        pspi_proto_contrastive_weight=0.01,
        module2_enabled=False,
        module2_use_state=True,
        module2_use_text=True,
        module2_diag_enabled=False,
        module2_diag_interval=50,
        module2_ct_text_feature=None,
        module2_pet_text_feature=None,
        module2_text_metadata=None,
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
        self.module2_enabled = bool(module2_enabled)
        self.module2_use_state = bool(module2_use_state)
        self.module2_use_text = bool(module2_use_text)
        self.module2_diag_enabled = bool(module2_diag_enabled)
        self.module2_diag_interval = int(module2_diag_interval)
        if self.module2_enabled:
            self._init_module2_fusion(
                pet_channels,
                module2_ct_text_feature,
                module2_pet_text_feature,
                module2_text_metadata,
            )
        else:
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

        # Legacy per-scale scalar prior scale (compatibility only; the new
        # experiment disables it and must not allocate trainable alpha).
        self.pspi_prior_scale_enabled = bool(pspi_prior_scale_enabled)
        self.pspi_prior_scale_init = float(pspi_prior_scale_init)
        if not 0.0 < self.pspi_prior_scale_init < 1.0:
            raise ValueError(
                f"pspi_prior_scale_init must be in (0,1), got {self.pspi_prior_scale_init!r}"
            )
        if self.pspi_enabled and self.pspi_prior_scale_enabled:
            initial_logit = math.log(
                self.pspi_prior_scale_init / (1.0 - self.pspi_prior_scale_init)
            )
            self.missing_prior_logits = nn.Parameter(
                torch.full((4,), initial_logit)
            )
        else:
            # No dangling trainable parameter when PSPI is off or scale is off.
            self.register_parameter("missing_prior_logits", None)

        # New scheme: CT-conditioned direct affine on Missing rows only.
        self.pspi_affine_enabled = bool(pspi_affine_enabled)
        if self.pspi_affine_enabled and self.pspi_prior_scale_enabled:
            raise ValueError(
                "pspi_affine_enabled=True is incompatible with "
                "pspi_prior_scale_enabled=True: the new scheme removes the "
                "learnable per-scale alpha; disable prior scale explicitly."
            )
        recon_w = float(pspi_reconstruction_weight)
        if not math.isfinite(recon_w) or recon_w < 0.0:
            raise ValueError(
                f"pspi_reconstruction_weight must be finite and >= 0, got {pspi_reconstruction_weight!r}"
            )
        if recon_w > 0.0 and not (self.pspi_enabled and self.pspi_affine_enabled):
            raise ValueError(
                "pspi_reconstruction_weight > 0 requires pspi_enabled=True "
                "and pspi_affine_enabled=True"
            )
        self.pspi_reconstruction_weight = recon_w
        proto_w = float(pspi_proto_contrastive_weight)
        if not math.isfinite(proto_w) or proto_w < 0.0:
            raise ValueError(
                f"pspi_proto_contrastive_weight must be finite and >= 0, got {pspi_proto_contrastive_weight!r}"
            )
        self.pspi_proto_contrastive_weight = proto_w
        if self.pspi_affine_enabled and self.pspi_enabled:
            self.pet_affine = CTConditionedPETAffine(pet_channels)
        else:
            self.pet_affine = None

    def _init_module2_fusion(self, pet_channels, ct_text, pet_text, text_metadata):
        """Construct StateTextCompetitiveFusion inside a CPU fork_rng guard.

        Creating it after all original modules are built AND inside fork_rng
        keeps the seed-determined initialization of decoder/module1/affine
        untouched, so the disabled-equivalence test can compare against the
        base commit.
        """
        import torch as _torch

        if ct_text is not None and pet_text is not None:
            ct_feature = _torch.as_tensor(ct_text).detach().float().cpu()
            pet_feature = _torch.as_tensor(pet_text).detach().float().cpu()
        elif self.module2_use_text:
            raise ValueError(
                'module2 text enabled but no text features supplied; '
                'builder must encode or read the pair cache first'
            )
        else:
            ct_feature = pet_feature = None
        metadata = dict(text_metadata) if text_metadata is not None else None
        with _torch.random.fork_rng(devices=[]):
            self.fusion = StateTextCompetitiveFusion(
                list(pet_channels),
                ct_text_feature=ct_feature,
                pet_text_feature=pet_feature,
                text_metadata=metadata,
                enabled=True,
                use_state=self.module2_use_state,
                use_text=self.module2_use_text,
                diag_enabled=self.module2_diag_enabled,
                diag_interval=self.module2_diag_interval,
            )
        if not (self.module2_use_state and self.module2_use_text):
            for name, param in self.fusion.named_parameters():
                if not param.requires_grad:
                    continue
                if not self.module2_use_state and '.missing_prompt' in name:
                    param.requires_grad_(False)
                elif not self.module2_use_text and ('.proj_' in name or '.gate_' in name
                                                    or '.missing_prompt' in name):
                    param.requires_grad_(False)

    def missing_prior_alpha_vals(self):
        """Return per-scale alpha_l = sigmoid(a_l) detached tensors (legacy).

        The new CT-affine scheme reports constant 1.0 here for old logs; the
        constant must not be described as a learned fusion weight.
        """
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

    def _fuse_modalities(self, ct_feats, pet_feats, pet_available):
        """Single downstream fusion boundary (AddFusion or Module-2).

        Disabled: original AddFusion call preserved exactly.
        Enabled: explicit [B] source state + validity. `pet_available` is 1
        for real PET, 0 for imputed PET; `pet_valid` additionally requires
        the Module-1 bank to be ready for Missing rows (cold-start rows are
        bypassed to strict CT inside the fusion). Never infer validity from
        all-zero PET.
        """
        if not self.module2_enabled:
            return self.fusion(ct_feats, pet_feats, None)
        ref = ct_feats[0]
        state = torch.as_tensor(pet_available, device=ref.device).reshape(-1)
        if state.numel() != ref.shape[0]:
            raise ValueError(
                f'pet_available must contain one state per sample: '
                f'got {state.numel()} for batch {ref.shape[0]}'
            )
        if not bool(torch.all((state == 0) | (state == 1)).item()):
            raise ValueError('pet_available values must be 0 or 1')
        bank_ready = bool(
            self.pspi_enabled and self.module1 is not None and self.module1.bank_ready
        )
        valid = state.bool() | bank_ready
        # AMP: ct_align ends with BatchNorm (autocast fp32 op) while the PET
        # backbone stays in the autocast dtype. AddFusion never cared (add
        # promotes); Module-2 enforces a strict shared-dtype contract, so the
        # integration boundary unifies to the CT dtype (differentiable).
        target_dtype = ref.dtype
        ct_feats = [
            c.to(dtype=target_dtype) if c.dtype != target_dtype else c
            for c in ct_feats
        ]
        pet_feats = [
            p.to(device=ref.device, dtype=target_dtype)
            if (p.device != ref.device or p.dtype != target_dtype) else p
            for p in pet_feats
        ]
        return self.fusion(
            ct_feats, pet_feats, state,
            pet_valid=valid.to(device=ref.device),
        )

    def _scaled_prior(self, pet_prior):
        """Apply legacy per-scale scalar alpha_l to the retrieved PET prior."""
        pet_for_fusion = []
        for scale_idx, prior_scale in enumerate(pet_prior):
            if self.missing_prior_logits is not None:
                alpha = torch.sigmoid(self.missing_prior_logits[scale_idx])
            else:
                alpha = prior_scale.new_tensor(1.0)
            pet_for_fusion.append(alpha * prior_scale)
        return pet_for_fusion

    def _affine_stats(self, gammas, betas, pet_comp, ref_tensor):
        """Detached per-scale affine/reconstruction monitoring stats."""
        stats = {}
        for s, (g, b, c) in enumerate(zip(gammas, betas, pet_comp)):
            with torch.no_grad():
                stats[f'affine_gamma_mean_s{s + 1}'] = float(g.detach().float().mean().item())
                stats[f'affine_gamma_std_s{s + 1}'] = float(g.detach().float().std().item())
                stats[f'affine_beta_rms_s{s + 1}'] = float(b.detach().float().pow(2).mean().sqrt().item())
                stats[f'compensated_pet_rms_s{s + 1}'] = float(c.detach().float().pow(2).mean().sqrt().item())
        return stats

    def _zero_reconstruction(self, ref_tensor):
        zero = ref_tensor.new_zeros((), dtype=torch.float32)
        return {
            'reconstruction_loss': zero,
            'reconstruction_active': False,
            'reconstruction_missing_samples': 0,
            'reconstruction_per_scale': {},
            'reconstruction_target_note': 'detached_same_sample_pet_real',
        }

    def _zero_affine_stats(self):
        stats = {}
        for s in range(1, 5):
            stats[f'affine_gamma_mean_s{s}'] = 0.0
            stats[f'affine_gamma_std_s{s}'] = 0.0
            stats[f'affine_beta_rms_s{s}'] = 0.0
            stats[f'compensated_pet_rms_s{s}'] = 0.0
        return stats

    def _compensate_missing_rows(
        self,
        ct_feats_missing,
        pet_real_missing,
        mask_missing,
        compute_reconstruction=True,
    ):
        """Central Missing compensation helper shared by train/infer paths.

        Returns dict with keys:
          pet_comp:      post-affine Missing PET actually fused (grad kept)
          pet_prior:     raw retrieved prior (for diagnostics only)
          gammas/betas:  CT-only affine parameters
          reconstruction: raw reconstruction dict from the loss helper
          module1_aux:   bank/entropy stats from retrieval
        Cold start (bank not ready): compensation is strictly zero (the whole
        affine is skipped before beta can leak), and reconstruction is zero.
        """
        if not self.pspi_enabled or self.module1 is None:
            raise RuntimeError('_compensate_missing_rows requires pspi_enabled=True')
        if not self.pspi_affine_enabled or self.pet_affine is None:
            raise RuntimeError('_compensate_missing_rows requires pspi_affine_enabled=True')
        pet_prior, aux = self.module1.retrieve_pet_prior(
            ct_feats_missing, return_attention=False
        )
        if not bool(aux.get('bank_ready', False)):
            # Cold start bypass: skip the entire affine so that even a
            # manually non-zero beta bias cannot produce output.
            zeros = [torch.zeros_like(p) for p in pet_prior]
            ref = ct_feats_missing[0]
            recon = self._zero_reconstruction(ref)
            return {
                'pet_comp': zeros,
                'pet_prior': pet_prior,
                'gammas': None,
                'betas': None,
                'reconstruction': recon,
                'module1_aux': aux,
                'affine_stats': self._zero_affine_stats(),
                'active': False,
            }
        pet_comp, gammas, betas = self.pet_affine(ct_feats_missing, pet_prior)
        if compute_reconstruction and self.training:
            recon = balanced_multi_scale_smooth_l1_reconstruction(
                pet_comp, pet_real_missing, mask_missing
            )
        else:
            ref = pet_comp[0]
            zero = ref.new_zeros((), dtype=torch.float32)
            recon = {
                'loss': zero,
                'active': False,
                'num_samples': 0,
                'per_scale': {},
            }
        stats = self._affine_stats(gammas, betas, pet_comp, pet_comp[0])
        return {
            'pet_comp': pet_comp,
            'pet_prior': pet_prior,
            'gammas': gammas,
            'betas': betas,
            'reconstruction': recon,
            'module1_aux': aux,
            'affine_stats': stats,
            'active': True,
        }

    def _attach_pspi_stats(self, out, proto_result=None, module1_aux=None, ref_tensor=None,
                           recon_result=None, affine_stats=None):
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
        if recon_result is not None:
            out['reconstruction_loss'] = recon_result.get(
                'loss', _zero(ref).to(dtype=torch.float32)
            )
            out['reconstruction_active'] = bool(recon_result.get('active', False))
            out['reconstruction_missing_samples'] = int(
                recon_result.get('num_samples', recon_result.get('num_missing', 0))
            )
            per_scale = recon_result.get('per_scale', {}) or {}
            for s in range(1, 5):
                out[f'reconstruction_fg_s{s}'] = float(per_scale.get(f's{s}_fg', 0.0))
                out[f'reconstruction_bg_s{s}'] = float(per_scale.get(f's{s}_bg', 0.0))
                out[f'reconstruction_rms_s{s}'] = float(per_scale.get(f's{s}_rms', 0.0))
        else:
            out['reconstruction_loss'] = _zero(ref).to(dtype=torch.float32)
            out['reconstruction_active'] = False
            out['reconstruction_missing_samples'] = 0
            for s in range(1, 5):
                out[f'reconstruction_fg_s{s}'] = 0.0
                out[f'reconstruction_bg_s{s}'] = 0.0
                out[f'reconstruction_rms_s{s}'] = 0.0
        if affine_stats is not None:
            for k, v in affine_stats.items():
                out[k] = float(v)
        else:
            for s in range(1, 5):
                out[f'affine_gamma_mean_s{s}'] = 0.0
                out[f'affine_gamma_std_s{s}'] = 0.0
                out[f'affine_beta_rms_s{s}'] = 0.0
                out[f'compensated_pet_rms_s{s}'] = 0.0
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
        if self.missing_prior_logits is not None:
            with torch.no_grad():
                alphas = [float(torch.sigmoid(v).item()) for v in self.missing_prior_logits]
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

    def _maybe_collect(self, ct_feats, pet_feats_real, mask, collect_module1_candidates=True):
        if not collect_module1_candidates:
            return None
        if not self.pspi_enabled or self.module1 is None:
            return None
        if not self.training:
            return None
        if not self.module1.config.collect_candidates_during_training:
            return None
        if mask is None:
            return None
        return self.module1.collect_candidates(ct_feats, pet_feats_real, mask)

    def _forward_full(self, ct, pet, target_size, mask=None, collect_module1_candidates=True):
        ct_feats = self._encode_ct(ct)
        pet_real_feats = self._encode_pet(pet)
        # Detached candidate collection never affects the Full prediction path.
        self._maybe_collect(
            ct_feats,
            pet_real_feats,
            mask,
            collect_module1_candidates=collect_module1_candidates,
        )

        proto_result = None
        module1_aux = None
        if self.pspi_enabled:
            if self.training and mask is not None and self.pspi_proto_contrastive_weight > 0.0:
                # PET prototype contrastive supervision (grad -> PET encoder).
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
            module1_aux = {
                "bank_ready": self.module1.bank_ready,
                "bank_version": int(self.module1.bank_version.item()),
                "attention_entropy": [0.0, 0.0, 0.0, 0.0],
                "normalized_attention_entropy": [0.0, 0.0, 0.0, 0.0],
            }

        # Full prediction = raw baseline: CT + real PET.
        # Full path NEVER calls retrieve_pet_prior, the CT affine, or
        # missing_prior_logits; reconstruction is strictly 0.
        fused_feats = self._fuse_modalities(
            ct_feats, pet_real_feats,
            torch.ones(ct.shape[0], device=ct.device, dtype=torch.long),
        )
        out = self._decode(fused_feats, target_size)
        return self._attach_pspi_stats(out, proto_result=proto_result, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _forward_missing(self, ct, pet, target_size, mask=None, collect_module1_candidates=True):
        ct_feats = self._encode_ct(ct)
        if self.pspi_enabled:
            if not self.training:
                # Missing inference: CT + bank + affine path only; no PET.
                if self.pspi_affine_enabled and self.pet_affine is not None:
                    if self.module1.bank_ready:
                        pet_prior, module1_aux = self.module1.retrieve_pet_prior(
                            ct_feats, return_attention=False
                        )
                        pet_comp, gammas, betas = self.pet_affine(ct_feats, pet_prior)
                        affine_stats = self._affine_stats(gammas, betas, pet_comp, pet_comp[0])
                        module1_aux = dict(module1_aux)
                        module1_aux['pet_prior'] = pet_prior
                        fused_feats = self._fuse_modalities(
                            ct_feats, pet_comp,
                            torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
                        )
                    else:
                        module1_aux = {
                            "bank_ready": False,
                            "bank_version": int(self.module1.bank_version.item()),
                            "attention_entropy": [0.0, 0.0, 0.0, 0.0],
                            "normalized_attention_entropy": [0.0, 0.0, 0.0, 0.0],
                        }
                        zeros = [torch.zeros_like(f) for f in ct_feats]
                        fused_feats = self._fuse_modalities(
                            ct_feats, zeros,
                            torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
                        )
                        affine_stats = self._zero_affine_stats()
                    out = self._decode(fused_feats, target_size)
                    return self._attach_pspi_stats(
                        out, proto_result=None, module1_aux=module1_aux,
                        ref_tensor=out['logits'], affine_stats=affine_stats,
                    )
                # Legacy path: scaled prior with availability preserved.
                pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
                module1_aux = dict(module1_aux)
                module1_aux['pet_prior'] = pet_prior
                fused_feats = self._fuse_modalities(
                    ct_feats, self._scaled_prior(pet_prior),
                    torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
                )
                out = self._decode(fused_feats, target_size)
                return self._attach_pspi_stats(out, proto_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])
            # Missing training: real PET is privileged supervision only, never enters logits.
            if pet is None:
                raise ValueError('Missing training requires real PET for privileged supervision')
            if mask is None:
                raise ValueError('Missing training requires mask')
            pet_real_feats = self._encode_pet(pet)
            self._maybe_collect(
                ct_feats,
                pet_real_feats,
                mask,
                collect_module1_candidates=collect_module1_candidates,
            )
            proto_result = None
            if self.pspi_proto_contrastive_weight > 0.0:
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_real_feats, mask)
            if self.pspi_affine_enabled and self.pet_affine is not None:
                comp = self._compensate_missing_rows(
                    ct_feats, pet_real_feats, mask.float(),
                    compute_reconstruction=True,
                )
                module1_aux = dict(comp['module1_aux'])
                module1_aux['pet_prior'] = comp['pet_prior']
                fused_feats = self._fuse_modalities(
                    ct_feats, comp['pet_comp'],
                    torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
                )
                out = self._decode(fused_feats, target_size)
                return self._attach_pspi_stats(
                    out, proto_result=proto_result, module1_aux=module1_aux,
                    ref_tensor=out['logits'], recon_result=comp['reconstruction'],
                    affine_stats=comp['affine_stats'],
                )
            pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
            module1_aux = dict(module1_aux)
            module1_aux['pet_prior'] = pet_prior
            fused_feats = self._fuse_modalities(
                ct_feats, self._scaled_prior(pet_prior),
                torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
            )
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=proto_result, module1_aux=module1_aux, ref_tensor=out['logits'])
        else:
            module1_aux = None
            if self.training:
                pet_feats_real = self._encode_pet(pet)
                pet_for_fusion = [torch.zeros_like(feat) for feat in pet_feats_real]
            else:
                pet_for_fusion = [torch.zeros_like(feat) for feat in ct_feats]
            fused_feats = self._fuse_modalities(
                ct_feats, pet_for_fusion,
                torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long),
            )
            out = self._decode(fused_feats, target_size)
            return self._attach_pspi_stats(out, proto_result=None, module1_aux=module1_aux, ref_tensor=out['logits'])

    def _expand_missing_to_full_batch(
        self, pet_comp_missing, missing_index, full_batch_size, ref_feats,
    ):
        """Scatter Missing-row compensation into a full-batch tensor.

        The affine only sees Missing rows (B_m); Full rows receive exact-zero
        placeholders so that torch.where assembly keeps the Full PET input
        untouched. Zero placeholders carry no grad_fn; the affine graph flows
        only through the Missing rows.
        """
        missing_idx = missing_index.to(dtype=torch.bool)
        device = ref_feats[0].device
        expanded = []
        for s, comp_m in enumerate(pet_comp_missing):
            like = ref_feats[s]
            full = like.new_zeros(like.shape)
            rows = torch.nonzero(missing_idx.to(device=device), as_tuple=False).flatten().long()
            if int(comp_m.shape[0]) != int(rows.numel()):
                raise ValueError(
                    f"[Mixed][S{s + 1}] compensation rows {int(comp_m.shape[0])} != "
                    f"missing count {int(rows.numel())}"
                )
            full = full.index_copy(0, rows, comp_m.to(dtype=like.dtype))
            expanded.append(full)
        return expanded

    def _assemble_mixed_pet_fusion(
        self, pet_real_feats, pet_comp_missing, missing_index, ref_tensor,
    ):
        """Out-of-place per-row assembly: Full rows keep P_real, Missing rows take pet_comp."""
        missing_idx = missing_index.to(dtype=torch.bool)
        device = pet_real_feats[0].device
        missing_rows = missing_idx.to(device=device).view(-1, 1, 1, 1)
        pet_for_fusion = []
        for s, (real_feat, comp_feat) in enumerate(zip(pet_real_feats, pet_comp_missing)):
            # comp_feat is already scattered to the full batch (Full rows = 0).
            if tuple(comp_feat.shape) != tuple(real_feat.shape):
                raise ValueError(
                    f"[Mixed][S{s + 1}] compensation shape {tuple(comp_feat.shape)} != "
                    f"real shape {tuple(real_feat.shape)}"
                )
            # Out-of-place selection: Full rows keep P_real, Missing rows use
            # pet_comp. torch.where preserves autograd through comp_feat.
            assembled = torch.where(missing_rows, comp_feat.to(real_feat.dtype), real_feat)
            if tuple(assembled.shape) != tuple(real_feat.shape):
                raise ValueError(
                    f"[Mixed][S{s + 1}] assembled shape mismatch: {tuple(assembled.shape)} "
                    f"vs {tuple(real_feat.shape)}"
                )
            pet_for_fusion.append(assembled)
        return pet_for_fusion

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None, collect_module1_candidates=True):
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size, mask=mask, collect_module1_candidates=collect_module1_candidates)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, pet, target_size, mask=mask, collect_module1_candidates=collect_module1_candidates)
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._maybe_collect(ct_feats, pet_feats_real, mask, collect_module1_candidates=collect_module1_candidates)
        if self.pspi_enabled and self.pspi_affine_enabled and self.pet_affine is not None:
            missing_index = pet_available.eq(0)
            num_missing = int(missing_index.sum().item())
            if num_missing == 0:
                raise RuntimeError('mixed affine path requires a non-empty Missing subset')
            ct_missing = [f[missing_index] for f in ct_feats]
            pet_real_missing = [f[missing_index] for f in pet_feats_real]
            mask_missing = mask[missing_index].float() if mask is not None else None
            if mask_missing is None and self.training:
                raise ValueError('mixed affine training requires mask for the Missing subset')
            if self.module1.bank_ready:
                pet_prior_m, _ = self.module1.retrieve_pet_prior(ct_missing, return_attention=False)
                pet_comp_m, gammas_m, betas_m = self.pet_affine(ct_missing, pet_prior_m)
                recon = None
                if self.training:
                    recon = balanced_multi_scale_smooth_l1_reconstruction(
                        pet_comp_m, pet_real_missing, mask_missing
                    )
                    recon_dict = recon
                else:
                    ref = pet_comp_m[0]
                    recon_dict = {
                        'loss': ref.new_zeros((), dtype=torch.float32),
                        'active': False,
                        'num_samples': 0,
                        'per_scale': {},
                    }
                affine_stats = self._affine_stats(gammas_m, betas_m, pet_comp_m, pet_comp_m[0])
                active = True
            else:
                # Cold start: Missing compensation strictly zero; skip affine.
                pet_comp_m = [torch.zeros_like(f[missing_index]) for f in pet_feats_real]
                ref = pet_comp_m[0]
                recon_dict = {
                    'loss': ref.new_zeros((), dtype=torch.float32),
                    'active': False,
                    'num_samples': 0,
                    'per_scale': {},
                }
                affine_stats = self._zero_affine_stats()
                active = False
            # Scatter the B_m-row compensation to full-batch tensors so the
            # assembly keeps Full rows on P_real and Missing rows on pet_comp.
            pet_comp_full = self._expand_missing_to_full_batch(
                pet_comp_m, missing_index, ct.shape[0], pet_feats_real
            )
            pet_for_fusion = self._assemble_mixed_pet_fusion(
                pet_feats_real, pet_comp_full, missing_index, pet_feats_real[0],
            )
            proto_result = None
            if self.training and mask is not None and self.pspi_proto_contrastive_weight > 0.0:
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
            module1_aux = {
                "bank_ready": self.module1.bank_ready,
                "bank_version": int(self.module1.bank_version.item()),
                "attention_entropy": [0.0, 0.0, 0.0, 0.0],
                "normalized_attention_entropy": [0.0, 0.0, 0.0, 0.0],
            }
            fused_feats = self._fuse_modalities(
                ct_feats, pet_for_fusion, pet_available,
            )
            out = self._decode(fused_feats, target_size)
            out = self._attach_pspi_stats(
                out, proto_result=proto_result, module1_aux=module1_aux,
                ref_tensor=out['logits'], recon_result=recon_dict,
                affine_stats=affine_stats,
            )
            out['reconstruction_missing_active'] = bool(active)
            return out
        if self.pspi_enabled:
            pet_prior, module1_aux = self.module1.retrieve_pet_prior(ct_feats, return_attention=False)
            module1_aux = dict(module1_aux)
            module1_aux['pet_prior'] = pet_prior
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            # Full samples use real PET; Missing samples use scaled PET prior (no leakage).
            prior_for_fusion = self._scaled_prior(pet_prior)
            pet_for_fusion = [
                real_feat * availability + prior_feat * (1.0 - availability)
                for real_feat, prior_feat in zip(pet_feats_real, prior_for_fusion)
            ]
            proto_result = None
            if self.training and mask is not None and self.pspi_proto_contrastive_weight > 0.0:
                proto_result = self.module1.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
        else:
            module1_aux = None
            proto_result = None
            availability = pet_available.view(-1, 1, 1, 1).to(dtype=pet_feats_real[0].dtype)
            pet_for_fusion = [feat * availability for feat in pet_feats_real]
        fused_feats = self._fuse_modalities(
            ct_feats, pet_for_fusion, pet_available,
        )
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
        collect_module1_candidates=True,
    ):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask, collect_module1_candidates=collect_module1_candidates)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, mask=mask, collect_module1_candidates=collect_module1_candidates)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size, mask=mask, collect_module1_candidates=collect_module1_candidates)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
