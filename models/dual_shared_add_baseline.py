import torch
import torch.nn as nn

from models.baseline_blocks import PrototypeReferencedPETAffineCalibration, StateAwareWeightedAddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.ct_pet_prototype_imputation import CrossScaleCTPETPrototypeMemory
from models.taskmoe_s4_refiner import TaskMoEStage4Refiner
from models.cross_scale_shared_taskmoe import CrossScaleSharedTaskMoE


def _parse_taskmoe_scales(taskmoe_scales):
    """Parse TaskMoE scale selection for ablation.

    Accepted examples:
      s4 | s3s4 | s2s3s4 | s1s2s3s4 | all
      s1,s2,s3,s4 | s2+s3+s4
    """
    text = str(taskmoe_scales or 's4').strip().lower()
    if text in ('all', 'full', 's1s2s3s4'):
        return ('s1', 's2', 's3', 's4')

    # Normalize separators then extract s1..s4 tokens in order.
    normalized = (
        text.replace('+', ',')
        .replace('_', ',')
        .replace('-', ',')
        .replace(' ', ',')
    )
    # Also allow compact forms like s3s4 / s2s3s4 without commas.
    compact = normalized.replace(',', '')
    presets = {
        's4': ('s4',),
        '4': ('s4',),
        's3s4': ('s3', 's4'),
        's34': ('s3', 's4'),
        '34': ('s3', 's4'),
        's2s3s4': ('s2', 's3', 's4'),
        's234': ('s2', 's3', 's4'),
        '234': ('s2', 's3', 's4'),
        's1s2s3s4': ('s1', 's2', 's3', 's4'),
        's1234': ('s1', 's2', 's3', 's4'),
        '1234': ('s1', 's2', 's3', 's4'),
    }
    if compact in presets:
        return presets[compact]

    found = []
    for token in normalized.split(','):
        token = token.strip()
        if not token:
            continue
        if token in ('s1', '1'):
            found.append('s1')
        elif token in ('s2', '2'):
            found.append('s2')
        elif token in ('s3', '3'):
            found.append('s3')
        elif token in ('s4', '4'):
            found.append('s4')
        else:
            raise ValueError(
                f'Unsupported taskmoe scale token={token!r} in {taskmoe_scales!r}'
            )
    # De-duplicate while preserving order s1->s4
    order = ('s1', 's2', 's3', 's4')
    selected = tuple(s for s in order if s in set(found))
    if not selected:
        raise ValueError(
            f'Unsupported taskmoe_scales={taskmoe_scales!r}; '
            'use s4 | s3s4 | s2s3s4 | s1s2s3s4/all | s1,s2,s3,s4'
        )
    return selected


def _is_taskmoe_param_name(name):
    return (
        name.startswith('taskmoe_s1.')
        or name.startswith('taskmoe_s2.')
        or name.startswith('taskmoe_s3.')
        or name.startswith('taskmoe_s4.')
        or name.startswith('cross_scale_taskmoe.')
    )


_TASKMOE_SCALE_INDEX = {'s1': 0, 's2': 1, 's3': 2, 's4': 3}


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
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, cppi_num_clusters=6, cppi_build_stage=3, cppi_output_dir=None, taskmoe_scales='s4', taskmoe_mode='independent'):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.stage2_moe_only = False
        self.taskmoe_enabled = True
        self.taskmoe_mode = str(taskmoe_mode or 'independent').strip().lower()
        if self.taskmoe_mode not in ('independent', 'cross_scale_shared'):
            raise ValueError(
                f'Unsupported taskmoe_mode={taskmoe_mode!r}; '
                'use independent or cross_scale_shared'
            )
        self.taskmoe_scales = _parse_taskmoe_scales(taskmoe_scales)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.pet_calibration = PrototypeReferencedPETAffineCalibration(channels=pet_channels)
        self.fusion = StateAwareWeightedAddFusion(num_scales=len(pet_channels))
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.prototype_memory = CrossScaleCTPETPrototypeMemory(
            channels=pet_channels,
            num_clusters=cppi_num_clusters,
            build_stage=cppi_build_stage,
            output_dir=cppi_output_dir,
        )
        self.taskmoe_s1 = None
        self.taskmoe_s2 = None
        self.taskmoe_s3 = None
        self.taskmoe_s4 = None
        self.cross_scale_taskmoe = None
        if len(pet_channels) < 4:
            raise ValueError(f'TaskMoE expects 4 encoder scales, got {len(pet_channels)}')

        if self.taskmoe_mode == 'cross_scale_shared':
            if self.taskmoe_scales != ('s1', 's2', 's3', 's4'):
                raise ValueError(
                    'cross_scale_shared TaskMoE is an all-scale module; use --taskmoe_scales all'
                )
            self.cross_scale_taskmoe = CrossScaleSharedTaskMoE(
                channels=pet_channels,
                expert_dim=128,
                num_experts=6,
                top_k=2,
                atom_num=32,
                atom_dim=256,
                prompt_hidden_channels=64,
                mlp_ratio=2.0,
                dropout=0.0,
                noisy_gating=True,
                noise_epsilon=1e-2,
                balance_loss_weight=0.1,
                zero_start=True,
            )
        else:
            for scale_name in self.taskmoe_scales:
                idx = _TASKMOE_SCALE_INDEX[scale_name]
                module = self._build_taskmoe(pet_channels[idx])
                setattr(self, f'taskmoe_{scale_name}', module)

    @staticmethod
    def _build_taskmoe(channels):
        return TaskMoEStage4Refiner(
            channels=channels,
            num_experts=6,
            top_k=2,
            prompt_atoms=32,
            prompt_dim=256,
            prompt_hidden_channels=64,
            mlp_ratio=2.0,
            dropout=0.0,
            noisy_gating=True,
            noise_epsilon=1e-2,
            balance_loss_weight=0.1,
            residual_mode='zero_start',
            residual_scale_init=0.0,
        )

    def _iter_active_taskmoe(self):
        """Yield (scale_name, feature_index, module) for independent TaskMoE scales."""
        if self.taskmoe_mode != 'independent':
            return
        for scale_name in self.taskmoe_scales:
            module = getattr(self, f'taskmoe_{scale_name}', None)
            if module is None:
                continue
            yield scale_name, _TASKMOE_SCALE_INDEX[scale_name], module

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('API-style baseline requires PET input before fusion-time masking')
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _refine_stage4(self, fused_feats):
        """Refine TaskMoE scales. Name kept for compatibility."""
        fused_feats = list(fused_feats)
        if not self.taskmoe_enabled:
            zero = fused_feats[-1].new_zeros((), dtype=torch.float32)
            return fused_feats, zero

        if self.taskmoe_mode == 'cross_scale_shared':
            if self.cross_scale_taskmoe is None:
                raise RuntimeError('cross_scale_shared mode requires cross_scale_taskmoe module')
            orig_dtypes = [feat.dtype for feat in fused_feats]
            with torch.cuda.amp.autocast(enabled=False):
                shared_input = [feat.float() for feat in fused_feats]
                result = self.cross_scale_taskmoe(shared_input)
            fused_feats = [
                out.to(dtype=dtype)
                for out, dtype in zip(result.features, orig_dtypes)
            ]
            return fused_feats, result.balance_loss.float()

        # independent: Sparse index_add_ is not AMP-safe. Run TaskMoE in fp32.
        aux_loss = None
        with torch.cuda.amp.autocast(enabled=False):
            for _, feat_idx, module in self._iter_active_taskmoe():
                feat = fused_feats[feat_idx]
                out, loss = module(feat.float())
                fused_feats[feat_idx] = out.to(dtype=feat.dtype)
                loss = loss.float()
                aux_loss = loss if aux_loss is None else (aux_loss + loss)
        if aux_loss is None:
            aux_loss = fused_feats[-1].new_zeros((), dtype=torch.float32)
        return fused_feats, aux_loss

    def _decode(self, fused_feats, target_size, aux=None):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {} if aux is None else aux
        return out

    def enable_stage2_moe_only(self):
        for p in self.parameters():
            p.requires_grad = False
        if self.taskmoe_mode == 'cross_scale_shared':
            if self.cross_scale_taskmoe is None:
                raise RuntimeError('cross_scale_shared mode requires cross_scale_taskmoe module')
            for p in self.cross_scale_taskmoe.parameters():
                p.requires_grad = True
        else:
            for _, _, module in self._iter_active_taskmoe():
                for p in module.parameters():
                    p.requires_grad = True
        self.stage2_moe_only = True

    def train(self, mode=True):
        super().train(mode)
        if self.stage2_moe_only:
            self.enc_ct.eval()
            self.enc_pet.eval()
            self.ct_align.eval()
            self.pet_calibration.eval()
            self.fusion.eval()
            self.decoder.eval()
            self.prototype_memory.eval()
            for scale_name in ('s1', 's2', 's3', 's4'):
                module = getattr(self, f'taskmoe_{scale_name}', None)
                if module is not None:
                    module.eval()
            if self.cross_scale_taskmoe is not None:
                self.cross_scale_taskmoe.eval()
            if self.taskmoe_mode == 'cross_scale_shared':
                self.cross_scale_taskmoe.train(mode)
            else:
                for _, _, module in self._iter_active_taskmoe():
                    module.train(mode)
        return self

    def _collect_cppi(self, ct_feats, pet_feats_real, mask):
        if self.stage2_moe_only:
            return None
        if self.training and mask is not None:
            return self.prototype_memory.collect(
                ct_feats=ct_feats,
                pet_feats=pet_feats_real,
                mask=mask,
                print_info=False,
                compute_report=False,
            )
        return None

    def _retrieve_cppi(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        return self.prototype_memory.retrieve(
            ct_feats,
            compute_report=compute_report,
            save_diagnostics=save_diagnostics,
            print_info=print_info,
            return_ct_reference=return_ct_reference,
        )

    def _forward_full(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._collect_cppi(ct_feats, pet_feats_real, mask)
        if self.prototype_memory.bank_ready:
            _, ct_reference_feats, _ = self._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_real,
                ct_reference_feats,
                reference_valid=True,
            )
        else:
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_real,
                None,
                reference_valid=False,
            )
        fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='full')
        fused_feats, moe_aux_loss = self._refine_stage4(fused_feats)
        return self._decode(
            fused_feats,
            target_size,
            aux={'taskmoe_balance_loss': moe_aux_loss},
        )

    def _forward_missing(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        if self.stage2_moe_only:
            pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_proxy,
                ct_reference_feats,
                reference_valid=self.prototype_memory.bank_ready,
            )
            fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='missing')
            fused_feats, moe_aux_loss = self._refine_stage4(fused_feats)
            return self._decode(
                fused_feats,
                target_size,
                aux={'taskmoe_balance_loss': moe_aux_loss},
            )

        pet_feats_real = self._encode_pet(pet)
        self._collect_cppi(ct_feats, pet_feats_real, mask)
        pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_feats_cal = self.pet_calibration(
            ct_feats,
            pet_feats_proxy,
            ct_reference_feats,
            reference_valid=self.prototype_memory.bank_ready,
        )
        fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='missing')
        fused_feats, moe_aux_loss = self._refine_stage4(fused_feats)
        return self._decode(
            fused_feats,
            target_size,
            aux={'taskmoe_balance_loss': moe_aux_loss},
        )

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._collect_cppi(ct_feats, pet_feats_real, mask)
        pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_selected = []
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        availability = availability.view(-1, 1, 1, 1)
        for feat_real, feat_proxy in zip(pet_feats_real, pet_feats_proxy):
            availability_mask = availability.to(device=feat_real.device, dtype=feat_real.dtype)
            pet_selected.append(feat_real * availability_mask + feat_proxy * (1.0 - availability_mask))
        pet_selected = self.pet_calibration(
            ct_feats,
            pet_selected,
            ct_reference_feats,
            reference_valid=self.prototype_memory.bank_ready,
        )
        fused_feats = self.fusion(
            ct_feats,
            pet_selected,
            mode='auto',
            pet_available=pet_available,
        )
        fused_feats, moe_aux_loss = self._refine_stage4(fused_feats)
        return self._decode(
            fused_feats,
            target_size,
            aux={'taskmoe_balance_loss': moe_aux_loss},
        )

    @torch.no_grad()
    def finalize_cppi_epoch(self, epoch, save_json=True, save_visualizations=False, print_info=True):
        return self.prototype_memory.finalize_epoch(
            epoch=epoch,
            save_json=save_json,
            save_visualizations=save_visualizations,
            print_info=print_info,
        )

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None):
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
