import torch
import torch.nn as nn

from models.baseline_blocks import PrototypeReferencedPETAffineCalibration, StateAwareWeightedAddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.ct_pet_prototype_imputation import CrossScaleCTPETPrototypeMemory
from models.taskmoe_s4_refiner import TaskMoEStage4Refiner


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
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, cppi_num_clusters=6, cppi_build_stage=3, cppi_output_dir=None):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.stage2_moe_only = False
        self.taskmoe_enabled = True
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
        self.taskmoe_s4 = TaskMoEStage4Refiner(
            channels=pet_channels[-1],
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
        fused_feats = list(fused_feats)
        if not self.taskmoe_enabled:
            zero = fused_feats[-1].new_zeros(())
            return fused_feats, zero
        # Sparse index_add_ is not AMP-safe (Half buffer vs Float source).
        # Run TaskMoE in fp32, then cast back to the fused feature dtype.
        f4 = fused_feats[3]
        with torch.cuda.amp.autocast(enabled=False):
            f4_out, moe_aux_loss = self.taskmoe_s4(f4.float())
        fused_feats[3] = f4_out.to(dtype=f4.dtype)
        return fused_feats, moe_aux_loss.to(dtype=f4.dtype)

    def _decode(self, fused_feats, target_size, aux=None):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {} if aux is None else aux
        return out

    def enable_stage2_moe_only(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.taskmoe_s4.parameters():
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
            self.taskmoe_s4.train(mode)
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
