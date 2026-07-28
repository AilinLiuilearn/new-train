import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.cipm_paired_modality_memory import CTIndexedPairedModalityMemory


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
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, use_cipm=False, cipm_num_slots=16, cipm_retrieval_temperature=0.1, cipm_max_tokens_per_batch=4096, cipm_max_cached_tokens=50000, cipm_positive_fraction=0.5, cipm_mask_threshold=0.5, cipm_outlier_fraction=0.05, cipm_init_kmeans_iters=20, cipm_update_kmeans_iters=3, cipm_seed=2026):
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
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.use_cipm = bool(use_cipm)
        if self.use_cipm:
            self.cipm = CTIndexedPairedModalityMemory(
                channels=pet_channels,
                num_slots=cipm_num_slots,
                retrieval_temperature=cipm_retrieval_temperature,
                max_tokens_per_batch=cipm_max_tokens_per_batch,
                max_cached_tokens=cipm_max_cached_tokens,
                positive_fraction=cipm_positive_fraction,
                mask_threshold=cipm_mask_threshold,
                outlier_fraction=cipm_outlier_fraction,
                init_kmeans_iters=cipm_init_kmeans_iters,
                update_kmeans_iters=cipm_update_kmeans_iters,
                seed=cipm_seed,
            )
        else:
            self.cipm = None

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

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _collect_enabled(self, pet_feats_real, mask, collect_memory):
        return bool(self.use_cipm and self.training and collect_memory and pet_feats_real is not None and mask is not None)

    def _forward_full(self, ct, pet, target_size, mask=None, collect_memory=None, return_cipm_diagnostics=False):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        should_collect = self._collect_enabled(pet_feats_real, mask, collect_memory)
        cipm_diagnostics = None
        if self.use_cipm:
            pet_for_fusion = self.cipm(
                ct_feats,
                pet_feats_real,
                mode='full',
                collect=should_collect,
                mask=mask,
                return_diagnostics=return_cipm_diagnostics,
            )
            if return_cipm_diagnostics:
                pet_for_fusion, cipm_diagnostics = pet_for_fusion
        else:
            pet_for_fusion = pet_feats_real
        fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
        out = self._decode(fused_feats, target_size)
        if return_cipm_diagnostics and cipm_diagnostics is not None:
            out['aux']['cipm'] = cipm_diagnostics
        return out

    def _forward_missing(self, ct, pet, target_size, mask=None, collect_memory=None, return_cipm_diagnostics=False):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        should_collect = self._collect_enabled(pet_feats_real, mask, collect_memory)
        cipm_diagnostics = None
        if self.use_cipm:
            pet_for_fusion = self.cipm(
                ct_feats,
                pet_feats_real,
                mode='missing',
                collect=should_collect,
                mask=mask,
                return_diagnostics=return_cipm_diagnostics,
            )
            if return_cipm_diagnostics:
                pet_for_fusion, cipm_diagnostics = pet_for_fusion
        else:
            pet_for_fusion = [torch.zeros_like(feat) for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_for_fusion, None)
        out = self._decode(fused_feats, target_size)
        if return_cipm_diagnostics and cipm_diagnostics is not None:
            out['aux']['cipm'] = cipm_diagnostics
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None, collect_memory=None, return_cipm_diagnostics=False):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        should_collect = self._collect_enabled(pet_feats_real, mask, collect_memory)
        if self.use_cipm:
            pet_proxy, cipm_diagnostics = self.cipm(
                ct_feats,
                pet_feats_real,
                mode='missing',
                collect=should_collect,
                mask=mask,
                return_diagnostics=True,
            )
            pet_feats_masked = []
            for feat_real, feat_proxy in zip(pet_feats_real, pet_proxy):
                availability_mask = pet_available.to(device=feat_real.device, dtype=feat_real.dtype).view(-1, 1, 1, 1)
                pet_feats_masked.append(feat_real * availability_mask + feat_proxy * (1.0 - availability_mask))
        else:
            pet_feats_masked = []
            for feat in pet_feats_real:
                availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
                pet_feats_masked.append(feat * availability_mask)
            cipm_diagnostics = None
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        out = self._decode(fused_feats, target_size)
        if return_cipm_diagnostics and cipm_diagnostics is not None:
            out['aux']['cipm'] = cipm_diagnostics
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None, collect_memory=None, return_cipm_diagnostics=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        if collect_memory is None:
            collect_memory = self.training
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask, collect_memory=collect_memory, return_cipm_diagnostics=return_cipm_diagnostics)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, mask=mask, collect_memory=collect_memory, return_cipm_diagnostics=return_cipm_diagnostics)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size, mask=mask, collect_memory=collect_memory, return_cipm_diagnostics=return_cipm_diagnostics)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')

    @torch.no_grad()
    def finalize_cipm_epoch(self, sync_distributed=False):
        if not self.use_cipm:
            return []
        return self.cipm.finalize_epoch(sync_distributed=sync_distributed)

    def print_cipm_report(self, print_per_slot=True):
        if self.use_cipm:
            self.cipm.print_memory_report(print_per_slot=print_per_slot)

    def reset_cipm_query_stats(self):
        if self.use_cipm:
            self.cipm.reset_query_stats()

    @property
    def cipm_ready(self):
        return bool(self.use_cipm and self.cipm.ready)

    @torch.no_grad()
    def visualize_cipm(self, ct, mask, save_dir, sample_index=0):
        if not self.use_cipm:
            return []
        was_training = self.training
        self.eval()
        try:
            ct_feats = self._encode_ct(ct)
            saved = []
            saved.extend(self.cipm.visualize_retrieval(ct_feats, mask=mask, sample_index=sample_index, save_dir=save_dir))
            saved.extend(self.cipm.visualize_cluster_pca(save_dir=save_dir))
            saved.append(self.cipm.visualize_slot_utilization(save_path=__import__('os').path.join(save_dir, 'slot_utilization.png')))
            return saved
        finally:
            self.train(was_training)
