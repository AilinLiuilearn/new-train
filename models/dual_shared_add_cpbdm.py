import torch
import torch.nn.functional as F

from models.cpbdm_distribution_memory import CTConditionedPETBenefitDistributionMemory
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


class DualSharedAddCPBDM(DualSharedAddPETCTBaseline):
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
        cpbdm_k=8,
        cpbdm_query_dim=16,
    ):
        super().__init__(ct_backbone, pet_backbone, ct_pretrained_path, pet_pretrained_path, in_channels, out_channels, decoder_channels, use_deep_supervision)
        self.cpbdm = CTConditionedPETBenefitDistributionMemory(
            decoder_channels=decoder_channels[-1],
            query_dim=cpbdm_query_dim,
            K=cpbdm_k,
        )

    def _forward_full(self, ct, pet, target_size):
        return super()._forward_full(ct, pet, target_size)

    def _forward_missing(self, ct, pet, target_size, capture_cpbdm_maps=False):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self.decoder(fused_feats, target_size, return_intermediates=True)
        d_ct = raw_out['decoder_feature']
        z_ct_native = raw_out['native_logits']
        z_corrected_native, cpbdm_info = self.cpbdm(d_ct, z_ct_native, capture_maps=capture_cpbdm_maps)
        z_missing = F.interpolate(z_corrected_native, size=target_size, mode='bilinear', align_corners=False)
        out = {'logits': z_missing, 'pred': z_missing, 'aux': {'cpbdm_memory_ready': bool(self.cpbdm.memory_ready.item()), **{k: v for k, v in cpbdm_info.items() if isinstance(v, (bool, int, float, str))}}}
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        pet_feats_masked = []
        for feat in pet_feats_real:
            availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
            pet_feats_masked.append(feat * availability_mask)
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self.decoder(fused_feats, target_size, return_intermediates=True)
        z_native = raw_out['native_logits']
        corrected_native, _ = self.cpbdm(raw_out['decoder_feature'], z_native, capture_maps=False)
        availability = pet_available.view(-1, 1, 1, 1).to(dtype=z_native.dtype)
        final_native = availability * z_native + (1 - availability) * corrected_native
        final_logits = F.interpolate(final_native, size=target_size, mode='bilinear', align_corners=False)
        out = {'logits': final_logits, 'pred': final_logits, 'aux': {'cpbdm_memory_ready': bool(self.cpbdm.memory_ready.item())}}
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', capture_cpbdm_maps=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, capture_cpbdm_maps=capture_cpbdm_maps)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')

    @torch.no_grad()
    def collect_cpbdm_candidates(self, ct, pet, target, target_size=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full_out = self.decoder(self.fusion(ct_feats, pet_feats, None), target_size, return_intermediates=True)
        zero_pet_feats = [torch.zeros_like(x) for x in pet_feats]
        ct_out = self.decoder(self.fusion(ct_feats, zero_pet_feats, None), target_size, return_intermediates=True)
        return self.cpbdm.collect_from_pair(ct_out['decoder_feature'], ct_out['native_logits'], full_out['native_logits'], target)

    def clear_cpbdm_cache(self):
        return self.cpbdm.clear_cache()

    def finalize_cpbdm_memory(self):
        return self.cpbdm.finalize_memory()

    def reset_cpbdm_retrieval_stats(self):
        return self.cpbdm.reset_retrieval_stats()

    def cpbdm_diagnostics(self):
        return self.cpbdm.diagnostics()

    def print_cpbdm_diagnostics(self):
        return self.cpbdm.print_diagnostics()

    def export_cpbdm_json(self, *args, **kwargs):
        return self.cpbdm.export_json(*args, **kwargs)

    def save_cpbdm_visualization(self, *args, **kwargs):
        return self.cpbdm.save_composite_visualization(*args, **kwargs)
