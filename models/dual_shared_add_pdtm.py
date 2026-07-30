import os

import torch
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.pdtm_runtime import PDTMRuntime


class DualSharedAddPDTM(DualSharedAddPETCTBaseline):
    def __init__(self, *args, pdtm_slots=8, pdtm_eps=1e-4, pdtm_max_pairs=256, **kwargs):
        decoder_channels = kwargs.get('decoder_channels', (512, 256, 128, 64))
        super().__init__(*args, **kwargs)
        self.pdtm_max_pairs = int(pdtm_max_pairs)
        self.pdtm = PDTMRuntime(channels=decoder_channels[-1], slots=pdtm_slots, eps=pdtm_eps)
        self._pdtm_pairs = []
        self._pdtm_examples = []

    def clear_pdtm_cache(self):
        self._pdtm_pairs = []
        self._pdtm_examples = []
        self.pdtm.reset_retrieval_stats()

    def _decode_with_intermediates(self, fused_feats, target_size):
        return self.decoder(fused_feats, target_size, return_intermediates=True)

    def _forward_full(self, ct, pet, target_size):
        return super()._forward_full(ct, pet, target_size)

    def _forward_missing(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self._decode_with_intermediates(fused_feats, target_size)
        d1_ct = raw_out['decoder_feature']
        d1_hat, pdtm_info = self.pdtm(d1_ct)
        native_logits_hat = self.decoder.seg_head(d1_hat)
        final_logits = F.interpolate(native_logits_hat, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': final_logits, 'pred': final_logits, 'aux': {**pdtm_info}}

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        pet_feats_masked = []
        for feat in pet_feats_real:
            availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
            pet_feats_masked.append(feat * availability_mask)
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self._decode_with_intermediates(fused_feats, target_size)
        d1 = raw_out['decoder_feature']
        d1_final = d1.clone()
        missing_idx = pet_available == 0
        aux = {'pdtm_memory_ready': bool(self.pdtm.memory_ready.item()), 'pdtm_selected_slot_mean': -1, 'pdtm_nearest_distance_mean': 0.0, 'pdtm_feature_change_ratio': 0.0}
        if bool(self.pdtm.memory_ready.item()) and missing_idx.any():
            d1_missing, aux = self.pdtm(d1[missing_idx])
            d1_final[missing_idx] = d1_missing
        native_logits = self.decoder.seg_head(d1_final)
        final_logits = F.interpolate(native_logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': final_logits, 'pred': final_logits, 'aux': aux}

    @torch.no_grad()
    def collect_pdtm_pairs(self, ct, pet, case_ids=None):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full_fused = self.fusion(ct_feats, pet_feats, None)
        full_out = self.decoder(full_fused, ct.shape[-2:], return_intermediates=True)
        zero_pet = [torch.zeros_like(feat) for feat in pet_feats]
        ct_fused = self.fusion(ct_feats, zero_pet, None)
        ct_out = self.decoder(ct_fused, ct.shape[-2:], return_intermediates=True)
        d1_full = full_out['decoder_feature']
        d1_ct = ct_out['decoder_feature']
        for i in range(d1_ct.shape[0]):
            if len(self._pdtm_pairs) >= self.pdtm_max_pairs:
                break
            c = d1_ct.shape[1]
            source_mean = d1_ct[i].mean(dim=(1, 2)).detach().cpu()
            target_mean = d1_full[i].mean(dim=(1, 2)).detach().cpu()
            self._pdtm_pairs.append({'source_mean': source_mean, 'source_covariance': torch.eye(c), 'target_mean': target_mean, 'target_covariance': torch.eye(c), 'delta_mean': (target_mean - source_mean).detach().cpu(), 'operator': torch.eye(c), 'paired_w2': torch.tensor(0.0), 'case_id': case_ids[i] if case_ids is not None else str(i), 'image_id': str(i)})
            if len(self._pdtm_examples) == 0:
                self._pdtm_examples = [d1_ct[i:i+1].detach().cpu(), d1_full[i:i+1].detach().cpu()]

    @torch.no_grad()
    def finalize_pdtm_memory(self):
        if not self._pdtm_pairs:
            self.pdtm.memory_ready.fill_(False)
            self.pdtm.valid_slots.zero_()
            return {'memory_ready': False, 'valid_slots': 0}
        n = min(self.pdtm.slots, len(self._pdtm_pairs))
        c = self.pdtm.channels
        means = torch.stack([x['source_mean'] for x in self._pdtm_pairs[:n]]).float()
        covs = torch.stack([x['source_covariance'] for x in self._pdtm_pairs[:n]]).float().reshape(n, c, c)
        delta = torch.stack([x['delta_mean'] for x in self._pdtm_pairs[:n]]).float()
        ops = torch.stack([x['operator'] for x in self._pdtm_pairs[:n]]).float().reshape(n, c, c)
        w2 = torch.stack([x['paired_w2'] for x in self._pdtm_pairs[:n]]).float()
        sizes = torch.ones(n, dtype=torch.long)
        self.pdtm.load_memory(means, covs, delta, ops, w2, sizes)
        return {'memory_ready': True, 'valid_slots': n}

    def pdtm_diagnostics(self):
        return self.pdtm.diagnostics()

    def export_pdtm_json(self, output_dir, tag):
        return self.pdtm.export_json(output_dir, tag)

    def save_pdtm_visualizations(self, output_dir, tag):
        os.makedirs(output_dir, exist_ok=True)
        return []
