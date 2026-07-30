import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import _check_tensor, _check_tensor_list
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.pdtm_runtime import PDTMRuntime, _matrix_sqrt, _matrix_inv_sqrt, _symmetrize


class DualSharedAddPDTM(DualSharedAddPETCTBaseline):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64),
                 use_deep_supervision=False,
                 pdtm_slots=8, pdtm_eps=1e-4, pdtm_max_pairs=256):
        super().__init__(
            ct_backbone=ct_backbone, pet_backbone=pet_backbone,
            ct_pretrained_path=ct_pretrained_path, pet_pretrained_path=pet_pretrained_path,
            in_channels=in_channels, out_channels=out_channels,
            decoder_channels=decoder_channels, use_deep_supervision=use_deep_supervision,
        )
        self.pdtm_max_pairs = int(pdtm_max_pairs)
        self.pdtm = PDTMRuntime(
            channels=decoder_channels[-1],
            slots=pdtm_slots,
            eps=pdtm_eps,
        )
        self._cache = []
        self._viz_cache = None

    def _forward_full(self, ct, pet, target_size):
        return super()._forward_full(ct, pet, target_size)

    def _forward_missing(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self.decoder(fused_feats, target_size, return_intermediates=True)
        d1_ct = raw_out['decoder_feature']
        d1_hat, pdtm_info = self.pdtm(d1_ct)
        native_logits_hat = self.decoder.seg_head(d1_hat)
        final_logits = F.interpolate(native_logits_hat, size=target_size, mode='bilinear', align_corners=False)
        _check_tensor('logits', final_logits)
        out = {'logits': final_logits, 'pred': final_logits, 'aux': dict(pdtm_info)}
        if self.use_deep_supervision and 'aux_logits' in raw_out:
            out['aux_logits'] = raw_out['aux_logits']
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        pet_feats_masked = []
        for feat in pet_feats_real:
            availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
            pet_feats_masked.append(feat * availability_mask)
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw_out = self.decoder(fused_feats, target_size, return_intermediates=True)
        d1 = raw_out['decoder_feature']
        missing_idx = (pet_available == 0)
        pdtm_info = {
            'pdtm_memory_ready': bool(self.pdtm.memory_ready.item()),
            'pdtm_selected_slot_mean': -1,
            'pdtm_nearest_distance_mean': 0.0,
            'pdtm_feature_change_ratio': 0.0,
        }
        if bool(self.pdtm.memory_ready.item()) and int(self.pdtm.valid_slots.item()) > 0 and missing_idx.any():
            d1_final = d1.clone()
            d1_missing = d1[missing_idx]
            d1_hat, info = self.pdtm(d1_missing)
            d1_final[missing_idx] = d1_hat
            pdtm_info.update(info)
        else:
            d1_final = d1
        native_logits = self.decoder.seg_head(d1_final)
        final_logits = F.interpolate(native_logits, size=target_size, mode='bilinear', align_corners=False)
        _check_tensor('logits', final_logits)
        out = {'logits': final_logits, 'pred': final_logits, 'aux': dict(pdtm_info)}
        if self.use_deep_supervision and 'aux_logits' in raw_out:
            out['aux_logits'] = raw_out['aux_logits']
        return out

    @torch.no_grad()
    def clear_pdtm_cache(self):
        self._cache = []
        self._viz_cache = None

    @torch.no_grad()
    def collect_pdtm_pairs(self, ct, pet, case_ids=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        full_fused = self.fusion(ct_feats, pet_feats_real, None)
        full_out = self.decoder(full_fused, ct.shape[-2:], return_intermediates=True)
        d1_full = full_out['decoder_feature']
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        ct_fused = self.fusion(ct_feats, pet_feats_masked, None)
        ct_out = self.decoder(ct_fused, ct.shape[-2:], return_intermediates=True)
        d1_ct = ct_out['decoder_feature']
        B = ct.shape[0]
        if case_ids is None:
            case_ids = [str(i) for i in range(B)]
        for i in range(B):
            pair = self._make_pair(d1_ct[i], d1_full[i], str(case_ids[i]))
            self._cache.append(pair)
        if self._viz_cache is None and B > 0:
            self._viz_cache = {
                'd1_ct': d1_ct[0].detach().float().cpu(),
                'd1_full': d1_full[0].detach().float().cpu(),
                'case_id': str(case_ids[0]),
            }

    def _make_pair(self, d1_ct, d1_full, case_id):
        eps = self.pdtm.eps
        x_ct = d1_ct.detach().float().permute(1, 2, 0).reshape(-1, self.pdtm.channels)
        m_ct = x_ct.mean(dim=0)
        centered_ct = x_ct - m_ct
        cov_ct = _symmetrize(centered_ct.t() @ centered_ct / max(1, x_ct.shape[0])) + eps * torch.eye(self.pdtm.channels, device=d1_ct.device, dtype=torch.float32)
        x_full = d1_full.detach().float().permute(1, 2, 0).reshape(-1, self.pdtm.channels)
        m_full = x_full.mean(dim=0)
        centered_full = x_full - m_full
        cov_full = _symmetrize(centered_full.t() @ centered_full / max(1, x_full.shape[0])) + eps * torch.eye(self.pdtm.channels, device=d1_full.device, dtype=torch.float32)
        src_sqrt = _matrix_sqrt(cov_ct, eps)
        src_inv_sqrt = _matrix_inv_sqrt(cov_ct, eps)
        middle = src_sqrt @ cov_full @ src_sqrt
        operator = _symmetrize(src_inv_sqrt @ _matrix_sqrt(middle, eps) @ src_inv_sqrt)
        delta_mean = m_full - m_ct
        mean_term = (m_ct - m_full).pow(2).sum()
        cov_term = torch.trace(cov_ct + cov_full - 2.0 * _matrix_sqrt(src_sqrt @ cov_full @ src_sqrt, eps))
        paired_w2 = float((mean_term + cov_term).clamp_min(0.0).item())
        return {
            'case_id': case_id,
            'source_mean': m_ct.cpu(),
            'source_covariance': cov_ct.cpu(),
            'delta_mean': delta_mean.cpu(),
            'operator': operator.cpu(),
            'paired_w2': paired_w2,
        }

    @torch.no_grad()
    def finalize_pdtm_memory(self):
        n = len(self._cache)
        if n == 0:
            print('[PDTM] WARNING: no pairs collected; memory stays not ready')
            self.pdtm.memory_ready.fill_(False)
            self.pdtm.valid_slots.fill_(0)
            return {'memory_ready': False, 'pair_count': 0, 'effective_slots': 0}
        K = self.pdtm.slots
        effective = min(K, n)
        if n <= K:
            medoid_indices = list(range(n))
            assignment = torch.arange(n)
        else:
            dist_matrix = torch.zeros(n, n, dtype=torch.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    d = self._bw2_between(self._cache[i]['source_mean'], self._cache[i]['source_covariance'],
                                          self._cache[j]['source_mean'], self._cache[j]['source_covariance'])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
            medoid_indices, assignment = self._k_medoids(dist_matrix, effective)
        for slot, idx in enumerate(medoid_indices):
            pair = self._cache[idx]
            self.pdtm.source_means[slot] = pair['source_mean'].float()
            self.pdtm.source_covariances[slot] = pair['source_covariance'].float()
            self.pdtm.delta_means[slot] = pair['delta_mean'].float()
            self.pdtm.operators[slot] = pair['operator'].float()
            self.pdtm.paired_w2[slot] = float(pair['paired_w2'])
        for slot in range(effective, K):
            self.pdtm.source_means[slot] = 0.0
            self.pdtm.source_covariances[slot] = torch.eye(self.pdtm.channels)
            self.pdtm.delta_means[slot] = 0.0
            self.pdtm.operators[slot] = torch.eye(self.pdtm.channels)
            self.pdtm.paired_w2[slot] = 0.0
        cluster_sizes = torch.zeros(K, dtype=torch.long)
        for c in range(effective):
            cluster_sizes[c] = int((assignment == c).sum().item())
        self.pdtm.cluster_sizes = cluster_sizes
        self.pdtm.valid_slots.fill_(effective)
        self.pdtm.memory_ready.fill_(True)
        selected_case_ids = [self._cache[idx]['case_id'] for idx in medoid_indices]
        report = {
            'memory_ready': True,
            'pair_count': n,
            'effective_slots': effective,
            'selected_case_ids': selected_case_ids,
            'cluster_sizes': cluster_sizes[:effective].tolist(),
            'paired_w2_mean': float(sum(p['paired_w2'] for p in self._cache) / n),
        }
        return report

    def _bw2_between(self, m1, c1, m2, c2):
        eps = self.pdtm.eps
        m1 = m1.double(); c1 = c1.double(); m2 = m2.double(); c2 = c2.double()
        mean_term = (m1 - m2).pow(2).sum()
        c1_sqrt = _matrix_sqrt(c1, eps).double()
        middle = c1_sqrt @ c2 @ c1_sqrt
        cov_term = torch.trace(c1 + c2 - 2.0 * _matrix_sqrt(middle, eps).double())
        return float((mean_term + cov_term).clamp_min(0.0).item())

    def _k_medoids(self, dist_matrix, k, max_iters=30):
        n = dist_matrix.shape[0]
        first = int(dist_matrix.sum(dim=1).argmin().item())
        medoids = [first]
        while len(medoids) < k:
            nearest = dist_matrix[:, medoids].min(dim=1).values.clone()
            for m in medoids:
                nearest[m] = -1.0
            medoids.append(int(nearest.argmax().item()))
        for _ in range(max_iters):
            assignment = dist_matrix[:, medoids].argmin(dim=1)
            updated = []
            for c in range(k):
                members = torch.where(assignment == c)[0]
                if members.numel() == 0:
                    blocked = list(set(medoids + updated))
                    nearest = dist_matrix[:, medoids].min(dim=1).values.clone()
                    for b in blocked:
                        nearest[b] = -1.0
                    updated.append(int(nearest.argmax().item()))
                    continue
                intra = dist_matrix[members][:, members]
                best = int(intra.sum(dim=1).argmin().item())
                updated.append(int(members[best].item()))
            if updated == medoids:
                break
            medoids = updated
        assignment = dist_matrix[:, medoids].argmin(dim=1)
        return medoids, assignment

    def pdtm_diagnostics(self):
        return self.pdtm.diagnostics()

    def export_pdtm_json(self, output_dir, tag):
        return self.pdtm.export_json(output_dir, tag)

    def save_pdtm_visualizations(self, output_dir, tag):
        if self._viz_cache is None:
            return []
        try:
            from models.pdtm_standalone import (
                save_visualizations, gaussian_from_feature, bures_wasserstein_squared,
                gaussian_ot_operator, apply_transport, estimate_gaussian, flatten_spatial,
            )
        except Exception as e:
            print(f'[PDTM] visualization import failed: {e}')
            return []
        d1_ct = self._viz_cache['d1_ct']
        d1_full = self._viz_cache['d1_full']
        case_id = self._viz_cache['case_id']
        eps = self.pdtm.eps
        src = estimate_gaussian(flatten_spatial(d1_ct), eps)
        tgt = estimate_gaussian(flatten_spatial(d1_full), eps)
        delta, op = gaussian_ot_operator(src, tgt, eps)
        transported = apply_transport(d1_ct, src, delta, op)
        import os
        os.makedirs(output_dir, exist_ok=True)
        reports = [{
            'selected_slot': 0,
            'retrieval_distances': [0.0],
            'w2_to_full_before': float(bures_wasserstein_squared(src, tgt, eps).item()),
            'w2_to_full_after': 0.0,
        }]
        return save_visualizations(
            d1_ct.unsqueeze(0), transported.unsqueeze(0), reports,
            type(output_dir)(output_dir), eps, 0, d1_full.unsqueeze(0),
        )
