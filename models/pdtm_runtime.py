import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetrize(x):
    return 0.5 * (x + x.transpose(-1, -2))


def _spd_eigh(x, eps):
    x = _symmetrize(x.float())
    vals, vecs = torch.linalg.eigh(x)
    vals = vals.clamp_min(eps)
    return vals, vecs


def _matrix_sqrt(x, eps):
    vals, vecs = _spd_eigh(x, eps)
    return _symmetrize((vecs * vals.sqrt().unsqueeze(-2)) @ vecs.transpose(-1, -2))


def _matrix_inv_sqrt(x, eps):
    vals, vecs = _spd_eigh(x, eps)
    return _symmetrize((vecs * vals.rsqrt().unsqueeze(-2)) @ vecs.transpose(-1, -2))


class PDTMRuntime(nn.Module):
    def __init__(self, channels, slots=8, eps=1e-4):
        super().__init__()
        self.channels = int(channels)
        self.slots = int(slots)
        self.eps = float(eps)
        eye = torch.eye(self.channels, dtype=torch.float32)
        self.register_buffer('memory_ready', torch.tensor(False))
        self.register_buffer('valid_slots', torch.zeros((), dtype=torch.long))
        self.register_buffer('source_means', torch.zeros(self.slots, self.channels, dtype=torch.float32))
        self.register_buffer('source_covariances', eye.unsqueeze(0).repeat(self.slots, 1, 1).contiguous())
        self.register_buffer('delta_means', torch.zeros(self.slots, self.channels, dtype=torch.float32))
        self.register_buffer('operators', eye.unsqueeze(0).repeat(self.slots, 1, 1).contiguous())
        self.register_buffer('paired_w2', torch.zeros(self.slots, dtype=torch.float32))
        self.register_buffer('cluster_sizes', torch.zeros(self.slots, dtype=torch.long))
        self._reset_stats()

    def _reset_stats(self):
        self._slot_hist = torch.zeros(self.slots, dtype=torch.long)
        self._nearest = []
        self._margins = []
        self._change_ratios = []

    def reset_retrieval_stats(self):
        self._reset_stats()

    def _current_cov(self, feat):
        x = feat.detach().float().flatten(2).transpose(1, 2)
        m = x.mean(dim=1, keepdim=True)
        centered = x - m
        cov = centered.transpose(1, 2) @ centered / max(1, x.shape[1])
        eye = torch.eye(self.channels, device=feat.device, dtype=torch.float32)
        return m.squeeze(1), _symmetrize(cov) + self.eps * eye

    def _bw2(self, mean, cov, slot):
        src_m = self.source_means[slot].float()
        src_c = _symmetrize(self.source_covariances[slot].float())
        mean_term = (mean - src_m).pow(2).sum()
        src_sqrt = _matrix_sqrt(src_c, self.eps)
        middle = src_sqrt @ cov @ src_sqrt
        cov_term = torch.trace(cov + src_c - 2.0 * _matrix_sqrt(middle, self.eps))
        return (mean_term + cov_term).clamp_min(0.0)

    def _apply(self, feat, slot):
        delta = self.delta_means[slot].to(dtype=feat.dtype, device=feat.device)
        op = self.operators[slot].to(dtype=torch.float32, device=feat.device)
        m = feat.mean(dim=(2, 3), keepdim=True)
        centered = feat - m
        transported = m + delta.view(1, -1, 1, 1) + torch.einsum('ij,bjhw->bihw', op, centered)
        return transported

    def forward(self, feat):
        if (not bool(self.memory_ready.item())) or int(self.valid_slots.item()) == 0:
            info = {
                'pdtm_memory_ready': False,
                'pdtm_selected_slot_mean': -1,
                'pdtm_nearest_distance_mean': 0.0,
                'pdtm_feature_change_ratio': 0.0,
            }
            return feat, info
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            mean, cov = self._current_cov(feat)
            valid = int(self.valid_slots.item())
            dists = []
            for k in range(valid):
                dists.append(self._bw2(mean[0], cov[0], k))
            dists = torch.stack(dists)
            slot = int(torch.argmin(dists).item())
            sorted_d = torch.sort(dists).values
            margin = float((sorted_d[1] - sorted_d[0]).item()) if sorted_d.numel() > 1 else 0.0
        transported = self._apply(feat, slot)
        ratio = float(torch.linalg.vector_norm((transported - feat).float()).item() / (torch.linalg.vector_norm(feat.float()).item() + self.eps))
        self._slot_hist[slot] += 1
        self._nearest.append(float(dists[slot].item()))
        self._margins.append(margin)
        self._change_ratios.append(ratio)
        info = {
            'pdtm_memory_ready': True,
            'pdtm_selected_slot_mean': slot,
            'pdtm_nearest_distance_mean': float(dists[slot].item()),
            'pdtm_feature_change_ratio': ratio,
            'pdtm_retrieval_margin_mean': margin,
        }
        return transported.to(dtype=feat.dtype), info

    def diagnostics(self):
        valid = int(self.valid_slots.item())
        return {
            'memory_ready': bool(self.memory_ready.item()),
            'valid_slots': valid,
            'slot_histogram': self._slot_hist[:valid].tolist() if valid else [],
            'nearest_distance_mean': float(sum(self._nearest) / len(self._nearest)) if self._nearest else 0.0,
            'retrieval_margin_mean': float(sum(self._margins) / len(self._margins)) if self._margins else 0.0,
            'feature_change_ratio_mean': float(sum(self._change_ratios) / len(self._change_ratios)) if self._change_ratios else 0.0,
            'delta_mean_norm_per_slot': torch.linalg.vector_norm(self.delta_means[:valid].float(), dim=1).tolist() if valid else [],
            'operator_frobenius_norm_per_slot': torch.linalg.matrix_norm(self.operators[:valid].float(), ord='fro', dim=(-2, -1)).tolist() if valid else [],
            'source_covariance_trace_per_slot': torch.einsum('kii->k', self.source_covariances[:valid].float()).tolist() if valid else [],
            'cluster_sizes': self.cluster_sizes[:valid].tolist() if valid else [],
            'paired_w2_per_slot': self.paired_w2[:valid].tolist() if valid else [],
        }

    def export_json(self, output_dir, tag):
        payload = self.diagnostics()
        path = f'{output_dir}/{tag}.json'
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        return path
