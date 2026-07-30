import json

import torch
import torch.nn as nn


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
        self.reset_retrieval_stats()

    def reset_retrieval_stats(self):
        self._slot_hist = torch.zeros(self.slots, dtype=torch.long)
        self._nearest = []
        self._margins = []
        self._change_ratios = []

    def _current_stats(self, feat):
        x = feat.detach().float().flatten(2).transpose(1, 2)
        mean = x.mean(dim=1)
        centered = x - mean.unsqueeze(1)
        cov = centered.transpose(1, 2) @ centered / max(1, x.shape[1])
        eye = torch.eye(self.channels, device=feat.device, dtype=torch.float32)
        return mean, _symmetrize(cov) + self.eps * eye

    def _bw2(self, mean, cov, slot):
        src_m = self.source_means[slot].float()
        src_c = _symmetrize(self.source_covariances[slot].float())
        mean_term = (mean - src_m).pow(2).sum()
        src_sqrt = _matrix_sqrt(src_c, self.eps)
        middle = src_sqrt @ cov @ src_sqrt
        cov_term = torch.trace(cov + src_c - 2.0 * _matrix_sqrt(middle, self.eps))
        return (mean_term + cov_term).clamp_min(0.0)

    def _transport(self, feat, selected_slots):
        delta = self.delta_means[selected_slots].to(dtype=feat.dtype, device=feat.device)
        operators = self.operators[selected_slots].to(dtype=feat.dtype, device=feat.device)
        live_mean = feat.mean(dim=(2, 3), keepdim=True)
        centered = feat - live_mean
        transported = live_mean + delta[:, :, None, None] + torch.einsum('bij,bjhw->bihw', operators, centered)
        return transported

    def forward(self, feat):
        if (not bool(self.memory_ready.item())) or int(self.valid_slots.item()) == 0:
            return feat, {
                'pdtm_memory_ready': False,
                'pdtm_selected_slot_mean': -1.0,
                'pdtm_nearest_distance_mean': 0.0,
                'pdtm_feature_change_ratio': 0.0,
            }
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            mean, cov = self._current_stats(feat)
            valid = int(self.valid_slots.item())
            dist_list = []
            for b in range(feat.shape[0]):
                dists_b = torch.stack([self._bw2(mean[b], cov[b], k) for k in range(valid)])
                dist_list.append(dists_b)
            dist_matrix = torch.stack(dist_list, dim=0)
            selected_slots = dist_matrix.argmin(dim=1)
            nearest = dist_matrix.gather(1, selected_slots[:, None]).squeeze(1)
            sorted_dists = torch.sort(dist_matrix, dim=1).values
            margins = sorted_dists[:, 1] - sorted_dists[:, 0] if valid > 1 else torch.zeros_like(nearest)
        transported = self._transport(feat, selected_slots)
        ratio = torch.linalg.vector_norm((transported - feat).float().reshape(feat.shape[0], -1), dim=1) / (torch.linalg.vector_norm(feat.float().reshape(feat.shape[0], -1), dim=1) + self.eps)
        self._last_selected_slots = selected_slots.detach().clone()
        for slot in selected_slots.tolist():
            self._slot_hist[slot] += 1
        self._nearest.extend([float(v) for v in nearest.detach().cpu().tolist()])
        self._margins.extend([float(v) for v in margins.detach().cpu().tolist()])
        self._change_ratios.extend([float(v) for v in ratio.detach().cpu().tolist()])
        info = {
            'pdtm_memory_ready': True,
            'pdtm_selected_slot_mean': float(selected_slots.float().mean().item()),
            'pdtm_nearest_distance_mean': float(nearest.float().mean().item()),
            'pdtm_feature_change_ratio': float(ratio.float().mean().item()),
            'pdtm_retrieval_margin_mean': float(margins.float().mean().item()),
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
