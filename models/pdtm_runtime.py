import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class _RetrievalRecord:
    selected_slot: int
    nearest_distance: float
    retrieval_margin: float
    feature_change_ratio: float


class PDTMRuntime(nn.Module):
    def __init__(self, channels: int, slots: int = 8, eps: float = 1e-4):
        super().__init__()
        self.channels = int(channels)
        self.slots = int(slots)
        self.eps = float(eps)
        eye = torch.eye(self.channels, dtype=torch.float32)
        self.register_buffer('memory_ready', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('valid_slots', torch.tensor(0, dtype=torch.long))
        self.register_buffer('source_means', torch.zeros(self.slots, self.channels, dtype=torch.float32))
        self.register_buffer('source_covariances', eye.unsqueeze(0).repeat(self.slots, 1, 1).contiguous())
        self.register_buffer('delta_means', torch.zeros(self.slots, self.channels, dtype=torch.float32))
        self.register_buffer('operators', eye.unsqueeze(0).repeat(self.slots, 1, 1).contiguous())
        self.register_buffer('paired_w2', torch.zeros(self.slots, dtype=torch.float32))
        self.register_buffer('cluster_sizes', torch.zeros(self.slots, dtype=torch.long))
        self._retrieval_records: List[_RetrievalRecord] = []
        self._slot_histogram = torch.zeros(self.slots, dtype=torch.long)

    def reset_retrieval_stats(self):
        self._retrieval_records = []
        self._slot_histogram.zero_()

    def _symmetrize(self, x):
        return 0.5 * (x + x.transpose(-1, -2))

    def _feature_mean_cov(self, feat):
        _, c, _, _ = feat.shape
        x = feat.flatten(2)
        mean = x.mean(dim=-1)
        centered = x - mean.unsqueeze(-1)
        cov = centered @ centered.transpose(-1, -2) / max(1, centered.shape[-1])
        cov = self._symmetrize(cov) + self.eps * torch.eye(c, device=feat.device, dtype=feat.dtype).unsqueeze(0)
        return mean, cov

    def _sqrtm(self, matrix):
        matrix = self._symmetrize(matrix.float())
        evals, evecs = torch.linalg.eigh(matrix)
        evals = evals.clamp_min(self.eps)
        return evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-1, -2)

    def _bw2(self, m0, s0, m1, s1):
        mean_term = (m0 - m1).pow(2).sum(dim=-1)
        s0 = self._symmetrize(s0.float())
        s1 = self._symmetrize(s1.float())
        s0_root = self._sqrtm(s0)
        middle = s0_root @ s1 @ s0_root
        middle_root = self._sqrtm(middle)
        tr = torch.diagonal(s0, dim1=-2, dim2=-1).sum(-1) + torch.diagonal(s1, dim1=-2, dim2=-1).sum(-1) - 2.0 * torch.diagonal(middle_root, dim1=-2, dim2=-1).sum(-1)
        return mean_term + tr

    def _select_slot(self, mean, cov):
        valid = int(self.valid_slots.item())
        if not bool(self.memory_ready.item()) or valid <= 0:
            return -1, None, None
        src_mean = self.source_means[:valid].float()
        src_cov = self.source_covariances[:valid].float()
        distances = self._bw2(mean.unsqueeze(0).expand(valid, -1), cov.unsqueeze(0).expand(valid, -1, -1), src_mean, src_cov)
        values, indices = torch.sort(distances)
        slot = int(indices[0].item())
        margin = float((values[1] - values[0]).item()) if valid > 1 else 0.0
        return slot, float(values[0].item()), margin

    def forward(self, feat):
        if feat.ndim != 4:
            raise ValueError('Expected [B,C,H,W]')
        b, c, _, _ = feat.shape
        if c != self.channels:
            raise ValueError(f'Expected channels={self.channels}, got {c}')
        if not bool(self.memory_ready.item()) or int(self.valid_slots.item()) <= 0:
            return feat, {'pdtm_memory_ready': False, 'pdtm_selected_slot_mean': -1, 'pdtm_nearest_distance_mean': 0.0, 'pdtm_feature_change_ratio': 0.0}
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                mean, cov = self._feature_mean_cov(feat.float())
                selected = []
                nearest = []
                outputs = feat.clone()
                for i in range(b):
                    slot, nearest_dist, margin = self._select_slot(mean[i], cov[i])
                    if slot < 0:
                        continue
                    selected.append(slot)
                    nearest.append(nearest_dist)
                    self._slot_histogram[slot] += 1
                    self._retrieval_records.append(_RetrievalRecord(slot, nearest_dist, margin, 0.0))
                if selected:
                    idx = torch.tensor(selected, device=feat.device, dtype=torch.long)
                    delta = self.delta_means[idx].to(dtype=feat.dtype, device=feat.device)
                    op = self.operators[idx].to(dtype=feat.dtype, device=feat.device)
                    m = feat.mean(dim=(2, 3), keepdim=True)
                    centered = feat - m
                    outputs = m + delta.view(b, c, 1, 1) + torch.einsum('bij,bjhw->bihw', op, centered)
                    ratio = (outputs - feat).flatten(1).norm(dim=1) / (feat.flatten(1).norm(dim=1).clamp_min(self.eps))
                    for j in range(len(selected)):
                        self._retrieval_records[-len(selected) + j] = _RetrievalRecord(self._retrieval_records[-len(selected) + j].selected_slot, self._retrieval_records[-len(selected) + j].nearest_distance, self._retrieval_records[-len(selected) + j].retrieval_margin, float(ratio[j].item()))
                info = {'pdtm_memory_ready': True, 'pdtm_selected_slot_mean': float(sum(selected) / max(1, len(selected))) if selected else -1, 'pdtm_nearest_distance_mean': float(sum(nearest) / max(1, len(nearest))) if nearest else 0.0, 'pdtm_feature_change_ratio': float(sum(r.feature_change_ratio for r in self._retrieval_records[-len(selected):]) / max(1, len(selected))) if selected else 0.0}
        return outputs.to(dtype=feat.dtype), info

    @torch.no_grad()
    def load_memory(self, source_means, source_covariances, delta_means, operators, paired_w2=None, cluster_sizes=None):
        n = min(self.slots, source_means.shape[0])
        self.source_means.zero_(); self.delta_means.zero_(); self.paired_w2.zero_(); self.cluster_sizes.zero_()
        eye = torch.eye(self.channels, dtype=self.source_covariances.dtype, device=self.source_covariances.device)
        self.source_covariances.copy_(eye.unsqueeze(0).repeat(self.slots, 1, 1))
        self.operators.copy_(eye.unsqueeze(0).repeat(self.slots, 1, 1))
        self.source_means[:n].copy_(source_means[:n].to(self.source_means.device, dtype=torch.float32))
        self.source_covariances[:n].copy_(source_covariances[:n].to(self.source_covariances.device, dtype=torch.float32))
        self.delta_means[:n].copy_(delta_means[:n].to(self.delta_means.device, dtype=torch.float32))
        self.operators[:n].copy_(operators[:n].to(self.operators.device, dtype=torch.float32))
        if paired_w2 is not None:
            self.paired_w2[:n].copy_(paired_w2[:n].to(self.paired_w2.device, dtype=torch.float32))
        if cluster_sizes is not None:
            self.cluster_sizes[:n].copy_(cluster_sizes[:n].to(self.cluster_sizes.device, dtype=torch.long))
        self.valid_slots.fill_(n)
        self.memory_ready.fill_(n > 0)

    def diagnostics(self):
        valid = int(self.valid_slots.item())
        return {'memory_ready': bool(self.memory_ready.item()), 'valid_slots': valid, 'slot_histogram': self._slot_histogram[:valid].detach().cpu().tolist(), 'nearest_distance_mean': float(sum(r.nearest_distance for r in self._retrieval_records) / max(1, len(self._retrieval_records))), 'retrieval_margin_mean': float(sum(r.retrieval_margin for r in self._retrieval_records) / max(1, len(self._retrieval_records))), 'feature_change_ratio_mean': float(sum(r.feature_change_ratio for r in self._retrieval_records) / max(1, len(self._retrieval_records))), 'delta_mean_norm_per_slot': self.delta_means[:valid].float().norm(dim=1).detach().cpu().tolist(), 'operator_frobenius_norm_per_slot': self.operators[:valid].float().flatten(1).norm(dim=1).detach().cpu().tolist(), 'source_covariance_trace_per_slot': self.source_covariances[:valid].float().diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu().tolist(), 'cluster_sizes': self.cluster_sizes[:valid].detach().cpu().tolist(), 'paired_w2_per_slot': self.paired_w2[:valid].detach().cpu().tolist()}

    def export_json(self, output_dir, tag):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{tag}.json')
        with open(path, 'w') as f:
            json.dump(self.diagnostics(), f, indent=2)
        return path
