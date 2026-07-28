from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.paam_affine_action_memory import (
    AffineActionWriter,
    LayerNorm2d,
    SharedAffineExecutor,
    _json_ready,
    _pairwise_cosine_mean,
    _safe_entropy,
    _tensor_stats,
    _to_float,
)


class ReliableAffineActionMemoryScale(nn.Module):
    QUERY_DIM = 32
    CACHE_CAPACITY = 12000
    KMEANS_ITERS = 20
    MAX_CANDIDATES_PER_IMAGE = 64

    def __init__(self, channels: int, K: int, scale_index: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.K = int(K)
        self.scale_index = int(scale_index)
        self.query_norm = LayerNorm2d(channels)
        self.query_proj = nn.Conv2d(channels, self.QUERY_DIM, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        self.register_buffer('keys', torch.zeros(K, self.QUERY_DIM))
        self.register_buffer('gamma_proto', torch.zeros(K, channels))
        self.register_buffer('beta_proto', torch.zeros(K, channels))
        self.register_buffer('slot_counts', torch.zeros(K))
        self.register_buffer('memory_ready', torch.tensor(False, dtype=torch.bool))
        self._cache_q: List[torch.Tensor] = []
        self._cache_gamma: List[torch.Tensor] = []
        self._cache_beta: List[torch.Tensor] = []
        self._cache_strength: List[torch.Tensor] = []
        self._retrieval_slot_hits = torch.zeros(K, dtype=torch.long)
        self._stats: Dict[str, Any] = {
            'retrieval_entropy': [], 'retrieval_max_similarity': [], 'retrieval_top1_weight': [],
            'retrieved_gamma_abs_mean': [], 'retrieved_beta_abs_mean': [],
            'candidate_strength': [], 'candidate_count': [], 'query_raw_abs_mean': [], 'query_centered_abs_mean': [], 'query_spatial_variance': [],
        }
        self._last_maps: Dict[str, torch.Tensor] = {}

    def make_query(self, ct: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q_raw = self.query_proj(self.query_norm(ct))
        q_centered = q_raw - q_raw.mean(dim=(2, 3), keepdim=True)
        q = F.normalize(q_centered, dim=1, eps=1e-6)
        q_spatial_var = q_raw.var(dim=(2, 3), unbiased=False).mean(dim=1)
        stats = {
            'query_raw_abs_mean': q_raw.abs().mean(),
            'query_centered_abs_mean': q_centered.abs().mean(),
            'query_spatial_variance': q_spatial_var.mean(),
        }
        return q, stats

    @torch.no_grad()
    def collect(self, query_map, gamma_star, beta_star, gate_star, query_stats=None):
        b, _, h, w = query_map.shape
        strength_map = gate_star.detach().squeeze(1) * (gamma_star.detach().abs().mean(dim=1) + beta_star.detach().abs().mean(dim=1))
        per_image = min(self.MAX_CANDIDATES_PER_IMAGE, max(1, (h * w) // 16))
        qs: List[torch.Tensor] = []
        gs: List[torch.Tensor] = []
        bs: List[torch.Tensor] = []
        ss: List[torch.Tensor] = []
        for bi in range(b):
            flat_strength = strength_map[bi].reshape(-1)
            topk = min(per_image, flat_strength.numel())
            _, idx = torch.topk(flat_strength, k=topk, largest=True, sorted=False)
            q_flat = query_map[bi].permute(1, 2, 0).reshape(-1, self.QUERY_DIM)
            g_flat = gamma_star[bi].permute(1, 2, 0).reshape(-1, self.channels)
            b_flat = beta_star[bi].permute(1, 2, 0).reshape(-1, self.channels)
            qs.append(q_flat[idx].detach().float().cpu())
            gs.append(g_flat[idx].detach().float().cpu())
            bs.append(b_flat[idx].detach().float().cpu())
            ss.append(flat_strength[idx].detach().float().cpu())
        q = torch.cat(qs, dim=0); gamma = torch.cat(gs, dim=0); beta = torch.cat(bs, dim=0); strength = torch.cat(ss, dim=0)
        self._cache_q.append(q); self._cache_gamma.append(gamma); self._cache_beta.append(beta); self._cache_strength.append(strength)
        self._trim_cache_if_needed()
        return {'selected_count': int(strength.numel()), 'strength_mean': _to_float(strength.mean()), 'strength_max': _to_float(strength.max())}

    @torch.no_grad()
    def _trim_cache_if_needed(self):
        total = sum(x.shape[0] for x in self._cache_strength)
        if total <= self.CACHE_CAPACITY:
            return
        q = torch.cat(self._cache_q, dim=0); gamma = torch.cat(self._cache_gamma, dim=0); beta = torch.cat(self._cache_beta, dim=0); strength = torch.cat(self._cache_strength, dim=0)
        keep = min(self.CACHE_CAPACITY, strength.numel())
        _, idx = torch.topk(strength, k=keep, largest=True, sorted=False)
        self._cache_q = [q[idx]]; self._cache_gamma = [gamma[idx]]; self._cache_beta = [beta[idx]]; self._cache_strength = [strength[idx]]

    @torch.no_grad()
    def finalize_memory(self):
        if not self._cache_q:
            return {'scale': self.scale_index, 'updated': False, 'reason': 'empty_cache', 'memory_ready': bool(self.memory_ready.item())}
        q = torch.cat(self._cache_q, dim=0).float(); gamma = torch.cat(self._cache_gamma, dim=0).float(); beta = torch.cat(self._cache_beta, dim=0).float(); strength = torch.cat(self._cache_strength, dim=0).float().clamp_min(1e-8)
        n = q.shape[0]
        if n < self.K:
            repeat = math.ceil(self.K / max(n, 1))
            q = q.repeat(repeat, 1)[:self.K]; gamma = gamma.repeat(repeat, 1)[:self.K]; beta = beta.repeat(repeat, 1)[:self.K]; strength = strength.repeat(repeat)[:self.K]
            n = q.shape[0]
        q_n = F.normalize(q, dim=1, eps=1e-6)
        action = torch.cat([gamma, beta], dim=1)
        action_n = F.normalize(action, dim=1, eps=1e-6)
        joint = F.normalize(torch.cat([q_n, action_n], dim=1), dim=1, eps=1e-6)
        centers = self._deterministic_farthest_init(joint, strength, self.K)
        assignment = torch.zeros(n, dtype=torch.long)
        for _ in range(self.KMEANS_ITERS):
            sim = joint @ centers.t()
            assignment = sim.argmax(dim=1)
            new_centers = []
            for j in range(self.K):
                mask = assignment == j
                if mask.any():
                    weighted = joint[mask] * strength[mask, None]
                    center = weighted.sum(dim=0) / strength[mask].sum().clamp_min(1e-8)
                    center = F.normalize(center, dim=0, eps=1e-6)
                else:
                    center = joint[sim.max(dim=1).values.argmin()]
                new_centers.append(center)
            new_centers_t = torch.stack(new_centers, dim=0)
            if torch.allclose(centers, new_centers_t, atol=1e-5, rtol=1e-4):
                centers = new_centers_t
                break
            centers = new_centers_t
        assignment = (joint @ centers.t()).argmax(dim=1)
        keys: List[torch.Tensor] = []; gammas: List[torch.Tensor] = []; betas: List[torch.Tensor] = []; counts: List[float] = []
        within_q, within_a = [], []
        for j in range(self.K):
            mask = assignment == j
            if not mask.any():
                fallback_idx = int(strength.argmax().item())
                mask = torch.zeros_like(assignment, dtype=torch.bool); mask[fallback_idx] = True
            wj = strength[mask]; denom = wj.sum().clamp_min(1e-8)
            key = F.normalize((q[mask] * wj[:, None]).sum(dim=0) / denom, dim=0, eps=1e-6)
            gamma_j = (gamma[mask] * wj[:, None]).sum(dim=0) / denom
            beta_j = (beta[mask] * wj[:, None]).sum(dim=0) / denom
            keys.append(key); gammas.append(gamma_j); betas.append(beta_j); counts.append(float(mask.sum().item()))
            if mask.sum() > 1:
                within_q.append(_pairwise_cosine_mean(q_n[mask].cpu()))
                within_a.append(_pairwise_cosine_mean(action_n[mask].cpu()))
        device = self.keys.device
        self.keys.copy_(torch.stack(keys).to(device=device, dtype=self.keys.dtype))
        self.gamma_proto.copy_(torch.stack(gammas).to(device=device, dtype=self.gamma_proto.dtype))
        self.beta_proto.copy_(torch.stack(betas).to(device=device, dtype=self.beta_proto.dtype))
        self.slot_counts.copy_(torch.tensor(counts, device=device, dtype=self.slot_counts.dtype))
        self.memory_ready.fill_(True)
        rep = {
            'scale': self.scale_index, 'updated': True, 'candidate_count': int(n), 'slot_counts': counts, 'slot_count_min': float(min(counts)), 'slot_count_max': float(max(counts)),
            'slot_count_ratio_max_min': float(max(counts) / max(min(counts), 1e-8)),
            'key_pairwise_cosine_mean': _pairwise_cosine_mean(self.keys.detach().cpu()),
            'action_pairwise_cosine_mean': _pairwise_cosine_mean(F.normalize(torch.cat([self.gamma_proto, self.beta_proto], dim=1).detach().cpu(), dim=1)),
            'joint_center_pairwise_cosine_mean': _pairwise_cosine_mean(centers.detach().cpu()),
            'within_cluster_query_cosine_mean': float(np.mean(within_q)) if within_q else 0.0,
            'within_cluster_action_cosine_mean': float(np.mean(within_a)) if within_a else 0.0,
            'gamma_proto_abs_mean': _to_float(self.gamma_proto.abs().mean()),
            'beta_proto_abs_mean': _to_float(self.beta_proto.abs().mean()),
            'memory_ready': True,
            'query_raw_abs_mean': _to_float(q.abs().mean()),
            'query_centered_abs_mean': _to_float((q - q.mean(dim=1, keepdim=True)).abs().mean()),
            'query_spatial_variance': _to_float(q.var(dim=1, unbiased=False).mean()),
        }
        self.clear_cache(); return rep

    @staticmethod
    @torch.no_grad()
    def _deterministic_farthest_init(action_n: torch.Tensor, strength: torch.Tensor, K: int) -> torch.Tensor:
        first = int(strength.argmax().item())
        chosen = [first]; centers = [action_n[first]]
        while len(centers) < K:
            current = torch.stack(centers, dim=0)
            max_sim = (action_n @ current.t()).max(dim=1).values
            max_sim[torch.tensor(chosen, dtype=torch.long)] = 1.0
            next_idx = int(max_sim.argmin().item())
            chosen.append(next_idx); centers.append(action_n[next_idx])
        return torch.stack(centers, dim=0)

    def clear_cache(self):
        self._cache_q.clear(); self._cache_gamma.clear(); self._cache_beta.clear(); self._cache_strength.clear()

    def retrieve(self, query_map: torch.Tensor, capture_visuals: bool = False):
        b, _, h, w = query_map.shape
        if not bool(self.memory_ready.item()):
            gamma = query_map.new_zeros((b, self.channels, h, w)); beta = query_map.new_zeros((b, self.channels, h, w))
            info = {'memory_ready': torch.tensor(False, device=query_map.device), 'entropy': query_map.new_zeros((b, h, w)), 'effective_slots': query_map.new_zeros((b, h, w)), 'reliability': query_map.new_zeros((b, h, w)), 'max_similarity': query_map.new_zeros((b, h, w)), 'top1_weight': query_map.new_zeros((b, h, w)), 'top1_slot': torch.zeros((b, h, w), device=query_map.device, dtype=torch.long), 'raw_gamma': gamma, 'raw_beta': beta, 'safe_gamma': gamma, 'safe_beta': beta}
            return type('Ret', (), {'gamma': gamma, 'beta': beta, 'info': info})
        q = query_map.permute(0, 2, 3, 1).reshape(-1, self.QUERY_DIM)
        q = F.normalize(q, dim=1, eps=1e-6)
        keys = F.normalize(self.keys, dim=1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        similarity = q @ keys.t()
        weights = torch.softmax(similarity * scale, dim=1)
        gamma_raw = weights @ self.gamma_proto
        beta_raw = weights @ self.beta_proto
        entropy = _safe_entropy(weights, dim=1)
        eff = entropy.exp()
        if self.K > 1:
            reliability = ((self.K - eff) / (self.K - 1)).clamp(0.0, 1.0)
        else:
            reliability = torch.ones_like(eff)
        gamma_safe = reliability[:, None] * gamma_raw
        beta_safe = reliability[:, None] * beta_raw
        gamma = gamma_safe.reshape(b, h, w, self.channels).permute(0, 3, 1, 2)
        beta = beta_safe.reshape(b, h, w, self.channels).permute(0, 3, 1, 2)
        top1_weight, top1_slot = weights.max(dim=1)
        top1_weight = top1_weight.reshape(b, h, w); top1_slot = top1_slot.reshape(b, h, w)
        max_similarity = similarity.max(dim=1).values.reshape(b, h, w)
        entropy = entropy.reshape(b, h, w); eff = eff.reshape(b, h, w); reliability_map = reliability.reshape(b, h, w)
        with torch.no_grad():
            self._retrieval_slot_hits += torch.bincount(top1_slot.detach().cpu().reshape(-1), minlength=self.K)
        info = {
            'memory_ready': torch.tensor(True, device=query_map.device), 'entropy': entropy, 'normalized_entropy': entropy / max(float(math.log(self.K)), 1e-6), 'effective_slots': eff, 'reliability': reliability_map,
            'max_similarity': max_similarity, 'top1_weight': top1_weight, 'top1_slot': top1_slot, 'raw_gamma': gamma_raw.reshape(b, h, w, self.channels).permute(0, 3, 1, 2), 'raw_beta': beta_raw.reshape(b, h, w, self.channels).permute(0, 3, 1, 2), 'safe_gamma': gamma, 'safe_beta': beta,
            'raw_gamma_abs_mean': gamma_raw.abs().mean(), 'safe_gamma_abs_mean': gamma_safe.abs().mean(), 'raw_beta_abs_mean': beta_raw.abs().mean(), 'safe_beta_abs_mean': beta_safe.abs().mean(), 'reliability_suppression_ratio': (1.0 - reliability_map).mean(), 'low_reliability_ratio': (reliability_map < 0.1).float().mean(),
        }
        if capture_visuals: self._store_last_maps(info, gamma, beta)
        return type('Ret', (), {'gamma': gamma, 'beta': beta, 'info': info})

    @torch.no_grad()
    def _store_last_maps(self, info, gamma, beta):
        self._last_maps['top1_slot'] = info['top1_slot'][0].detach().cpu(); self._last_maps['entropy'] = info['entropy'][0].detach().cpu(); self._last_maps['reliability'] = info['reliability'][0].detach().cpu(); self._last_maps['effective_slots'] = info['effective_slots'][0].detach().cpu(); self._last_maps['raw_gamma_norm'] = info['raw_gamma'][0].float().norm(dim=0).detach().cpu(); self._last_maps['safe_gamma_norm'] = gamma[0].float().norm(dim=0).detach().cpu(); self._last_maps['raw_beta_norm'] = info['raw_beta'][0].float().norm(dim=0).detach().cpu(); self._last_maps['safe_beta_norm'] = beta[0].float().norm(dim=0).detach().cpu()

    def reset_epoch_stats(self):
        self._retrieval_slot_hits.zero_(); self._last_maps.clear()

    def diagnostics(self):
        total_hits = max(int(self._retrieval_slot_hits.sum().item()), 1)
        slot_hits = self._retrieval_slot_hits.tolist(); util = [float(v)/total_hits for v in slot_hits]
        return {'scale': self.scale_index, 'channels': self.channels, 'K': self.K, 'memory_ready': bool(self.memory_ready.item()), 'cache_count': sum(x.shape[0] for x in self._cache_strength), 'slot_counts_build': self.slot_counts.detach().cpu().tolist(), 'slot_usage': slot_hits, 'slot_utilization': util, 'active_slots_retrieval': int(sum(v > 0 for v in slot_hits)), 'key_pairwise_cosine_mean': _pairwise_cosine_mean(self.keys.detach().cpu()) if bool(self.memory_ready.item()) else 0.0, 'gamma_prototype': _tensor_stats(self.gamma_proto), 'beta_prototype': _tensor_stats(self.beta_proto)}


class PETReliableAffineActionMemory(nn.Module):
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512), K: int = 8) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.K = int(K)
        self.writers = nn.ModuleList([AffineActionWriter(c, relation_dim=ReliableAffineActionMemoryScale.QUERY_DIM) for c in self.channels])
        self.executors = nn.ModuleList([SharedAffineExecutor(c) for c in self.channels])
        self.memories = nn.ModuleList([ReliableAffineActionMemoryScale(c, K=self.K, scale_index=i + 1) for i, c in enumerate(self.channels)])
        self.current_epoch = 0
        self.route_counts = {'full': 0, 'missing': 0}
        self._last_forward: Dict[str, Any] = {}
        self._last_visual_maps: Dict[str, Dict[str, torch.Tensor]] = {}

    def begin_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.route_counts = {'full': 0, 'missing': 0}
        self._last_forward = {}
        self._last_visual_maps = {}
        for m in self.memories: m.reset_epoch_stats()

    def forward(self, ct_features, pet_features=None, route='full', update_memory=False, capture_visuals=False):
        fused_features: List[torch.Tensor] = []
        per_scale_info = []
        if route not in {'full', 'missing'}:
            raise ValueError(route)
        if route == 'full' and pet_features is None:
            raise ValueError('Full requires pet_features')
        for idx, (ct, writer, executor, memory) in enumerate(zip(ct_features, self.writers, self.executors, self.memories)):
            query, q_stats = memory.make_query(ct)
            gamma_star = beta_star = None; true_exec = None; write_report = None
            if route == 'full' and pet_features is not None:
                gamma_star, beta_star = writer(ct, pet_features[idx])
                _, true_exec = executor(ct, gamma_star, beta_star)
            elif self.training and pet_features is not None:
                with torch.no_grad():
                    gamma_star, beta_star = writer(ct.detach(), pet_features[idx].detach())
                    _, true_exec = executor(ct.detach(), gamma_star, beta_star)
                if update_memory:
                    write_report = memory.collect(query.detach(), gamma_star, beta_star, true_exec['gate'], query_stats=q_stats)
            if route == 'full':
                used_gamma, used_beta = gamma_star, beta_star
                retrieval_info = None
                used_source = 'current_real_pet_affine'
            else:
                retrieval = memory.retrieve(query, capture_visuals=capture_visuals)
                used_gamma, used_beta = retrieval.gamma, retrieval.beta
                retrieval_info = retrieval.info
                used_source = 'delayed_memory_retrieval'
            fused, exec_info = executor(ct, used_gamma, used_beta)
            fused_features.append(fused)
            summary = {
                'scale': idx + 1, 'route': route, 'used_affine_source': used_source, 'memory_ready': bool(memory.memory_ready.item()), 'ct_shape': list(ct.shape),
                'gate_mean': _to_float(exec_info['gate'].mean()), 'correction_abs_mean': _to_float(exec_info['correction'].abs().mean()), 'correction_to_ct_l2_ratio': _to_float(exec_info['correction_ratio'].mean()),
                'query_raw_abs_mean': _to_float(q_stats['query_raw_abs_mean']), 'query_centered_abs_mean': _to_float(q_stats['query_centered_abs_mean']), 'query_spatial_variance': _to_float(q_stats['query_spatial_variance']),
            }
            if gamma_star is not None:
                summary.update({'true_gamma_abs_mean': _to_float(gamma_star.abs().mean()), 'true_beta_abs_mean': _to_float(beta_star.abs().mean())})
            if write_report is not None: summary['memory_write'] = write_report
            if retrieval_info is not None:
                summary.update({k: _to_float(v.mean() if torch.is_tensor(v) and v.ndim > 0 else v) for k, v in retrieval_info.items() if k in {'entropy', 'normalized_entropy', 'effective_slots', 'reliability', 'raw_gamma_abs_mean', 'safe_gamma_abs_mean', 'raw_beta_abs_mean', 'safe_beta_abs_mean', 'reliability_suppression_ratio', 'low_reliability_ratio'}})
            per_scale_info.append(summary)
            if capture_visuals:
                visual = {'gate': exec_info['gate'][0, 0].detach().cpu(), 'correction_norm': exec_info['correction'][0].float().norm(dim=0).detach().cpu(), 'used_gamma_norm': used_gamma[0].float().norm(dim=0).detach().cpu(), 'used_beta_norm': used_beta[0].float().norm(dim=0).detach().cpu()}
                if retrieval_info is not None:
                    visual.update({'reliability': retrieval_info['reliability'][0].detach().cpu(), 'effective_slots': retrieval_info['effective_slots'][0].detach().cpu(), 'raw_gamma_norm': retrieval_info['raw_gamma'][0].float().norm(dim=0).detach().cpu(), 'safe_gamma_norm': retrieval_info['safe_gamma'][0].float().norm(dim=0).detach().cpu(), 'raw_beta_norm': retrieval_info['raw_beta'][0].float().norm(dim=0).detach().cpu(), 'safe_beta_norm': retrieval_info['safe_beta'][0].float().norm(dim=0).detach().cpu()})
                self._last_visual_maps[f's{idx + 1}'] = visual
        forward_info = {'epoch': self.current_epoch, 'route': route, 'K': self.K, 'leakage_guard': 'PASS: Missing 当前融合只使用上一轮冻结记忆。' if route == 'missing' else 'N/A: Full 使用当前真实 PET 仿射作用。', 'scales': per_scale_info}
        self._last_forward = forward_info
        return fused_features, forward_info

    @torch.no_grad()
    def finalize_epoch_memory(self):
        return {'epoch': self.current_epoch, 'K': self.K, 'scales': [m.finalize_memory() for m in self.memories]}

    def diagnostics(self):
        return {'epoch': self.current_epoch, 'K': self.K, 'scales': [m.diagnostics() for m in self.memories], 'last_forward': self._last_forward}

    def print_diagnostics(self):
        print('PAAM-Reliable diagnostics')
        for sd in self.diagnostics()['scales']:
            print(sd)

    def export_diagnostics(self, output_dir, epoch=None, split='val_missing'):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        epoch_id = self.current_epoch if epoch is None else int(epoch)
        stem = f'epoch_{epoch_id:03d}_{split}'
        json_path = out / f'{stem}_paam_reliable_diagnostics.json'
        with json_path.open('w', encoding='utf-8') as f:
            import json
            json.dump(_json_ready(self.diagnostics()), f, ensure_ascii=False, indent=2)
        if self._last_visual_maps:
            for scale_name, maps in self._last_visual_maps.items():
                for name, tensor in maps.items():
                    path = out / f'{stem}_{scale_name}_{name}.png'
                    self._save_map(tensor, path, f'{scale_name} {name}')
        return {'json': str(json_path)}

    @staticmethod
    def _save_map(tensor, path, title):
        arr = tensor.detach().float().cpu().numpy(); plt.figure(figsize=(6, 5)); plt.imshow(arr); plt.title(title); plt.axis('off'); plt.colorbar(); plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()
