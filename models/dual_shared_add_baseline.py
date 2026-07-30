import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe


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


@dataclass
class _PDTMRecord:
    case_id: str
    source_mean: torch.Tensor
    source_covariance: torch.Tensor
    target_mean: torch.Tensor
    target_covariance: torch.Tensor
    delta_mean: torch.Tensor
    operator: torch.Tensor
    paired_w2: torch.Tensor


class PDTMRuntime(nn.Module):
    def __init__(self, channels, slots=8, eps=1e-4):
        super().__init__()
        self.channels = int(channels)
        self.slots = int(slots)
        self.eps = float(eps)
        eye = torch.eye(self.channels, dtype=torch.float32)
        self.register_buffer('memory_ready', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('valid_slots', torch.tensor(0, dtype=torch.long))
        self.register_buffer('source_means', torch.zeros(self.slots, self.channels))
        self.register_buffer('source_covariances', eye.unsqueeze(0).repeat(self.slots, 1, 1))
        self.register_buffer('delta_means', torch.zeros(self.slots, self.channels))
        self.register_buffer('operators', eye.unsqueeze(0).repeat(self.slots, 1, 1))
        self.register_buffer('paired_w2', torch.zeros(self.slots))
        self.register_buffer('cluster_sizes', torch.zeros(self.slots, dtype=torch.long))
        self._retrieval_stats = []

    def reset_retrieval_stats(self):
        self._retrieval_stats = []

    def _sym(self, x):
        return 0.5 * (x + x.transpose(-1, -2))

    def _sqrtm(self, x):
        x = self._sym(x.float())
        e, v = torch.linalg.eigh(x)
        e = e.clamp_min(self.eps)
        return v @ torch.diag_embed(torch.sqrt(e)) @ v.transpose(-1, -2)

    def _bw2(self, m0, s0, m1, s1):
        dm = (m0 - m1).pow(2).sum(dim=-1)
        s0 = self._sym(s0.float()); s1 = self._sym(s1.float())
        s0_root = self._sqrtm(s0)
        mid = s0_root @ s1 @ s0_root
        mid_root = self._sqrtm(mid)
        tr = torch.diagonal(s0, dim1=-2, dim2=-1).sum(-1) + torch.diagonal(s1, dim1=-2, dim2=-1).sum(-1) - 2 * torch.diagonal(mid_root, dim1=-2, dim2=-1).sum(-1)
        return dm + tr

    def _feat_stats(self, feat):
        x = feat.flatten(2)
        m = x.mean(dim=-1)
        xc = x - m.unsqueeze(-1)
        cov = xc @ xc.transpose(-1, -2) / max(1, xc.shape[-1]) + self.eps * torch.eye(feat.shape[1], device=feat.device, dtype=feat.dtype).unsqueeze(0)
        return m, self._sym(cov)

    def forward(self, feat):
        if not bool(self.memory_ready.item()) or int(self.valid_slots.item()) <= 0:
            return feat, {'pdtm_memory_ready': False, 'pdtm_selected_slot_mean': -1, 'pdtm_nearest_distance_mean': 0.0, 'pdtm_feature_change_ratio': 0.0}
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                mean, cov = self._feat_stats(feat.float())
                valid = int(self.valid_slots.item())
                src_mean = self.source_means[:valid].float()
                src_cov = self.source_covariances[:valid].float()
                out = feat.clone()
                selected = []
                dists = []
                ratios = []
                for i in range(feat.shape[0]):
                    w2 = self._bw2(mean[i].unsqueeze(0).expand(valid, -1), cov[i].unsqueeze(0).expand(valid, -1, -1), src_mean, src_cov)
                    idx = int(torch.argmin(w2).item())
                    selected.append(idx)
                    dists.append(float(w2[idx].item()))
                    m = feat[i].mean(dim=(1, 2), keepdim=True)
                    centered = feat[i] - m
                    delta = self.delta_means[idx].to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1)
                    op = self.operators[idx].to(device=feat.device, dtype=feat.dtype)
                    out[i] = m + delta + torch.einsum('ij,jhw->ihw', op, centered)
                    ratios.append(float((out[i] - feat[i]).flatten().norm() / feat[i].flatten().norm().clamp_min(self.eps)))
                self._retrieval_stats.extend([{'selected_slot': s, 'nearest_distance': d, 'feature_change_ratio': r} for s, d, r in zip(selected, dists, ratios)])
                info = {'pdtm_memory_ready': True, 'pdtm_selected_slot_mean': float(sum(selected)/len(selected)) if selected else -1, 'pdtm_nearest_distance_mean': float(sum(dists)/len(dists)) if dists else 0.0, 'pdtm_feature_change_ratio': float(sum(ratios)/len(ratios)) if ratios else 0.0}
        return out.to(dtype=feat.dtype), info

    @torch.no_grad()
    def load_memory(self, source_means, source_covariances, delta_means, operators, paired_w2=None, cluster_sizes=None):
        n = min(self.slots, source_means.shape[0])
        eye = torch.eye(self.channels, device=self.source_covariances.device, dtype=self.source_covariances.dtype)
        self.source_means.zero_(); self.delta_means.zero_(); self.paired_w2.zero_(); self.cluster_sizes.zero_()
        self.source_covariances.copy_(eye.unsqueeze(0).repeat(self.slots, 1, 1)); self.operators.copy_(eye.unsqueeze(0).repeat(self.slots, 1, 1))
        self.source_means[:n].copy_(source_means[:n].to(self.source_means.device, dtype=torch.float32))
        self.source_covariances[:n].copy_(source_covariances[:n].to(self.source_covariances.device, dtype=torch.float32))
        self.delta_means[:n].copy_(delta_means[:n].to(self.delta_means.device, dtype=torch.float32))
        self.operators[:n].copy_(operators[:n].to(self.operators.device, dtype=torch.float32))
        if paired_w2 is not None:
            self.paired_w2[:n].copy_(paired_w2[:n].to(self.paired_w2.device, dtype=torch.float32))
        if cluster_sizes is not None:
            self.cluster_sizes[:n].copy_(cluster_sizes[:n].to(self.cluster_sizes.device, dtype=torch.long))
        self.valid_slots.fill_(n); self.memory_ready.fill_(n > 0)

    def diagnostics(self):
        valid = int(self.valid_slots.item())
        return {'memory_ready': bool(self.memory_ready.item()), 'valid_slots': valid, 'slot_histogram': [], 'nearest_distance_mean': float(sum(r['nearest_distance'] for r in self._retrieval_stats) / max(1, len(self._retrieval_stats))), 'retrieval_margin_mean': 0.0, 'feature_change_ratio_mean': float(sum(r['feature_change_ratio'] for r in self._retrieval_stats) / max(1, len(self._retrieval_stats))), 'delta_mean_norm_per_slot': self.delta_means[:valid].float().norm(dim=1).detach().cpu().tolist(), 'operator_frobenius_norm_per_slot': self.operators[:valid].float().flatten(1).norm(dim=1).detach().cpu().tolist(), 'source_covariance_trace_per_slot': self.source_covariances[:valid].float().diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu().tolist(), 'cluster_sizes': self.cluster_sizes[:valid].detach().cpu().tolist(), 'paired_w2_per_slot': self.paired_w2[:valid].detach().cpu().tolist()}


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, pdtm_slots=8, pdtm_eps=1e-4, pdtm_max_pairs=256):
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
        self.pdtm = PDTMRuntime(channels=decoder_channels[-1], slots=pdtm_slots, eps=pdtm_eps)
        self.pdtm_max_pairs = int(pdtm_max_pairs)
        self._pdtm_pairs = []
        self._pdtm_examples = []

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

    def _forward_full(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused_feats = self.fusion(ct_feats, pet_feats, None)
        return self._decode(fused_feats, target_size)

    def _forward_missing(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        if pet is None:
            pet_feats_real = [torch.zeros_like(feat) for feat in ct_feats]
        else:
            pet_feats_real = self._encode_pet(pet)
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw = self.decoder(fused_feats, target_size, return_intermediates=True)
        d1_hat, info = self.pdtm(raw['decoder_feature'])
        logits = self.decoder.seg_head(d1_hat)
        out = {'logits': F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False), 'pred': None, 'aux': info}
        out['pred'] = out['logits']
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        pet_feats_masked = []
        for feat in pet_feats_real:
            availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
            pet_feats_masked.append(feat * availability_mask)
        fused_feats = self.fusion(ct_feats, pet_feats_masked, None)
        raw = self.decoder(fused_feats, target_size, return_intermediates=True)
        d1 = raw['decoder_feature']
        d1_final = d1.clone()
        missing_idx = pet_available == 0
        aux = {'pdtm_memory_ready': bool(self.pdtm.memory_ready.item()), 'pdtm_selected_slot_mean': -1, 'pdtm_nearest_distance_mean': 0.0, 'pdtm_feature_change_ratio': 0.0}
        if bool(self.pdtm.memory_ready.item()) and missing_idx.any():
            d1_final[missing_idx], aux = self.pdtm(d1[missing_idx])
        native_logits = self.decoder.seg_head(d1_final)
        final_logits = F.interpolate(native_logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': final_logits, 'pred': final_logits, 'aux': aux}

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full': return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing': return self._forward_missing(ct, pet, target_size)
        if forward_mode == 'auto':
            if pet_available is None: pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')

    @torch.no_grad()
    def collect_pdtm_pairs(self, ct, pet, case_ids=None):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full = self.decoder(self.fusion(ct_feats, pet_feats, None), ct.shape[-2:], return_intermediates=True)
        ct_only = self.decoder(self.fusion(ct_feats, [torch.zeros_like(f) for f in pet_feats], None), ct.shape[-2:], return_intermediates=True)
        for i in range(ct.shape[0]):
            if len(self._pdtm_pairs) >= self.pdtm_max_pairs: break
            c = ct_only['decoder_feature'].shape[1]
            src = ct_only['decoder_feature'][i].mean(dim=(1,2)).detach().cpu()
            tgt = full['decoder_feature'][i].mean(dim=(1,2)).detach().cpu()
            self._pdtm_pairs.append(_PDTMRecord(case_ids[i] if case_ids is not None else str(i), src, torch.eye(c), tgt, torch.eye(c), (tgt-src), torch.eye(c), torch.tensor(0.0)))
            if len(self._pdtm_examples) == 0: self._pdtm_examples = [ct_only['decoder_feature'][i:i+1].detach().cpu(), full['decoder_feature'][i:i+1].detach().cpu()]

    @torch.no_grad()
    def finalize_pdtm_memory(self):
        if not self._pdtm_pairs:
            self.pdtm.memory_ready.fill_(False); self.pdtm.valid_slots.zero_(); return {'memory_ready': False, 'valid_slots': 0}
        n = min(self.pdtm.slots, len(self._pdtm_pairs)); c = self.pdtm.channels
        means = torch.stack([x.source_mean for x in self._pdtm_pairs[:n]]).float(); covs = torch.stack([x.source_covariance for x in self._pdtm_pairs[:n]]).float().reshape(n,c,c)
        delta = torch.stack([x.delta_mean for x in self._pdtm_pairs[:n]]).float(); ops = torch.stack([x.operator for x in self._pdtm_pairs[:n]]).float().reshape(n,c,c)
        w2 = torch.stack([x.paired_w2 for x in self._pdtm_pairs[:n]]).float(); sizes = torch.ones(n, dtype=torch.long)
        self.pdtm.load_memory(means, covs, delta, ops, w2, sizes)
        return {'memory_ready': True, 'valid_slots': n}

    def pdtm_diagnostics(self):
        return self.pdtm.diagnostics()

    def export_pdtm_json(self, output_dir, tag):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{tag}.json')
        with open(path, 'w') as f:
            json.dump(self.pdtm.diagnostics(), f, indent=2)
        return path

    def save_pdtm_visualizations(self, output_dir, tag):
        os.makedirs(output_dir, exist_ok=True)
        return []
