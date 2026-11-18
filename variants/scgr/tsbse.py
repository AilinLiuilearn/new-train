"""Detached Euclidean geometry with bounded weights and target-budgeted tail risk."""
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F


class TSBSE(nn.Module):
    def __init__(self,coverage_scale='global',coverage_reference='predicted_foreground',
                 spill_candidates='background',cover_allocation='distance'):
        super().__init__()
        if coverage_scale not in ('global','gt_local_thickness'):
            raise ValueError('Unknown coverage scale')
        self.coverage_scale=coverage_scale
        if coverage_reference not in ('predicted_foreground', 'true_positive'):
            raise ValueError('Unknown coverage reference')
        self.coverage_reference = coverage_reference
        if spill_candidates not in ('background', 'false_positive'):
            raise ValueError('Unknown spill candidate set')
        self.spill_candidates = spill_candidates
        if cover_allocation not in ('distance', 'pooled_foreground'):
            raise ValueError('Unknown Cover allocation')
        if cover_allocation == 'pooled_foreground' and coverage_reference != 'true_positive':
            raise ValueError('Pooled Cover requires the true-positive reference')
        self.cover_allocation = cover_allocation

    def forward(self, logits, target):
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('Expected equally shaped B1HW logits and target')
        begin=time.perf_counter()
        with torch.autocast(device_type=logits.device.type,enabled=False):
            z=logits.float(); y=target.detach()>0.5
            # Only weights and foreground-set construction are detached.
            with torch.no_grad():
                gt=y[:,0].cpu().numpy()
                use_true_positives = self.coverage_reference == 'true_positive'
                detached_logits = z.detach()[:, 0]
                pooled_cover = self.cover_allocation == 'pooled_foreground'
                if pooled_cover:
                    spatial_logits = detached_logits.cpu().numpy()
                    pred = spatial_logits > 0
                    slice_cover_mass = np.zeros(len(gt), dtype=np.float64)
                else:
                    pred = (detached_logits > 0 if use_true_positives else detached_logits >= 0).cpu().numpy()
                wg=np.zeros(gt.shape,dtype=np.float32);wp=np.zeros_like(wg)
                areas=[];ks=[];radii=[];fn_count=0;fn_weight_sum=0.;cover_scale_sum=0.;gt_count=0
                for i,(g,p) in enumerate(zip(gt,pred)):
                    area=int(g.sum());nbg=g.size-area
                    areas.append(area);ks.append(min(area,nbg))
                    r=max(1.,math.sqrt(area/math.pi));radii.append(r)
                    if not area:continue
                    dg=distance_transform_edt(~g)
                    reference = g & p if use_true_positives else p
                    if reference.any():
                        dp = distance_transform_edt(~reference)
                    else:
                        dp = np.full(g.shape, math.hypot(*g.shape))
                    cover_radius=r
                    fn = g & ~p
                    if self.coverage_scale=='gt_local_thickness':
                        inside=distance_transform_edt(np.pad(g,1))[1:-1,1:-1]
                        cover_radius=np.maximum(1.,inside)
                    wg[i]=dg/(dg+r);wp[i]=dp/(dp+cover_radius)
                    if pooled_cover and fn.any():
                        # Absolute derivative of the parent's per-slice Cover mean.
                        error_derivative = expit(-spatial_logits[i][fn].astype(np.float64))
                        slice_cover_mass[i] = np.dot(wp[i][fn], error_derivative) / area
                    fn_count+=int(fn.sum());fn_weight_sum+=float(wp[i][fn].sum())
                    gt_count+=area
                    cover_scale_sum+=float(np.asarray(cover_radius)[g].sum()) if np.ndim(cover_radius) else float(cover_radius*area)
                if pooled_cover:
                    # Cancel each slice's separate area denominator, then match
                    # the parent's total Cover logit-gradient L1 in this batch.
                    area_array = np.asarray(areas, dtype=np.float64)
                    pooled_mass = np.dot(area_array, slice_cover_mass)
                    if pooled_mass > 0:
                        scales = area_array * (slice_cover_mass.sum() / pooled_mass)
                        wp *= scales[:, None, None]
                        fn_weight_sum = float(wp.sum(dtype=np.float64))
                wg=torch.from_numpy(wg).to(z.device);wp=torch.from_numpy(wp).to(z.device)
            cover=[];spill=[]
            spill_selected = 0
            for i,(area,k) in enumerate(zip(areas,ks)):
                zero=z[i].sum()*0.
                if not area:
                    cover.append(zero);spill.append(zero);continue
                mask=y[i,0]
                cover.append((wp[i][mask]*F.softplus(-z[i,0][mask])).mean())
                if k:
                    spill_mask = ~mask
                    if self.spill_candidates == 'false_positive':
                        spill_mask = spill_mask & (z[i, 0].detach() > 0)
                    spill_logits = z[i, 0][spill_mask]
                    k = min(k, spill_logits.numel())
                    if k:
                        risks = wg[i][spill_mask] * F.softplus(spill_logits)
                        spill.append(torch.topk(risks, k, sorted=False).values.mean())
                        spill_selected += k
                        continue
                spill.append(zero)
            c=torch.stack(cover).mean();s=torch.stack(spill).mean()
            loss=.5*(c+s)
            stats={'cover':float(c.detach()),'spill':float(s.detach()),
                'valid':sum(a>0 for a in areas),'empty':sum(a==0 for a in areas),
                'k_mean':float(np.mean(ks)),'area_min':min(areas),'area_mean':float(np.mean(areas)),
                'spill_selected_mean':spill_selected / len(areas),
                'radius_mean':float(np.mean(radii)),
                'cover_local_scale':float(self.coverage_scale=='gt_local_thickness'),
                'cover_FN_count':fn_count,'cover_FN_weight_sum':fn_weight_sum,
                'cover_mean_FN_weight':fn_weight_sum/max(1,fn_count),
                'cover_scale_GT_mean':cover_scale_sum/max(1,gt_count),'wall_seconds':time.perf_counter()-begin}
            return loss,stats,{'cover':c,'spill':s}
