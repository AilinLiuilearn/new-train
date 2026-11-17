"""Detached Euclidean geometry with bounded weights and target-budgeted tail risk."""
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F


class TSBSE(nn.Module):
    def __init__(self,coverage_scale='global'):
        super().__init__()
        if coverage_scale not in ('global','gt_local_thickness'):
            raise ValueError('Unknown coverage scale')
        self.coverage_scale=coverage_scale

    def forward(self, logits, target):
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('Expected equally shaped B1HW logits and target')
        begin=time.perf_counter()
        with torch.autocast(device_type=logits.device.type,enabled=False):
            z=logits.float(); y=target.detach()>0.5
            # Only weights and foreground-set construction are detached.
            with torch.no_grad():
                gt=y[:,0].cpu().numpy()
                pred=(z.detach()[:,0]>=0).cpu().numpy()
                wg=np.zeros(gt.shape,dtype=np.float32);wp=np.zeros_like(wg)
                areas=[];ks=[];radii=[];fn_count=0;fn_weight_sum=0.;cover_scale_sum=0.;gt_count=0
                for i,(g,p) in enumerate(zip(gt,pred)):
                    area=int(g.sum());nbg=g.size-area
                    areas.append(area);ks.append(min(area,nbg))
                    r=max(1.,math.sqrt(area/math.pi));radii.append(r)
                    if not area:continue
                    dg=distance_transform_edt(~g)
                    dp=distance_transform_edt(~p) if p.any() else np.full(g.shape,math.hypot(*g.shape))
                    cover_radius=r
                    if self.coverage_scale=='gt_local_thickness':
                        inside=distance_transform_edt(np.pad(g,1))[1:-1,1:-1]
                        cover_radius=np.maximum(1.,inside)
                    wg[i]=dg/(dg+r);wp[i]=dp/(dp+cover_radius)
                    fn=g&~p;fn_count+=int(fn.sum());fn_weight_sum+=float(wp[i][fn].sum())
                    gt_count+=area
                    cover_scale_sum+=float(np.asarray(cover_radius)[g].sum()) if np.ndim(cover_radius) else float(cover_radius*area)
                wg=torch.from_numpy(wg).to(z.device);wp=torch.from_numpy(wp).to(z.device)
            cover=[];spill=[]
            for i,(area,k) in enumerate(zip(areas,ks)):
                zero=z[i].sum()*0.
                if not area:
                    cover.append(zero);spill.append(zero);continue
                mask=y[i,0]
                cover.append((wp[i][mask]*F.softplus(-z[i,0][mask])).mean())
                if k:
                    risks=wg[i][~mask]*F.softplus(z[i,0][~mask])
                    spill.append(torch.topk(risks,k,sorted=False).values.mean())
                else:spill.append(zero)
            c=torch.stack(cover).mean();s=torch.stack(spill).mean()
            loss=.5*(c+s)
            stats={'cover':float(c.detach()),'spill':float(s.detach()),
                'valid':sum(a>0 for a in areas),'empty':sum(a==0 for a in areas),
                'k_mean':float(np.mean(ks)),'area_min':min(areas),'area_mean':float(np.mean(areas)),
                'radius_mean':float(np.mean(radii)),
                'cover_local_scale':float(self.coverage_scale=='gt_local_thickness'),
                'cover_FN_count':fn_count,'cover_FN_weight_sum':fn_weight_sum,
                'cover_mean_FN_weight':fn_weight_sum/max(1,fn_count),
                'cover_scale_GT_mean':cover_scale_sum/max(1,gt_count),'wall_seconds':time.perf_counter()-begin}
            return loss,stats,{'cover':c,'spill':s}
