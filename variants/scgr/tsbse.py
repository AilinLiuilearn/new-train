"""Detached Euclidean geometry with bounded weights and target-budgeted tail risk."""
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F


class TSBSE(nn.Module):
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
                areas=[];ks=[];radii=[]
                for i,(g,p) in enumerate(zip(gt,pred)):
                    area=int(g.sum());nbg=g.size-area
                    areas.append(area);ks.append(min(area,nbg))
                    r=max(1.,math.sqrt(area/math.pi));radii.append(r)
                    if not area:continue
                    dg=distance_transform_edt(~g)
                    dp=distance_transform_edt(~p) if p.any() else np.full(g.shape,math.hypot(*g.shape))
                    wg[i]=dg/(dg+r);wp[i]=dp/(dp+r)
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
                'radius_mean':float(np.mean(radii)),'wall_seconds':time.perf_counter()-begin}
            return loss,stats,{'cover':c,'spill':s}
