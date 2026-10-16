"""Trainable V=I+AB values; CT keys are buffers with stable slot identities.

Full paired ridge initializes values once. Afterwards only Missing segmentation
backprop updates A/B; key refresh never rewrites parameters or optimizer state.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.ct_full_matrix import MatrixBank


class LowRankValueBank(MatrixBank):
    def __init__(self, channels, value_rank=8, value_init='full', **kwargs):
        super().__init__(channels, **kwargs)
        if not 1 <= value_rank <= channels or value_init not in ('full','identity'):
            raise ValueError('require 1<=rank<=channels and value_init full/identity')
        self.value_rank, self.value_init = value_rank, value_init
        self.value_a = nn.Parameter(torch.zeros(self.clusters,channels,value_rank))
        generator=torch.Generator().manual_seed(1729+channels)
        b=torch.randn(self.clusters,value_rank,channels,generator=generator)/channels**.5
        self.value_b = nn.Parameter(b)

    def effective_transforms(self):
        eye=torch.eye(self.channels,device=self.value_a.device,dtype=self.value_a.dtype)
        return eye[None]+self.value_a@self.value_b

    def apply_value(self, query, slot):
        return query+(query@self.value_b[slot].T)@self.value_a[slot].T

    @torch.no_grad()
    def refresh(self):
        if not bool(self.valid):
            if not super().refresh():return False
            if self.value_init=='full':
                try:
                    with torch.autocast(device_type=self.keys.device.type,enabled=False):
                        residual=self.transforms.float()-torch.eye(self.channels,device=self.keys.device)
                        u,s,vh=torch.linalg.svd(residual,full_matrices=False)
                        root=s[:,:self.value_rank].clamp_min(0).sqrt()
                        a=u[:,:,:self.value_rank]*root[:,None,:]
                        b=root[:,:,None]*vh[:,:self.value_rank,:]
                        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise RuntimeError('nonfinite SVD')
                        self.value_a.copy_(a);self.value_b.copy_(b)
                except RuntimeError:
                    self.valid.fill_(False);self.updates.sub_(1);self.refresh_failures.add_(1)
                    self.refresh_stats={'status':'value_initialization_failed'}
                    return False
            self.transforms.copy_(self.effective_transforms())
            self._value_statistics()
            return True
        # Assign directly to current slots. No reclustering/permutation/reinitialization.
        self.refresh_attempts.add_(1)
        n=int(self.size)
        if n<self.min_candidates:
            self.refresh_failures.add_(1);self.refresh_stats={'status':'insufficient_candidates'};return False
        x=self.pool_ct[:n].float();w=self.pool_weight[:n].float()
        unit=F.normalize(x,dim=1);assignment=(unit@self.keys.T).argmax(1)
        support=torch.bincount(assignment,minlength=self.clusters)
        for j in range(self.clusters):
            ids=assignment==j
            if int(support[j])<self.min_cluster_samples:continue
            direction=F.normalize((unit[ids]*w[ids,None]).sum(0),dim=0)
            self.keys[j].copy_(F.normalize(self.decay*self.keys[j]+(1-self.decay)*direction,dim=0))
        current=self.effective_transforms()
        self.matrix_drift.copy_((current-self.transforms).flatten(1).norm(dim=1))
        self.transforms.copy_(current)  # diagnostic snapshot only, never used for reading
        self.support.copy_(support);self._value_statistics(assignment)
        self.updates.add_(1);self.refresh_stats={'status':'keys_updated_values_gradient_only'}
        return True

    @torch.no_grad()
    def _value_statistics(self, assignment=None):
        n=int(self.size);x=self.pool_ct[:n].float();y=self.pool_full[:n].float();w=self.pool_weight[:n].float()
        if assignment is None:assignment=(F.normalize(x,dim=1)@self.keys.T).argmax(1)
        for j in range(self.clusters):
            ids=assignment==j
            if not ids.any():
                # support=0 marks absent data; retain last defined errors.
                continue
            ww=w[ids];pred=self.apply_value(x[ids],j)
            self.fit_error[j]=((pred-y[ids]).square().mean(1)*ww).sum()/ww.sum()
            self.identity_error[j]=((x[ids]-y[ids]).square().mean(1)*ww).sum()/ww.sum()
