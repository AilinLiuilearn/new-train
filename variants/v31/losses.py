"""Targets are computed AFTER synchronized image/GT augmentation, in collate workers."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
from scipy.ndimage import distance_transform_edt,binary_dilation
from utils.seg_losses import BCEDiceLoss


def targets(mask,pet_radius=3,distance_clip=16,boundary_width=3):
    y=np.asarray(mask,dtype=bool).squeeze()
    if y.ndim!=2:raise ValueError('Expected a 2D binary mask')
    yy,xx=np.ogrid[-pet_radius:pet_radius+1,-pet_radius:pet_radius+1]
    pet=binary_dilation(y,structure=xx*xx+yy*yy<=pet_radius*pet_radius)
    if not y.any():signed=np.full(y.shape,-distance_clip,dtype=np.float32);near=np.zeros_like(y)
    elif y.all():signed=np.full(y.shape,distance_clip,dtype=np.float32);near=np.zeros_like(y)
    else:
        signed=distance_transform_edt(y)-distance_transform_edt(~y)
        near=(~y)&(distance_transform_edt(~y)<=boundary_width)
    weight=1+4*(np.abs(signed)<=boundary_width)
    return {k:torch.from_numpy(np.asarray(v,dtype=np.float32)[None].copy()) for k,v in
            {'pet_target':pet,'ct_target':np.clip(signed/distance_clip,-1,1),'ct_weight':weight,'near_bg':near}.items()}


def collate(items):
    items=[dict(x,**targets(x['mask'].numpy()>.5)) for x in items]
    return default_collate(items)


def masked_mean(value,mask):
    total=mask.sum();return (value*mask).sum()/total.clamp_min(1),bool(total.detach()>0)


def balanced_gate(logits,error):
    losses=[]
    for z,y in zip(logits,error):
        v=F.binary_cross_entropy_with_logits(z,y.float(),reduction='none');terms=[]
        for m in [y,~y]:
            part,valid=masked_mean(v,m)
            if valid:terms.append(part)
        if terms:losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0


def protection(change,base,y,near):
    correct=(base.float().sigmoid()>.5)==(y>.5);losses=[]
    for v,c,t,n in zip(change.float().square(),correct,y>.5,near>.5):
        terms=[]
        for mask in [c&t,c&~t&n]:
            value,valid=masked_mean(v,mask)
            if valid:terms.append(value)
        if terms:losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean() if losses else change.sum()*0


def objective(o,batch,phase):
    y=batch['mask'].float();seg,_=BCEDiceLoss()(o['logits'].float(),y)
    if phase=='joint':
        pet,_=BCEDiceLoss()(o['pet_logits'].float(),batch['pet_target'])
        v=F.smooth_l1_loss(o['ct_distance'].float(),batch['ct_target'],reduction='none')
        w=batch['ct_weight'];ct=((v*w).flatten(1).sum(1)/w.flatten(1).sum(1)).mean()
        total=seg+.1*pet+.1*ct;parts={'seg':seg,'pet':pet,'ct':ct}
    else:
        error=(o['base_logits'].float().sigmoid()>.5)!=(y>.5)
        gate=balanced_gate(o['gate_logits'],error)
        keep=protection(o['change'],o['base_logits'],y,batch['near_bg'])
        total=seg+.1*gate+.1*keep;parts={'seg':seg,'gate':gate,'keep':keep}
    return total,{'total':float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
