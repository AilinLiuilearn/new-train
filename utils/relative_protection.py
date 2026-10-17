"""Single-model detached Add comparison and class-balanced correction loss.

No external weights or second model. The Add logits come from the same
current network and are detached. GT is used only in training. This is
our adaptation, NOT PNDC's native Reverse Audit/Forward Checksum algorithm.
"""
import torch
from torch.nn import functional as F


def relative_protection_loss(logits, reference_logits, target):
    if (logits.shape != reference_logits.shape or logits.shape != target.shape
            or logits.ndim != 4 or logits.shape[1] != 1
            or logits.device != reference_logits.device or logits.device != target.device):
        raise ValueError('Protection requires matching B,1,H,W logits/reference/binary target on one device')
    if (not torch.isfinite(logits).all() or not torch.isfinite(reference_logits).all()
            or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all()):
        raise ValueError('Protection requires finite logits and binary GT')
    with torch.autocast(device_type=logits.device.type,enabled=False):
        ref = reference_logits.detach().float()
        labels = target.detach().float()
        correct = (ref >= 0).eq(labels.bool())
        excess = (F.binary_cross_entropy_with_logits(logits.float(),labels,reduction='none')
                  - F.binary_cross_entropy_with_logits(ref,labels,reduction='none')).relu()
        terms,stats = [],{}
        for cls,name in ((1,'fg'),(0,'bg')):
            selected = correct & (labels == cls)
            count = selected.sum()
            stats[f'protect_correct_{name}'] = count.detach()
            # Empty classes contribute zero and do not halve a nonempty class.
            if bool(count):
                terms.append((excess*selected).sum()/count)
        loss = torch.stack(terms).mean() if terms else logits.float().sum()*0
    return loss,stats
