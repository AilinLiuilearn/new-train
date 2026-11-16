"""Soft occupancy supervision. Never call the base binary-target helper here."""
import torch
from torch.nn import functional as F

EPS = 1e-6


def squared_dice_loss(probability, target, nonempty):
    axes = (1, 2, 3)
    score = (2 * (probability * target).sum(axes) + EPS) / (
        probability.square().sum(axes) + target.square().sum(axes) + EPS)
    return (1 - score) * nonempty.float()


def localization_loss(logits, target):
    if set(logits) != {'ct', 'pet'} or any(len(x) != 4 for x in logits.values()):
        raise ValueError('Eight localization heads required')
    if target.ndim == 3:
        target = target.unsqueeze(1)
    details, terms = {}, []
    with torch.autocast(device_type=target.device.type, enabled=False):
        target = target.float()
        nonempty = target.flatten(1).sum(1) > 0
        for modality in ['ct', 'pet']:
            for scale, raw in enumerate(logits[modality], 1):
                z = raw.float()
                y = F.interpolate(target, size=z.shape[-2:], mode='area')
                bce = F.binary_cross_entropy_with_logits(z, y, reduction='none').mean((1, 2, 3))
                dice = squared_dice_loss(z.sigmoid(), y, nonempty)
                terms.append((bce + dice).mean())
                details[f'{modality}_s{scale}_loc_bce'] = bce.detach().mean()
                details[f'{modality}_s{scale}_loc_dice'] = dice.detach().mean()
        total = torch.stack(terms).mean()
        details['loc_bce'] = torch.stack([v for k, v in details.items() if k.endswith('_bce')]).mean()
        details['loc_dice'] = torch.stack([v for k, v in details.items() if k.endswith('_dice')]).mean()
        details['loc_nonempty_fraction'] = nonempty.float().mean()
        details['loc_total'] = total.detach()
    return total, details
