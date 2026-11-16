import torch
from torch.nn import functional as F


def score_loss(scores,target):
    if len(scores)!=4:raise ValueError('Expected four pairs of native-grid support logits')
    losses=[];stats={}
    for s,pair in enumerate(scores,1):
        for name,z in zip(('local','context'),pair):
            y=F.interpolate(target.float(),size=z.shape[-2:],mode='area')
            loss=F.binary_cross_entropy_with_logits(z.float(),y)
            losses.append(loss)
            stats[f's{s}_{name}_bce']=float(loss.detach())
            stats[f's{s}_{name}_target_fraction']=float(y.mean())
            with torch.no_grad():
                probability=z.detach().float().sigmoid()
                stats[f's{s}_{name}_foreground_score']=float((probability*y).sum()/y.sum().clamp_min(1e-8))
                stats[f's{s}_{name}_background_score']=float((probability*(1-y)).sum()/(1-y).sum().clamp_min(1e-8))
                stats[f's{s}_{name}_foreground_valid']=float((y.flatten(1).sum(1)>0).float().mean())
    return torch.stack(losses).mean(),stats


def objective(model,output,target,criterion):
    primary,parts=criterion(output['logits'].float(),target)
    auxiliary,stats=score_loss(output['support_logits'],target)
    enabled=model.cfg['score_enabled']
    total=primary+model.cfg['score_weight']*auxiliary if enabled else primary
    return total,{'total':float(total.detach()),'bce':float(parts['loss_bce']),
        'dice_loss':float(parts['loss_dice']),'segmentation':float(primary.detach()),
        'score_bce_diagnostic':float(auxiliary.detach()),
        'score_weighted':float((model.cfg['score_weight']*auxiliary).detach()) if enabled else 0.,**stats}
