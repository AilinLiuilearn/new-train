import torch
from utils.seg_losses import prepare_binary_target


def calibration_loss(ct_logits,full_logits,target,delta=.05):
    with torch.autocast(device_type=full_logits.device.type,enabled=False):
        y=prepare_binary_target(full_logits.float(),target)
        # Online CT reference; no backward path to CT prediction from this term.
        pc=ct_logits.detach().float().sigmoid();pf=full_logits.float().sigmoid()
        correct=pc.gt(.5).eq(y.bool())
        qc=y*pc+(1-y)*(1-pc);qf=y*pf+(1-y)*(1-pf)
        reference=(correct*qc).detach()
        hinge=(qc-qf-delta).clamp_min(0)
        terms=[];stats={}
        for name,region in [('fg',y),('bg',1-y)]:
            weight=reference*region
            term=(weight*hinge).sum()/(weight.sum()+1e-6)
            terms.append(term)
            stats['cal_'+name+'_pixels']=(weight>0).float().sum()
            stats['cal_'+name+'_active_pixels']=((weight>0)&(hinge.detach()>0)).float().sum()
            stats['cal_'+name+'_loss']=term.detach()
        loss=.5*(terms[0]+terms[1])
    return loss,stats


def objective(model,output,target,criterion):
    losses={};stats={}
    for name,key in [('full','logits'),('ct','ct_logits'),('pet','pet_logits')]:
        value,parts=criterion(output[key].float(),target)
        losses[name]=value
        stats[name+'_bce']=float(parts['loss_bce']);stats[name+'_dice_loss']=float(parts['loss_dice'])
        stats[name+'_seg']=float(value.detach())
    total=sum(losses.values())
    if model.cfg['calibration_enabled']:
        cal,parts=calibration_loss(output['ct_logits'],output['logits'],target,model.cfg['calibration_delta'])
        total=total+model.cfg['calibration_weight']*cal
        stats.update({k:float(v) for k,v in parts.items()});stats['calibration']=float(cal.detach())
    else:stats['calibration']=0.
    if model.cfg['semantic_enabled']:
        semantic,parts=model.semantic(output['semantic_features'],target)
        total=total+model.cfg['semantic_weight']*semantic
        stats.update({k:float(v) for k,v in parts.items()});stats['semantic']=float(semantic.detach())
    else:stats['semantic']=0.
    stats.update(total=float(total.detach()),bce=stats['full_bce'],dice_loss=stats['full_dice_loss'])
    return total,stats
