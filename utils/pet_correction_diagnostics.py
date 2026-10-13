"""Sparse, detached correction diagnostics; never alter retrieval or write memory.

Source reliability is NOT a calibrated probability of query correctness.
Coverage means enough usable entries, NOT a correct anatomical/lesion match.
Foreground metadata refers to source training cells; it never routes retrieval.
"""
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F


def scalar_tree(value):
    if isinstance(value,torch.Tensor):
        if value.numel()!=1:
            raise ValueError('JSON diagnostics require scalar tensors, not feature maps')
        value=value.detach().cpu().item()
    if isinstance(value,dict):
        return {str(k):scalar_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [scalar_tree(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def queue_snapshot(model):
    result={}
    for s,retriever in enumerate(model.retrievers):
        bank=retriever.memory
        available=bank.allowed(-1,-1,int(model.memory_step))
        # Transfer only small diagnostic buffers; no full key/value CPU copies.
        rel=bank.reliability[available].detach().float().cpu()
        known=bank.audit_known[available].detach().cpu()
        fg=bank.foreground[available].detach().cpu()
        gain=bank.source_gain[available].detach().float().cpu()
        age=(model.memory_step-bank.written_at[available]).detach().float().cpu()
        ids=bank.sample_ids[available].detach().cpu()
        result[s]={'stored':int(bank.valid.sum()),'available':rel.numel(),
            'unique_slices':int(ids.unique().numel()),'audit_known':int(known.sum()),
            'foreground_fraction':float(fg[known].float().mean()) if known.any() else None,
            'source_gain':float(gain[known].mean()) if known.any() else None,
            'source_rel_mean':float(rel.mean()) if rel.numel() else None,
            'source_rel_p10':float(rel.quantile(.1)) if rel.numel() else None,
            'source_rel_p90':float(rel.quantile(.9)) if rel.numel() else None,
            'age_mean':float(age.mean()) if age.numel() else None,
            'value_rms':float(bank.values[available].detach().float().square().mean().sqrt())
                        if rel.numel() else None}
    return result


def report_memory(task,diagnostics,path,epoch,batch,committed=None):
    record=scalar_tree({'epoch':epoch,'batch':batch,'memory_step':int(task.model.memory_step),
        'bank_ema_active':task.bank_ema_active,
        'bank_ema_updates':task.bank_ema.updates if task.bank_ema else 0,
        'eval_weights':task.eval_weights_tag(),'retrieval_active':bool(task.model.retrieval_active),
        'committed_tokens':committed,'queue':queue_snapshot(task.model),
        'sampled_batch':diagnostics})
    with open(path,'a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
    def fmt(x):
        return 'n/a' if x is None else f'{x:.4f}'
    for s,q in record['queue'].items():
        candidate=(record['sampled_batch'] or {}).get('candidates',{}).get(s,{})
        written='n/a' if committed is None else committed[int(s)]
        print(f'[BANK S{int(s)+1}] available={q["available"]}/{task.config.memory_capacity} '
              f'candidates={candidate.get("tokens","n/a")} written={written} '
              f'slices={q["unique_slices"]} fg={fmt(q["foreground_fraction"])} '
              f'source_rel={fmt(q["source_rel_mean"])} '
              f'p10/p90={fmt(q["source_rel_p10"])}/{fmt(q["source_rel_p90"])} '
              f'gain={fmt(q["source_gain"])} value_rms={fmt(q["value_rms"])} '
              f'age_steps={fmt(q["age_mean"])}',flush=True)
        retrieval=(record['sampled_batch'] or {}).get('retrieval',{}).get(s)
        if retrieval is not None:
            print(f'[READ S{int(s)+1}] coverage={fmt(retrieval["coverage"])} '
                  f'top1_cos={fmt(retrieval["top1_cos"])} '
                  f'retrieved_source_rel={fmt(retrieval["source_rel"])} '
                  f'attn_entropy={fmt(retrieval["attention_entropy"])} '
                  f'known_fg_mass={fmt(retrieval["known_fg_mass"])} '
                  f'corr_ratio={fmt(retrieval["correction_ratio"])}',flush=True)
    return record


def _image(x,norm_mode):
    x=x.detach().float().cpu()
    # Dataset repeats grayscale to RGB before normalization.
    x=(x[0]+1.6)/3.2 if norm_mode=='cipa' else x[0]*.229+.485
    return x.clamp(0,1).numpy()


@torch.no_grad()
def visualize_memory(task,loader,directory,epoch):
    """Few fixed validation slices, diagnostic GT audit, NO writes/selection changes.

Missing on/off uses the SAME main segmenter and SAME source query features.
Source Full probes are separate explanatory passes; neither PET nor GT enters
the Missing forward. PNGs include GT contours; NPZ preserves exact map values.
"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from models.pet_correction_memory import audit_regions
    from tasks.pet_correction import batch_ids

    active=task.eval_model(); source=task.memory_source_model()
    states={id(m):(m,m.training) for m in (active,source)}
    generator=getattr(loader,'generator',None)
    generator_state=generator.get_state() if generator is not None else None
    devices=[task.device.index if task.device.index is not None else torch.cuda.current_device()] if task.device.type=='cuda' else []
    files=[]
    try:
        for m,_ in states.values():
            m.eval()
        with torch.random.fork_rng(devices=devices):
            batch=next(iter(loader),None)
            if batch is None:
                return files
            n=min(task.config.memory_vis_samples,batch['ct'].shape[0])
            selected={k:v[:n] for k,v in batch.items() if isinstance(v,(torch.Tensor,list,tuple))}
            ct=selected['ct'].to(task.device); pet=selected['pet'].to(task.device)
            gt=selected['mask'].to(task.device).float()
            sid,gid=batch_ids(selected,task.device)
            queries=source._encode_ct(ct)
            on=active(ct,None,forward_mode='missing',sample_ids=sid,group_ids=gid,
                      query_features=queries,return_diagnostics=True,return_maps=True)
            off=active(ct,None,forward_mode='missing',sample_ids=sid,group_ids=gid,
                       query_features=queries,retrieval_override=False)
            full=active(ct,pet,forward_mode='full',sample_ids=sid,group_ids=gid)
            controlled=source.controlled_comparison(queries,source.fusion(queries,source._encode_pet(pet)))
            audits=[audit_regions(a,b,gt,d.shape[-2:],task.config.memory_gain_margin,
                task.config.memory_min_support,task.config.memory_max_negative_fraction)
                for a,b,d in zip(controlled['reference_logits'],controlled['plus_logits'],controlled['delta'])]
            folder=os.path.join(directory,f'epoch_{epoch:03d}'); os.makedirs(folder,exist_ok=True)
            for i in range(n):
                name=re.sub(r'[^A-Za-z0-9_.-]','_',str(selected['image_id'][i]))
                name=f'{i:02d}_{name}'
                arrays={'ct':_image(ct[i],task.config.norm_mode),'pet':_image(pet[i],task.config.norm_mode),
                    'gt':gt[i,0].cpu().numpy(),'full_prob':full['logits'][i,0].float().sigmoid().cpu().numpy(),
                    'missing_off_prob':off['logits'][i,0].float().sigmoid().cpu().numpy(),
                    'missing_on_prob':on['logits'][i,0].float().sigmoid().cpu().numpy()}
                arrays['prob_change']=arrays['missing_on_prob']-arrays['missing_off_prob']
                def draw(ax,value,title,cmap='viridis',vmin=None,vmax=None):
                    # Keep the GT contour on the original grid: downsampling GT
                    # to S3/S4 can erase the very tiny lesions we want to inspect.
                    display=F.interpolate(torch.as_tensor(value)[None,None].float(),
                        size=arrays['gt'].shape,mode='nearest')[0,0].numpy()
                    im=ax.imshow(display,cmap=cmap,vmin=vmin,vmax=vmax)
                    contour=arrays['gt']
                    if contour.any() and not contour.all():
                        ax.contour(contour,levels=[.5],colors='red',linewidths=.5)
                    ax.set_title(title,fontsize=8); ax.axis('off')
                    return im
                fig,axes=plt.subplots(1,7,figsize=(20,3))
                for ax,key in zip(axes,arrays):
                    if key=='prob_change':
                        draw(ax,arrays[key],key,'coolwarm',-1,1)
                    else:
                        draw(ax,arrays[key],key,'gray' if key in ('ct','pet','gt') else 'viridis',0,1)
                fig.suptitle(f'{name} | {task.eval_weights_tag()} | retrieval_active={bool(active.retrieval_active)} | red=GT')
                output=os.path.join(folder,name+'_predictions.png'); fig.savefig(output,dpi=130,bbox_inches='tight'); plt.close(fig); files.append(output)
                fig,axes=plt.subplots(4,6,figsize=(17,11))
                for s,audit in enumerate(audits):
                    ref=controlled['reference_logits'][s][i,0].float().sigmoid().cpu().numpy()
                    plus=controlled['plus_logits'][s][i,0].float().sigmoid().cpu().numpy()
                    panels=[('reference_prob',ref),('plus_prob',plus)]+[(key,audit[key][i,0].float().cpu().numpy())
                        for key in ('gain','eligible','reliability','foreground')]
                    for ax,(key,value) in zip(axes[s],panels):
                        arrays[f's{s+1}_source_{key}']=value
                        limit=float(np.abs(value).max()) or 1.
                        im=draw(ax,value,f'S{s+1} source {key}', 'coolwarm' if key=='gain' else 'viridis',
                                -limit if key=='gain' else 0,limit if key=='gain' else 1)
                        fig.colorbar(im,ax=ax,fraction=.045)
                fig.suptitle('Bank-source controlled audit; validation GT is diagnostic only; NO writes')
                output=os.path.join(folder,name+'_source_audit.png'); fig.savefig(output,dpi=120,bbox_inches='tight'); plt.close(fig); files.append(output)
                fig,axes=plt.subplots(4,6,figsize=(17,11))
                for s in range(4):
                    info=on['retrieval_diagnostics'].get(s,{})
                    maps=info.get('maps',{})
                    for ax,key in zip(axes[s],('coverage','top1_cos','source_rel','attention_entropy','known_fg_mass','correction_rms')):
                        value=maps[key][i,0].cpu().numpy() if key in maps else np.zeros(queries[s].shape[-2:],dtype=np.float32)
                        arrays[f's{s+1}_query_{key}']=value
                        im=draw(ax,value,f'S{s+1} {key}',vmin=-1 if key=='top1_cos' else 0,
                                vmax=None if key=='correction_rms' else 1)
                        fig.colorbar(im,ax=ax,fraction=.045)
                    if info:
                        for key,value in maps.items():
                            arrays[f's{s+1}_query_{key}']=value[i,0].cpu().numpy()
                fig.suptitle('Actual Missing retrieval; coverage is availability, source_rel is NOT query correctness')
                output=os.path.join(folder,name+'_retrieval.png'); fig.savefig(output,dpi=120,bbox_inches='tight'); plt.close(fig); files.append(output)
                output=os.path.join(folder,name+'_maps.npz'); np.savez_compressed(output,**arrays); files.append(output)
                with open(os.path.join(folder,name+'_metadata.json'),'w') as f:
                    json.dump({'image_id':str(selected['image_id'][i]),'epoch':epoch,
                        'eval_weights':task.eval_weights_tag(),'bank_ema_active':task.bank_ema_active,
                        'retrieval_active':bool(active.retrieval_active),'queue':queue_snapshot(active),
                        'gt_usage':'plot/source audit only; never Missing input; no writes'},f,indent=2,allow_nan=False)
    finally:
        for m,previous in states.values():
            m.train(previous)
        if generator_state is not None:
            generator.set_state(generator_state)
        plt.close('all')
    return files
