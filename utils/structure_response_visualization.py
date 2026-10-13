"""Fixed-slice Full feature inspection; does not change model/RNG/selection."""
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F


def scalar_diagnostics(info):
    result={}
    for s,values in info.items():
        result[str(s)]={}
        for name,value in values.items():
            if name=='maps':continue
            if isinstance(value,torch.Tensor):value=float(value.detach())
            result[str(s)][name]=None if isinstance(value,float) and not math.isfinite(value) else value
    return result


def log_fusion(info,path,epoch,batch,weights='online'):
    measured=scalar_diagnostics(info)
    with open(path,'a') as f:
        f.write(json.dumps({'epoch':epoch,'batch':batch,'weights':weights,'scales':measured},allow_nan=False)+'\n')
    for s,values in measured.items():
        print(f'[FUSION S{int(s)+1}] rCT={values["ct_residual_ratio"]:.4f} '
              f'rPET={values["pet_residual_ratio"]:.4f} edge_rms={values["edge_rms"]:.4f} '
              f'PETcontrast3/5/7={values["pet_contrast3_rms"]:.4f}/'
              f'{values["pet_contrast5_rms"]:.4f}/{values["pet_contrast7_rms"]:.4f} '
              f'CAentropy_CT/PET={values["ct_ca_entropy"]:.4f}/{values["pet_ca_entropy"]:.4f} '
              f'temp_CT/PET={values["ct_temperature"]:.4f}/{values["pet_temperature"]:.4f}',flush=True)
    return measured


@torch.no_grad()
def visualize_fusion(model,loader,cfg,device,epoch,weights='raw'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original_mode=model.training
    generator=getattr(loader,'generator',None)
    generator_state=generator.get_state() if generator is not None else None
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    paths=[]
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            batch=next(iter(loader),None)
            if batch is None:return paths
            n=min(cfg.fusion_vis_samples,batch['ct'].shape[0])
            ct=batch['ct'][:n].to(device);pet=batch['pet'][:n].to(device);gt=batch['mask'][:n].float().cpu().numpy()
            out=model(ct,pet,forward_mode='full',return_diagnostics=True,return_maps=True)
            add=model(ct,pet,forward_mode='full',use_add=True)
            info=out['fusion_diagnostics'];summary=scalar_diagnostics(info)
            folder=os.path.join(cfg.checkpoint_dir,'fusion_visualizations',f'epoch_{epoch:03d}')
            os.makedirs(folder,exist_ok=True)
            for i in range(n):
                name=re.sub('[^A-Za-z0-9_.-]','_',str(batch.get('image_id',list(range(n)))[i]))
                name=f'{i:02d}_{name}'
                def image(x):
                    z=x[0].detach().float().cpu()
                    return (((z+1.6)/3.2) if cfg.norm_mode=='cipa' else z*.229+.485).clamp(0,1).numpy()
                arrays={'ct':image(ct[i]),'pet':image(pet[i]),'gt':gt[i,0],
                    'add_counterfactual_prob':add['logits'][i,0].float().sigmoid().cpu().numpy(),
                    'fusion_prob':out['logits'][i,0].float().sigmoid().cpu().numpy()}
                arrays['prob_change']=arrays['fusion_prob']-arrays['add_counterfactual_prob']
                def draw(ax,value,title,signed=False,image_gray=False):
                    display=F.interpolate(torch.as_tensor(value)[None,None].float(),size=arrays['gt'].shape,
                                          mode='nearest')[0,0].numpy()
                    if signed:
                        limit=float(np.abs(value).max()) or 1.
                        im=ax.imshow(display,cmap='coolwarm',vmin=-limit,vmax=limit)
                    else:
                        im=ax.imshow(display,cmap='gray' if image_gray else 'viridis',vmin=0,
                                     vmax=1 if image_gray or 'prob' in title else None)
                    if arrays['gt'].any() and not arrays['gt'].all():
                        ax.contour(arrays['gt'],levels=[.5],colors='red',linewidths=.5)
                    ax.set_title(title,fontsize=8);ax.axis('off');return im
                fig,axes=plt.subplots(1,6,figsize=(18,3),constrained_layout=True)
                for ax,(key,value) in zip(axes,arrays.items()):
                    draw(ax,value,key,signed=key=='prob_change',image_gray=key in ('ct','pet','gt'))
                fig.suptitle(f'{name} | {weights} | red=GT | Add counterfactual uses SAME learned encoders/decoder')
                file=os.path.join(folder,name+'_predictions.png');fig.savefig(file,dpi=130,bbox_inches='tight');plt.close(fig);paths.append(file)
                if info:
                    fig,axes=plt.subplots(4,6,figsize=(17,11),constrained_layout=True)
                    for s in range(4):
                        maps=info[s]['maps']
                        for key,value in maps.items():arrays[f's{s+1}_{key}']=value[i].cpu().numpy()
                        for ax,key in zip(axes[s],('ct_structure','pet_contrast5_rms','ct_enhanced','pet_enhanced','delta_ct','delta_pet')):
                            value=maps[key][i,0].cpu().numpy()
                            im=draw(ax,value,f'S{s+1} {key}');fig.colorbar(im,ax=ax,fraction=.045)
                    fig.suptitle('Feature/contrast/residual RMS; these are NOT lesion probability maps; red=original GT')
                    file=os.path.join(folder,name+'_features.png');fig.savefig(file,dpi=120,bbox_inches='tight');plt.close(fig);paths.append(file)
                    fig,axes=plt.subplots(4,2,figsize=(9,14),constrained_layout=True)
                    for s in range(4):
                        for ax,key in zip(axes[s],('ct_ca','pet_ca')):
                            matrix=info[s]['maps'][key][i].cpu().numpy().mean(0)
                            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
                            ax.set_title(f'S{s+1} {key}: mean over heads',fontsize=9)
                            ax.set_xlabel('key channel within head');ax.set_ylabel('query channel within head')
                            fig.colorbar(im,ax=ax,fraction=.045)
                    fig.suptitle('CHANNEL attention matrices, not spatial correspondences')
                    file=os.path.join(folder,name+'_channel_attention.png');fig.savefig(file,dpi=110,bbox_inches='tight');plt.close(fig);paths.append(file)
                file=os.path.join(folder,name+'_maps.npz');np.savez_compressed(file,**arrays);paths.append(file)
                with open(os.path.join(folder,name+'_metadata.json'),'w') as f:
                    json.dump({'image_id':name,'epoch':epoch,'weights':weights,'fusion_enabled':cfg.fusion_enabled,
                               'batch_scale_diagnostics':summary,'counterfactual':'same learned backbone/decoder, NOT independently trained Add baseline'},f,indent=2,allow_nan=False)
    finally:
        model.train(original_mode)
        if generator_state is not None:generator.set_state(generator_state)
        plt.close('all')
    return paths
