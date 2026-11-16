"""One offline four-slice figure after selection, never another full validation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,local_paths,gpu_lock_path


def main(args):
    require_myenv();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from datasets.pclt20k_seg import PCLT20KSegDataset,_records_from_ids
    from variants.scgr.inference import load_model
    torch.set_num_threads(4)
    ids=['0523_009','0424_086','0446_008','0054_005']
    root=local_paths(args.paths)['data_root']
    ds=PCLT20KSegDataset(_records_from_ids(root,ids),train=False,norm_mode='imagenet')
    with open(gpu_lock_path(),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=load_model(args.checkpoint,'cuda');fig,axes=plt.subplots(4,5,figsize=(14,13))
        records=[];packed=[]
        for row,identity in enumerate(ids):
            sample=ds[row]
            with torch.no_grad():out=model(sample['ct'][None].cuda(),sample['pet'][None].cuda(),capture=True)
            pred=(out['logits'].sigmoid()[0,0]>.5).cpu().numpy();gt=sample['mask'][0].numpy()>.5
            ct=sample['ct'][0].numpy()*.229+.485;pet=sample['pet'][0].numpy()*.229+.485
            axes[row,0].imshow(ct,cmap='gray',vmin=0,vmax=1);axes[row,1].imshow(pet,cmap='gray',vmin=0,vmax=1)
            axes[row,2].imshow(ct,cmap='gray',vmin=0,vmax=1)
            if gt.any():axes[row,2].contour(gt,levels=[.5],colors=['lime'],linewidths=.7)
            if pred.any():axes[row,2].contour(pred,levels=[.5],colors=['magenta'],linewidths=.7)
            error=np.zeros((512,512,3));error[gt&pred]=[.2,.6,.2];error[gt&~pred]=[.1,.5,1];error[~gt&pred]=[1,.2,.1]
            axes[row,3].imshow(error)
            gate=out['capture']['refiner']['1']['acceptance'][0,0].float().cpu().numpy()
            axes[row,4].imshow(gate,vmin=0,vmax=1,cmap='viridis')
            for ax,title in zip(axes[row],[identity+' CT','PET','GT green / pred magenta','TP green / FN blue / FP red','S1 residual acceptance (not probability)']):
                ax.set_title(title,fontsize=8);ax.axis('off')
            records.append({'image_id':identity,'tp':int((gt&pred).sum()),'fp':int((~gt&pred).sum()),'fn':int((gt&~pred).sum())})
            packed.append(np.packbits(pred.ravel()))
        fig.suptitle('Fixed historical diagnostic examples, not a representative sample. Same PNG pixel grid.',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.96),h_pad=2)
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        fig.savefig(output/'EXAMPLES.png',dpi=140);plt.close(fig)
        np.savez_compressed(output/'predictions.npz',ids=np.array(ids),packed=np.stack(packed))
        (output/'EXAMPLES.json').write_text(json.dumps({'selection':'Four IDs fixed from historical M1 diagnosis, no selection by current outputs',
            'scope':'Qualitative only; not full validation, not extra checkpoint selection; gate is not tumor probability',
            'checkpoint':str(Path(args.checkpoint).resolve()),'examples':records},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');main(p.parse_args())
