"""Error locations from the existing fusion validation pass, no extra inference."""
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
import torch


class ErrorMetrics:
    def __init__(self):self.counts=defaultdict(int)

    @torch.no_grad()
    def update(self,logits,target):
        prob=logits.float().sigmoid().cpu().numpy()[:,0];ys=target.cpu().numpy()[:,0]>.5
        structure=ndi.generate_binary_structure(2,1)
        for q,y in zip(prob,ys):
            p=q>.5;fn=y&~p;fp=~y&p
            inside=ndi.distance_transform_edt(y);outside=ndi.distance_transform_edt(~y)
            c={'gt':y.sum(),'tp':(y&p).sum(),'fn':fn.sum(),'fp':fp.sum(),
               'gt_inner3':(y&(inside<=3)).sum(),'gt_deep':(y&(inside>3)).sum(),
               'fn_inner3':(fn&(inside<=3)).sum(),'fn_deep':(fn&(inside>3)).sum(),
               'fp_outer3':(fp&(outside<=3)).sum(),
               'fp_outer3_12':(fp&(outside>3)&(outside<=12)).sum(),
               'fp_far12':(fp&(outside>12)).sum(),
               'fp_confident':(fp&(q>.9)).sum(),'fn_confident':(fn&(q<.1)).sum()}
            c['fp_empty_gt']=0
            if not y.any():
                # No GT foreground means distance to a lesion is undefined.
                c.update(fp_outer3=0,fp_outer3_12=0,fp_far12=0,fp_empty_gt=fp.sum())
            labels,n=ndi.label(p,structure)
            c['fp_detached']=(p&~np.isin(labels,np.unique(labels[y]))).sum()
            gl,gn=ndi.label(y,structure);areas=np.bincount(gl.ravel(),minlength=gn+1)
            hits=np.bincount(gl[p].ravel(),minlength=gn+1)
            small=(areas>0)&(areas<=256);small[0]=False
            c['small_components']=small.sum();c['small_components_hit']=((hits>0)&small).sum()
            c['small_pixels']=areas[small].sum();c['small_pixels_hit']=hits[small].sum()
            for k,v in c.items():self.counts[k]+=int(v)

    def compute(self):
        c=dict(self.counts)
        def ratio(k,d):return c.get(k,0)/c[d] if c.get(d,0) else None
        return {'counts':c,'fn_inner3_rate':ratio('fn_inner3','gt_inner3'),
                'fn_deep_rate':ratio('fn_deep','gt_deep'),
                'fp_outer3_fraction':ratio('fp_outer3','fp'),
                'fp_far12_fraction':ratio('fp_far12','fp'),
                'small_component_coverage':ratio('small_components_hit','small_components'),
                'small_pixel_sensitivity':ratio('small_pixels_hit','small_pixels'),
                'scope':'512 PNG pixels; GT/pred 2D4-connected components; small<=256px, not clinical lesions. Same fusion validation pass.'}
