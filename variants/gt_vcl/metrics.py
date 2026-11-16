"""One-pass auxiliary predictions and descriptive agreement counts, not causal attribution."""
import numpy as np
from scipy import ndimage
import torch
from variants.bcsi_adapt.metrics import SmallTargetMetrics


class JointMetrics:
    def __init__(self):
        self.counts={name:np.zeros(3,dtype=np.int64) for name in ['ct','pet']}
        self.patterns={name:np.zeros(8,dtype=np.int64) for name in ['fg','bg']}
        self.small=SmallTargetMetrics()
        self.boundary_pred=self.boundary_gt=self.boundary_matched_pred=self.boundary_matched_gt=0

    @torch.no_grad()
    def update_output(self,output,target):
        self.small.update(output['logits'],target)
        y=(target>=.5).cpu().numpy()[:,0]
        pred={name:(output[key].float().sigmoid()>.5).cpu().numpy()[:,0]
              for name,key in [('ct','ct_logits'),('pet','pet_logits'),('full','logits')]}
        for name in ['ct','pet']:
            p=pred[name]
            self.counts[name]+=np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum()],dtype=np.int64)
        code=(pred['ct']==y).astype(np.uint8)*4+(pred['pet']==y).astype(np.uint8)*2+(pred['full']==y).astype(np.uint8)
        self.patterns['fg']+=np.bincount(code[y],minlength=8)
        self.patterns['bg']+=np.bincount(code[~y],minlength=8)
        for p,g in zip(pred['full'],y):
            bp=p&~ndimage.binary_erosion(p,structure=np.ones((3,3)),border_value=0)
            bg=g&~ndimage.binary_erosion(g,structure=np.ones((3,3)),border_value=0)
            self.boundary_pred+=int(bp.sum());self.boundary_gt+=int(bg.sum())
            if bp.any() and bg.any():
                self.boundary_matched_pred+=int((bp&(ndimage.distance_transform_edt(~bg)<=3)).sum())
                self.boundary_matched_gt+=int((bg&(ndimage.distance_transform_edt(~bp)<=3)).sum())

    def compute(self):
        result={'small_targets':self.small.compute(),'single_modal':{},'agreement_counts':{},
                'scope':'Same batch predictions; CT-referenced agreement, not causal PET contribution'}
        for name,(tp,fp,fn) in self.counts.items():
            result['single_modal'][name]={'dice':float(2*tp/max(2*tp+fp+fn,1)),
                                         'precision':float(tp/max(tp+fp,1)),
                                         'sensitivity':float(tp/max(tp+fn,1))}
        for name,v in self.patterns.items():
            result['agreement_counts'][name]={'eight_states_CPF':v.tolist(),
                'ct_wrong_pet_and_full_correct':int(v[3]),'both_single_wrong_full_correct':int(v[1]),
                'ct_correct_full_wrong':int(v[4]+v[6]),'ct_correct_full_correct':int(v[5]+v[7])}
        precision=self.boundary_matched_pred/max(self.boundary_pred,1)
        recall=self.boundary_matched_gt/max(self.boundary_gt,1)
        result['full_boundary_f1_tol3_micro']=2*precision*recall/max(precision+recall,1e-12)
        return result
