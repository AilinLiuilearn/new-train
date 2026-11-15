import numpy as np
import torch
from torch.nn import functional as F


class RoleMetrics:
    def __init__(self):
        self.s={};self.pos=np.zeros(2048,dtype=np.int64);self.neg=self.pos.copy()

    def add(self,k,v):self.s[k]=self.s.get(k,0.)+float(v)

    @torch.no_grad()
    def update(self,o,b):
        y=b['mask']>.5;p=o['pet_logits'].float().sigmoid()>.5
        self.add('pet_tp',(p&y).sum());self.add('pet_area',p.sum());self.add('gt_area',y.sum())
        near=(b['ct_target'].abs()*16<=3)
        self.add('ct_band_abs_error',((o['ct_distance']-b['ct_target']).abs()*16*near).sum())
        self.add('ct_band_pixels',near.sum())
        def edge(a):
            eroded=1-F.max_pool2d(F.pad(1-a.float(),(1,1,1,1),value=1),3,stride=1)
            return a&~(eroded>.5)
        c=edge(o['ct_distance']>=0);t=edge(y)
        self.add('ct_boundary_total',c.sum());self.add('gt_boundary_total',t.sum())
        self.add('ct_boundary_matched',(c&(F.max_pool2d(t.float(),7,1,3)>.5)).sum())
        self.add('gt_boundary_matched',(t&(F.max_pool2d(c.float(),7,1,3)>.5)).sum())
        if 'base_logits' not in o:return
        before=o['base_logits'].float().sigmoid()>.5;after=o['logits'].float().sigmoid()>.5;err=before!=y
        repaired=err&(after==y);damaged=~err&(after!=y)
        for label,mask in [('all',torch.ones_like(y)),('fg',y),('bg',~y),
                           ('near_bg',b['near_bg']>.5),('far_bg',~y&~(b['near_bg']>.5))]:
            self.add(label+'_errors',(err&mask).sum());self.add(label+'_correct',(~err&mask).sum())
            self.add(label+'_repaired',(repaired&mask).sum());self.add(label+'_damaged',(damaged&mask).sum())
        gate=o['gate'];indices=(gate*2048).long().clamp(0,2047)
        self.pos+=torch.bincount(indices[err],minlength=2048).cpu().numpy()
        self.neg+=torch.bincount(indices[~err],minlength=2048).cpu().numpy()
        self.add('gate_error_sum',gate[err].sum());self.add('gate_correct_sum',gate[~err].sum())
        self.add('delta_abs',o['delta'].abs().sum());self.add('change_abs',o['change'].abs().sum());self.add('pixels',y.numel())

    def compute(self):
        s=self.s;ratio=lambda x,y:s.get(x,0)/max(s.get(y,0),1)
        prec=ratio('ct_boundary_matched','ct_boundary_total');rec=ratio('gt_boundary_matched','gt_boundary_total')
        result={'counts':s,'pet_candidate_precision':ratio('pet_tp','pet_area'),
                'pet_candidate_recall':ratio('pet_tp','gt_area'),'pet_candidate_area_ratio':ratio('pet_area','gt_area'),
                'ct_band_distance_MAE_pixels':ratio('ct_band_abs_error','ct_band_pixels'),
                'ct_boundary_F1_tolerance3':2*prec*rec/max(prec+rec,1e-12)}
        if 'all_errors' in s:
            tp=np.cumsum(self.pos[::-1]);fp=np.cumsum(self.neg[::-1]);ap=None
            if tp[-1]>0:
                recall=tp/tp[-1]
                precision=np.divide(tp,tp+fp,out=np.ones_like(tp,dtype=np.float64),where=(tp+fp)>0)
                ap=float(np.trapz(np.r_[1.,precision],np.r_[0.,recall]))
            result.update(gate_pr_auc_hist2048=ap,gate_error_mean=ratio('gate_error_sum','all_errors'),
                          gate_correct_mean=ratio('gate_correct_sum','all_correct'),
                          residual_abs_mean=ratio('delta_abs','pixels'),actual_change_abs_mean=ratio('change_abs','pixels'))
            result['gate_histogram_32bins']={'error':self.pos.reshape(32,64).sum(1).tolist(),
                                            'correct':self.neg.reshape(32,64).sum(1).tolist()}
            for group in ['all','fg','bg','near_bg','far_bg']:
                result[group]={'repair_rate':ratio(group+'_repaired',group+'_errors'),
                               'harmful_rate':ratio(group+'_damaged',group+'_correct'),
                               'net_repaired_pixels':s.get(group+'_repaired',0)-s.get(group+'_damaged',0)}
        return result
