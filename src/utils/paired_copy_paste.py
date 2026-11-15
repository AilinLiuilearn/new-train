"""Train-only paired, context-preserving Copy-Paste. Maps select donors only."""
from collections import OrderedDict
import json
from pathlib import Path
import cv2
import numpy as np
from utils.dual_state_maps import MapCache, sha256, fingerprints, train_records

POLICY = {'version': 1, 'context': 'max(4,round(.1*max(width,height)))',
          'placement': 'body_relative_center_jitter_0.08', 'placement_attempts': 12,
          'body_threshold': 8, 'minimum_body_coverage': .95,
          'maximum_ring_mean_difference': 32., 'maximum_ring_std_difference': 32.,
          'scale': 1., 'cross_case': True, 'maximum_patch_side_fraction': .4,
          'control_size_matching': 'dual-reference floor(log2(foreground_area)) bins'}
COUNTERS = ('samples','attempts','success','fallback_no_donor','fallback_placement',
            'selected_1','selected_2','both_available','selected_1_when_both',
            'selected_2_when_both','pasted_pixels','pasted_foreground_pixels',
            'recipient_foreground_pixels','donor_reads','ring_mean_difference_sum',
            'patch_width_sum','patch_height_sum','placement_trials','success_1','success_2') + tuple(f'area_bin_{i}' for i in range(20))


def body_mask(ct):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((ct > POLICY['body_threshold']).astype(np.uint8), connectivity=8)
    if n <= 1: return np.zeros_like(ct, dtype=np.uint8), None
    label = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    binary = (labels == label).astype(np.uint8)
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary);cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
    x,y,w,h = cv2.boundingRect(filled)
    return filled,(int(x),int(y),int(w),int(h))


def candidates(mask,difference):
    """Whole GT 2D components; error-only background never becomes a donor lesion."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out=[];h,w=mask.shape
    for label in range(1,n):
        x,y,bw,bh,area = (int(v) for v in stats[label])
        context=max(4,round(.1*max(bw,bh)))
        x0,y0,x1,y1=max(0,x-context),max(0,y-context),min(w,x+bw+context),min(h,y+bh+context)
        if max(x1-x0,y1-y0)>POLICY['maximum_patch_side_fraction']*min(h,w):continue
        patch_labels=labels[y0:y1,x0:x1]
        # Do not cut through a second labeled target at the rectangular seam.
        if np.any((patch_labels!=0)&(patch_labels!=label)):continue
        region=labels==label
        counts=[int(np.sum(difference[region]==i)) for i in range(4)]
        out.append({'label':label,'bbox':[x0,y0,x1,y1],'area':area,'errors':counts,
                    'flags':[i for i in (1,2) if counts[i]>0],
                    'hard':bool(sum(counts[1:]))})
    return out


def raw_record(record,size):
    from datasets.pclt20k_seg import _imread_grayscale,_resize_gray
    arrays=[_imread_grayscale(record[k]) for k in ('ct_path','pet_path','mask_path')]
    if any(a is None for a in arrays):raise FileNotFoundError(record)
    if any(a.shape!=arrays[0].shape for a in arrays):raise ValueError('Unregistered donor')
    if not set(np.unique(arrays[2]).tolist()).issubset({0,1,255}):raise ValueError('Invalid donor GT')
    if arrays[2].max()<=1:arrays[2]=arrays[2]*np.uint8(255)
    if arrays[0].shape!=(size,size):
        arrays=[_resize_gray(a,size,nearest=i==2) for i,a in enumerate(arrays)]
    return tuple(arrays)


class PasteBank:
    def __init__(self,index_path,map_dir,config,expected_identity=None,selection='overlap'):
        if selection not in ('overlap','dominance'):raise ValueError('Invalid paste selection')
        self.index_path=Path(index_path).resolve()
        self.index=json.loads(self.index_path.read_text())
        self.maps=MapCache(map_dir,config)
        if not self.index.get('complete') or self.index.get('policy')!=POLICY:
            raise ValueError('Incomplete/incompatible Copy-Paste index')
        if self.index['map_manifest_sha256']!=self.maps.identity['manifest_sha256']:
            raise ValueError('Copy-Paste map identity mismatch')
        self.records=train_records(config,self.maps.manifest['train_split_sha256'])
        if self.index['image_ids']!=[r['image_id'] for r in self.records]:raise ValueError('Paste train identity mismatch')
        self.size=config['image_size_2d'];self.content_hash=self.maps.manifest['content_hash']
        self.identity={'index_sha256':sha256(self.index_path),**self.maps.identity,'policy':POLICY,'selection':selection}
        if expected_identity is not None and self.identity!=expected_identity:raise ValueError('Copy-Paste resume identity changed')
        self.pools={'random':[],'hard':[],1:[],2:[]}
        for item in self.index['candidates']:
            self.pools['random'].append(item)
            if item['hard']:self.pools['hard'].append(item)
            flags=item['flags'] if selection=='overlap' else ([1] if item['errors'][1]>item['errors'][2]
                    else [2] if item['errors'][2]>item['errors'][1] else [])
            for category in flags:self.pools[category].append(item)
        self.cache=OrderedDict()
        self.binned={mode:{} for mode in ('random','hard')}
        for mode in self.binned:
            for item in self.pools[mode]:
                key=int(np.floor(np.log2(max(1,item['area']))))
                self.binned[mode].setdefault(key,[]).append(item)

    def read(self,idx):
        record=self.records[idx];name=record['image_id']
        if fingerprints(record,self.content_hash)!=self.maps.samples[name]['files']:
            raise ValueError(f'Donor fingerprint changed: {name}')
        if idx not in self.cache:
            self.cache[idx]=raw_record(record,self.size)
            if len(self.cache)>8:self.cache.popitem(last=False)
        self.cache.move_to_end(idx)
        return self.cache[idx]

    def paste(self,ct,pet,mask,recipient_idx,mode,probability,trace=False):
        if mode not in ('random','hard','dual_state') or not 0<=probability<=1:
            raise ValueError('Invalid Copy-Paste policy')
        info={key:0. for key in COUNTERS};info['samples']=1.
        details={}
        if probability==0 or np.random.random()>=probability:return ct,pet,mask,info,details
        info['attempts']=1.
        category=None
        if mode=='dual_state':
            available=[i for i in (1,2) if self.pools[i]]
            if not available:
                info['fallback_no_donor']=1.;return ct,pet,mask,info,details
            both=len(available)==2
            category=available[np.random.randint(len(available))] if both else available[0]
            info['selected_'+str(category)]=1.;info['both_available']=float(both)
            info['selected_'+str(category)+'_when_both']=float(both)
        pool=self.pools[category if category is not None else mode]
        if mode!='dual_state':
            # Controls match the dual-state reference donor-size distribution.
            reference_classes=[i for i in (1,2) if self.pools[i]]
            if reference_classes:
                reference_class=reference_classes[np.random.randint(len(reference_classes))]
                reference_pool=self.pools[reference_class]
                reference=reference_pool[np.random.randint(len(reference_pool))]
                key=int(np.floor(np.log2(max(1,reference['area']))))
                pool=self.binned[mode].get(key,[])
        recipient=self.records[recipient_idx]
        donor=None
        # Bounded rejection, never same-case self retrieval.
        for _ in range(32):
            if not pool:break
            item=pool[np.random.randint(len(pool))]
            if self.records[item['record_idx']]['case_id']!=recipient['case_id']:
                donor=item;break
        if donor is None:
            info['fallback_no_donor']=1.;return ct,pet,mask,info,details
        dct,dpet,dgt=self.read(donor['record_idx']);info['donor_reads']=1.
        x0,y0,x1,y1=donor['bbox'];pw,ph=x1-x0,y1-y0
        patch_ct=dct[y0:y1,x0:x1];patch_pet=dpet[y0:y1,x0:x1];patch_gt=dgt[y0:y1,x0:x1]
        body,rb=body_mask(ct)
        db=self.index['body_bboxes'][donor['record_idx']]
        if rb is None or db is None:
            info['fallback_placement']=1.;return ct,pet,mask,info,details
        bx,by,bw,bh=rb;dx,dy,dw,dh=db
        relx=((x0+x1)/2-dx)/max(dw,1);rely=((y0+y1)/2-dy)/max(dh,1)
        ring=np.ones((ph,pw),bool);margin=max(1,min(4,ph//4,pw//4))
        ring[margin:ph-margin,margin:pw-margin]=False
        source_mean=float(patch_ct[ring].mean());source_std=float(patch_ct[ring].std())
        location=None;mean_diff=0.
        for _ in range(POLICY['placement_attempts']):
            info['placement_trials']+=1
            cx=bx+(relx+np.random.uniform(-.08,.08))*bw
            cy=by+(rely+np.random.uniform(-.08,.08))*bh
            x,y=int(round(cx-pw/2)),int(round(cy-ph/2))
            if x<0 or y<0 or x+pw>ct.shape[1] or y+ph>ct.shape[0]:continue
            if np.any(mask[y:y+ph,x:x+pw]>0):continue
            if float(body[y:y+ph,x:x+pw].mean())<POLICY['minimum_body_coverage']:continue
            target=ct[y:y+ph,x:x+pw]
            mean_diff=abs(float(target[ring].mean())-source_mean)
            if mean_diff>POLICY['maximum_ring_mean_difference']:continue
            if abs(float(target[ring].std())-source_std)>POLICY['maximum_ring_std_difference']:continue
            location=(x,y);break
        if location is None:
            info['fallback_placement']=1.;return ct,pet,mask,info,details
        x,y=location
        ct,pet,mask=ct.copy(),pet.copy(),mask.copy()
        ct[y:y+ph,x:x+pw]=patch_ct;pet[y:y+ph,x:x+pw]=patch_pet;mask[y:y+ph,x:x+pw]=patch_gt
        info.update(success=1.,pasted_pixels=float(pw*ph),pasted_foreground_pixels=float((patch_gt>0).sum()),
                    recipient_foreground_pixels=float((mask>0).sum()),ring_mean_difference_sum=mean_diff,
                    patch_width_sum=float(pw),patch_height_sum=float(ph))
        if category is not None:info['success_'+str(category)]=1.
        info['area_bin_'+str(min(19,int(np.floor(np.log2(max(1,donor['area']))))))]=1.
        if trace:
            details={'donor_id':self.records[donor['record_idx']]['image_id'],'donor_bbox':donor['bbox'],
                     'recipient_bbox':[x,y,x+pw,y+ph],'category':category,'donor_errors':donor['errors'],
                     'selected_target_area':donor['area'],'context_ring_mean_difference':mean_diff}
        return ct,pet,mask,info,details


def summarize(counts):
    s={k:float(v) for k,v in counts.items()}
    s['success_per_attempt']=s.get('success',0)/max(s.get('attempts',0),1)
    s['success_per_sample']=s.get('success',0)/max(s.get('samples',0),1)
    s['mean_pasted_foreground']=s.get('pasted_foreground_pixels',0)/max(s.get('success',0),1)
    s['mean_ring_mean_difference']=s.get('ring_mean_difference_sum',0)/max(s.get('success',0),1)
    return s
