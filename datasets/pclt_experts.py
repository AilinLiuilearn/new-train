"""Paired grayscale PNG loader using existing split parsing/patient guardrails.

case/id_CT.png, case/id_PET.png, case/id_mask.png; case=id.split('_')[0].
Masks accept binary 0/1 or 0/255. CT-only never opens a PET file.
PNG intensity is not HU/SUV; normalization is per-image z-score or unit [0,1].
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.pclt20k_seg import (_read_list, _records_from_ids, _check_patient_disjoint,
                                 _make_loader)

class SliceDataset(Dataset):
    def __init__(self, records, size, norm, train=False, paired=True, augment=True):
        self.records, self.size, self.norm = records, size, norm
        self.train, self.paired, self.augment = train, paired, augment

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        keys = ['ct','pet'] if self.paired else ['ct']
        images = [cv2.imread(r[k+'_path'], cv2.IMREAD_GRAYSCALE) for k in keys]
        mask = cv2.imread(r['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None or any(x is None for x in images): raise FileNotFoundError(r)
        if any(x.shape != mask.shape for x in images):
            raise ValueError(f'unaligned image/mask dimensions: {r["image_id"]}')
        if not set(np.unique(mask)).issubset({0,1,255}):
            raise ValueError(f'mask must contain only 0/1/255: {r["mask_path"]}')
        mask = (mask > 0).astype(np.uint8)
        images = [cv2.resize(x, (self.size,self.size), interpolation=cv2.INTER_LINEAR) for x in images]
        mask = cv2.resize(mask, (self.size,self.size), interpolation=cv2.INTER_NEAREST)
        if self.train and self.augment:
            # Shared geometric draw; nearest interpolation for labels.
            if np.random.rand() < .5:
                images = [x[:,::-1].copy() for x in images]; mask = mask[:,::-1].copy()
            if np.random.rand() < .7:
                mat = cv2.getRotationMatrix2D((self.size/2,self.size/2), np.random.uniform(-20,20), np.random.uniform(.9,1.1))
                mat[:,2] += np.random.uniform(-.1,.1,2)*self.size
                images = [cv2.warpAffine(x, mat, (self.size,self.size), flags=cv2.INTER_LINEAR) for x in images]
                mask = cv2.warpAffine(mask, mat, (self.size,self.size), flags=cv2.INTER_NEAREST)
        out = {'mask':torch.from_numpy(mask[None].astype(np.float32)), 'case_id':r['case_id'], 'image_id':r['image_id']}
        for key, im in zip(keys, images):
            x = im.astype(np.float32)/255.
            if self.norm == 'zscore': x = (x-x.mean())/max(float(x.std()),1e-6)
            out[key] = torch.from_numpy(np.ascontiguousarray(x[None]))
        return out

def split_signature(cfg):
    return {s:hashlib.sha256((Path(cfg.root)/getattr(cfg,s+'_split')).read_bytes()).hexdigest()
            for s in ['train','val','test']}

def loaders(cfg, output_dir):
    records = []
    for s in ['train','val','test']:
        ids = _read_list(str(Path(cfg.root)/getattr(cfg,s+'_split')))
        if not ids: raise ValueError(f'{s} split absent or empty')
        if len(ids) != len(set(ids)): raise ValueError(f'{s} split contains duplicate slice IDs')
        rows = _records_from_ids(cfg.root, ids, ct_only=cfg.mode=='ct')
        for r in rows:
            for k in ['ct_path','mask_path'] + ([] if cfg.mode=='ct' else ['pet_path']):
                if not Path(r[k]).is_file(): raise FileNotFoundError(r[k])
        records.append(rows)
    same = cfg.val_split == cfg.test_split
    if same and not cfg.allow_val_equals_test: raise ValueError('val=test requires --allow_val_equals_test')
    _check_patient_disjoint(*records, allow_val_equals_test=same and cfg.allow_val_equals_test)
    if same: print('[WARN] val=test: development evaluation, not independent test performance',flush=True)
    summary = {s:{'slices':len(rs),'cases':sorted({r['case_id'] for r in rs})}
               for s,rs in zip(['train','val','test'],records)}
    Path(output_dir,'split_summary.json').write_text(json.dumps(summary,indent=2))
    result=[]
    for i,rs in enumerate(records):
        ds=SliceDataset(rs,cfg.image_size,cfg.norm,train=i==0,paired=cfg.mode!='ct',augment=not cfg.no_augment)
        result.append(_make_loader(ds,cfg.batch_size,cfg.num_workers,i==0,i==0,cfg.seed+11+6*i,True))
    if not len(result[0]): raise ValueError('no complete training batch')
    return result
