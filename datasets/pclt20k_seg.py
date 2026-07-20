# -*- coding: utf-8 -*-
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
except Exception:
    cv2 = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _read_list(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return [x.strip() for x in f if x.strip()]


def _imread_grayscale(path):
    if cv2 is not None:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.array(Image.open(path).convert('L')) if os.path.isfile(path) else None


def _resize_gray(img, size, nearest=False):
    if cv2 is not None:
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(img, (size, size), interpolation=interp)
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.array(Image.fromarray(img).resize((size, size), mode))


def _normalize_ch(x, mode='cipa'):
    x = x.astype(np.float32) / 255.0
    x = x[None, ...]
    rgb = np.repeat(x, 3, axis=0)
    if mode == 'cipa':
        return rgb * 3.2 - 1.6
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


class PCLT20KSegDataset(Dataset):
    def __init__(self, records, image_size=512, train=False, random_state=2023, aug_mode='cipa', norm_mode='imagenet'):
        self.records = records
        self.image_size = image_size
        self.train = train
        self.aug_mode = aug_mode
        self.norm_mode = norm_mode
        self.rng = random.Random(random_state)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        ct = _imread_grayscale(record['ct_path'])
        pet = _imread_grayscale(record['pet_path'])
        mask = _imread_grayscale(record['mask_path'])
        if ct is None or pet is None or mask is None:
            raise FileNotFoundError(record)
        if ct.shape != (self.image_size, self.image_size):
            ct = _resize_gray(ct, self.image_size)
            pet = _resize_gray(pet, self.image_size)
            mask = _resize_gray(mask, self.image_size, nearest=True)
        ct_t = torch.tensor(_normalize_ch(ct, self.norm_mode), dtype=torch.float32)
        pet_t = torch.tensor(_normalize_ch(pet, self.norm_mode), dtype=torch.float32)
        mask = (mask.astype(np.float32) / 255.0)
        mask = (mask[None, ...] >= 0.5).astype(np.float32)
        return {
            'ct': ct_t,
            'pet': pet_t,
            'mask': torch.tensor(mask),
            'image_id': record.get('image_id', str(idx)),
            'case_id': record.get('case_id', str(idx)),
            'slice_id': record.get('slice_id', record.get('image_id', str(idx))),
            'idx': idx,
        }


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(dataset, batch_size, num_workers, shuffle, drop_last, seed, pin_memory=True):
    g = torch.Generator(); g.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory, worker_init_fn=_seed_worker, generator=g)


def _records_from_ids(root, ids):
    out = []
    for image_id in ids:
        case_id = image_id.split('_')[0]
        case_dir = os.path.join(root, case_id)
        ct = os.path.join(case_dir, f'{image_id}_CT.png')
        pet = os.path.join(case_dir, f'{image_id}_PET.png')
        mask = os.path.join(case_dir, f'{image_id}_mask.png')
        out.append({'image_id': image_id, 'case_id': case_id, 'slice_id': image_id, 'ct_path': ct, 'pet_path': pet, 'mask_path': mask})
    return out


def _split_summary(records):
    by_case = Counter(r['case_id'] for r in records)
    return {'case_count': len(by_case), 'slice_count': len(records)}


def get_pclt20k_loaders_cipa_aligned(root, image_size=512, batch_size=8, num_workers=4, random_state=2023, pin_memory=True, aug_mode='cipa', norm_mode='cipa', train_split_file='train.txt', val_split_file='val.txt', test_split_file='test.txt'):
    train_ids = _read_list(os.path.join(root, train_split_file))
    val_ids = _read_list(os.path.join(root, val_split_file))
    test_ids = _read_list(os.path.join(root, test_split_file))
    if train_ids is None or val_ids is None or test_ids is None:
        raise FileNotFoundError(root)
    train_records = _records_from_ids(root, train_ids)
    val_records = _records_from_ids(root, val_ids)
    test_records = _records_from_ids(root, test_ids)
    for name, recs in [('train', train_records), ('val', val_records), ('test', test_records)]:
        if not recs:
            raise ValueError(f'{name} split is empty')
        if any(not (os.path.isfile(r['ct_path']) and os.path.isfile(r['pet_path']) and os.path.isfile(r['mask_path'])) for r in recs):
            raise FileNotFoundError(f'{name} split contains missing PET/CT/mask files')
    train_cases = {r['case_id'] for r in train_records}
    val_cases = {r['case_id'] for r in val_records}
    test_cases = {r['case_id'] for r in test_records}
    if train_cases & val_cases or train_cases & test_cases or val_cases & test_cases:
        raise ValueError('train/val/test splits overlap on case_id')
    split_summary = {
        'train': {**_split_summary(train_records), 'case_ids': sorted(train_cases)},
        'val': {**_split_summary(val_records), 'case_ids': sorted(val_cases)},
        'test': {**_split_summary(test_records), 'case_ids': sorted(test_cases)},
    }
    with open(os.path.join(root, 'split_summary.json'), 'w') as f:
        json.dump(split_summary, f, indent=2)
    train_ds = PCLT20KSegDataset(train_records, image_size=image_size, train=True, random_state=random_state, aug_mode=aug_mode, norm_mode=norm_mode)
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False, random_state=random_state, aug_mode='none', norm_mode=norm_mode)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False, random_state=random_state, aug_mode='none', norm_mode=norm_mode)
    return _make_loader(train_ds, batch_size, num_workers, True, True, random_state+11, pin_memory), _make_loader(val_ds, batch_size, num_workers, False, False, random_state+17, pin_memory), _make_loader(test_ds, batch_size, num_workers, False, False, random_state+23, pin_memory)


def get_pclt20k_loaders(*args, **kwargs):
    return get_pclt20k_loaders_cipa_aligned(*args, **kwargs)
