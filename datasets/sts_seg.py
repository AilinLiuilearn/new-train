# -*- coding: utf-8 -*-
import json
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.image_augmentation import (
    randomBrightnessContrast,
    randomGaussianBlur,
    randomGaussianNoise,
    randomHorizontalFlip,
    randomShiftScaleRotate,
    randomVerticalFlip,
    randomcrop_lesion_center,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


STS_AUG_PRESETS = {
    'none': {
        'crop_u': 0.0, 'shift_rotate_u': 0.0, 'hflip_u': 0.0, 'vflip_u': 0.0,
        'bc_u': 0.0, 'noise_u': 0.0, 'blur_u': 0.0,
        'shift_limit': 0.0, 'scale_limit': 0.0, 'aspect_limit': 0.0, 'rotate_limit': 0.0,
    },
    'stable': {
        'crop_u': 0.0, 'shift_rotate_u': 0.35, 'hflip_u': 0.5, 'vflip_u': 0.0,
        'bc_u': 0.20, 'noise_u': 0.0, 'blur_u': 0.0,
        'shift_limit': 0.04, 'scale_limit': 0.06, 'aspect_limit': 0.03, 'rotate_limit': 8.0,
    },
    'light': {
        'crop_u': 0.15, 'shift_rotate_u': 0.45, 'hflip_u': 0.5, 'vflip_u': 0.1,
        'bc_u': 0.25, 'noise_u': 0.10, 'blur_u': 0.05,
        'shift_limit': 0.06, 'scale_limit': 0.08, 'aspect_limit': 0.04, 'rotate_limit': 12.0,
    },
    'default': {
        'crop_u': 0.35, 'shift_rotate_u': 0.55, 'hflip_u': 0.5, 'vflip_u': 0.2,
        'bc_u': 0.35, 'noise_u': 0.20, 'blur_u': 0.10,
        'shift_limit': 0.08, 'scale_limit': 0.12, 'aspect_limit': 0.05, 'rotate_limit': 15.0,
    },
}


def _pid_from_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem[:3]


def _read_split_json(path):
    with open(path, 'r') as f:
        split = json.load(f)
    return {
        'train': set(split.get('train_pids', split.get('train', []))),
        'val': set(split.get('val_pids', split.get('val', []))),
        'test': set(split.get('test_pids', split.get('test', []))),
    }


def scan_sts_records(root, require_pet=True):
    ct_dir = os.path.join(root, 'ct')
    pet_dir = os.path.join(root, 'pet')
    mask_dir = os.path.join(root, 'mask')
    if not os.path.isdir(ct_dir) or not os.path.isdir(pet_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'STS root must contain ct/, pet/, mask/: {root}')

    ct_names = {x for x in os.listdir(ct_dir) if x.lower().endswith('.png')}
    pet_names = {x for x in os.listdir(pet_dir) if x.lower().endswith('.png')}
    mask_names = {x for x in os.listdir(mask_dir) if x.lower().endswith('.png')}
    paired = sorted(ct_names & mask_names & pet_names)
    names = paired if require_pet else sorted(ct_names & mask_names)

    records = []
    for name in names:
        pid = _pid_from_name(name)
        image_id = os.path.splitext(name)[0]
        pet_path = os.path.join(pet_dir, name)
        records.append({
            'image_id': image_id,
            'case_id': pid,
            'ct_path': os.path.join(ct_dir, name),
            'pet_path': pet_path if os.path.isfile(pet_path) else None,
            'mask_path': os.path.join(mask_dir, name),
        })
    return records


def make_patient_split(records, random_state=2023, val_ratio=0.1, test_ratio=0.2):
    pids = sorted({r['case_id'] for r in records})
    rng = random.Random(int(random_state))
    rng.shuffle(pids)
    n = len(pids)
    n_test = max(1, int(round(n * float(test_ratio))))
    n_val = max(1, int(round(n * float(val_ratio))))
    n_train = max(1, n - n_val - n_test)
    train_pids = sorted(pids[:n_train])
    val_pids = sorted(pids[n_train:n_train + n_val])
    test_pids = sorted(pids[n_train + n_val:])
    return train_pids, val_pids, test_pids


def split_records(records, split_json=None, random_state=2023, val_ratio=0.1, test_ratio=0.2):
    if split_json:
        split = _read_split_json(split_json)
        train_pids, val_pids, test_pids = split['train'], split['val'], split['test']
    else:
        tr, va, te = make_patient_split(records, random_state, val_ratio, test_ratio)
        train_pids, val_pids, test_pids = set(tr), set(va), set(te)

    train_records = [r for r in records if r['case_id'] in train_pids]
    val_records = [r for r in records if r['case_id'] in val_pids]
    test_records = [r for r in records if r['case_id'] in test_pids]
    return train_records, val_records, test_records


def _normalize_rgb(single_channel):
    rgb = np.repeat(single_channel, 3, axis=0)
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def _normalize_slice(gray):
    gray = gray.astype(np.float32)
    max_v = 65535.0 if gray.max() > 255.0 else 255.0
    gray = np.clip(gray / max_v, 0.0, 1.0)
    return gray[None, ...]


def _preset_config(name):
    if name not in STS_AUG_PRESETS:
        raise ValueError(f'Unknown STS augmentation preset: {name}. Choices: {sorted(STS_AUG_PRESETS)}')
    return dict(STS_AUG_PRESETS[name])


class STSSegDataset(Dataset):
    def __init__(self, records, image_size=512, train=False, random_state=2023, aug_preset='stable'):
        self.records = list(records)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.rng = random.Random(int(random_state))
        self.aug_preset = aug_preset
        self.aug = _preset_config(aug_preset)

    def __len__(self):
        return len(self.records)

    def _load_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _resize(self, ct, pet, mask):
        if ct.shape[:2] == (self.image_size, self.image_size):
            return ct, pet, mask
        ct = cv2.resize(ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        pet = cv2.resize(pet, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return ct, pet, mask

    def _augment(self, img, mask):
        if not self.train or self.aug_preset == 'none':
            return img, mask
        aug = self.aug
        if aug['crop_u'] > 0:
            img, mask = randomcrop_lesion_center(img, mask, u=aug['crop_u'], crop_range=(0.88, 0.97), jitter_ratio=0.06)
        if aug['shift_rotate_u'] > 0:
            img, mask = randomShiftScaleRotate(
                img,
                mask,
                shift_limit=(-aug['shift_limit'], aug['shift_limit']),
                scale_limit=(-aug['scale_limit'], aug['scale_limit']),
                aspect_limit=(-aug['aspect_limit'], aug['aspect_limit']),
                rotate_limit=(-aug['rotate_limit'], aug['rotate_limit']),
                u=aug['shift_rotate_u'],
            )
        img, mask = randomHorizontalFlip(img, mask, u=aug['hflip_u'])
        img, mask = randomVerticalFlip(img, mask, u=aug['vflip_u'])
        img, mask = randomBrightnessContrast(img, mask, brightness_limit=0.08, contrast_limit=0.08, u=aug['bc_u'])
        img, mask = randomGaussianNoise(img, mask, var_limit=(2.0, 8.0), u=aug['noise_u'])
        img, mask = randomGaussianBlur(img, mask, kernel_range=(3, 5), u=aug['blur_u'])
        return img, mask

    def __getitem__(self, idx):
        record = self.records[idx]
        ct = self._load_gray(record['ct_path'])
        pet = self._load_gray(record['pet_path'])
        mask = self._load_gray(record['mask_path'])
        ct, pet, mask = self._resize(ct, pet, mask)

        img = np.stack([pet, ct], axis=-1)
        img, mask = self._augment(img, mask)
        img = img.astype(np.float32).transpose(2, 0, 1)
        mask = (mask.astype(np.float32) > 0).astype(np.float32)
        mask = mask[None, ...]

        pet_ch = _normalize_slice(img[0])
        ct_ch = _normalize_slice(img[1])
        return {
            'ct': torch.tensor(_normalize_rgb(ct_ch), dtype=torch.float32),
            'pet': torch.tensor(_normalize_rgb(pet_ch), dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'pet_available': True,
            'idx': idx,
            'case_id': record['case_id'],
            'image_id': record['image_id'],
        }


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(dataset, batch_size, num_workers, shuffle, drop_last, seed, pin_memory=True):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _print_summary(train_records, val_records, test_records, aug_preset):
    def describe(name, records):
        c = Counter(r['case_id'] for r in records)
        print(f'  {name}: slices={len(records)} patients={len(c)} min/max per patient={min(c.values()) if c else 0}/{max(c.values()) if c else 0}')
    print('[STS] patient-level split')
    describe('train', train_records)
    describe('val', val_records)
    describe('test', test_records)
    print(f'  ← 数据增强: train only, preset={aug_preset}')
    print('  ← CT/PET归一化: PNG range -> /255 or /65535 + ImageNet 标准化')


def get_sts_loaders(root, image_size=512, batch_size=8, num_workers=4, random_state=2023,
                    split_json=None, val_ratio=0.1, test_ratio=0.2, pin_memory=True,
                    aug_preset='stable'):
    records = scan_sts_records(root, require_pet=True)
    train_records, val_records, test_records = split_records(records, split_json, random_state, val_ratio, test_ratio)
    if not train_records or not val_records or not test_records:
        raise RuntimeError(f'Invalid STS split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}')
    _print_summary(train_records, val_records, test_records, aug_preset)
    train_ds = STSSegDataset(train_records, image_size=image_size, train=True, random_state=random_state, aug_preset=aug_preset)
    val_ds = STSSegDataset(val_records, image_size=image_size, train=False, random_state=random_state, aug_preset='none')
    test_ds = STSSegDataset(test_records, image_size=image_size, train=False, random_state=random_state, aug_preset='none')
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(val_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 23, pin_memory=pin_memory),
    )
