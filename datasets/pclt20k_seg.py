# -*- coding: utf-8 -*-
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.image_augmentation import (
    elasticTransform,
    randomHorizontalFlip,
    randomShiftScaleRotate,
    randomVerticalFlip,
    randomcrop,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _read_list(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return [x.strip() for x in f if x.strip()]


def _scan_records(root):
    records = []
    for case_id in sorted(os.listdir(root)):
        case_dir = os.path.join(root, case_id)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if not fname.endswith('_CT.png'):
                continue
            base = fname.replace('_CT.png', '')
            image_id = f'{case_id}_{base.split("_")[-1]}' if '_' in base else f'{case_id}_{base}'
            ct_path = os.path.join(case_dir, fname)
            pet_path = os.path.join(case_dir, f'{base}_PET.png')
            mask_path = os.path.join(case_dir, f'{base}_mask.png')
            if not os.path.isfile(mask_path):
                continue
            records.append({
                'image_id': image_id,
                'case_id': case_id,
                'ct_path': ct_path,
                'pet_path': pet_path if os.path.isfile(pet_path) else None,
                'mask_path': mask_path,
            })
    return records


def _collect_records(root, train_ids=None, test_ids=None, val_ids=None, val_ratio=0.1, random_state=2023):
    records = _scan_records(root)
    if train_ids is None or test_ids is None:
        cases = sorted({r['case_id'] for r in records})
        random.Random(random_state).shuffle(cases)
        n = len(cases)
        t_end, v_end = int(n * 0.7), int(n * 0.85)
        train_cases = set(cases[:t_end])
        val_cases = set(cases[t_end:v_end])
        test_cases = set(cases[v_end:])
        return (
            [r for r in records if r['case_id'] in train_cases],
            [r for r in records if r['case_id'] in val_cases],
            [r for r in records if r['case_id'] in test_cases],
        )

    train_records = [r for r in records if r['image_id'] in set(train_ids)]
    test_records = [r for r in records if r['image_id'] in set(test_ids)]
    if val_ids is not None:
        val_set = set(val_ids)
        val_records = [r for r in records if r['image_id'] in val_set]
        train_records = [r for r in train_records if r['image_id'] not in val_set]
    else:
        cases = sorted({r['case_id'] for r in train_records})
        random.Random(random_state).shuffle(cases)
        n_val = max(1, int(len(cases) * val_ratio))
        val_cases = set(cases[:n_val])
        val_records = [r for r in train_records if r['case_id'] in val_cases]
        train_records = [r for r in train_records if r['case_id'] not in val_cases]
    return train_records, val_records, test_records


def _split_complete_incomplete(train_records, missing_rate, random_state=2023):
    rng = random.Random(random_state)
    complete, incomplete = [], []
    for record in train_records:
        if record['pet_path'] is None:
            incomplete.append(record)
        elif missing_rate > 0 and rng.random() < missing_rate:
            incomplete.append(record)
        else:
            complete.append(record)
    return complete, incomplete


def _normalize_rgb(single_channel):
    rgb = np.repeat(single_channel, 3, axis=0)
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def _normalize_ct_slice(ct_uint8):
    ct = ct_uint8.astype(np.float32) / 255.0
    return ct[None, ...]


def _normalize_pet_slice(pet_uint8):
    pet = pet_uint8.astype(np.float32) / 255.0
    return pet[None, ...]


class PCLT20KSegDataset(Dataset):
    def __init__(self, records, image_size=512, train=False, pet_available_list=None, random_state=2023, aug_strong=False):
        self.records = records
        self.image_size = image_size
        self.train = train
        self.aug_strong = aug_strong
        self.rng = random.Random(random_state)
        self.pet_available = list(pet_available_list) if pet_available_list is not None else [r['pet_path'] is not None for r in records]

    def __len__(self):
        return len(self.records)

    def _load_image(self, path, fallback=None):
        if path is None:
            return np.zeros_like(fallback)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img if img is not None else np.zeros_like(fallback)

    def _resize(self, ct, pet, mask):
        if ct.shape[:2] == (self.image_size, self.image_size):
            return ct, pet, mask
        ct = cv2.resize(ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        pet = cv2.resize(pet, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return ct, pet, mask

    def _augment(self, img, mask):
        if not self.train:
            return img, mask

        if self.aug_strong:
            img, mask = randomShiftScaleRotate(
                img,
                mask,
                shift_limit=(-0.15, 0.15),
                scale_limit=(-0.12, 0.12),
                rotate_limit=(-30, 30),
                u=0.7,
            )
            img, mask = randomHorizontalFlip(img, mask, u=0.5)
            img, mask = randomVerticalFlip(img, mask, u=0.3)
            img, mask = randomcrop(img, mask, u=0.5)
            img, mask = elasticTransform(img, mask, alpha=60, sigma=7, u=0.4)
        else:
            img, mask = randomShiftScaleRotate(
                img,
                mask,
                shift_limit=(-0.05, 0.05),
                scale_limit=(-0.05, 0.05),
                rotate_limit=(-8, 8),
                u=0.45,
            )
            img, mask = randomHorizontalFlip(img, mask, u=0.5)

        return img, mask

    def __getitem__(self, idx):
        record = self.records[idx]
        ct = cv2.imread(record['ct_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE)
        assert ct is not None and mask is not None
        pet = self._load_image(record['pet_path'], fallback=ct)

        ct, pet, mask = self._resize(ct, pet, mask)
        img = np.stack([pet, ct], axis=-1)
        img, mask = self._augment(img, mask)

        img = img.astype(np.float32).transpose(2, 0, 1)
        mask = (mask.astype(np.float32) / 255.0)
        if mask.ndim == 2:
            mask = mask[None, ...]
        else:
            mask = mask.transpose(2, 0, 1)
        mask = (mask >= 0.5).astype(np.float32)

        pet_ch = _normalize_pet_slice(img[0])
        ct_ch = _normalize_ct_slice(img[1])
        ct_rgb = _normalize_rgb(ct_ch)
        if self.pet_available[idx]:
            pet_rgb = _normalize_rgb(pet_ch)
        else:
            pet_rgb = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        return {
            'ct': torch.tensor(ct_rgb, dtype=torch.float32),
            'pet': torch.tensor(pet_rgb, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'pet_available': self.pet_available[idx],
            'idx': idx,
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


def get_pclt20k_loaders_cipa_aligned(root, image_size=512, batch_size=8, num_workers=4, random_state=2023, aug_strong=False, pin_memory=True):
    train_ids = _read_list(os.path.join(root, 'train.txt'))
    test_ids = _read_list(os.path.join(root, 'test.txt'))
    if train_ids is None or test_ids is None:
        raise FileNotFoundError(f'未找到 train.txt 或 test.txt，请确认数据路径: {root}')

    all_records = _scan_records(root)
    train_records = [r for r in all_records if r['image_id'] in set(train_ids)]
    test_records = [r for r in all_records if r['image_id'] in set(test_ids)]
    complete_records = [r for r in train_records if r['pet_path'] is not None]

    print(f'[CIPA对齐] 训练(complete配对): {len(complete_records)}  测试/验证: {len(test_records)}')
    print('  ← 对比 CIPA: train.txt 全量用于训练, test.txt 既做 early stop 也是最终报告')
    print(f'  ← 数据增强: {"strong" if aug_strong else "weak"}')
    print('  ← PET归一化: /255 + ImageNet 标准化')

    train_ds = PCLT20KSegDataset(
        complete_records,
        image_size=image_size,
        train=True,
        pet_available_list=[True] * len(complete_records),
        random_state=random_state,
        aug_strong=aug_strong,
    )
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False)
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
    )


def get_pclt20k_loaders(root, image_size=512, batch_size=8, num_workers=4, missing_rate=0.0, val_ratio=0.1, val_ids=None, random_state=2023, use_case_split=False, aug_strong=False, pin_memory=True):
    train_txt = os.path.join(root, 'train.txt')
    test_txt = os.path.join(root, 'test.txt')
    val_txt = os.path.join(root, 'val.txt')

    if use_case_split:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)
    elif os.path.isfile(train_txt) and os.path.isfile(test_txt):
        train_ids = _read_list(train_txt)
        test_ids = _read_list(test_txt)
        if val_ids is None and os.path.isfile(val_txt):
            val_ids = _read_list(val_txt)
        train_records, val_records, test_records = _collect_records(
            root,
            train_ids,
            test_ids,
            val_ids=val_ids,
            val_ratio=val_ratio,
            random_state=random_state,
        )
    else:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)

    complete_records, _ = _split_complete_incomplete(train_records, missing_rate, random_state)

    print(f'教师：complete {len(complete_records)} 验证 {len(val_records)} 测试 {len(test_records)}')
    print(f'  ← 数据增强: {"strong" if aug_strong else "weak"}')
    print('  ← PET归一化: /255 + ImageNet 标准化')

    train_ds = PCLT20KSegDataset(complete_records, image_size=image_size, train=True, pet_available_list=[True] * len(complete_records), random_state=random_state, aug_strong=aug_strong)
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False)
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(val_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 23, pin_memory=pin_memory),
    )

