# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.image_augmentation import randomShiftScaleRotate, randomHorizontalFlip, randomVerticalFlip, randomcrop, elasticTransform


def _read_list(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return [x.strip() for x in f if x.strip()]


def _collect_records(root, train_ids=None, test_ids=None, val_ids=None, val_ratio=0.1, random_state=2023):
    records = []
    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if not fname.endswith('_CT.png'):
                continue
            base = fname.replace('_CT.png', '')
            image_id = f"{name}_{base.split('_')[-1]}" if '_' in base else f"{name}_{base}"
            ct_path = os.path.join(case_dir, fname)
            pet_path = os.path.join(case_dir, f"{base}_PET.png")
            mask_path = os.path.join(case_dir, f"{base}_mask.png")
            if not os.path.isfile(mask_path):
                continue
            records.append({'image_id': image_id, 'case_id': name, 'ct_path': ct_path,
                            'pet_path': pet_path if os.path.isfile(pet_path) else None, 'mask_path': mask_path})
    if train_ids is None or test_ids is None:
        cases = list(set(r['case_id'] for r in records))
        random.Random(random_state).shuffle(cases)
        n = len(cases)
        t_end, v_end = int(n * 0.7), int(n * 0.85)
        train_c, val_c, test_c = set(cases[:t_end]), set(cases[t_end:v_end]), set(cases[v_end:])
        return ([r for r in records if r['case_id'] in train_c],
                [r for r in records if r['case_id'] in val_c],
                [r for r in records if r['case_id'] in test_c])
    train_records = [r for r in records if r['image_id'] in set(train_ids)]
    test_records = [r for r in records if r['image_id'] in set(test_ids)]
    if val_ids is not None:
        val_set = set(val_ids)
        val_records = [r for r in records if r['image_id'] in val_set]
        train_records = [r for r in train_records if r['image_id'] not in val_set]
    else:
        cases = list(set(r['case_id'] for r in train_records))
        random.Random(random_state).shuffle(cases)
        n_val = max(1, int(len(cases) * val_ratio))
        val_cases = set(cases[:n_val])
        val_records = [r for r in train_records if r['case_id'] in val_cases]
        train_records = [r for r in train_records if r['case_id'] not in val_cases]
    return train_records, val_records, test_records


def _split_complete_incomplete(train_records, missing_rate, random_state=2023):
    rng = random.Random(random_state)
    complete, incomplete = [], []
    for r in train_records:
        if r['pet_path'] is None:
            incomplete.append(r)
        elif missing_rate <= 0:
            complete.append(r)
        elif rng.random() < missing_rate:
            incomplete.append(r)
        else:
            complete.append(r)
    return complete, incomplete


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

    def _load_and_augment(self, idx):
        r = self.records[idx]
        ct = cv2.imread(r['ct_path'], cv2.IMREAD_GRAYSCALE)
        pet = cv2.imread(r['pet_path'], cv2.IMREAD_GRAYSCALE) if r['pet_path'] else np.zeros_like(ct)
        mask = cv2.imread(r['mask_path'], cv2.IMREAD_GRAYSCALE)
        assert ct is not None and mask is not None
        if pet is None:
            pet = np.zeros_like(ct)
        if ct.shape[:2] != (self.image_size, self.image_size):
            ct = cv2.resize(ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            pet = cv2.resize(pet, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if ct.ndim == 2:
            ct = np.expand_dims(ct, 2)
        if pet.ndim == 2:
            pet = np.expand_dims(pet, 2)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, 2)
        img = np.concatenate([pet, ct], axis=2)
        if self.train:
            img, mask = randomShiftScaleRotate(img, mask, shift_limit=(-0.15, 0.15), scale_limit=(-0.12, 0.12), u=0.7)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticalFlip(img, mask, u=0.5)
            img, mask = randomcrop(img, mask, u=0.6)
            img, mask = elasticTransform(img, mask, alpha=80, sigma=8, u=0.8)
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.squeeze(2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32)
        mask = mask[None, ...] if mask.ndim == 2 else mask.transpose(2, 0, 1)
        mask = mask / 255.0
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        return img[0:1], img[1:2], mask

    def __getitem__(self, idx):
        pet_ch, ct_ch, mask = self._load_and_augment(idx)
        use_pet = self.pet_available[idx]
        if not use_pet:
            pet_ch = np.zeros_like(pet_ch)
        return {'ct': torch.tensor(ct_ch, dtype=torch.float32), 'pet': torch.tensor(pet_ch, dtype=torch.float32), 'mask': torch.tensor(mask, dtype=torch.float32), 'pet_available': use_pet, 'idx': idx}


def get_pclt20k_loaders_cipa_aligned(root, image_size=512, batch_size=8, num_workers=4, random_state=2023, aug_strong=False):
    train_ids = _read_list(os.path.join(root, 'train.txt'))
    test_ids = _read_list(os.path.join(root, 'test.txt'))
    if train_ids is None or test_ids is None:
        raise FileNotFoundError(f'未找到 train.txt 或 test.txt，请确认数据路径: {root}')

    all_records = []
    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if not fname.endswith('_CT.png'):
                continue
            base = fname.replace('_CT.png', '')
            image_id = f"{name}_{base.split('_')[-1]}" if '_' in base else f"{name}_{base}"
            ct_path = os.path.join(case_dir, fname)
            pet_path = os.path.join(case_dir, f"{base}_PET.png")
            mask_path = os.path.join(case_dir, f"{base}_mask.png")
            if not os.path.isfile(mask_path):
                continue
            all_records.append({'image_id': image_id, 'case_id': name, 'ct_path': ct_path,
                                'pet_path': pet_path if os.path.isfile(pet_path) else None, 'mask_path': mask_path})
    train_records = [r for r in all_records if r['image_id'] in set(train_ids)]
    test_records = [r for r in all_records if r['image_id'] in set(test_ids)]
    complete_records = [r for r in train_records if r['pet_path'] is not None]
    print(f'[CIPA对齐] 训练(complete配对): {len(complete_records)}  测试/验证: {len(test_records)}')
    print('  ← 对比 CIPA: train.txt 全量用于训练, test.txt 既做 early stop 也是最终报告')
    train_ds = PCLT20KSegDataset(complete_records, image_size=image_size, train=True, pet_available_list=[True] * len(complete_records), random_state=random_state, aug_strong=aug_strong)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


def get_pclt20k_loaders(root, image_size=512, batch_size=8, num_workers=4, missing_rate=0.0, val_ratio=0.1, val_ids=None, random_state=2023, use_case_split=False, aug_strong=False):
    train_txt, test_txt, val_txt = os.path.join(root, 'train.txt'), os.path.join(root, 'test.txt'), os.path.join(root, 'val.txt')
    if use_case_split:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)
    elif os.path.isfile(train_txt) and os.path.isfile(test_txt):
        train_ids, test_ids = _read_list(train_txt), _read_list(test_txt)
        if val_ids is None and os.path.isfile(val_txt):
            val_ids = _read_list(val_txt)
        train_records, val_records, test_records = _collect_records(root, train_ids, test_ids, val_ids=val_ids, val_ratio=val_ratio, random_state=random_state)
    else:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)
    complete_records, _ = _split_complete_incomplete(train_records, missing_rate, random_state)
    print(f'教师：complete {len(complete_records)} 验证 {len(val_records)} 测试 {len(test_records)}')
    train_ds = PCLT20KSegDataset(complete_records, image_size=image_size, train=True, pet_available_list=[True] * len(complete_records), random_state=random_state, aug_strong=aug_strong)
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
    )


def get_pclt20k_loaders_student(root, image_size=512, batch_size=8, num_workers=4, missing_rate=0.3, val_ratio=0.1, val_ids=None, random_state=2023, use_case_split=False, aug_strong=False):
    train_txt, test_txt, val_txt = os.path.join(root, 'train.txt'), os.path.join(root, 'test.txt'), os.path.join(root, 'val.txt')
    if use_case_split:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)
    elif os.path.isfile(train_txt) and os.path.isfile(test_txt):
        train_ids, test_ids = _read_list(train_txt), _read_list(test_txt)
        if val_ids is None and os.path.isfile(val_txt):
            val_ids = _read_list(val_txt)
        train_records, val_records, test_records = _collect_records(root, train_ids, test_ids, val_ids=val_ids, val_ratio=val_ratio, random_state=random_state)
    else:
        train_records, val_records, test_records = _collect_records(root, random_state=random_state)
    complete_records, incomplete_records = _split_complete_incomplete(train_records, missing_rate, random_state)
    print(f'学生：配对 {len(complete_records)} 非配对 {len(incomplete_records)} 验证 {len(val_records)} 测试 {len(test_records)}')
    train_paired_ds = PCLT20KSegDataset(complete_records, image_size=image_size, train=True, pet_available_list=[True] * len(complete_records), random_state=random_state, aug_strong=aug_strong)
    train_mri_ds = PCLT20KSegDataset(incomplete_records, image_size=image_size, train=True, pet_available_list=[False] * len(incomplete_records), random_state=random_state, aug_strong=aug_strong)
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False)
    train_paired_loader = torch.utils.data.DataLoader(train_paired_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True) if len(complete_records) > 0 else None
    train_mri_loader = torch.utils.data.DataLoader(train_mri_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True) if len(incomplete_records) > 0 else None
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_paired_loader, train_mri_loader, val_loader, test_loader
