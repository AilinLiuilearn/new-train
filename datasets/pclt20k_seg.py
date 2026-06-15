# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
except Exception:
    cv2 = None

try:
    from utils.image_augmentation import (
        randomHorizontalFlip,
        randomShiftScaleRotate,
        randomVerticalFlip,
        randomcrop,
        randomcrop_lesion_center,
        elasticTransform,
        randomBrightnessContrast,
        randomGaussianNoise,
        randomGaussianBlur,
    )
except Exception:
    def _noop(img, mask, *args, **kwargs):
        return img, mask

    randomHorizontalFlip = _noop
    randomShiftScaleRotate = _noop
    randomVerticalFlip = _noop
    randomcrop = _noop
    randomcrop_lesion_center = _noop
    elasticTransform = _noop
    randomBrightnessContrast = _noop
    randomGaussianNoise = _noop
    randomGaussianBlur = _noop

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


def _records_from_ids(root, ids):
    records = []
    missing = []
    for image_id in ids:
        parts = image_id.split('_')
        if len(parts) < 2:
            missing.append(image_id)
            continue
        case_id = parts[0]
        case_dir = os.path.join(root, case_id)
        base = image_id
        ct_path = os.path.join(case_dir, f'{base}_CT.png')
        pet_path = os.path.join(case_dir, f'{base}_PET.png')
        mask_path = os.path.join(case_dir, f'{base}_mask.png')
        if not os.path.isfile(ct_path) or not os.path.isfile(mask_path):
            missing.append(image_id)
            continue
        records.append({
            'image_id': image_id,
            'case_id': case_id,
            'ct_path': ct_path,
            'pet_path': pet_path if os.path.isfile(pet_path) else None,
            'mask_path': mask_path,
        })
    if missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(f'划分文件中有 {len(missing)} 个样本缺少 CT 或 mask，例如: {preview}')
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


def _normalize_cipa_rgb(single_channel):
    rgb = np.repeat(single_channel, 3, axis=0)
    return rgb * 3.2 - 1.6


def _normalize_ct_slice(ct_uint8):
    ct = ct_uint8.astype(np.float32) / 255.0
    return ct[None, ...]


def _normalize_pet_slice(pet_uint8):
    pet = pet_uint8.astype(np.float32) / 255.0
    return pet[None, ...]


def _imread_grayscale(path):
    if cv2 is not None:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if path is None or not os.path.isfile(path):
        return None
    return np.array(Image.open(path).convert('L'))


def _resize_gray(img, size, nearest=False):
    if cv2 is not None:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(img, (size, size), interpolation=interpolation)
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.array(Image.fromarray(img).resize((size, size), mode))


class PCLT20KSegDataset(Dataset):
    def __init__(self, records, image_size=512, train=False, pet_available_list=None, random_state=2023, aug_mode='cipa', norm_mode='imagenet'):
        self.records = records
        self.image_size = image_size
        self.train = train
        self.aug_mode = aug_mode
        self.norm_mode = norm_mode
        self.rng = random.Random(random_state)
        self.pet_available = list(pet_available_list) if pet_available_list is not None else [r['pet_path'] is not None for r in records]

    def __len__(self):
        return len(self.records)

    def _load_image(self, path, fallback=None):
        if path is None:
            return np.zeros_like(fallback)
        img = _imread_grayscale(path)
        return img if img is not None else np.zeros_like(fallback)

    def _resize(self, ct, pet, mask):
        if ct.shape[:2] == (self.image_size, self.image_size):
            return ct, pet, mask
        ct = _resize_gray(ct, self.image_size, nearest=False)
        pet = _resize_gray(pet, self.image_size, nearest=False)
        mask = _resize_gray(mask, self.image_size, nearest=True)
        return ct, pet, mask

    def _augment(self, img, mask):
        if not self.train or self.aug_mode == 'none':
            return img, mask

        if self.aug_mode == 'cipa':
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomcrop(img, mask)
            return img, mask

        img, mask = randomcrop_lesion_center(img, mask, u=0.4, crop_range=(0.80, 0.95))
        img, mask = randomShiftScaleRotate(
            img, mask,
            shift_limit=(-0.1, 0.1),
            scale_limit=(-0.15, 0.15),
            rotate_limit=(-20, 20),
            u=0.6,
        )
        img, mask = randomHorizontalFlip(img, mask, u=0.5)
        img, mask = randomVerticalFlip(img, mask, u=0.3)
        img, mask = elasticTransform(img, mask, alpha=60, sigma=7, u=0.3)
        img, mask = randomBrightnessContrast(img, mask, brightness_limit=0.12, contrast_limit=0.12, u=0.4)
        img, mask = randomGaussianNoise(img, mask, var_limit=(3.0, 15.0), u=0.25)
        img, mask = randomGaussianBlur(img, mask, kernel_range=(3, 5), u=0.15)
        return img, mask

    def __getitem__(self, idx):
        record = self.records[idx]
        ct = _imread_grayscale(record['ct_path'])
        mask = _imread_grayscale(record['mask_path'])
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
        normalize_rgb = _normalize_cipa_rgb if self.norm_mode == 'cipa' else _normalize_rgb
        ct_rgb = normalize_rgb(ct_ch)
        if self.pet_available[idx]:
            pet_rgb = normalize_rgb(pet_ch)
        else:
            pet_rgb = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        return {
            'ct': torch.tensor(ct_rgb, dtype=torch.float32),
            'pet': torch.tensor(pet_rgb, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'pet_available': self.pet_available[idx],
            'idx': idx,
        }


class PCLT20KTextProxyAlignedDataset(PCLT20KSegDataset):
    """CIPA-aligned PET/CT segmentation dataset with training-time PET modality dropout.

    Validation/test always return real PET. Missing-modality evaluation is handled
    centrally in task.evaluate() for reproducibility.
    """

    def __init__(self, records, image_size=512, train=False, pet_drop_prob=0.0, random_state=2023, aug_mode='cipa', norm_mode='imagenet'):
        super().__init__(
            records,
            image_size=image_size,
            train=train,
            pet_available_list=[r['pet_path'] is not None for r in records],
            random_state=random_state,
            aug_mode=aug_mode,
            norm_mode=norm_mode,
        )
        self.pet_drop_prob = float(pet_drop_prob)

    def __getitem__(self, idx):
        record = self.records[idx]
        ct = _imread_grayscale(record['ct_path'])
        mask = _imread_grayscale(record['mask_path'])
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
        if self.norm_mode == 'cipa':
            ct_out = ct_ch * 3.2 - 1.6
            pet_out = pet_ch * 3.2 - 1.6
        else:
            ct_out = ct_ch
            pet_out = pet_ch

        pet_available = 1 if record['pet_path'] is not None else 0
        if self.train and random.random() < self.pet_drop_prob:
            pet_out = np.zeros_like(pet_out, dtype=np.float32)
            pet_available = 0

        image = np.concatenate([pet_out, ct_out], axis=0)
        image_id = record.get('image_id', str(idx))
        parts = image_id.split('_')
        slice_id = parts[-1] if len(parts) > 1 else image_id

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'ct': torch.tensor(ct_out, dtype=torch.float32),
            'pet': torch.tensor(pet_out, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'pet_available': torch.tensor(pet_available, dtype=torch.long),
            'case_id': record.get('case_id', parts[0] if parts else ''),
            'slice_id': slice_id,
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


def get_pclt20k_loaders_cipa_aligned(root, image_size=512, batch_size=8, num_workers=4, random_state=2023, pin_memory=True, aug_mode='cipa', norm_mode='imagenet', train_split_file='train.txt', val_split_file='val.txt', test_split_file='test.txt'):
    train_ids = _read_list(os.path.join(root, train_split_file))
    val_ids = _read_list(os.path.join(root, val_split_file))
    test_ids = _read_list(os.path.join(root, test_split_file))
    if train_ids is None or val_ids is None or test_ids is None:
        raise FileNotFoundError(f'未找到 train.txt、val.txt 或 test.txt，请确认数据路径: {root}')

    train_records = _records_from_ids(root, train_ids)
    val_records = _records_from_ids(root, val_ids)
    test_records = _records_from_ids(root, test_ids)
    complete_records = [r for r in train_records if r['pet_path'] is not None]

    print(f'[CIPA对齐] 训练(complete配对): {len(complete_records)}  验证: {len(val_records)}  测试: {len(test_records)}')
    print('  ← 使用当前划分: train.txt 训练, val.txt early stop/选最佳, test.txt 最终报告')
    print(f'  ← 数据增强: {aug_mode}')
    print(f'  ← CT/PET归一化: {norm_mode}')

    train_ds = PCLT20KSegDataset(
        complete_records,
        image_size=image_size,
        train=True,
        pet_available_list=[True] * len(complete_records),
        random_state=random_state,
        aug_mode=aug_mode,
        norm_mode=norm_mode,
    )
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False, aug_mode='none', norm_mode=norm_mode)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False, aug_mode='none', norm_mode=norm_mode)
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(val_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 23, pin_memory=pin_memory),
    )


def _resolve_textproxy_train_list(root, train_list):
    candidates = [train_list]
    if train_list != 'train_orgian.txt':
        candidates.append('train_orgian.txt')
    candidates.append('train_original.txt')
    seen = []
    for name in candidates:
        if name in seen:
            continue
        seen.append(name)
        path = os.path.join(root, name)
        ids = _read_list(path)
        if ids is not None:
            return name, ids
    raise FileNotFoundError(
        f'未找到训练划分文件。已尝试: {", ".join(seen)}；数据路径: {root}'
    )


def get_pclt20k_loaders_textproxy_aligned(root, image_size=512, batch_size=8, num_workers=4, random_state=2023, pin_memory=True, aug_mode='none', norm_mode='cipa', train_list='train_orgian.txt', val_list='test.txt', test_list='test.txt', pet_drop_prob=0.4):
    used_train_list, train_ids = _resolve_textproxy_train_list(root, train_list)
    val_ids = _read_list(os.path.join(root, val_list))
    test_ids = _read_list(os.path.join(root, test_list))
    if val_ids is None or test_ids is None:
        raise FileNotFoundError(f'未找到验证/测试划分文件: val={val_list}, test={test_list}；数据路径: {root}')

    train_records = _records_from_ids(root, train_ids)
    val_records = _records_from_ids(root, val_ids)
    test_records = _records_from_ids(root, test_ids)
    train_records = [r for r in train_records if r['pet_path'] is not None]

    print(f'[TextProxy-CIPA对齐] 训练: {len(train_records)}  验证: {len(val_records)}  测试: {len(test_records)}')
    print(f'  ← 使用划分: train={used_train_list}, val={val_list}, test={test_list}')
    print(f'  ← 数据增强: {aug_mode}')
    print(f'  ← CT/PET归一化: {norm_mode}')
    print(f'  ← train PET modality dropout: pet_drop_prob={pet_drop_prob}')
    print('  ← val/test return real PET; missing-mode eval is controlled by task.evaluate()')

    train_ds = PCLT20KTextProxyAlignedDataset(
        train_records,
        image_size=image_size,
        train=True,
        pet_drop_prob=pet_drop_prob,
        random_state=random_state,
        aug_mode=aug_mode,
        norm_mode=norm_mode,
    )
    val_ds = PCLT20KTextProxyAlignedDataset(
        val_records,
        image_size=image_size,
        train=False,
        random_state=random_state,
        aug_mode='none',
        norm_mode=norm_mode,
    )
    test_ds = PCLT20KTextProxyAlignedDataset(
        test_records,
        image_size=image_size,
        train=False,
        random_state=random_state,
        aug_mode='none',
        norm_mode=norm_mode,
    )
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(val_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 23, pin_memory=pin_memory),
    )


def get_pclt20k_loaders(root, image_size=512, batch_size=8, num_workers=4, missing_rate=0.0, val_ratio=0.1, val_ids=None, random_state=2023, use_case_split=False, pin_memory=True, aug_mode='cipa', norm_mode='imagenet'):
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
    print(f'  ← 数据增强: {aug_mode}')
    print(f'  ← CT/PET归一化: {norm_mode}')

    train_ds = PCLT20KSegDataset(complete_records, image_size=image_size, train=True, pet_available_list=[True] * len(complete_records), random_state=random_state, aug_mode=aug_mode, norm_mode=norm_mode)
    val_ds = PCLT20KSegDataset(val_records, image_size=image_size, train=False, aug_mode='none', norm_mode=norm_mode)
    test_ds = PCLT20KSegDataset(test_records, image_size=image_size, train=False, aug_mode='none', norm_mode=norm_mode)
    return (
        _make_loader(train_ds, batch_size, num_workers, True, True, random_state + 11, pin_memory=pin_memory),
        _make_loader(val_ds, batch_size, num_workers, False, False, random_state + 17, pin_memory=pin_memory),
        _make_loader(test_ds, batch_size, num_workers, False, False, random_state + 23, pin_memory=pin_memory),
    )

