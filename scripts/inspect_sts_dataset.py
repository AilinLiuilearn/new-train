# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.sts_seg import scan_sts_records


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {'count': 0}
    return {
        'count': int(arr.size),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'p1': float(np.percentile(arr, 1)),
        'p50': float(np.percentile(arr, 50)),
        'p99': float(np.percentile(arr, 99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--max_stat_images', type=int, default=0, help='0 means all paired records')
    parser.add_argument('--output_json', type=str, default=None)
    args = parser.parse_args()

    records = scan_sts_records(args.root, require_pet=True)
    ct_names = {x for x in os.listdir(os.path.join(args.root, 'ct')) if x.lower().endswith('.png')}
    pet_names = {x for x in os.listdir(os.path.join(args.root, 'pet')) if x.lower().endswith('.png')}
    mask_names = {x for x in os.listdir(os.path.join(args.root, 'mask')) if x.lower().endswith('.png')}
    paired_names = {os.path.basename(r['ct_path']) for r in records}
    by_pid = Counter(r['case_id'] for r in records)

    sample_records = records if args.max_stat_images <= 0 else records[:args.max_stat_images]
    ct_values, pet_values, mask_areas = [], [], []
    empty_masks = 0
    for r in sample_records:
        ct = cv2.imread(r['ct_path'], cv2.IMREAD_UNCHANGED)
        pet = cv2.imread(r['pet_path'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(r['mask_path'], cv2.IMREAD_UNCHANGED)
        if ct is None or pet is None or mask is None:
            continue
        ct_values.extend([float(ct.min()), float(ct.max()), float(ct.mean())])
        pet_values.extend([float(pet.min()), float(pet.max()), float(pet.mean())])
        area = float((mask > 0).mean())
        mask_areas.append(area)
        empty_masks += int(area == 0.0)

    out = {
        'root': args.root,
        'counts': {
            'ct': len(ct_names),
            'pet': len(pet_names),
            'mask': len(mask_names),
            'paired': len(records),
            'ct_unpaired_examples': sorted(ct_names - paired_names)[:20],
            'pet_unpaired_examples': sorted(pet_names - paired_names)[:20],
            'mask_unpaired_examples': sorted(mask_names - paired_names)[:20],
        },
        'patients': {
            'count': len(by_pid),
            'min_slices': min(by_pid.values()) if by_pid else 0,
            'max_slices': max(by_pid.values()) if by_pid else 0,
            'slice_counts': dict(sorted(by_pid.items())),
        },
        'sampled_for_stats': len(sample_records),
        'ct_summary_values_min_max_mean_stream': _stats(ct_values),
        'pet_summary_values_min_max_mean_stream': _stats(pet_values),
        'mask_area_ratio': _stats(mask_areas),
        'empty_masks': empty_masks,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
