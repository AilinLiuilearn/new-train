# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.sts_seg import make_patient_split, scan_sts_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()

    records = scan_sts_records(args.root, require_pet=True)
    train_pids, val_pids, test_pids = make_patient_split(records, args.seed, args.val_ratio, args.test_ratio)
    by_pid = Counter(r['case_id'] for r in records)

    def n_slices(pids):
        return sum(by_pid[p] for p in pids)

    split = {
        'dataset': 'STS_dicom_aligned',
        'root': args.root,
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'total_patients': len(by_pid),
        'total_slices': len(records),
        'train_pids': train_pids,
        'val_pids': val_pids,
        'test_pids': test_pids,
        'summary': {
            'train_patients': len(train_pids),
            'val_patients': len(val_pids),
            'test_patients': len(test_pids),
            'train_slices': n_slices(train_pids),
            'val_slices': n_slices(val_pids),
            'test_slices': n_slices(test_pids),
        },
        'slice_counts_by_pid': dict(sorted(by_pid.items())),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(json.dumps(split['summary'], indent=2, ensure_ascii=False))
    print(f'Saved split: {args.output}')


if __name__ == '__main__':
    main()
