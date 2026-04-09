# -*- coding: utf-8 -*-
"""
从 train.txt 按 val_ratio 分出验证集，生成 val.txt 并更新 train.txt。
只需执行一次。用法：
  python split_val.py --root ../data/PCLT20K [--val_ratio 0.1]
"""

import os
import sys
import argparse

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from datasets.pclt20k_seg import split_train_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/PCLT20K', help='PCLT20K 根目录')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--random_state', type=int, default=2023)
    args = parser.parse_args()
    split_train_val(args.root, val_ratio=args.val_ratio, random_state=args.random_state)


if __name__ == '__main__':
    main()
