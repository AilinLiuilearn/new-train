# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--random_state', type=int, default=2023)
    args = p.parse_args()
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'), map_location='cpu')
    cfg = SegMDTConfig(args={**ckpt['config'], 'root': args.root, 'random_state': args.random_state})
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.model.load_state_dict(ckpt['model'])
    print('loaded', ckpt['epoch'])


if __name__ == '__main__':
    main()
