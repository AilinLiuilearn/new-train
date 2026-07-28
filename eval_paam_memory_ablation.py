import argparse
import copy
import json
import os

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from configs.seg_mdt import SegMDTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=('normal', 'memory_off', 'shuffled_value'), default='normal')
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_dict = ckpt.get('config', {})
    cfg = SegMDTConfig.parse_arguments()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    model = build_mdt_seg_teacher(cfg)['model']
    model.load_state_dict(ckpt['model'])
    backups = []
    for mem in getattr(model, 'paam').memories:
        backups.append((mem.keys.clone(), mem.gamma_proto.clone(), mem.beta_proto.clone(), mem.memory_ready.clone()))
    if args.mode == 'memory_off':
        for mem in model.paam.memories:
            mem.memory_ready.fill_(False)
    elif args.mode == 'shuffled_value':
        g = torch.Generator().manual_seed(2026)
        for mem in model.paam.memories:
            idx = torch.randperm(mem.K, generator=g)
            mem.gamma_proto.copy_(mem.gamma_proto[idx])
            mem.beta_proto.copy_(mem.beta_proto[idx])
    print(json.dumps({'mode': args.mode, 'status': 'loaded'}, indent=2))
    for mem, backup in zip(model.paam.memories, backups):
        mem.keys.copy_(backup[0]); mem.gamma_proto.copy_(backup[1]); mem.beta_proto.copy_(backup[2]); mem.memory_ready.copy_(backup[3])


if __name__ == '__main__':
    main()
