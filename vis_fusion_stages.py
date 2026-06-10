# -*- coding: utf-8 -*-
"""Standalone visualization for PET-guided wavelet fusion stage outputs."""

import argparse
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from utils.vis_teacher import save_segmentation_diagnostics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _load_dataset_module():
    root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f'[vis_fusion_stages] loaded checkpoint: {checkpoint_path}')
    print(f'[vis_fusion_stages] load status: {msg}')


class _VisTask:
    def __init__(self, model, device):
        self.networks = {'model': model}
        self.device = device


def parse_args():
    parser = argparse.ArgumentParser('Visualize fusion stage outputs')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./fusion_stage_vis')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--gpus', type=str, nargs='+', default=['0'])
    parser.add_argument('--image_size_2d', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=2023)
    parser.add_argument('--cipa_aligned', type=str2bool, default=True)
    parser.add_argument('--backbone', type=str, default='convnext_nano')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--decoder_type', type=str, default='light')
    parser.add_argument('--use_tcpm', type=str2bool, default=False)
    parser.add_argument('--fusion_type', type=str, default='pet_window_wavelet')
    parser.add_argument('--wavelet_window_sizes', type=int, nargs='+', default=[8, 8, 4, 4])
    parser.add_argument('--wavelet_heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--wavelet_sr_ratios', type=int, nargs='+', default=[4, 4, 2, 1])
    parser.add_argument('--wavelet_attn_ratio', type=float, default=0.25)
    parser.add_argument('--wavelet_conv_ratio', type=float, default=0.125)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    args = parse_args()
    device = torch.device('cuda', int(args.gpus[0])) if torch.cuda.is_available() else torch.device('cpu')

    cfg = SimpleNamespace(
        backbone=args.backbone,
        pretrained_path=args.pretrained_path,
        use_tcpm=args.use_tcpm,
        decoder_type=args.decoder_type,
        fusion_type=args.fusion_type,
        wavelet_window_sizes=args.wavelet_window_sizes,
        wavelet_heads=args.wavelet_heads,
        wavelet_sr_ratios=args.wavelet_sr_ratios,
        wavelet_attn_ratio=args.wavelet_attn_ratio,
        wavelet_conv_ratio=args.wavelet_conv_ratio,
    )
    model = build_mdt_seg_teacher(cfg)['model']
    _load_model_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    dataset_mod = _load_dataset_module()
    if args.cipa_aligned:
        _, val_loader, _ = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            args.root,
            args.image_size_2d,
            args.batch_size,
            args.num_workers,
            args.random_state,
            pin_memory=True,
        )
    else:
        _, val_loader, _ = dataset_mod.get_pclt20k_loaders(
            args.root,
            args.image_size_2d,
            args.batch_size,
            args.num_workers,
            val_ratio=0.1,
            random_state=args.random_state,
            use_case_split=True,
            pin_memory=True,
        )

    task = _VisTask(model, device)
    save_segmentation_diagnostics(
        task=task,
        loader=val_loader,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
