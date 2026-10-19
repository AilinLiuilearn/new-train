"""Two-step dual ConvNeXtV2-Nano Full-Add preflight; no dataset/task checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.base import str2bool
from scripts.check_mit_b1_baselines import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--nano_pretrained_path',
        default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.batch_size < 2 or args.image_size < 32 or args.image_size % 32:
        raise ValueError('Use batch_size>=2 and image_size>=32 divisible by 32')
    check(args, 'full', backbone='convnextv2_nano',
          pretrained_path=args.nano_pretrained_path)


if __name__ == '__main__':
    main()
