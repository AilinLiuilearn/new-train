"""Evaluate this branch's checkpoint; never reload encoder pretraining."""
import argparse
import json
from pathlib import Path

from configs.full import FullConfig
from models.build_model import build_model
from tasks.segmentation import SegmentationTask
from train_full import _loaders


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--root')
    p.add_argument('--test_split_file')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--output')
    args=p.parse_args()
    ckpt=SegmentationTask.load_state_dicts(args.checkpoint)
    cfg=FullConfig(args=ckpt['config'])
    cfg.device=args.device
    if args.root:cfg.root=args.root
    if args.test_split_file:cfg.test_split_file=args.test_split_file
    task=SegmentationTask({'model':build_model(cfg,load_encoder_pretrained=False)},cfg)
    tag=ckpt['eval_weights']
    if tag not in ('raw','ema') or (tag=='ema' and ckpt.get('model_ema') is None):
        raise RuntimeError('Invalid recorded eval weights')
    task.model.load_state_dict(ckpt['model_ema'] if tag=='ema' else ckpt['model'],strict=True)
    _,_,loader=_loaders(cfg)
    result=task.evaluate(loader,eval_mode='full',model=task.model)
    result.update(weights=tag,checkpoint=str(Path(args.checkpoint).resolve()),
                  fusion=cfg.fusion,hd95_mode=cfg.hd95_mode,hd95_unit='pixel')
    print(json.dumps(result,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
