"""Evaluate saved prototype checkpoint; never collects or changes memories."""
import argparse
import os

from configs.base import str2bool
from run_decoder_fg_bg_prototype_add import ForegroundBackgroundConfig, _loaders
from models.fg_bg_prototype_add_model import build_fg_bg_model
from tasks.fg_bg_prototype_seg import ForegroundBackgroundSegTask
from utils.run_common import eval_missing_rates, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--use_ema', type=str2bool, default=None,
                   help='Default: weights used for checkpoint selection')
    args = p.parse_args()
    ckpt = ForegroundBackgroundSegTask.load_state_dicts(args.checkpoint)
    if ckpt.get('experiment_type') != 'decoder_fg_bg_scale_prototype_add':
        raise ValueError('expected a decoder prototype checkpoint')
    saved = dict(ckpt['config'])
    saved.pop('checkpoint_dir', None)
    saved.update(pretrained=False, ct_pretrained_path=None, pet_pretrained_path=None,
                 ema_enabled=False, aug_mode='none')
    if args.root is not None:
        saved['root'] = args.root
    cfg = ForegroundBackgroundConfig(args=saved)
    seed_everything(cfg.random_state)
    task = ForegroundBackgroundSegTask(build_fg_bg_model(cfg), cfg)
    use_ema = ckpt['eval_weights'] == 'ema' if args.use_ema is None else args.use_ema
    state = ckpt.get('model_ema') if use_ema else ckpt['model']
    if state is None:
        raise ValueError('EMA weights requested but not stored')
    task.model.load_state_dict(state, strict=True)
    task.model.eval()
    _, _, test = _loaders(cfg)
    output = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output, exist_ok=True)
    eval_missing_rates(task, task.model, test, cfg.random_state, output, 'final_test',
                       rates=cfg.final_test_missing_rates, weights_tag='ema' if use_ema else 'raw')


if __name__ == '__main__':
    main()
