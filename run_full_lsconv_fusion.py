"""Full-only Add -> LSConv at all four scales; same baseline training loop."""
import os
import csv
import json
import sys
import uuid
from pathlib import Path

import torch

from configs.base import str2bool
from models.components.lsconv_fusion import LSConvFusion, STAGE_CHANNELS
from models.components.lsconv_ska import BACKENDS
from models.full_lsconv_fusion import (
    BASELINE_COMMIT, CONFIG_FIELDS, UPSTREAM_COMMIT, build_full_lsconv_model)
from run_full_add_baseline import FullAddConfig, main as run_full


def check_runtime(require_cuda, enabled, backend):
    """Reject accidental CPU training before touching the experiment directory."""
    needs_cuda = require_cuda or (enabled and backend == 'triton')
    if needs_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            'LSConv training requires CUDA, but this Python cannot use it. '
            f'python={sys.executable}; torch={torch.__version__}; '
            f'torch.version.cuda={torch.version.cuda}; '
            f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")!r}. '
            'Run nvidia-smi and inspect this Python environment first. '
            'No experiment files have been created. '
            'CPU smoke tests require --require_cuda false --lsconv_backend torch.')


def prepare_run_directory(cfg):
    # Do not access cfg.checkpoint_dir: that property creates the directory.
    path = Path(cfg.checkpoint_root) / 'MDT' / cfg.hash
    saved_path = path / 'config_args.json'
    if not saved_path.exists():
        return
    refusal = 'This run has existing results; use a fresh --hash (no resume support)'
    if not cfg.retry_failed_start:
        raise FileExistsError(refusal)
    saved = json.loads(saved_path.read_text())
    if (saved.get('experiment_type') != 'full_lsconv_fusion'
            or saved.get('model_arch') != 'full_lsconv_fusion'
            or saved.get('hash') != cfg.hash):
        raise FileExistsError(refusal)
    allowed = {'config_args.json', 'split_summary.json',
               'train_log.csv', 'train_log_readable.txt'}
    if any(not item.is_file() or item.name not in allowed for item in path.iterdir()):
        raise FileExistsError(refusal)
    log = path / 'train_log.csv'
    if log.exists():
        with log.open(newline='') as f:
            rows = [row for row in csv.reader(f) if row]
        if len(rows) > 1 or (rows and rows[0][0] != 'epoch'):
            raise FileExistsError(refusal)
    readable = path / 'train_log_readable.txt'
    if readable.exists() and readable.read_text() != 'Training Log\n' + '=' * 64 + '\n':
        raise FileExistsError(refusal)
    # Retain every startup file instead of deleting or overwriting it.
    backup = path.with_name(path.name + '.failed-start-' + uuid.uuid4().hex[:8])
    path.rename(backup)
    print(f'[INFO] archived failed startup without epoch/checkpoint: {backup}', flush=True)


class FullLSConvConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_lsconv_fusion'
                action.choices = ('full_lsconv_fusion',)
        parser.add_argument('--lsconv_enabled', type=str2bool, default=True)
        parser.add_argument('--require_cuda', type=str2bool, default=True,
                            help='Refuse CPU training; false is only for explicit CPU smoke tests')
        parser.add_argument('--retry_failed_start', type=str2bool, default=False,
                            help='Archive startup-only files and retry; refuses epochs/checkpoints')
        parser.add_argument('--lsconv_backend', choices=BACKENDS, default='auto',
                            help='auto: official Triton on CUDA, torch reference on CPU; '
                                 'CUDA errors never silently fall back')
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        check_runtime(cfg.require_cuda, cfg.lsconv_enabled, cfg.lsconv_backend)
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.lsnet_upstream_commit = UPSTREAM_COMMIT
        cfg.lsconv_stage_channels = list(STAGE_CHANNELS)
        cfg.lsconv_stage_order = [1, 2, 3, 4]
        cfg.lsconv_final_bn_init = 'zeros'
        cfg.lsconv_large_kernel, cfg.lsconv_small_kernel, cfg.lsconv_groups = 7, 3, 8
        prepare_run_directory(cfg)
        return cfg


def main():
    return run_full(config_cls=FullLSConvConfig, model_builder=build_full_lsconv_model,
                    expected_fusion=LSConvFusion, experiment_type='full_lsconv_fusion',
                    result_config_fields=CONFIG_FIELDS, extra_grad_modules=('pet_align', 'fusion'))


if __name__ == '__main__':
    main()
