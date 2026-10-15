"""Isolated Full experiment switches and non-destructive startup validation."""
import csv
import json
import math
import sys
import uuid
from pathlib import Path

import torch

from configs.base import str2bool
from models.components.lesion_guided_fusion import DEFAULT_WIDTHS, STAGE_CHANNELS, LOCAL_KERNELS
from models.full_lesion_guided_fusion import BASELINE_COMMIT
from run_full_add_baseline import FullAddConfig


def validate_config(cfg):
    if (not math.isfinite(cfg.lesion_aux_loss_weight) or cfg.lesion_aux_loss_weight < 0):
        raise ValueError('lesion_aux_loss_weight must be finite and nonnegative')
    if len(cfg.lesion_widths) != 4 or any(w < 2 for w in cfg.lesion_widths):
        raise ValueError('lesion_widths must contain four integers >=2')
    if not cfg.lesion_fusion_enabled and cfg.lesion_aux_loss_enabled and cfg.lesion_aux_loss_weight > 0:
        raise ValueError('Disable lesion_aux_loss_enabled when the fusion module is disabled')
    if (cfg.accumulation_steps != 1 or cfg.optimizer != 'adamw'
            or cfg.use_deep_supervision or cfg.deep_supervision):
        raise ValueError('Full comparison requires AdamW, accumulation_steps=1, no decoder deep supervision')


def prepare_run_directory(cfg):
    folder = Path(cfg.checkpoint_root) / 'MDT' / cfg.hash
    if not folder.exists() or not any(folder.iterdir()):
        return
    refusal = 'Existing run contains results or unknown files; use a fresh --hash (no resume support)'
    saved_path = folder / 'config_args.json'
    if not cfg.retry_failed_start or not saved_path.is_file():
        raise FileExistsError(refusal)
    saved = json.loads(saved_path.read_text())
    if any(saved.get(k) != v for k, v in {
        'experiment_type': 'full_lesion_guided_fusion',
        'model_arch': 'full_lesion_guided_fusion', 'hash': cfg.hash}.items()):
        raise FileExistsError(refusal)
    allowed = {'config_args.json', 'split_summary.json', 'train_log.csv', 'train_log_readable.txt'}
    if any(not p.is_file() or p.name not in allowed for p in folder.iterdir()):
        raise FileExistsError(refusal)
    log = folder / 'train_log.csv'
    if log.exists():
        with log.open(newline='') as f:
            rows = [row for row in csv.reader(f) if row]
        if len(rows) > 1 or (rows and rows[0][0] != 'epoch'):
            raise FileExistsError(refusal)
    readable = folder / 'train_log_readable.txt'
    if readable.exists() and readable.read_text() != 'Training Log\n' + '=' * 64 + '\n':
        raise FileExistsError(refusal)
    backup = folder.with_name(folder.name + '.failed-start-' + uuid.uuid4().hex[:8])
    folder.rename(backup)
    print(f'[INFO] archived startup-only run: {backup}', flush=True)


class FullLesionGuidedConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        p = FullAddConfig.model_parser()
        for a in p._actions:
            if a.dest == 'model_arch':
                a.default = 'full_lesion_guided_fusion'
                a.choices = ('full_lesion_guided_fusion',)
        for name in ('fusion_enabled', 'correction_enabled', 'guidance_enabled', 'aux_loss_enabled'):
            p.add_argument('--lesion_' + name, type=str2bool, default=True)
        p.add_argument('--lesion_aux_loss_weight', type=float, default=.1)
        p.add_argument('--lesion_widths', type=int, nargs=4, default=list(DEFAULT_WIDTHS))
        p.add_argument('--require_cuda', type=str2bool, default=True)
        p.add_argument('--retry_failed_start', type=str2bool, default=False)
        return p

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        validate_config(cfg)
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA required but unavailable; python={sys.executable}; '
                               f'torch={torch.__version__}; torch.version.cuda={torch.version.cuda}. '
                               'No experiment files have been created. CPU smoke: --require_cuda false')
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.lesion_stage_channels = list(STAGE_CHANNELS)
        cfg.lesion_stage_order = [1, 2, 3, 4]
        cfg.lesion_local_kernels = list(LOCAL_KERNELS)
        cfg.lesion_output_init = 'zeros'
        cfg.lesion_aux_target = 'upsample_logits_to_original_gt'
        prepare_run_directory(cfg)
        return cfg
