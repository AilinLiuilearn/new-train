"""Isolated Full experiment switches and non-destructive startup validation."""
import csv
import json
import sys
import uuid
from pathlib import Path

import torch

from configs.base import str2bool
from models.components.cmx_fusion import STAGE_CHANNELS, STAGE_HEADS, UPSTREAM_COMMIT, UPSTREAM_BLOB
from models.full_cmx_fusion import BASELINE_COMMIT
from models.components import skff_fusion
from run_full_add_baseline import FullAddConfig


def validate_config(cfg):
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
        'experiment_type': 'full_cmx_fusion',
        'model_arch': 'full_cmx_fusion', 'hash': cfg.hash}.items()):
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


class FullCMXConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        p = FullAddConfig.model_parser()
        for a in p._actions:
            if a.dest == 'model_arch':
                a.default = 'full_cmx_fusion'
                a.choices = ('full_cmx_fusion',)
        p.add_argument('--cmx_ffm_enabled', type=str2bool, default=True)
        p.add_argument('--cmx_frm_enabled', type=str2bool, default=False)
        p.add_argument('--fusion_backend', choices=('cmx', 'skff'), default='cmx',
                       help='Choose one fusion; skff disables CMX FFM/FRM')
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
        cfg.cmx_stage_channels = list(STAGE_CHANNELS)
        cfg.cmx_stage_heads = list(STAGE_HEADS)
        cfg.cmx_stage_order = [1, 2, 3, 4]
        cfg.cmx_reduction = 1
        cfg.cmx_initialization = 'official'
        cfg.cmx_output_mode = 'replace_add'
        cfg.cmx_upstream_commit = UPSTREAM_COMMIT
        cfg.cmx_source_blob = UPSTREAM_BLOB
        if cfg.fusion_backend == 'skff':
            if cfg.cmx_frm_enabled:
                raise ValueError('SKFF replaces CMX; --cmx_frm_enabled must be false')
            cfg.cmx_ffm_enabled = False
            cfg.skff_stage_channels = list(skff_fusion.STAGE_CHANNELS)
            cfg.skff_stage_order = [1, 2, 3, 4]
            cfg.skff_height = 2
            cfg.skff_reduction = 8
            cfg.skff_output_scale = 1
            cfg.skff_initialization = 'official'
            cfg.skff_upstream_commit = skff_fusion.UPSTREAM_COMMIT
            cfg.skff_source_blob = skff_fusion.UPSTREAM_BLOB
        prepare_run_directory(cfg)
        return cfg
