#!/usr/bin/env python
# Temporary Stage2 DP-PGFA pipeline smoke test. Not a full training run.
import argparse
import gc
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_prompt_pgfa import FULL_TEXT, MISSING_TEXT
from models.dual_shared_add_baseline import DP_SCALE_CHANNELS, DP_SCALE_NAMES
from run_mdt_seg import (
    _count_group_trainable,
    _count_parameters,
    _dp_core_grad_norm,
    _dp_out_proj_grad_norm,
    _load_stage1_for_dp,
    _print_stage2_startup,
    _trainable_grads_finite,
    _zero_step_identity_check,
    module_grad_norm,
)
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss


class Cfg:
    def __init__(self, scales, hash_name):
        self.checkpoint_root = os.path.join(ROOT, 'checkpoints_new')
        self.hash = hash_name
        self.task = 'MDT'
        self.ct_backbone = 'convnextv2_nano'
        self.pet_backbone = 'mit_b1'
        self.ct_pretrained_path = os.path.join(ROOT, 'pretrained/convnextv2_nano')
        self.pet_pretrained_path = os.path.join(ROOT, 'pretrained/mit-b1')
        self.no_encoder_pretrained = False
        self.decoder_channels = (512, 256, 128, 64)
        self.use_deep_supervision = False
        self.deep_supervision = False
        self.cppi_num_clusters = 6
        self.cppi_build_stage = 4
        self.dp_pgfa_enabled = True
        self.dp_pgfa_scales = scales
        self.dp_text_tower_path = os.path.join(ROOT, 'pretrained/biomedbert_text_tower')
        self.dp_biomedclip_model_path = os.path.join(ROOT, 'pretrained/biomedclip_model')
        self.dp_window_size = 8
        self.dp_depth = 2
        self.dp_prompt_len = 128
        self.dp_compress_ratio = 8
        self.dp_use_task_prompt = True
        self.dp_use_text_prompt = True
        self.learning_rate = 8e-5
        self.weight_decay = 1e-4
        self.mixed_precision = True
        self.loss_smooth = 1.0
        self.bce_weight = 1.0
        self.dice_weight = 1.0
        self.stage1_checkpoint = os.path.join(
            ROOT,
            'checkpoints_new/MDT/e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar',
        )

    @property
    def checkpoint_dir(self):
        return os.path.join(self.checkpoint_root, 'MDT', self.hash)


def _make_batch(device, batch_size=2, image_size=512):
    return {
        'ct': torch.randn(batch_size, 1, image_size, image_size, device=device),
        'pet': torch.randn(batch_size, 1, image_size, image_size, device=device),
        'mask': (torch.rand(batch_size, 1, image_size, image_size, device=device) > 0.8).float(),
    }


def _assert_no_text_encoder(model):
    bad = [n for n, _ in model.named_parameters() if 'biomed' in n.lower() or 'bert' in n.lower() or 'text_tower' in n.lower()]
    if bad:
        raise RuntimeError(f'Text encoder params leaked into model: {bad[:10]}')


def _grad_step(task, batch, route, amp_enabled, steps=1):
    criterion_stats = []
    for step in range(steps):
        task.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, _, outputs, _ = task.train_step(batch, forward_mode=route)
        if task.scaler.is_enabled():
            task.scaler.scale(loss).backward()
            task.scaler.unscale_(task.optimizer)
        else:
            loss.backward()
        ok, bad = _trainable_grads_finite(task.model)
        if not ok:
            raise RuntimeError(f'nonfinite grad at step={step + 1} param={bad}')
        stats = {
            'loss': float(loss.detach()),
            'dp': module_grad_norm(torch.nn.ModuleList([a for _, a in task.model._iter_dp_adapters(active_only=True)])),
            'out_proj': _dp_out_proj_grad_norm(task.model),
            'core': _dp_core_grad_norm(task.model),
            'enc_ct': module_grad_norm(task.model.enc_ct),
            'decoder': module_grad_norm(task.model.decoder),
            'outputs': outputs,
        }
        criterion_stats.append(stats)
        if task.scaler.is_enabled():
            task.scaler.step(task.optimizer)
            task.scaler.update()
        else:
            task.optimizer.step()
    return criterion_stats


def run_smoke(scales, hash_name, batch_size=2, do_backward=True):
    print(f'\n===== PIPELINE SMOKE scales={scales} =====', flush=True)
    cfg = Cfg(scales=scales, hash_name=hash_name)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    stage1_info = _load_stage1_for_dp(model, cfg.stage1_checkpoint)
    model.enable_stage2_dp_only()
    _print_stage2_startup(model, cfg, stage1_info)
    _assert_no_text_encoder(model)

    task = MDTSegTeacher(networks, cfg)
    batch = _make_batch(device, batch_size=batch_size)
    zero = _zero_step_identity_check(task.model, batch, device, amp_enabled=True)

    # Full / Missing / Auto forwards
    task.model.train()
    with torch.cuda.amp.autocast(enabled=True):
        out_full = task.model(batch['ct'], pet=batch['pet'], forward_mode='full', mask=batch['mask'])
        out_missing = task.model(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=batch['mask'])
        avail = torch.tensor([1, 0][: batch['ct'].shape[0]], device=device, dtype=torch.long)
        if batch['ct'].shape[0] == 1:
            avail = torch.tensor([0], device=device, dtype=torch.long)
        out_auto = task.model(
            batch['ct'], pet=batch['pet'], pet_available=avail, forward_mode='auto', mask=batch['mask']
        )

    # Shape assertions on active scales via fused path
    ct_feats = task.model._encode_ct(batch['ct'])
    pet_proxy, ct_ref, _ = task.model._retrieve_cppi(ct_feats, return_ct_reference=True)
    ct_ref = [x.detach() for x in ct_ref]
    pet_cal = task.model.pet_calibration(ct_feats, pet_proxy, ct_ref, reference_valid=True)
    fused = task.model.fusion(ct_feats, pet_cal, mode='missing')
    refined, stats = task.model._refine_dp_pgfa(fused, route='missing')
    for scale_name in task.model.dp_pgfa_scales:
        idx = DP_SCALE_NAMES.index(scale_name)
        c = DP_SCALE_CHANNELS[scale_name]
        assert fused[idx].shape[1] == c
        assert refined[idx].shape == fused[idx].shape
        print(f'[SHAPE] {scale_name} {tuple(fused[idx].shape)} -> {tuple(refined[idx].shape)}', flush=True)

    # Missing route invariants
    assert task.model._last_missing_used_real_pet is False
    assert task.model._last_missing_cppi_collect is False
    assert task.model._last_missing_cppi_retrieve is True
    print('[MISSING ROUTE]', flush=True)
    print('real_pet_encoder_called=False', flush=True)
    print('cppi_collect_called=False', flush=True)
    print('cppi_retrieve_called=True', flush=True)
    print('selected_text=Missing', flush=True)

    # Text constants unchanged
    assert FULL_TEXT.startswith('This fused feature combines CT structural information with detailed')
    assert MISSING_TEXT.startswith('This fused feature combines CT structural information with smooth')

    grad_report = {}
    if do_backward:
        full_stats = _grad_step(task, batch, 'full', amp_enabled=True, steps=1)
        missing_stats = _grad_step(task, batch, 'missing', amp_enabled=True, steps=2)
        # after step1 core may be ~0; after step2/3 should become nonzero typically
        grad_report = {
            'dp_grad': missing_stats[-1]['dp'],
            'dp_out_proj_grad': missing_stats[-1]['out_proj'],
            'dp_core_grad_step1': full_stats[0]['core'],
            'dp_core_grad_step2_or_3': missing_stats[-1]['core'],
            'stage1_grad': missing_stats[-1]['enc_ct'],
            'decoder_grad': missing_stats[-1]['decoder'],
            'nonfinite': 0,
            'loss_finite': all(math.isfinite(s['loss']) for s in full_stats + missing_stats),
        }
        if grad_report['dp_grad'] <= 0:
            raise RuntimeError('DP grad must be > 0')
        if grad_report['stage1_grad'] != 0 or grad_report['decoder_grad'] != 0:
            raise RuntimeError('Stage1/decoder grads must be 0')
        print('[GRADIENT]', flush=True)
        for k, v in grad_report.items():
            print(f'{k}={v}', flush=True)

    peak = None
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'[AMP] PASS=True peak_memory_mb={peak:.1f}', flush=True)

    total, trainable = _count_parameters(task.model)
    report = {
        'scales': list(task.model.dp_pgfa_scales),
        'stage1': stage1_info,
        'zero': zero,
        'grad': grad_report,
        'peak_memory_mb': peak,
        'trainable': trainable,
        'total': total,
        'text_dim': task.model.dp_text_embedding_dim,
        'text_cosine': task.model.dp_full_missing_text_cosine,
        'dp_trainable': _count_group_trainable(task.model, ('dp_pgfa_',)),
        'stage1_trainable': _count_group_trainable(
            task.model,
            ('enc_ct.', 'enc_pet.', 'ct_align.', 'pet_calibration.', 'fusion.', 'prototype_memory.'),
        ),
        'decoder_trainable': _count_group_trainable(task.model, ('decoder.',)),
        'full_ok': out_full['logits'].shape[0] == batch_size,
        'missing_ok': out_missing['logits'].shape[0] == batch_size,
        'auto_ok': out_auto['logits'].shape[0] == batch_size,
    }
    print(f'[SMOKE] scales={scales} PASS', flush=True)
    del task, model, networks, batch
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('s4', 'all', 'both'), default='both')
    args = parser.parse_args()
    reports = {}
    if args.mode in ('s4', 'both'):
        reports['s4'] = run_smoke('s4', 'smoke-dp-pgfa-s4', batch_size=2, do_backward=True)
    if args.mode in ('all', 'both'):
        reports['all'] = run_smoke('all', 'smoke-dp-pgfa-all', batch_size=1, do_backward=True)
    print('\n===== SUMMARY =====', flush=True)
    for name, rep in reports.items():
        print(name, {k: rep[k] for k in ('scales', 'peak_memory_mb', 'zero', 'grad') if k in rep}, flush=True)


if __name__ == '__main__':
    main()
