# -*- coding: utf-8 -*-
"""Check whether PG-MTR auxiliary losses are active in a checkpoint.

This script loads a checkpoint, runs one Full forward pass and one Missing forward
pass, and prints:
- segmentation loss
- PG-MTR route loss
- PG-MTR memory loss
- whether the auxiliary losses are finite / non-zero
- a backward-gradient sanity check for the Full auxiliary loss

It is intentionally lightweight and does not start training.
"""

import argparse
import os

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _load_dataset_module():
    import importlib.util

    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_loaders(config):
    dataset_mod = _load_dataset_module()
    return dataset_mod.get_pclt20k_loaders_textproxy_aligned(
        config.root,
        config.image_size_2d,
        config.batch_size,
        config.num_workers,
        config.random_state,
        pin_memory=getattr(config, 'pin_memory', True),
        aug_mode=getattr(config, 'aug_mode', 'cipa'),
        norm_mode=getattr(config, 'norm_mode', 'cipa'),
        train_list=getattr(config, 'train_list', 'train_original.txt'),
        val_list=getattr(config, 'val_list', 'test.txt'),
        test_list=getattr(config, 'test_list', 'test.txt'),
        pet_drop_prob=getattr(config, 'train_pet_drop_prob', 0.0),
    )


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_checkpoint(task, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            task.load_model_state_dict(v, ckpt[k], strict=False)
    return ckpt


def _first_batch(loader):
    return next(iter(loader))


def _print_loss_block(title, outputs, loss_seg, aux_losses):
    route_loss = aux_losses.get('pg_mtr_route_loss')
    mem_loss = aux_losses.get('pg_mtr_mem_loss')
    route_v = float(route_loss.detach().cpu()) if torch.is_tensor(route_loss) else float(route_loss or 0.0)
    mem_v = float(mem_loss.detach().cpu()) if torch.is_tensor(mem_loss) else float(mem_loss or 0.0)
    print(f'[{title}] seg_loss={float(loss_seg.detach().cpu()):.6f} route_loss={route_v:.6f} mem_loss={mem_v:.6f}')
    if isinstance(outputs, dict):
        diag = outputs.get('diagnostics', {})
        keys = [k for k in sorted(diag.keys()) if k.startswith('pg_mtr_')]
        if keys:
            print(f'[{title}] diagnostics:')
            for k in keys:
                v = diag[k]
                if torch.is_tensor(v):
                    v = float(v.detach().float().mean().cpu())
                else:
                    v = float(v)
                print(f'  {k}={v:.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size_2d', type=int, default=512)
    parser.add_argument('--root', required=True)
    parser.add_argument('--train_list', default='train_original.txt')
    parser.add_argument('--val_list', default='test.txt')
    parser.add_argument('--test_list', default='test.txt')
    parser.add_argument('--ct_backbone', default='convnextv2_nano')
    parser.add_argument('--pet_backbone', default='mit_b1')
    parser.add_argument('--ct_pretrained_path', default=None)
    parser.add_argument('--pet_pretrained_path', default=None)
    parser.add_argument('--pg_mtr_stages', default='all')
    parser.add_argument('--pg_mtr_num_tokens', type=int, default=8)
    parser.add_argument('--pg_mtr_temperature', type=float, default=0.07)
    parser.add_argument('--pg_mtr_detach_bank_missing', action='store_true')
    parser.add_argument('--use_aligned_loader', action='store_true')
    parser.add_argument('--aug_mode', default='cipa')
    parser.add_argument('--norm_mode', default='cipa')
    args = parser.parse_args()

    config = SegMDTConfig()
    config.model_arch = 'dual_decoder_pg_mtr_retrieval'
    config.ct_backbone = args.ct_backbone
    config.pet_backbone = args.pet_backbone
    config.ct_pretrained_path = args.ct_pretrained_path
    config.pet_pretrained_path = args.pet_pretrained_path
    config.pg_mtr_stages = args.pg_mtr_stages
    config.pg_mtr_num_tokens = args.pg_mtr_num_tokens
    config.pg_mtr_temperature = args.pg_mtr_temperature
    config.pg_mtr_detach_bank_missing = bool(args.pg_mtr_detach_bank_missing)
    config.batch_size = args.batch_size
    config.image_size_2d = args.image_size_2d
    config.root = args.root
    config.train_list = args.train_list
    config.val_list = args.val_list
    config.test_list = args.test_list
    config.use_aligned_loader = bool(args.use_aligned_loader)
    config.aug_mode = args.aug_mode
    config.norm_mode = args.norm_mode
    config.gpus = [int(x) for x in str(args.gpus).split(',') if x.strip()]
    config.random_state = getattr(config, 'random_state', 2026)
    config.num_workers = getattr(config, 'num_workers', 4)
    config.pin_memory = getattr(config, 'pin_memory', True)
    config.mixed_precision = False
    config.learning_rate = getattr(config, 'learning_rate', 8e-5)
    config.decoder_lr = getattr(config, 'decoder_lr', 8e-5)
    config.weight_decay = getattr(config, 'weight_decay', 1e-4)
    config.optimizer = getattr(config, 'optimizer', 'adamw')
    config.bce_weight = getattr(config, 'bce_weight', 1.0)
    config.dice_weight = getattr(config, 'dice_weight', 1.0)
    config.loss_smooth = getattr(config, 'loss_smooth', 1.0)
    config.pos_weight = getattr(config, 'pos_weight', None)

    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpus[0])

    loaders = _build_loaders(config)
    if isinstance(loaders, (tuple, list)) and len(loaders) >= 3:
        train_loader, val_loader, test_loader = loaders[:3]
    else:
        raise RuntimeError('Unexpected loader return value.')

    model_dict = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(model_dict, config)

    _load_checkpoint(task, args.checkpoint_path)
    model = _unwrap_model(task.networks['model'])
    model.eval()

    batch = _first_batch(test_loader)
    ct = batch['ct'].float().to(task.device)
    pet = batch['pet'].float().to(task.device)
    mask = batch['mask'].float().to(task.device)
    target_size = mask.shape[-2:]

    print(f'Loaded checkpoint: {args.checkpoint_path}')
    print(f'Active stages: {model.pg_mtr.active_stage_numbers}')

    route_weight = float(getattr(config, 'pg_mtr_route_weight', 0.1))
    mem_weight = float(getattr(config, 'pg_mtr_mem_weight', 0.05))

    with torch.no_grad():
        full_out = model._forward_full(ct, pet, target_size, mask=mask)
        full_pred = full_out['logits']
        full_seg_loss, _, _ = task._compute_segmentation_loss(full_out, mask)
        print('\n=== FULL FORWARD ===')
        _print_loss_block('FULL', full_out, full_seg_loss, full_out.get('aux_losses', {}))
        print(f'[FULL] logits finite={bool(torch.isfinite(full_pred).all())} shape={tuple(full_pred.shape)}')
        print(f'[FULL] route_loss.requires_grad={bool(full_out.get("aux_losses", {}).get("pg_mtr_route_loss").requires_grad)}')
        print(f'[FULL] mem_loss.requires_grad={bool(full_out.get("aux_losses", {}).get("pg_mtr_mem_loss").requires_grad)}')

        missing_out = model._forward_missing(ct, target_size)
        missing_pred = missing_out['logits']
        missing_seg_loss, _, _ = task._compute_segmentation_loss(missing_out, mask)
        print('\n=== MISSING FORWARD ===')
        _print_loss_block('MISSING', missing_out, missing_seg_loss, missing_out.get('aux_losses', {}))
        print(f'[MISSING] logits finite={bool(torch.isfinite(missing_pred).all())} shape={tuple(missing_pred.shape)}')

    # Backward sanity check for Full auxiliary loss only.
    model.zero_grad(set_to_none=True)
    full_out = model._forward_full(ct, pet, target_size, mask=mask)
    aux_losses = full_out.get('aux_losses', {})
    route_loss = aux_losses.get('pg_mtr_route_loss')
    mem_loss = aux_losses.get('pg_mtr_mem_loss')
    aux_total = route_weight * route_loss + mem_weight * mem_loss
    print('\n=== BACKWARD SANITY CHECK (FULL AUX ONLY) ===')
    print(f'pg_mtr_route_loss={float(route_loss.detach().cpu()):.6f}')
    print(f'pg_mtr_mem_loss={float(mem_loss.detach().cpu()):.6f}')
    print(f'aux_total.requires_grad={bool(aux_total.requires_grad)}')
    if aux_total.requires_grad:
        aux_total.backward()
        print('backward_ran=True')
    else:
        print('backward_ran=False (aux losses are detached / zero in this checkpoint path)')
    print('pg_mtr has grad:', any(p.grad is not None for p in model.pg_mtr.parameters() if p.requires_grad))
    print('full_decoder has grad:', any(p.grad is not None for p in model.full_decoder.parameters() if p.requires_grad))
    print('missing_decoder has grad:', any(p.grad is not None for p in model.missing_decoder.parameters() if p.requires_grad))
    print('enc_ct has grad:', any(p.grad is not None for p in model.enc_ct.parameters() if p.requires_grad))
    print('enc_pet has grad:', any(p.grad is not None for p in model.enc_pet.parameters() if p.requires_grad))
    print('retrieval_adapters has grad:', any(p.grad is not None for p in model.retrieval_adapters.parameters() if p.requires_grad))
    print('pg_mtr.shared_memory_tokens.grad is None:', model.pg_mtr.shared_memory_tokens.grad is None)
    print('pg_mtr.shared_token_key.weight.grad is None:', model.pg_mtr.shared_token_key.weight.grad is None)
    print('pg_mtr.shared_token_value.weight.grad is None:', model.pg_mtr.shared_token_value.weight.grad is None)
    writer_ct_q = model.pg_mtr.stage_modules[str(model.pg_mtr.writer_stage)].ct_query_proj
    print('pg_mtr.writer ct_query_proj has grad:', any(p.grad is not None for p in writer_ct_q.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()
