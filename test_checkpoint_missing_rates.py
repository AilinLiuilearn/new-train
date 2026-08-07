# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_mdt_seg_teacher
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.seg_losses import BCEDiceLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a CPPI checkpoint at 0% and 100% PET missing rates.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(checkpoint_config, checkpoint_path):
    config = dict(checkpoint_config)
    config['checkpoint_root'] = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    config['hash'] = os.path.basename(os.path.dirname(checkpoint_path))
    config['checkpoint_dir'] = os.path.dirname(checkpoint_path)
    return SimpleNamespace(**config)


@torch.no_grad()
def evaluate(model, loader, device, criterion, forward_mode):
    model.eval()
    metrics = SegmentationMetricsCIPA(threshold=0.5)
    total_loss = 0.0
    sample_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        ct = batch['ct'].to(device, non_blocking=True)
        pet = batch['pet'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True).float()
        outputs = model(ct, pet=pet, forward_mode=forward_mode, mask=None)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, _ = criterion(logits, mask)
        metrics.update(logits, mask)
        batch_size = int(ct.shape[0])
        total_loss += float(loss) * batch_size
        sample_count += batch_size
        if batch_idx % 100 == 0:
            print(
                f'[{forward_mode}] batch={batch_idx}/{len(loader)} samples={sample_count}',
                flush=True,
            )
    result = metrics.compute()
    result['total_loss'] = total_loss / max(1, sample_count)
    result['num_samples'] = sample_count
    result['forward_mode'] = forward_mode
    return result


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = build_config(checkpoint['config'], checkpoint_path)
    seed_everything(int(getattr(config, 'random_state', 2023)))

    eval_batch_size = args.batch_size or int(getattr(config, 'batch_size', 16))
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        config.root,
        config.image_size_2d,
        eval_batch_size,
        args.num_workers,
        config.random_state,
        config.pin_memory,
        config.aug_mode,
        config.norm_mode,
        config.train_split_file,
        config.val_split_file,
        config.test_split_file,
        checkpoint_dir=os.path.dirname(checkpoint_path),
    )

    model = build_mdt_seg_teacher(config)['model']
    load_result = model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    criterion = BCEDiceLoss(
        smooth=config.loss_smooth,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
    )

    print(
        f'[CHECKPOINT] epoch={checkpoint.get("epoch")} '
        f'best_joint={checkpoint.get("best_joint")} '
        f'bank_version={int(model.prototype_memory.bank_version.item())} '
        f'ready_slots={int(model.prototype_memory.prototype_ready.sum().item())}/'
        f'{model.prototype_memory.prototype_ready.numel()}',
        flush=True,
    )
    print(f'[LOAD] {load_result}', flush=True)
    print(f'[TEST] samples={len(test_loader.dataset)} batches={len(test_loader)}', flush=True)

    results = {
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_best_joint': checkpoint.get('best_joint'),
        'threshold': 0.5,
        'zero_percent_missing': evaluate(
            model, test_loader, device, criterion, forward_mode='full'
        ),
        'one_hundred_percent_missing': evaluate(
            model, test_loader, device, criterion, forward_mode='missing'
        ),
    }
    output_path = args.output or os.path.join(
        os.path.dirname(checkpoint_path), 'test_missing_0_100_results.json'
    )
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f'[SAVED] {output_path}', flush=True)


if __name__ == '__main__':
    main()
