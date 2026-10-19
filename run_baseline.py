"""Train independent CT-only or Full aligned-add baseline on all training rows."""
import json
import os
import time
import torch
from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_dual_model
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import seed_everything, optimizer_step_succeeded
from utils.train_logger import init_train_log, append_epoch_log


def loaders(cfg):
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir, ct_only=cfg.baseline == 'ct_only',
        allow_val_equals_test=cfg.allow_val_equals_test)


def evaluate(task, loader, model):
    if task.config.baseline == 'ct_only':
        return {'ct_only': task.evaluate(loader, 'missing', model=model)}
    full = task.evaluate(loader, 'full', model=model)
    missing = task.evaluate(loader, 'missing', model=model)
    return {'full': full, 'missing': missing,
            'joint_dice': 0.5 * (full['dice'] + missing['dice'])}


def main():
    cfg = SegMDTConfig.parse_arguments()
    if cfg.optimizer != 'adamw' or cfg.accumulation_steps != 1 or cfg.deep_supervision or cfg.use_deep_supervision:
        raise ValueError('requires AdamW, accumulation=1, no deep supervision')
    if (cfg.ct_backbone, cfg.pet_backbone) != ('convnextv2_nano', 'mit_b1'):
        raise ValueError('fixed heterogeneous encoders required')
    directory = cfg.checkpoint_dir
    if os.path.exists(os.path.join(directory, 'config_args.json')):
        raise FileExistsError('existing run: choose a new --hash')
    seed_everything(cfg.random_state)
    cfg.experiment_type = 'clean_ct_full_add_hetero_lr8e5'
    cfg.save()
    with open(os.path.join(directory, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2)
    train, val, test = loaders(cfg)
    if not len(train):
        raise ValueError('empty training loader')
    task = MDTSegTeacher(build_dual_model(cfg), cfg)
    task.scheduler = get_cosine_scheduler(task.optimizer, cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train), steps_per_epoch=len(train),
        min_lr=cfg.cosine_min_lr, flat_ratio=cfg.lr_flat_ratio)
    log = os.path.join(directory, 'train_log.csv')
    headers = ['val_full_dice', 'val_missing_dice', 'joint_dice', 'ema_updates', 'skipped_updates', 'epoch_time']
    init_train_log(log, extra_headers=headers)
    best, best_epoch, stale = -1., 0, 0
    best_path = os.path.join(directory, 'ckpt.best.pth.tar')
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        task.model.train()
        task.begin_epoch(epoch)
        loss_sum = samples = skipped = 0
        grad_sum = 0.
        for batch in train:
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and task.device.type == 'cuda'):
                loss, _ = task.train_step(batch)
            task.scaler.scale(loss).backward()
            task.scaler.unscale_(task.optimizer)
            if cfg.grad_clip > 0:
                grad_sum += float(torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), cfg.grad_clip))
            if optimizer_step_succeeded(task):
                task.scheduler.step()
                task.update_ema()
            else:
                skipped += 1
            task.global_batch_step += 1
            n = len(batch['ct'])
            samples += n
            loss_sum += float(loss.detach()) * n
        result = evaluate(task, val, task.eval_model())
        primary = result['ct_only'] if cfg.baseline == 'ct_only' else result['full']
        # Select each independent baseline by its own trained operating condition.
        score = primary['dice']
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            task.save_checkpoint(best_path, epoch, best, best_epoch, result)
        else:
            stale += 1
        task.save_checkpoint(os.path.join(directory, 'ckpt.last.pth.tar'), epoch, best, best_epoch, result)
        append_epoch_log(log, epoch, loss_sum / samples, primary,
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=grad_sum / len(train),
            extra_metrics={'val_full_dice': result.get('full', {}).get('dice', float('nan')),
                'val_missing_dice': result.get('missing', {}).get('dice', float('nan')),
                'joint_dice': result.get('joint_dice', float('nan')),
                'ema_updates': task.ema.updates if task.ema else 0,
                'skipped_updates': skipped, 'epoch_time': time.time() - start})
        print(f'[epoch {epoch}] {cfg.baseline} dice={score:.4f} best={best:.4f} results={result}', flush=True)
        if stale >= cfg.early_stop_patience:
            break
    checkpoint = task.load_state_dicts(best_path)
    tag = checkpoint['eval_weights']
    task.model.load_state_dict(checkpoint['model_ema'] if tag == 'ema' else checkpoint['model'])
    final = {'epoch': checkpoint['epoch'], 'weights': tag,
             'selection': cfg.baseline, 'metrics': evaluate(task, test, task.model)}
    with open(os.path.join(directory, 'final_test.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(final, flush=True)


if __name__ == '__main__':
    main()
