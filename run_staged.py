"""Clean CT-only pretraining -> dual-decoder Full/Missing joint training."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from datasets.staged import loaders, split_signature
from models.staged_petct import StagedPETCT
from utils.ema import ModelEMA
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_cosine_scheduler
from utils.run_common import seed_everything, module_grad_norm
from utils.seg_losses import BCEDiceLoss

FORMAT = 'ct_pretrain_dual_decoder_v1'


def load(path):
    # Only load your own trusted local checkpoints (legacy files include RNG objects).
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic_save(payload, path):
    tmp = str(path)+'.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected(checkpoint, preference='auto'):
    tag = checkpoint.get('eval_weights') if preference == 'auto' else preference
    if tag not in ('raw', 'ema'):
        raise ValueError('Checkpoint has no valid eval_weights; explicitly pass --stage1_weights raw or ema')
    key = 'model_ema' if tag == 'ema' else 'model'
    state = checkpoint.get(key)
    if state is None and tag == 'raw':
        state = checkpoint.get('state_dict')
        if state is None and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            state = checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Selected {tag} weights are absent')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state, tag


def import_stage1(model, path, cfg, signature):
    checkpoint = load(path)
    old = checkpoint.get('config', {})
    if old.get('stage', 1) != 1 or old.get('baseline', 'ct_only') != 'ct_only':
        raise ValueError('The source must be a CT-only stage-1 checkpoint')
    checked, unknown = [], []
    expected = {'ct_backbone': 'convnextv2_nano', 'decoder_channels': [512, 256, 128, 64],
                'image_size_2d': cfg.image_size_2d, 'norm_mode': cfg.norm_mode,
                'aug_mode': cfg.aug_mode, 'train_split_file': cfg.train_split_file,
                'val_split_file': cfg.val_split_file, 'test_split_file': cfg.test_split_file,
                'use_deep_supervision': False, 'deep_supervision': False}
    for key, wanted in expected.items():
        if key not in old:
            unknown.append(key)
            continue
        value = list(old[key]) if key == 'decoder_channels' else old[key]
        if value != wanted:
            raise ValueError(f'Stage-1 protocol mismatch: {key}={value!r}, expected {wanted!r}')
        checked.append(key)
    if 'splits' in checkpoint and checkpoint['splits'] != signature:
        raise ValueError('Stage-1 split contents differ from this run')
    state, tag = selected(checkpoint, cfg.stage1_weights)
    report = model.import_stage1_state(state)
    report.update(path=str(Path(path).resolve()), sha256=file_sha(path),
                  epoch=checkpoint.get('epoch'), weights=tag, checked_metadata=checked,
                  unavailable_metadata=unknown,
                  split_hash_verified='splits' in checkpoint,
                  optimizer_reset=True, scheduler_reset=True, ema_reset=True)
    print('[TRANSFER] '+json.dumps(report), flush=True)
    if unknown or 'splits' not in checkpoint:
        print('[TRANSFER] Legacy checkpoint: architecture is strict; unavailable data metadata '
              'cannot establish identical preprocessing/split contents. See transfer.json.', flush=True)
    return report


def forward_loss(model, batch, stage, device, criterion):
    ct = batch['ct'].to(device, non_blocking=True)
    target = batch['mask'].to(device, non_blocking=True)
    if stage == 1:
        out = model(ct, mode='missing')
        loss = criterion(out['ct_logits'], target)[0]
        return loss, {'ct_loss': float(loss.detach()), 'full_loss': 0.,
                      'ct_samples': len(ct), 'full_samples': 0, 'missing_samples': len(ct)}
    full = batch['is_full'].to(device=device, dtype=torch.bool)
    indices = full.nonzero(as_tuple=True)[0]
    if indices.numel()*2 != len(ct):
        raise ValueError('Stage 2 requires exact half Full rows')
    # Subset before device transfer and before PET encoding. Missing PET rows are unused.
    pet = batch['pet'][batch['is_full'].bool()].to(device, non_blocking=True)
    out = model(ct, pet, mode='mixed', full_indices=indices)
    lc = criterion(out['ct_logits'], target)[0]
    lf = criterion(out['full_logits'], target.index_select(0, indices))[0]
    return .5*(lc+lf), {'ct_loss': float(lc.detach()), 'full_loss': float(lf.detach()),
                        'ct_samples': len(ct), 'full_samples': len(indices),
                        'missing_samples': len(ct)-len(indices)}


@torch.inference_mode()
def evaluate(model, loader, cfg, device, criterion, tag, weights):
    model.eval()
    names = ['missing'] if cfg.stage == 1 or cfg.eval_mode == 'missing' else ['full', 'missing']
    metrics = {k: SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode) for k in names}
    sums = {k: 0. for k in names}
    count = 0
    print(f'[{tag}] start modes={names} batches={len(loader)} weights={weights}', flush=True)
    for step, batch in enumerate(loader, 1):
        ct, y = batch['ct'].to(device), batch['mask'].to(device)
        # Existing baseline evaluates FP32. Reuse one CT pass for both modes.
        if names == ['missing']:
            out = model(ct, mode='missing')
            logits = {'missing': out['ct_logits']}
        else:
            out = model(ct, batch['pet'].to(device), mode='full')
            logits = {'full': out['full_logits'], 'missing': out['ct_logits']}
        for key, value in logits.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f'Nonfinite {tag} {key}')
            metrics[key].update(value, y)
            sums[key] += float(criterion(value, y)[0])*len(ct)
        count += len(ct)
        if step == 1 or step % cfg.log_interval == 0 or step == len(loader):
            print(f'[{tag}] {step}/{len(loader)}', flush=True)
    if not count:
        raise ValueError('Empty evaluation loader')
    result = {k: {**metrics[k].compute(), 'total_loss': sums[k]/count,
                  'eval_mode': k, 'weights': weights} for k in names}
    if 'full' in result:
        result['joint_dice'] = .5*(result['full']['dice']+result['missing']['dice'])
    return result


def rng_state(train):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'loader': train.generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, train):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); train.generator.set_state(state['loader'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=int, choices=[1, 2])
    p.add_argument('--output', help='New run directory, never overwritten')
    p.add_argument('--stage1_checkpoint', help='Trusted CT-only checkpoint, required for new stage 2')
    p.add_argument('--stage1_weights', choices=['auto', 'raw', 'ema'], default='auto')
    p.add_argument('--resume', help='Resume last.pt at an epoch boundary using saved config')
    p.add_argument('--evaluate', help='Evaluate selected raw/EMA checkpoint, no training')
    p.add_argument('--eval_mode', choices=['both', 'missing'], default='both',
                   help='Evaluation only; missing requires no PET files')
    p.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--train_split_file', default='train_original.txt')
    p.add_argument('--val_split_file', default='val.txt')
    p.add_argument('--test_split_file', default='test.txt')
    p.add_argument('--allow_val_equals_test', action='store_true')
    p.add_argument('--ct_pretrained_path', default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    p.add_argument('--pet_pretrained_path', default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    p.add_argument('--no_pretrained', action='store_true', help='Random encoders: synthetic checks only')
    p.add_argument('--image_size_2d', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--aug_mode', choices=['cipa', 'none'], default='cipa')
    p.add_argument('--norm_mode', choices=['imagenet', 'cipa'], default='imagenet')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--learning_rate', type=float, default=8e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--cosine_warmup', type=int, default=3)
    p.add_argument('--cosine_min_lr', type=float, default=1e-6)
    p.add_argument('--lr_flat_ratio', type=float, default=.3)
    p.add_argument('--mixed_precision', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ema_enabled', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ema_decay', type=float, default=.999)
    p.add_argument('--ema_start_epoch', type=int, default=3)
    p.add_argument('--grad_clip', type=float, default=5.)
    p.add_argument('--random_state', type=int, default=2023)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--hd95_mode', choices=['cipa_official', 'surface', 'legacy'], default='cipa_official')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p


def validate(cfg, evaluating=False):
    if cfg.stage not in (1, 2) or not cfg.output:
        raise ValueError('--stage and --output are required for new runs')
    if cfg.stage == 2 and (cfg.batch_size < 2 or cfg.batch_size % 2):
        raise ValueError('Stage 2 requires an even batch_size >= 2')
    if cfg.batch_size < 1 or cfg.image_size_2d < 64 or cfg.image_size_2d % 32:
        raise ValueError('batch_size >= 1; image_size_2d must be >=64 and a multiple of 32')
    if cfg.hd95_mode == 'cipa_official' and cfg.image_size_2d != 512:
        raise ValueError('cipa_official HD95 requires image_size_2d=512; use surface for small smoke tests')
    if cfg.epochs < 1 or not 0 <= cfg.cosine_warmup < cfg.epochs:
        raise ValueError('epochs must exceed cosine_warmup')
    if cfg.log_interval < 1 or cfg.num_workers < 0 or not 0 < cfg.learning_rate:
        raise ValueError('invalid log interval/workers/lr')
    if not 0 <= cfg.lr_flat_ratio < 1 or not 0 <= cfg.ema_decay < 1:
        raise ValueError('invalid schedule/EMA')
    if not evaluating and cfg.eval_mode != 'both':
        raise ValueError('--eval_mode missing is evaluation-only')


def save_environment(output, device):
    versions = {}
    for name in ['torch', 'torchvision', 'timm', 'transformers', 'numpy', 'opencv-python-headless', 'MedPy']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
    (output/'environment.json').write_text(json.dumps({'versions': versions, 'python': sys.version,
        'commit': commit, 'dirty': bool(dirty), 'cuda': torch.version.cuda,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}, indent=2))


def append_log(output, row):
    with (output/'train.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    path = output/'train.csv'
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume and args.evaluate:
        raise ValueError('Choose --resume or --evaluate')
    checkpoint = load(args.resume or args.evaluate) if args.resume or args.evaluate else None
    if checkpoint and checkpoint.get('format') != FORMAT:
        raise ValueError('Resume/evaluate needs a new staged checkpoint; use --stage1_checkpoint for legacy CT weights')
    cfg = argparse.Namespace(**checkpoint['config']) if checkpoint else args
    if checkpoint:
        supplied = set((argv if argv is not None else sys.argv[1:]))
        allowed = {'--evaluate', '--resume', '--device'}
        if args.evaluate: allowed |= {'--root', '--eval_mode'}
        unknown = {x.split('=')[0] for x in supplied if x.startswith('--')} - allowed
        if unknown:
            raise ValueError(f'Saved config is authoritative; unsupported overrides: {sorted(unknown)}')
        cfg.device = args.device
        if args.evaluate:
            if '--root' in supplied or any(x.startswith('--root=') for x in supplied): cfg.root = args.root
            cfg.eval_mode = args.eval_mode
    validate(cfg, bool(args.evaluate))
    if not checkpoint and cfg.stage == 2 and not cfg.stage1_checkpoint:
        raise ValueError('New stage 2 requires --stage1_checkpoint')
    if checkpoint is None:
        cfg.output = str(Path(cfg.output).resolve())
        cfg.root = str(Path(cfg.root).resolve())
    output = Path(cfg.output)
    if not checkpoint and output.exists() and any(output.iterdir()):
        raise FileExistsError('Run directory is not empty; use a new --output or --resume last.pt')
    if args.resume:
        if Path(args.resume).name != 'last.pt':
            raise ValueError('Resume last.pt; evaluate best.pt instead')
        if not (output/'best.pt').is_file():
            raise FileNotFoundError('Resume requires the original output directory and best.pt')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.random_state)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device)
    signature = split_signature(cfg)
    if checkpoint and signature != checkpoint['splits']:
        raise ValueError('Dataset split contents changed')
    print(f'[START] stage={cfg.stage} device={device} batch={cfg.batch_size} lr={cfg.learning_rate} '
          f'size={cfg.image_size_2d} epochs={cfg.epochs}', flush=True)
    print('[DATA] checking split membership and PNG paths', flush=True)
    train, val, test = loaders(cfg, output)
    if not len(train): raise ValueError('No complete train batch; reduce batch_size')
    print(f'[DATA] train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)} '
          f'train_batches={len(train)} dropped_per_epoch={len(train.dataset)%cfg.batch_size}', flush=True)
    fresh = checkpoint is None
    model = StagedPETCT(cfg.stage,
        ct_pretrained_path=cfg.ct_pretrained_path if fresh and cfg.stage == 1 and not cfg.no_pretrained else None,
        pet_pretrained_path=cfg.pet_pretrained_path if fresh and cfg.stage == 2 and not cfg.no_pretrained else None)
    source = None
    if checkpoint:
        model.load_state_dict(selected(checkpoint)[0] if args.evaluate else checkpoint['model'], strict=True)
        source = checkpoint['source']
    elif cfg.stage == 2:
        source = import_stage1(model, cfg.stage1_checkpoint, cfg, signature)
        (output/'transfer.json').write_text(json.dumps(source, indent=2))
    model.to(device)
    if not all(p.requires_grad for p in model.parameters()):
        raise AssertionError('This baseline freezes no parameters')
    print(f'[MODEL] params={sum(p.numel() for p in model.parameters()):,} all trainable; '
          'align=BN decoder=GN, no retrieval/distillation', flush=True)
    criterion = BCEDiceLoss()
    if args.evaluate:
        result = {'epoch': checkpoint['epoch'], 'weights': checkpoint['eval_weights'],
                  'metrics': evaluate(model, test, cfg, device, criterion, 'test', checkpoint['eval_weights'])}
        (output/f'reeval_{time.time_ns()}.json').write_text(json.dumps(result, indent=2))
        print(result, flush=True)
        return result
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_scheduler(optimizer, cfg.epochs, warmup_steps=cfg.cosine_warmup*len(train),
        min_lr=cfg.cosine_min_lr, steps_per_epoch=len(train), flat_ratio=cfg.lr_flat_ratio)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg.mixed_precision)
    ema = ModelEMA(model, cfg.ema_decay, warmup=True, device=device) if cfg.ema_enabled else None
    active, best, best_epoch, start_epoch, global_step = False, -1., 0, 1, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        active, best, best_epoch = checkpoint['ema_active'], checkpoint['best'], checkpoint['best_epoch']
        start_epoch, global_step = checkpoint['epoch']+1, checkpoint['global_step']
        if ema:
            ema.model.load_state_dict(checkpoint['model_ema'], strict=True)
            ema.updates = checkpoint['ema_updates']
        restore_rng(checkpoint['rng'], train)
        # A crash after last.pt but before log append can omit a row; never duplicate newer rows.
        log = output/'train.jsonl'
        if log.exists():
            rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            rows = [r for r in rows if r['epoch'] <= checkpoint['epoch']]
            if checkpoint.get('log_row') and not any(r['epoch'] == checkpoint['epoch'] for r in rows):
                rows.append(checkpoint['log_row'])
            log.unlink(); (output/'train.csv').unlink(missing_ok=True)
            for row in rows: append_log(output, row)
        elif checkpoint.get('log_row'):
            append_log(output, checkpoint['log_row'])
    else:
        (output/'config.json').write_text(json.dumps(vars(cfg), indent=2))
        (output/'splits.json').write_text(json.dumps(signature, indent=2))
        save_environment(output, device)
    for epoch in range(start_epoch, cfg.epochs+1):
        if hasattr(train.batch_sampler, 'set_epoch'): train.batch_sampler.set_epoch(epoch)
        if ema and not active and epoch > cfg.ema_start_epoch:
            ema.reset(model); active = True
        model.train()
        sums = {'loss': 0., 'ct_loss': 0., 'full_loss': 0.}
        counts = {'ct_samples': 0, 'full_samples': 0, 'missing_samples': 0}
        skipped = 0
        begin = time.time()
        grads = {}
        print(f'[TRAIN {epoch}/{cfg.epochs}] start batches={len(train)}', flush=True)
        for step, batch in enumerate(train, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, stats = forward_loss(model, batch, cfg.stage, device, criterion)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float('inf'))
            if not scaler.is_enabled() and not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            if step == 1:
                grads = {'grad_'+name: module_grad_norm(module) for name, module in model.named_children()
                         if any(p.requires_grad for p in module.parameters())}
                print('[GRAD first batch] '+json.dumps(grads), flush=True)
            before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= before:
                scheduler.step(); global_step += 1
                if active: ema.update(model)
            else: skipped += 1
            n = len(batch['mask'])
            for key in sums: sums[key] += (float(loss.detach()) if key == 'loss' else stats[key])*n
            for key in counts: counts[key] += stats[key]
            if step == 1 or step % cfg.log_interval == 0 or step == len(train):
                elapsed = time.time()-begin
                print(f'[TRAIN {epoch}] {step}/{len(train)} loss={float(loss.detach()):.4f} '
                      f'ct={stats["ct_loss"]:.4f} full={stats["full_loss"]:.4f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.3g} elapsed={elapsed:.0f}s '
                      f'eta={elapsed/step*(len(train)-step):.0f}s', flush=True)
        if cfg.stage == 2 and counts['full_samples'] != counts['missing_samples']:
            raise AssertionError('Full/Missing count mismatch')
        weights = 'ema' if active else 'raw'
        result = evaluate(ema.model if active else model, val, cfg, device, criterion, 'val', weights)
        score = result['missing']['dice'] if cfg.stage == 1 else result['joint_dice']
        if not np.isfinite(score): raise RuntimeError('Nonfinite selection Dice')
        improved = score > best
        if improved: best, best_epoch = score, epoch
        row = {'epoch': epoch, **{'train_'+k: v/counts['ct_samples'] for k, v in sums.items()},
               **counts, **grads, 'selection_dice': score, 'best': best, 'best_epoch': best_epoch,
               'weights': weights, 'lr': optimizer.param_groups[0]['lr'], 'skipped_updates': skipped,
               'ema_updates': ema.updates if ema else 0, 'epoch_seconds': time.time()-begin}
        for key, value in result.items():
            if isinstance(value, dict): row.update({key+'_'+k: v for k, v in value.items()})
            else: row[key] = value
        payload = {'format': FORMAT, 'config': vars(cfg), 'splits': signature, 'source': source,
            'epoch': epoch, 'global_step': global_step, 'model': model.state_dict(),
            'model_ema': ema.state_dict() if ema else None, 'ema_active': active,
            'ema_updates': ema.updates if ema else 0, 'eval_weights': weights,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best': best, 'best_epoch': best_epoch,
            'val': result, 'rng': rng_state(train), 'log_row': row}
        if improved: atomic_save(payload, output/'best.pt')
        atomic_save(payload, output/'last.pt')
        append_log(output, row)
        print(f'[EPOCH {epoch}] score={score:.4f} best={best:.4f}@{best_epoch} {result}', flush=True)
    best_checkpoint = load(output/'best.pt')
    model.load_state_dict(selected(best_checkpoint)[0], strict=True)
    result = {'epoch': best_checkpoint['epoch'], 'weights': best_checkpoint['eval_weights'],
              'metrics': evaluate(model, test, cfg, device, criterion, 'final_test', best_checkpoint['eval_weights'])}
    (output/'final_test.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    return result


if __name__ == '__main__':
    main()
