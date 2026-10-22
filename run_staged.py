"""Clean paired Full/Missing training, with optional CT-only task initialization."""
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
from models.components.backbones import _resolve_weight_file
from utils.ema import ModelEMA
from utils.amp import make_grad_scaler, autocast_context
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_cosine_scheduler
from utils.run_common import seed_everything, module_grad_norm
from utils.seg_losses import BCEDiceLoss

FORMAT = 'ct_pretrain_dual_decoder_v2_paired'
LEGACY_FORMAT = 'ct_pretrain_dual_decoder_v1'


def patience_exhausted(epoch, best_epoch, patience):
    return bool(patience and epoch-best_epoch >= patience)


def load(path):
    # Only load your own trusted local checkpoints (legacy files include RNG objects).
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def prepare_output(output):
    if not output.exists() or not any(output.iterdir()):
        output.mkdir(parents=True, exist_ok=True)
        return
    # A failure before optimizer/EMA initialization leaves only these audit files.
    # Preserve the entire failed startup instead of deleting or overwriting it.
    startup_files = {'split_summary.json', 'transfer.json', 'initialization.json'}
    entries = list(output.iterdir())
    if all(x.is_file() and x.name in startup_files for x in entries):
        archive = output.with_name(output.name+f'.failed-start-{time.time_ns()}')
        output.rename(archive)
        output.mkdir(parents=True)
        print(f'[RETRY] preserved failed startup at {archive}', flush=True)
        return
    raise FileExistsError('Run directory is not empty; use a new --output or --resume last.pt')


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
    if old.get('decoder_norm', 'GN') != model.decoder_norm:
        raise ValueError('Stage-1 protocol mismatch: decoder_norm')
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


def fusion_kwargs(cfg):
    """Serialized architecture is authoritative; legacy checkpoints retain add."""
    return {'fusion_type': getattr(cfg, 'fusion_type', 'add'),
            'fusion_config': getattr(cfg, 'fusion_config', None)}


def normalize_fusion_config(cfg):
    from copy import deepcopy
    from models.components.local_region_residual_fusion import DEFAULT_FUSION_CONFIG
    cfg.fusion_type = getattr(cfg, 'fusion_type', 'add')
    saved = getattr(cfg, 'fusion_config', None)
    cfg.fusion_config = deepcopy(DEFAULT_FUSION_CONFIG if saved is None else saved)
    if saved is None:
        cfg.fusion_config['use_gaussian'] = getattr(cfg, 'fusion_gaussian', True)
    cfg.fusion_gaussian = cfg.fusion_config['use_gaussian']


def build_fresh_model(cfg):
    """Shared by training and preflight; never called to restore a checkpoint."""
    normalize_fusion_config(cfg)
    joint = getattr(cfg, 'joint_from_pretrained', False)
    load_ct = not cfg.no_pretrained and (cfg.stage == 1 or joint)
    load_pet = not cfg.no_pretrained and cfg.stage == 2
    if load_ct and not cfg.ct_pretrained_path:
        raise ValueError('CT pretrained path is required; --no_pretrained is for smoke tests only')
    if load_pet and not cfg.pet_pretrained_path:
        raise ValueError('PET pretrained path is required; --no_pretrained is for smoke tests only')
    return StagedPETCT(cfg.stage,
        decoder_norm=getattr(cfg, 'decoder_norm', 'GN'), **fusion_kwargs(cfg),
        ct_pretrained_path=cfg.ct_pretrained_path if load_ct else None,
        pet_pretrained_path=cfg.pet_pretrained_path if load_pet else None)


def joint_initialization_report(cfg):
    report = {'initialization': 'random_smoke' if cfg.no_pretrained else 'backbone_pretrained',
              'stage1_checkpoint': None, 'aligners': 'random',
              'decoder_norm': getattr(cfg, 'decoder_norm', 'GN'),
              'ct_decoder': 'random', 'full_decoder': 'independent_copy_of_random_ct_decoder'}
    if not cfg.no_pretrained:
        for name in ('ct', 'pet'):
            path = Path(_resolve_weight_file(getattr(cfg, name+'_pretrained_path'), name.upper()+'_Encoder')).resolve()
            report[name] = {'weight_file': str(path), 'sha256': file_sha(path)}
    return report


def forward_loss(model, batch, stage, device, criterion):
    ct = batch['ct'].to(device, non_blocking=True)
    target = batch['mask'].to(device, non_blocking=True)
    if stage == 1:
        out = model(ct, mode='missing')
        loss = criterion(out['ct_logits'], target)[0]
        return loss, {'ct_loss': float(loss.detach()), 'full_loss': 0.,
                      'ct_samples': len(ct), 'full_samples': 0, 'missing_samples': len(ct)}
    full = batch['is_full']
    if full.ndim != 1 or len(full) != len(ct) or not full.bool().all():
        raise ValueError('Paired stage 2 requires real PET for every training row; half-missing batches are unsupported')
    pet = batch['pet'].to(device, non_blocking=True)
    out = model(ct, pet, mode='full')
    lc = criterion(out['ct_logits'], target)[0]
    lf = criterion(out['full_logits'], target)[0]
    return .5*(lc+lf), {'ct_loss': float(lc.detach()), 'full_loss': float(lf.detach()),
                        'ct_samples': len(ct), 'full_samples': len(ct),
                        'missing_samples': len(ct)}


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
    init = p.add_mutually_exclusive_group()
    init.add_argument('--stage1_checkpoint', help='Initialize stage 2 from a trusted CT-only checkpoint')
    init.add_argument('--joint_from_pretrained', action='store_true',
                      help='Direct paired training: initialize BOTH encoders from local generic weights, no CT-only checkpoint')
    p.add_argument('--stage1_weights', choices=['auto', 'raw', 'ema'], default='auto')
    p.add_argument('--resume', help='Resume last.pt at an epoch boundary using saved config')
    p.add_argument('--save_last_checkpoint', action=argparse.BooleanOptionalAction, default=False,
                   help='Opt in to last.pt recovery checkpoints; default keeps best.pt only')
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
    p.add_argument('--decoder_norm', choices=['GN', 'BN'], default='GN',
                   help='Independent decoder normalization; old checkpoints default to GN')
    p.add_argument('--fusion_type', choices=['add', 'local_region_residual'], default='add')
    p.add_argument('--fusion_gaussian', action=argparse.BooleanOptionalAction, default=True)
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
    p.add_argument('--early_stopping_patience', type=int, default=5,
                   help='Stop after 5 unimproved selection epochs by default; 0 disables early stopping')
    return p


def clean_legacy_augmentation_config(cfg):
    """Keep saved baseline/module checkpoints compatible without old experiments."""
    for name in ('crop_guidance', 'copy_paste'):
        if getattr(cfg, name, 'none') != 'none':
            raise ValueError('Removed augmentation experiment; use the archived branch for '+name)
    for name in ('crop_guidance', 'guidance_map_dir', 'guided_crop_prob',
                 'guidance_region_sampling', 'guidance_identity', 'copy_paste',
                 'copy_paste_prob', 'paste_index', 'paste_selection', 'copy_paste_identity'):
        if hasattr(cfg, name):
            delattr(cfg, name)


def validate(cfg, evaluating=False):
    if getattr(cfg, 'fusion_type', 'add') != 'add' and (cfg.stage != 2 or cfg.image_size_2d < 256 or cfg.image_size_2d % 128):
        raise ValueError('residual fusion requires stage=2, image_size >=256 and divisible by128 (reflect kernels and 4x4 windows)')
    if getattr(cfg, 'decoder_norm', 'GN') not in ('GN', 'BN'):
        raise ValueError('Invalid decoder_norm')
    if cfg.early_stopping_patience < 0:
        raise ValueError('Invalid early_stopping_patience')
    if cfg.stage not in (1, 2) or not cfg.output:
        raise ValueError('--stage and --output are required for new runs')
    if getattr(cfg, 'joint_from_pretrained', False) and cfg.stage != 2:
        raise ValueError('--joint_from_pretrained requires --stage 2 (the dual-path architecture)')
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
    if checkpoint and checkpoint.get('format') not in (FORMAT, LEGACY_FORMAT):
        raise ValueError('Resume/evaluate needs a new staged checkpoint; use --stage1_checkpoint for legacy CT weights')
    if (args.resume and checkpoint.get('format') == LEGACY_FORMAT
            and checkpoint['config']['stage'] == 2):
        raise ValueError('Cannot resume old 50% stage-2 routing with the paired protocol. '
                         'Keep that run for evaluation; start a new --output from --stage1_checkpoint. '
                         'To continue the old protocol, use its original commit.')
    cfg = argparse.Namespace(**checkpoint['config']) if checkpoint else args
    # Checkpoints from before direct joint training had only CT-task initialization.
    cfg.joint_from_pretrained = getattr(cfg, 'joint_from_pretrained', False)
    cfg.decoder_norm = getattr(cfg, 'decoder_norm', 'GN')
    normalize_fusion_config(cfg)
    clean_legacy_augmentation_config(cfg)
    cfg.early_stopping_patience = getattr(cfg, 'early_stopping_patience', 0)
    cfg.save_last_checkpoint = getattr(cfg, 'save_last_checkpoint', False)
    cfg._evaluation_only = bool(args.evaluate)
    cfg.training_protocol = ('ct_only' if cfg.stage == 1 else
        'half_missing_v1' if checkpoint and checkpoint.get('format') == LEGACY_FORMAT else 'paired_full_missing')
    if checkpoint:
        supplied = set((argv if argv is not None else sys.argv[1:]))
        allowed = {'--evaluate', '--resume', '--device'}
        if args.resume: allowed |= {'--save_last_checkpoint', '--no-save_last_checkpoint'}
        if args.evaluate: allowed |= {'--root', '--eval_mode'}
        unknown = {x.split('=')[0] for x in supplied if x.startswith('--')} - allowed
        if unknown:
            raise ValueError(f'Saved config is authoritative; unsupported overrides: {sorted(unknown)}')
        cfg.device = args.device
        if args.resume and any(x.split('=')[0] in ('--save_last_checkpoint', '--no-save_last_checkpoint') for x in supplied):
            cfg.save_last_checkpoint = args.save_last_checkpoint
        if args.evaluate:
            if '--root' in supplied or any(x.startswith('--root=') for x in supplied): cfg.root = args.root
            cfg.eval_mode = args.eval_mode
    validate(cfg, bool(args.evaluate))
    if not checkpoint and cfg.stage == 2 and not (cfg.stage1_checkpoint or cfg.joint_from_pretrained):
        raise ValueError('Choose --joint_from_pretrained or --stage1_checkpoint for a new dual-path run')
    if checkpoint is None:
        cfg.output = str(Path(cfg.output).resolve())
        cfg.root = str(Path(cfg.root).resolve())
    output = Path(cfg.output)
    if not checkpoint:
        prepare_output(output)
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
          f'size={cfg.image_size_2d} epochs={cfg.epochs} protocol={cfg.training_protocol}', flush=True)
    if cfg.stage == 2 and not args.evaluate:
        print(f'[PAIRED] each batch: Full={cfg.batch_size} CT-only={cfg.batch_size}; '
              'one shared CT encode; loss=0.5*CT_loss+0.5*Full_loss; no ranking/compensation', flush=True)
    print('[DATA] checking split membership and PNG paths', flush=True)
    train, val, test = loaders(cfg, output)
    if not len(train): raise ValueError('No complete train batch; reduce batch_size')
    print(f'[DATA] train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)} '
          f'train_batches={len(train)} dropped_per_epoch={len(train.dataset)%cfg.batch_size}', flush=True)
    fresh = checkpoint is None
    model = build_fresh_model(cfg) if fresh else StagedPETCT(cfg.stage, decoder_norm=cfg.decoder_norm, **fusion_kwargs(cfg))
    source = None
    if checkpoint:
        model.load_state_dict(selected(checkpoint)[0] if args.evaluate else checkpoint['model'], strict=True)
        source = checkpoint['source']
    elif cfg.stage == 2 and cfg.joint_from_pretrained:
        source = joint_initialization_report(cfg)
        (output/'initialization.json').write_text(json.dumps(source, indent=2))
        print('[INIT] '+json.dumps(source)+'; no CT-only task weights loaded', flush=True)
    elif cfg.stage == 2:
        source = import_stage1(model, cfg.stage1_checkpoint, cfg, signature)
        (output/'transfer.json').write_text(json.dumps(source, indent=2))
    model.to(device)
    if not all(p.requires_grad for p in model.parameters()):
        raise AssertionError('This baseline freezes no parameters')
    print(f'[MODEL] params={sum(p.numel() for p in model.parameters()):,} all trainable; '
          f'align=BN decoder={cfg.decoder_norm}, no retrieval/distillation', flush=True)
    criterion = BCEDiceLoss()
    if args.evaluate:
        result = {'epoch': checkpoint['epoch'], 'weights': checkpoint['eval_weights'],
                  'metrics': evaluate(model, test, cfg, device, criterion, 'test', checkpoint['eval_weights'])}
        (output/f'reeval_{time.time_ns()}.json').write_text(json.dumps(result, indent=2))
        print(result, flush=True)
        return result
    print(f'[CHECKPOINT] best.pt only={not cfg.save_last_checkpoint}; save_last_checkpoint={cfg.save_last_checkpoint}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_scheduler(optimizer, cfg.epochs, warmup_steps=cfg.cosine_warmup*len(train),
        min_lr=cfg.cosine_min_lr, steps_per_epoch=len(train), flat_ratio=cfg.lr_flat_ratio)
    scaler = make_grad_scaler(device, cfg.mixed_precision)
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
            for row in rows:
                # Old stage-1 logs predate this column; keep CSV headers aligned
                # with later rows when safely continuing unchanged CT training.
                append_log(output, {'epoch': row['epoch'], 'training_protocol': cfg.training_protocol, **row})
        elif checkpoint.get('log_row'):
            row = checkpoint['log_row']
            append_log(output, {'epoch': row['epoch'], 'training_protocol': cfg.training_protocol, **row})
    else:
        (output/'config.json').write_text(json.dumps(vars(cfg), indent=2))
        (output/'splits.json').write_text(json.dumps(signature, indent=2))
        save_environment(output, device)
    if args.resume:
        # Storage policy can change without changing the saved scientific protocol.
        (output/'config.json').write_text(json.dumps(vars(cfg), indent=2))
    for epoch in range(start_epoch, cfg.epochs+1):
        if epoch == start_epoch and patience_exhausted(start_epoch-1, best_epoch, cfg.early_stopping_patience):
            print(f'[EARLY STOP] saved run already exhausted patience={cfg.early_stopping_patience}', flush=True)
            break
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
            with autocast_context(device, enabled=scaler.is_enabled()):
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
        if cfg.stage == 2 and not (counts['ct_samples'] == counts['full_samples'] == counts['missing_samples']):
            raise AssertionError('Every CT sample must receive both Full and Missing supervision')
        weights = 'ema' if active else 'raw'
        result = evaluate(ema.model if active else model, val, cfg, device, criterion, 'val', weights)
        score = result['missing']['dice'] if cfg.stage == 1 else result['joint_dice']
        if not np.isfinite(score): raise RuntimeError('Nonfinite selection Dice')
        improved = score > best
        if improved: best, best_epoch = score, epoch
        print('[COVERAGE] '+json.dumps(counts)+' (Full/Missing are overlapping paths, not disjoint samples)', flush=True)
        row = {'epoch': epoch, 'training_protocol': cfg.training_protocol,
               **{'train_'+k: v/counts['ct_samples'] for k, v in sums.items()},
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
        if cfg.save_last_checkpoint:
            atomic_save(payload, output/'last.pt')
        append_log(output, row)
        print(f'[EPOCH {epoch}] score={score:.4f} best={best:.4f}@{best_epoch} {result}', flush=True)
        if patience_exhausted(epoch, best_epoch, cfg.early_stopping_patience):
            print(f'[EARLY STOP] {cfg.early_stopping_patience} epochs without strict selection improvement; '
                  f'best={best:.6f}@{best_epoch}', flush=True)
            break
    best_checkpoint = load(output/'best.pt')
    model.load_state_dict(selected(best_checkpoint)[0], strict=True)
    result = {'epoch': best_checkpoint['epoch'], 'weights': best_checkpoint['eval_weights'],
              'metrics': evaluate(model, test, cfg, device, criterion, 'final_test', best_checkpoint['eval_weights'])}
    (output/'final_test.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    return result


if __name__ == '__main__':
    main()
