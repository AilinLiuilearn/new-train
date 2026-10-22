"""Build deterministic train-only empirical Full/Missing error maps, FP32."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets.pclt20k_seg import PCLT20KSegDataset
from run_staged import load
from utils.dual_state_maps import (MODEL, SCHEMA, atomic_json, commit, difference_map,
                                   fingerprints, map_model, sha256, train_records)
from utils.dual_state_augmentation import components


def build(args):
    started = time.monotonic()
    path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Cache directory is nonempty; use a new directory (no implicit overwrite/resume)')
    if args.batch_size < 1 or args.num_workers < 0 or args.log_interval < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Invalid batch/workers/log interval/limit')
    print(f'[MAP] start checkpoint={path} output={output}', flush=True)
    checkpoint = load(path)
    config = dict(checkpoint['config'])
    if args.root:
        config['root'] = str(Path(args.root).resolve())
    records = train_records(config, checkpoint['splits']['train'])
    all_ids = [r['image_id'] for r in records]
    if args.limit is not None:
        records = records[:args.limit]
    # Stat all train sources up front, without touching val/test images or labels.
    files = {r['image_id']: fingerprints(r, args.content_hash) for r in records}
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model, weights = map_model(checkpoint, device)
    bn_before = {k: v.detach().clone() for k, v in model.named_buffers()}
    environment = path.parent/'environment.json'
    source_commit = json.loads(environment.read_text()).get('commit') if environment.is_file() else None
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': SCHEMA, 'complete': False, 'partial': args.limit is not None,
                'count': 0, 'expected_train_count': len(all_ids), 'image_ids': [], 'samples': {},
                'source': {'path': str(path), 'sha256': sha256(path), 'epoch': checkpoint['epoch'],
                           'weights': weights, 'commit': source_commit},
                'current_commit': commit(), 'model': MODEL, 'baseline_config': config,
                'baseline_source': checkpoint.get('source'), 'splits': checkpoint['splits'],
                'train_split_sha256': checkpoint['splits']['train'],
                'threshold': 'sigmoid_fp32 > 0.5', 'content_hash': args.content_hash,
                'device': str(device), 'inference_batch': args.batch_size,
                'precision': 'FP32', 'augmentation': 'none', 'shuffle': False, 'drop_last': False}
    atomic_json(output/'manifest.json', manifest)
    ds = PCLT20KSegDataset(records, image_size=config['image_size_2d'], train=False,
                          aug_mode='none', norm_mode=config['norm_mode'])
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False)
    totals = {'pixels': [0]*4, 'regions': [0]*4, 'samples_with_class': [0]*4,
              'both_differential': 0, 'no_differential': 0, 'background_false_positive_regions': 0}
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            out = model(batch['ct'].to(device), batch['pet'].to(device), mode='full')
            maps = difference_map(out['ct_logits'], out['full_logits'], batch['mask'].to(device))
            for j, (image_id, array) in enumerate(zip(batch['image_id'], maps)):
                pixel_counts = np.bincount(array.ravel(), minlength=4).tolist()
                region_counts = [components(array == i)[0]-1 for i in range(4)]
                background_regions = 0
                gt = batch['mask'][j, 0].numpy() > .5
                for category in (1, 2, 3):
                    n, labels, _, _ = components(array == category)
                    touching = set(np.unique(labels[gt]).tolist())
                    background_regions += sum(i not in touching for i in range(1, n))
                filename = image_id+'.png'
                temp = output/(filename+'.tmp')
                Image.fromarray(array).save(temp, format='PNG')
                temp.replace(output/filename)
                # Detect concurrent changes to source data while maps were produced.
                record = records[manifest['count']]
                if fingerprints(record, args.content_hash) != files[image_id]:
                    raise ValueError(f'Data changed during map build: {image_id}')
                manifest['samples'][image_id] = {'map': filename, 'map_sha256': sha256(output/filename),
                    'files': files[image_id], 'pixels': pixel_counts, 'regions': region_counts,
                    'background_false_positive_regions': background_regions}
                manifest['image_ids'].append(image_id)
                manifest['count'] += 1
                for key, values in (('pixels', pixel_counts), ('regions', region_counts)):
                    totals[key] = [a+b for a, b in zip(totals[key], values)]
                totals['samples_with_class'] = [a+int(b > 0) for a, b in zip(totals['samples_with_class'], pixel_counts)]
                totals['both_differential'] += int(pixel_counts[1] > 0 and pixel_counts[2] > 0)
                totals['no_differential'] += int(pixel_counts[1]+pixel_counts[2] == 0)
                totals['background_false_positive_regions'] += background_regions
            if step == 1 or step % args.log_interval == 0 or step == len(loader):
                elapsed = time.monotonic()-started
                print(f'[MAP] {step}/{len(loader)} samples={manifest["count"]}/{len(records)} '
                      f'elapsed={elapsed:.1f}s eta={elapsed/max(manifest["count"],1)*(len(records)-manifest["count"]):.1f}s', flush=True)
    if manifest['image_ids'] != [r['image_id'] for r in records]:
        raise AssertionError('Train coverage/order mismatch')
    if any(not torch.equal(value, bn_before[key]) for key, value in model.named_buffers()):
        raise AssertionError('Model buffers changed during inference')
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError('Map model must remain frozen in eval with no gradients')
    manifest.update(complete=True, coverage=totals, elapsed_seconds=time.monotonic()-started,
                    buffers_unchanged=True, train_only=True,
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
    atomic_json(output/'manifest.json', manifest)
    print('[MAP DONE] '+json.dumps({k:manifest[k] for k in ('count','partial','coverage','elapsed_seconds','peak_allocated_bytes')}), flush=True)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--root', help='Explicit data-root migration; split identities must match')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--limit', type=int, help='Debug cache only; never accepted by training')
    p.add_argument('--content_hash', action='store_true')
    return p


if __name__ == '__main__':
    torch.set_num_threads(1)
    build(parser().parse_args())
