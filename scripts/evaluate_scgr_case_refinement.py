"""Evaluate frozen T predictions before and after the confirmed case refinement."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from medpy.metric.binary import hd95
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, gpu_lock_path, local_paths, require_myenv, sha
from datasets.pclt20k_seg import PCLT20KSegDataset, _records_from_ids
from utils.metrics_seg import hd95_cipa_official_pair
from variants.scgr.case_refinement import refine_case, valid_neighbors
from variants.scgr.inference import load_model


def read_image(root, identity, suffix):
    path = root / identity.split('_')[0] / f'{identity}_{suffix}.png'
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_target(root, identity):
    image = read_image(root, identity, 'mask')
    if image.shape != (512, 512):
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    return image > 0 if image.max() <= 1 else image >= 128


def cropped_hd95(pred, target):
    if not target.any():
        return hd95_cipa_official_pair(pred, target)
    if not pred.any():
        pred = pred.copy()
        pred[256, 256] = True
    y, x = np.nonzero(pred | target)
    box = np.s_[y.min():y.max() + 1, x.min():x.max() + 1]
    return float(hd95(pred[box], target[box]))


def measure(mask, target):
    tp = int((mask & target).sum())
    return (tp, int((mask & ~target).sum()), int((~mask & target).sum()),
            cropped_hd95(mask, target), int(tp == 0), int(not mask.any()))


def score_case(ids, root, packed, logits, index, allowed_edges=None):
    ids = sorted(ids, key=lambda name: int(name.rsplit('_', 1)[1]))
    consecutive = [int(b.rsplit('_', 1)[1]) - int(a.rsplit('_', 1)[1]) == 1
                   for a, b in zip(ids, ids[1:])]
    if allowed_edges is None and not all(consecutive):
        raise ValueError('Expected consecutive numbered slices per patient')
    take = [index[identity] for identity in ids]
    masks = np.unpackbits(packed[take], axis=-1).astype(bool)
    ct = [read_image(root, identity, 'CT') for identity in ids]
    pet = [read_image(root, identity, 'PET') for identity in ids]
    valid = valid_neighbors(ct, pet)
    if allowed_edges is not None:
        valid &= np.asarray([adjacent and (a, b) in allowed_edges
                             for adjacent, a, b in zip(consecutive, ids, ids[1:])], dtype=bool)
    output = refine_case(masks, logits[take], ct, valid)
    scores = []
    for identity, a, b in zip(ids, masks, output):
        gt = load_target(root, identity)
        before = measure(a, gt)
        after = before if np.array_equal(a, b) else measure(b, gt)
        scores.append((identity, before, after))
    return scores


@torch.inference_mode()
def predict(root, ids, checkpoint):
    model = load_model(checkpoint, 'cuda')
    dataset = PCLT20KSegDataset(_records_from_ids(str(root), ids), train=False,
                                norm_mode=model.cfg['norm_mode'])
    loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
    packed = np.empty((len(ids), 512, 64), dtype=np.uint8)
    logits = np.empty((len(ids), 128, 128), dtype=np.float32)
    head = {}

    def native_prediction(_module, _args, output):
        head['logit'] = output

    offset = 0
    with model.decoder_full.seg_head.register_forward_hook(native_prediction):
        for step, batch in enumerate(loader, 1):
            if not batch['is_full'].all():
                raise ValueError('All prediction records must contain paired CT and PET')
            names = list(batch['image_id'])
            end = offset + len(names)
            if names != ids[offset:end]:
                raise ValueError('Prediction order differs from the development split')
            final = model(batch['ct'].cuda(non_blocking=True), batch['pet'].cuda(non_blocking=True))
            logits[offset:end] = head['logit'][:, 0].float().cpu().numpy()
            packed[offset:end] = np.packbits(final['logits'][:, 0].float().sigmoid().cpu().numpy() > .5,
                                            axis=-1)
            offset = end
            if step == 1 or step % 100 == 0 or end == len(ids):
                print(f'PREDICT {end}/{len(ids)}', flush=True)
    assert offset == len(ids)
    encoders = {'ct_backbone': model.cfg.get('ct_backbone', 'convnextv2_nano'),
                'pet_backbone': model.cfg.get('pet_backbone', 'mit_b1')}
    return packed, logits, encoders, model.cfg.get('mlrf_ablation')


def summarize(values):
    tp, fp, fn, hd, zero, empty = np.asarray(values).sum(axis=0)
    return {'dice_percent': float(200 * tp / (2 * tp + fp + fn)), 'hd95': float(hd / len(values)),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'zero_overlap': int(zero), 'empty': int(empty)}


def main(args):
    require_myenv()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(local_paths(args.paths)['data_root'])
    split = args.split or REPO / 'splits/dev.txt'
    if args.split:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        expected = checkpoint['manifest']['splits'].get('test')
        if not expected or sha(split) != expected['sha256']:
            raise ValueError('Evaluation list must match the held-out test recorded during training')
        if root != Path(checkpoint['config']['root']).resolve():
            raise ValueError('Evaluation root must match the trained dataset')
        del checkpoint
    ids = split.read_text().split()
    index = {identity: i for i, identity in enumerate(ids)}
    assert len(index) == len(ids)
    allowed_edges = None
    if args.adjacent_edges:
        if not args.split:
            raise ValueError('Explicit adjacency requires an external holdout split')
        allowed_edges = set()
        for line in args.adjacent_edges.read_text().splitlines():
            a, b = line.split()
            if a not in index or b not in index or a.split('_')[0] != b.split('_')[0]:
                raise ValueError('Each adjacency pair must belong to one held-out case')
            allowed_edges.add((a, b))
    patients = defaultdict(list)
    for identity in ids:
        patients[identity.split('_')[0]].append(identity)
    with gpu_lock_path().open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        packed, logits, encoders, mlrf_ablation = predict(root, ids, args.checkpoint)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(score_case, case, root, packed, logits, index, allowed_edges)
                for case in patients.values()]
        for job in jobs:
            rows.extend(job.result())
    assert len(rows) == len(ids) and {r[0] for r in rows} == set(index)
    baseline, refined = (summarize([r[k] for r in rows]) for k in (1, 2))
    report = {
        'checkpoint_sha256': sha(args.checkpoint),
        ('test_split_sha256' if args.split else 'development_split_sha256'): sha(split),
        'adjacent_edges_sha256': sha(args.adjacent_edges) if args.adjacent_edges else None,
        'encoders': encoders, 'mlrf_ablation': mlrf_ablation,
        'patients': len(patients), 'slices': len(rows),
        'without_case_refinement': baseline, 'with_case_refinement': refined,
        'delta_dice_pp': refined['dice_percent'] - baseline['dice_percent'],
        'delta_hd95': refined['hd95'] - baseline['hd95'],
        'scope': ('Frozen validation-selected checkpoint, held-out test cohort, GT only scored.'
                  if args.split else 'Frozen Dice-selected checkpoint, reused development cohort, GT only scored.'),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'CASE_SLICES.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['identity', 'arm', 'tp', 'fp', 'fn', 'hd95', 'zero_overlap', 'empty'])
        for identity, raw, post in rows:
            writer.writerow((identity, 'raw', *raw))
            writer.writerow((identity, 'case_refinement', *post))
    (args.output / 'CASE_REFINEMENT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', type=Path, help='Held-out test manifest recorded during training')
    parser.add_argument('--adjacent-edges', type=Path, help='Physically verified holdout slice pairs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    main(parser.parse_args())
