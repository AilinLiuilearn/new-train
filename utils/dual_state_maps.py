"""Strict, train-only offline map cache. No model/task weights enter training."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image
import torch

SCHEMA = 1
MODEL = {'ct_backbone': 'convnextv2_nano', 'pet_backbone': 'mit_b1',
         'channels': [64, 128, 320, 512], 'decoder_channels': [512, 256, 128, 64],
         'align_norm': 'BN', 'decoder_norm': 'GN', 'fusion': 'add', 'decoders': 2}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def checked_ids(ids):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate image_id')
    for item in ids:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', item) or '..' in item:
            raise ValueError(f'Unsafe image_id: {item!r}')
    return ids


def train_records(config, expected_sha):
    from datasets.pclt20k_seg import _records_from_ids
    split = Path(config['root'])/config['train_split_file']
    if sha256(split) != expected_sha:
        raise ValueError('Train split SHA256 mismatch')
    ids = checked_ids([x.strip() for x in split.read_text().splitlines() if x.strip()])
    return _records_from_ids(config['root'], ids)


def fingerprints(record, content_hash=False):
    result = {}
    for key in ('ct_path', 'pet_path', 'mask_path'):
        path = Path(record[key])
        st = path.stat()
        result[key] = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
        if content_hash:
            result[key]['sha256'] = sha256(path)
    return result


def difference_map(missing_logits, full_logits, target):
    if missing_logits.shape != full_logits.shape or missing_logits.shape != target.shape:
        raise ValueError('Logit/target shape mismatch')
    if missing_logits.ndim != 4 or missing_logits.shape[1] != 1:
        raise ValueError('Expected Bx1xHxW')
    if not all(torch.isfinite(x).all() for x in (missing_logits, full_logits, target)):
        raise ValueError('Nonfinite logits/target')
    if not ((target == 0) | (target == 1)).all():
        raise ValueError('Target must be binary')
    em = (missing_logits.float().sigmoid() > .5) != target.bool()
    ef = (full_logits.float().sigmoid() > .5) != target.bool()
    return (em.to(torch.uint8)+2*ef.to(torch.uint8))[:, 0].cpu().numpy()


def map_model(checkpoint, device):
    from run_staged import FORMAT, selected
    from models.staged_petct import StagedPETCT
    cfg = checkpoint.get('config', {})
    if checkpoint.get('format') != FORMAT or cfg.get('stage') != 2 or cfg.get('training_protocol') != 'paired_full_missing':
        raise ValueError('Need stage=2 paired_full_missing checkpoint in current format')
    for key, value in MODEL.items():
        if key in cfg and cfg[key] != value:
            raise ValueError(f'Model config mismatch: {key}')
    if cfg.get('norm_mode') not in ('imagenet', 'cipa') or not isinstance(cfg.get('image_size_2d'), int):
        raise ValueError('Missing/invalid preprocessing config')
    tag = checkpoint.get('eval_weights')
    if tag not in ('raw', 'ema') or not isinstance(checkpoint.get('model_ema' if tag == 'ema' else 'model'), dict):
        raise ValueError('Invalid eval_weights or absent selected weights')
    state, tag = selected(checkpoint)
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint weights')
    model = StagedPETCT(2)  # No fresh-model or generic pretrained loading here.
    model.load_state_dict(state, strict=True)
    model.float().to(device).eval().requires_grad_(False)
    return model, tag


class MapCache:
    """No worker cache: one uint8 image is held per access, bounded memory."""
    def __init__(self, directory, config, records=None, expected_identity=None):
        self.directory = Path(directory).resolve()
        path = self.directory/'manifest.json'
        self.manifest = m = json.loads(path.read_text())
        if m.get('schema') != SCHEMA or not m.get('complete') or m.get('partial'):
            raise ValueError('Incomplete/partial/unsupported map cache')
        if m.get('model') != MODEL or m.get('threshold') != 'sigmoid_fp32 > 0.5':
            raise ValueError('Map model/threshold mismatch')
        for key in ('norm_mode', 'image_size_2d', 'train_split_file'):
            if config[key] != m['baseline_config'][key]:
                raise ValueError(f'Cache preprocessing mismatch: {key}')
        actual = train_records(config, m['train_split_sha256'])
        if records is not None and [r['image_id'] for r in records] != [r['image_id'] for r in actual]:
            raise ValueError('Cache dataset image_id mismatch')
        ids = checked_ids([r['image_id'] for r in actual])
        if ids != m['image_ids'] or set(ids) != set(m['samples']) or len(ids) != m['count']:
            raise ValueError('Cache sample identity/count mismatch')
        self.samples = m['samples']
        self.size = config['image_size_2d']
        print(f'[CACHE] validating {len(actual)} train file fingerprints', flush=True)
        for r in actual:
            entry = self.samples[r['image_id']]
            if fingerprints(r, m['content_hash']) != entry['files']:
                raise ValueError(f'Dataset file fingerprint mismatch: {r["image_id"]}')
            if entry['map'] != r['image_id']+'.png' or sha256(self.directory/entry['map']) != entry['map_sha256']:
                raise ValueError(f'Map filename/hash mismatch: {r["image_id"]}')
        self.identity = {'manifest_sha256': sha256(path),
                         'source_checkpoint_sha256': m['source']['sha256'],
                         'context_rule': __import__('utils.dual_state_augmentation', fromlist=['CONTEXT_RULE']).CONTEXT_RULE}
        if expected_identity is not None and expected_identity != self.identity:
            raise ValueError('Resume map identity changed')

    def read(self, image_id):
        entry = self.samples[image_id]
        path = self.directory/entry['map']
        if sha256(path) != entry['map_sha256']:
            raise ValueError(f'Map changed since validation: {image_id}')
        with Image.open(path) as image:
            array = np.array(image)
        if array.dtype != np.uint8 or array.shape != (self.size, self.size) or array.max() > 3:
            raise ValueError(f'Invalid map dtype/shape/classes: {image_id}')
        return array
