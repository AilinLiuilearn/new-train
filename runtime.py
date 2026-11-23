"""Machine-local paths, preflight, immutable run identity and GPU coordination."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def parser(variant):
    p = argparse.ArgumentParser(description=f'{variant}: fresh task training, myenv, best-only')
    p.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    p.add_argument('--config', default=str(REPO / 'configs' / f'{variant}.json'))
    p.add_argument('--run', required=True, help='Unique run directory; never reuse an old run')
    p.add_argument('--output', help='Optional absolute run directory')
    p.add_argument('--gpu', help='One physical GPU index or UUID; sets CUDA_VISIBLE_DEVICES')
    p.add_argument('--arm', choices=['full', 'm1'], default='full')
    p.add_argument('--allow-dirty', action='store_true', help='Record uncommitted source hashes explicitly')
    p.set_defaults(resume=None)
    return p


def require_myenv():
    if Path(sys.prefix).name != 'myenv':
        raise RuntimeError(f'Use conda environment myenv, not {sys.prefix}; invoke scripts/myenv.sh')


def local_paths(path):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Copy paths.example.json to {p} and set this machine paths')
    obj = json.loads(p.read_text())
    return {k: str(Path(obj[k]).expanduser().resolve())
            for k in ('data_root', 'pretrained_root', 'output_root')}


def resolve_config(args, variant):
    locations = local_paths(args.paths)
    cfg = json.loads(Path(args.config).read_text())
    encoder_directories = {'convnextv2_nano': 'convnextv2_nano', 'mit_b1': 'mit-b1'}
    ct_backbone = cfg.get('ct_backbone', 'convnextv2_nano')
    pet_backbone = cfg.get('pet_backbone', 'mit_b1')
    if ct_backbone not in encoder_directories or pet_backbone not in encoder_directories:
        raise ValueError('Unsupported CT or PET backbone')
    cfg.update(root=locations['data_root'],
               ct_pretrained_path=str(Path(locations['pretrained_root']) / encoder_directories[ct_backbone]),
               pet_pretrained_path=str(Path(locations['pretrained_root']) / encoder_directories[pet_backbone]),
               train_split_file=str(REPO / 'splits/train.txt'),
               val_split_file=str(REPO / 'splits/dev.txt'),
               test_split_file=str(REPO / 'splits/dev.txt'),
               save_last_checkpoint=False, resume=None, device='cuda', arm=args.arm)
    if split_directory := getattr(args, 'split_directory', None):
        split_directory = Path(split_directory).expanduser().resolve()
        shared_eval = getattr(args, 'reuse_val_as_test', False)
        cfg.update({f'{name}_split_file': str(split_directory / f'{name}.txt')
                    for name in ('train', 'val', 'test')}, allow_val_equals_test=shared_eval)
        if shared_eval:
            if (split_directory / 'val.txt').read_bytes() != (split_directory / 'test.txt').read_bytes():
                raise ValueError('Shared validation/test requires identical val.txt and test.txt')
            cfg['test_split_file'] = cfg['val_split_file']
    if cfg.get('initial_checkpoint') or cfg.get('stage1_checkpoint') or cfg.get('evaluate'):
        raise ValueError('Training must not load historical task checkpoints')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run) or args.run in ('.', '..'):
        raise ValueError('Run name must be a simple directory name')
    out = Path(args.output).expanduser().resolve() if args.output else Path(locations['output_root']) / args.run
    cfg['output'] = str(out)
    return argparse.Namespace(**cfg), out, locations


def verify_resources(locations, cfg=None):
    from datasets.pclt20k_seg import _read_list, _records_from_ids, _check_patient_disjoint
    records = {}
    splits = {}
    external = cfg is not None
    paths = ({name: Path(getattr(cfg, f'{name}_split_file')) for name in ('train', 'val', 'test')}
             if external else {'train': REPO / 'splits/train.txt', 'val': REPO / 'splits/dev.txt'})
    for name, p in paths.items():
        ids = _read_list(p)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'Missing, empty or duplicated split entries: {p}')
        recs = _records_from_ids(locations['data_root'], ids)
        for rec in recs:
            for key in ('ct_path', 'pet_path', 'mask_path'):
                if not Path(rec[key]).is_file():
                    raise FileNotFoundError(rec[key])
        records[name] = recs
        splits[name] = {'sha256': sha(p), 'slices': len(recs)}
    shared_eval = not external or (cfg.allow_val_equals_test and cfg.val_split_file == cfg.test_split_file)
    _check_patient_disjoint(records['train'], records['val'], records.get('test', records['val']),
                            allow_val_equals_test=shared_eval)
    weights = {}
    for name, entry in json.loads((REPO / 'resources/pretrained_manifest.json').read_text()).items():
        p = Path(locations['pretrained_root']) / entry['relative_file']
        if not p.is_file() or sha(p) != entry['sha256']:
            raise RuntimeError(f'Pretrained identity mismatch: {p}')
        weights[name] = {'file': str(p), 'sha256': entry['sha256']}
    return {'generic_weights': weights, 'splits': splits,
            'text_json_sha256': sha(REPO / 'resources/biomedclip_roles.json'),
            'scope': ('Reused development set; no independent test claim' if shared_eval
                      else 'Patient-disjoint held-out test, reserved for final evaluation')}


def git_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty': bool(git('status', '--porcelain')), 'origin': git('remote', 'get-url', 'origin')}


def environment():
    import torch
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'torchvision', 'numpy', 'scipy', 'timm', 'transformers', 'MedPy', 'opencv-python']}
    return {'python': sys.version, 'executable': sys.executable, 'conda_prefix': sys.prefix,
            'host': platform.node(), 'packages': versions, 'torch_cuda': torch.version.cuda,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def source_files():
    for root in ['src', 'variants', 'scripts', 'configs', 'resources', 'splits']:
        for p in sorted((REPO / root).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.log'):
                yield p
    yield REPO / 'runtime.py'


def prepare_run(args, variant):
    require_myenv()
    if args.gpu is not None:
        if not re.fullmatch(r'(\d+|GPU-[A-Za-z0-9-]+)', args.gpu):
            raise ValueError('Specify exactly one GPU index or UUID')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for formal training')
    cfg, out, locations = resolve_config(args, variant)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Output is not empty; use a new run name: {out}')
    existing_parent = out.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 2 * 1024**3:
        raise RuntimeError('Need at least 2 GiB free before starting')
    identity = git_identity()
    if identity['dirty'] and not args.allow_dirty:
        raise RuntimeError('Commit code before training, or explicitly pass --allow-dirty')
    manifest = (verify_resources(locations, cfg) if getattr(args, 'split_directory', None)
                else verify_resources(locations))
    if text_cache := getattr(args, 'text_cache', None):
        manifest['text_json_sha256'] = sha(text_cache)
    out.mkdir(parents=True, exist_ok=True)
    record = {'variant': variant, 'git': identity, 'environment': environment(),
              'config': vars(cfg), 'resources': manifest,
              'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in source_files()},
              'checkpoint_policy': 'best.pt only; selected raw/EMA weights, no resume state'}
    (out / 'run_identity.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'resolved_config.json').write_text(json.dumps(vars(cfg), indent=2) + '\n')
    return out, cfg, manifest


def gpu_lock_path():
    import torch
    # nvidia-smi identifies the physical device even when another process uses a different alias.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] or '0'
    proc = subprocess.run(['nvidia-smi', '-i', visible, '--query-gpu=uuid', '--format=csv,noheader'],
                          text=True, capture_output=True, check=True)
    gpu = proc.stdout.strip()
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', gpu):
        raise RuntimeError('Cannot identify one physical GPU for locking')
    p = Path.home() / '.cache/ct_pet_research/locks' / (gpu + '.lock')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def snapshot_sources(out):
    with tarfile.open(Path(out) / 'code_snapshot.tar.gz', 'w:gz') as tf:
        for p in source_files():
            tf.add(p, arcname=str(p.relative_to(REPO)), recursive=False)
