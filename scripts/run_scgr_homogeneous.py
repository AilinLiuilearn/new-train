"""Train homogeneous CT/PET alternatives to SCGR T and score the complete model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, git_identity, local_paths, require_myenv, sha


BACKBONES = ('mit_b1', 'convnextv2_nano')
HETEROGENEOUS_REFERENCE = {
    'encoders': {'ct_backbone': 'convnextv2_nano', 'pet_backbone': 'mit_b1'},
    'development_split_sha256': '87888bee32380d3806ff8e65eac8da268b272994d2550b50fe6570a07c43171e',
    'checkpoint_sha256': '2fd48124254bd2c2798318fe4d538ae5e240d3f48938163703d1aead09092d89',
    'without_case_refinement': {'dice_percent': 78.9488123395166, 'hd95': 16.16871927712624},
    'with_case_refinement': {'dice_percent': 79.58489177944149, 'hd95': 14.565184390880258,
                             'zero_overlap': 411},
}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def run_child(cmd, log, root, interrupt):
    with log.open('w') as stream:
        child = subprocess.Popen(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        write(root / 'CHILD.json', {'pid': child.pid, 'command': cmd})
        while True:
            try:
                return child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if (root / 'STOP').exists():
                    interrupt(child)


def campaign(args, root):
    source = git_identity()
    if source['dirty']:
        raise RuntimeError('Commit experiment code before launch')
    base = load(REPO / 'configs/scgr_T.json')
    groups = [args.only] if args.only else list(BACKBONES)
    split = sha(REPO / 'splits/dev.txt')
    if split != HETEROGENEOUS_REFERENCE['development_split_sha256']:
        raise ValueError('Development split differs from the reference')
    write(root / 'PLAN.json', {
        'git': source, 'groups': groups, 'training': base, 'gpu': args.gpu,
        'control': HETEROGENEOUS_REFERENCE, 'pid': os.getpid(),
        'design': 'Independent pretrained encoders with common task initialization. Fixed T and case refinement.',
        'selection': 'Best single full-fusion development Dice selected during training. Run case refinement once per chosen checkpoint.',
        'save': 'Two best-only runs; no snapshot of optimizer state and no large prediction caches.',
    })
    records = {}

    def status(state, **details):
        write(root / 'STATUS.json', {'state': state, 'pid': os.getpid(),
                                     'updated': time.time(), 'completed': list(records), **details})

    for position, name in enumerate(groups):
        pending = groups[position + 1:]
        if (root / 'STOP').exists():
            status('stopped_by_request', pending=groups[position:])
            return
        if git_identity() != source:
            raise RuntimeError('Source changed after campaign launch')
        cfg = root / f'{name}.json'
        write(cfg, {**base, 'ct_backbone': name, 'pet_backbone': name})
        output = root / name
        common = ['--paths', str(Path(args.paths).resolve()), '--output', str(output), '--gpu', args.gpu]
        cmd = [sys.executable, '-u', '-m', 'variants.scgr.train', '--recipe', 'T',
               '--config', str(cfg), '--run', f'{args.campaign}_{name}', *common]
        status('training', active=name, pending=pending, log=str(root / f'{name}.log'))
        ret = run_child(cmd, root / f'{name}.log', root,
                        lambda child: (output / 'STOP').touch() if output.exists() else None)
        if (root / 'STOP').exists():
            status('stopped_by_request', active=name, pending=pending)
            return
        if ret != 0:
            raise RuntimeError(f'{name}: training failed, see {root / (name + ".log")}')
        result = load(output / 'RESULTS.json')
        if not result['complete'] or load(output / 'status.json')['state'] != 'complete':
            status('stopped_by_request', active=name, pending=pending)
            return
        checkpoint = output / 'best.pt'
        cmd = [sys.executable, '-u', 'scripts/evaluate_scgr_case_refinement.py',
               '--checkpoint', str(checkpoint), *common]
        status('evaluating_case_refinement', active=name, pending=pending,
               log=str(root / f'{name}_case.log'))
        ret = run_child(cmd, root / f'{name}_case.log', root, lambda child: child.terminate())
        if (root / 'STOP').exists():
            status('stopped_by_request', active=name, pending=pending)
            return
        if ret != 0:
            raise RuntimeError(f'{name}: case evaluation failed, see {root / (name + "_case.log")}')
        case_metrics = load(output / 'CASE_REFINEMENT.json')
        if case_metrics['development_split_sha256'] != split or any(
                case_metrics['encoders'][key] != name for key in ('ct_backbone', 'pet_backbone')):
            raise RuntimeError('Case result differs from the scheduled comparison')
        records[name] = {'case_metrics': case_metrics, 'best_epoch': result['best_epoch'],
                         'weights': result['weights'], 'selection_metrics': result['metrics']}
        write(root / 'COMPARISON.json', {'reference': HETEROGENEOUS_REFERENCE, 'homogeneous': records,
                                        'scope': 'Fixed seed and reused development split.'})
        lines = ['# SCGR T encoder comparison', '',
                 '| Encoders | Best epoch | Raw Dice % | Raw HD95 | Full Dice % | Full HD95 | Zero overlap |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        candidates = [('ConvNeXtV2-Nano CT + MiT-B1 PET', 6, HETEROGENEOUS_REFERENCE)]
        candidates.extend((key, row['best_epoch'], row['case_metrics']) for key, row in records.items())
        for label, epoch, entry in candidates:
            raw, full = (entry[k] for k in ('without_case_refinement', 'with_case_refinement'))
            lines.append(f"| {label} | {epoch} | {raw['dice_percent']:.4f} | {raw['hd95']:.4f} | "
                         f"{full['dice_percent']:.4f} | {full['hd95']:.4f} | {full['zero_overlap']} |")
        (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    status('complete')


def main(args):
    require_myenv()
    if Path(args.campaign).name != args.campaign or args.campaign in ('.', '..'):
        raise ValueError('Use a simple campaign directory name')
    root = Path(local_paths(args.paths)['output_root']) / args.campaign
    root.mkdir(parents=False, exist_ok=False)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            campaign(args, root)
        except Exception as error:
            progress = load(root / 'STATUS.json') if (root / 'STATUS.json').exists() else {}
            write(root / 'STATUS.json', {**progress, 'state': 'needs_analysis', 'error': str(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--only', choices=BACKBONES)
    main(parser.parse_args())
