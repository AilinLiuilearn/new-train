"""One user-authorized full-model run, immutable source and no follow-up queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, require_myenv, local_paths, git_identity, sha


def dump(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def analysis(root, result, returncode):
    metrics = result.get('metrics')
    if not metrics:
        (root / 'FINAL_ANALYSIS.md').write_text(
            f'# LPI-BI V1\n\nNo completed validation result. Return code: {returncode}. '
            'Inspect logs; no conclusion about performance or automatic next experiment.\n')
        return
    lines = ['# LPI-BI V1 full-model result', '',
             f"Best Dice epoch: {result['best_epoch']} ({result['weights']}).",
             f"Dice: {100*metrics['dice']:.4f}%; paired HD95: {metrics['hd95']:.4f}.",
             f"Sensitivity: {100*metrics['sensitivity']:.4f}%; precision: {100*metrics['precision']:.4f}%."]
    if 'reference_M1' in result:
        ref = result['reference_M1']['result']['metrics']
        lines += ['', f"Same-protocol M1: {100*ref['dice']:.4f}% / {ref['hd95']:.4f}.",
                  f"Dice difference: {result['delta_M1_dice_pp']:+.4f} pp; "
                  f"HD95 difference: {result['delta_M1_hd95']:+.4f}."]
    lines += ['', f"Dice>=78.8% and HD95<17: {result['target_788_hd17_met']}.",
              f"Dice>=78.9% and HD95<17: {result['target_789_hd17_met']}.", '',
              result.get('assessment', 'Partial run; final conclusion pending.'), '',
              'A full candidate alone cannot isolate auxiliary supervision, the spatial bias, or channel pooling. '
              'If the target is missed, use losses, localization quality, per-scale updates and false-positive/negative '
              'changes to form hypotheses; training snapshots alone do not establish a cause. '
              'Only the requested run was authorized here; no automatic tuning or next experiment.', '',
              'Single seed, reused development set; no independent test or stability claim.']
    (root / 'FINAL_ANALYSIS.md').write_text('\n'.join(lines) + '\n')


def main(args):
    require_myenv()
    identity = git_identity()
    if identity['dirty']:
        raise RuntimeError('Commit source before launch')
    checks = json.loads((REPO / 'docs/LPI_BI_GPU_PREFLIGHT.json').read_text())
    if not checks['passed']:
        raise RuntimeError('GPU preflight did not pass')
    for name, expected in checks['source_sha256'].items():
        if sha(REPO / name) != expected:
            raise RuntimeError('Preflight source changed: ' + name)
    root = Path(local_paths(args.paths)['output_root']) / args.campaign
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'PLAN.json').exists():
            raise RuntimeError('Existing campaign, choose a new directory')
        (root / 'logs').mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'variants.lpi_bi.train',
                   '--run', args.campaign + '_full', '--paths', str(Path(args.paths).resolve()),
                   '--output', str(root / 'full'), '--gpu', args.gpu]
        dump(root / 'PLAN.json', {'git': identity, 'pid': os.getpid(), 'groups': ['full'],
                                  'gpu': args.gpu, 'command': command, 'preflight': checks,
                                  'next_experiments': []})
        def status(state, **extra):
            dump(root / 'STATUS.json', {'state': state, 'pid': os.getpid(),
                                        'time_epoch': time.time(), **extra})
        if (root / 'STOP').exists():
            status('stopped_by_request'); return
        status('running', active='full')
        with (root / 'logs/full.log').open('w') as stream:
            child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            dump(root / 'CHILD.json', {'pid': child.pid, 'group': 'full'})
            while child.poll() is None:
                if (root / 'STOP').exists() and (root / 'full').exists():
                    (root / 'full/STOP').touch()
                time.sleep(10)
        output = root / 'full'
        result = json.loads((output / 'RESULTS.json').read_text()) if (output / 'RESULTS.json').exists() else {}
        state = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        stopped = (root / 'STOP').exists() or state.get('state') == 'stopped_by_request'
        complete = child.returncode == 0 and result.get('complete') and state.get('state') == 'complete'
        final_state = 'stopped_by_request' if stopped else 'complete' if complete else 'needs_analysis'
        analysis(root, result, child.returncode)
        dump(root / 'RESULTS.json', {'state': final_state, 'result': result, 'returncode': child.returncode})
        status(final_state, result=result, returncode=child.returncode, next_experiments=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        import traceback
        traceback.print_exc()
        root = Path(local_paths(args.paths)['output_root']) / args.campaign
        if root.exists():
            dump(root / 'STATUS.json', {'state': 'needs_analysis', 'error': repr(error)})
        raise SystemExit(1)
