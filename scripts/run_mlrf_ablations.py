"""Train four standalone MLRF controls after the TCSR ablation queue."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, git_identity, local_paths, require_myenv, sha
from run_scgr_homogeneous import load, run_child, write
from run_tcsr_ablations import DESIGNS as TCSR_DESIGNS


ABLATIONS = {
    'no_gaussian': ({'use_gaussian': False}, 'CT local + PET regional; Gaussian disabled'),
    'ct_local_only': ({'use_regional': False}, 'CT local with Gaussian, PET identity'),
    'pet_regional_only': ({'use_local': False, 'use_gaussian': False}, 'CT identity, PET regional'),
    'swap_roles_no_gaussian': ({'swap_roles': True, 'use_gaussian': False},
                               'CT regional + PET local; Gaussian disabled'),
}


def main(args):
    require_myenv()
    if Path(args.campaign).name != args.campaign or args.campaign in ('.', '..'):
        raise ValueError('Choose a simple campaign directory name')
    identity = git_identity()
    if identity['dirty']:
        raise RuntimeError('Commit code before launching the campaign')
    output_root = Path(local_paths(args.paths)['output_root'])
    root = output_root / args.campaign
    dependency = args.after if args.after.is_absolute() else output_root / args.after
    reference_path = output_root / 'rerun_m1_concat_ctalign_tsbci_p3_20261115/m1'
    base = load(REPO / 'configs/rerun_m1_p3.json')
    historical = load(reference_path / 'run_identity.json')
    reference = load(reference_path / 'RESULTS.json')
    if not reference['complete'] or any(historical['config'][k] != v for k, v in base.items()):
        raise RuntimeError('Historical full MLRF protocol differs')
    for split, file in (('train', 'train.txt'), ('val', 'dev.txt')):
        if historical['resources']['splits'][split]['sha256'] != sha(REPO / 'splits' / file):
            raise RuntimeError('Historical MLRF data split differs')
    root.mkdir(exist_ok=False)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {'full_mlrf_reference': {k: reference[k] for k in ('metrics', 'best_epoch', 'weights')}}
        plan = list(ABLATIONS)

        def publish(state, **kw):
            write(root / 'STATUS.json', {'state': state, 'completed': list(rows)[1:],
                                         'updated': time.time(), **kw})

        def summarize():
            write(root / 'COMPARISON.json', {'groups': rows})
            lines = ['# Standalone MLRF ablations', '',
                     '| Group | Best epoch | Weights | Dice % | HD95 | Precision % | Sensitivity % |',
                     '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
            for key, row in rows.items():
                m = row['metrics']
                lines.append(f"| {key} | {row['best_epoch']} | {row['weights']} | {100*m['dice']:.4f} | "
                             f"{m['hd95']:.4f} | {100*m['precision']:.4f} | {100*m['sensitivity']:.4f} |")
            lines.extend(('', 'Compare the swap against no_gaussian. Single-seed development set.'))
            (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')

        write(root / 'PLAN.json', {
            'git': identity, 'after': str(dependency), 'groups': plan, 'config': base,
            'overrides': ABLATIONS, 'reference': str(reference_path),
            'fixed': 'Both encoders and aligned inputs, all four scales, residual addition, GN decoder, direct output.',
            'interpretation': 'Role swap compared against no_gaussian. PET-regional-only vs no_gaussian isolates CT local without Gaussian. CT-local-only vs full isolates PET regional.',
        })
        summarize()
        publish('waiting_for_tcsr_ablations', pending=plan)
        try:
            while True:
                if (root / 'STOP').exists():
                    publish('stopped_by_request', pending=plan)
                    return
                status = dependency / 'STATUS.json'
                if status.is_file():
                    upstream = load(status)
                    if upstream['state'] == 'complete':
                        if set(upstream['completed']) != set(TCSR_DESIGNS):
                            raise RuntimeError('Expected the complete planned TCSR ablation queue')
                        break
                    if upstream['state'] in ('stopped_by_request', 'needs_analysis', 'waiting_for_encoder_resolution'):
                        publish('waiting_for_tcsr_resolution', pending=plan, upstream=upstream)
                        return
                time.sleep(15)

            for position, (group, (override, label)) in enumerate(ABLATIONS.items()):
                pending = plan[position+1:]
                if (root / 'STOP').exists():
                    publish('stopped_by_request', pending=plan[position:])
                    return
                if git_identity() != identity:
                    raise RuntimeError('Ablation source changed after launch')
                config = root / f'{group}.json'
                write(config, {**base, 'mlrf_ablation': group,
                               'fusion_config': {**base['fusion_config'], **override}})
                output = root / group
                log = root / f'{group}.log'
                cmd = [sys.executable, '-u', '-m', 'variants.rerun.train', '--variant', 'm1',
                       '--config', str(config), '--run', f'{args.campaign}_{group}',
                       '--paths', str(Path(args.paths).resolve()), '--output', str(output), '--gpu', args.gpu]
                publish('training', active=group, pending=pending, label=label, log=str(log))
                code = run_child(cmd, log, root,
                                 lambda child: (output / 'STOP').touch() if output.exists() else None)
                if (root / 'STOP').exists() or (output / 'STOP').exists():
                    publish('stopped_by_request', pending=plan[position:])
                    return
                if code != 0:
                    raise RuntimeError(f'{group} failed; see {log}')
                result = load(output / 'RESULTS.json')
                if not result['complete'] or load(output / 'status.json')['state'] != 'complete':
                    raise RuntimeError(f'{group} did not finish training')
                rows[group] = {k: result[k] for k in ('metrics', 'best_epoch', 'weights')}
                summarize()
            publish('complete')
        except Exception as error:
            prior = load(root / 'STATUS.json')
            write(root / 'STATUS.json', {**prior, 'state': 'needs_analysis', 'error': str(error)})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    p.add_argument('--campaign', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--after', type=Path, default=Path('tcsr_ablations_20261120'))
    main(p.parse_args())
