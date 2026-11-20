"""Train four MLRF controls in the complete T model after the TCSR queue."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, git_identity, local_paths, require_myenv, sha
from run_scgr_homogeneous import HETEROGENEOUS_REFERENCE, load, run_child, write
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
    reference_path = output_root / 'scgr_postfusion_20261116/T'
    base = load(REPO / 'configs/scgr_T.json')
    historical = load(reference_path / 'run_identity.json')
    reference = load(reference_path / 'RESULTS.json')
    if not reference['complete'] or any(historical['config'][k] != v for k, v in base.items()):
        raise RuntimeError('Historical complete T protocol differs')
    for split, file in (('train', 'train.txt'), ('val', 'dev.txt')):
        if historical['resources']['splits'][split]['sha256'] != sha(REPO / 'splits' / file):
            raise RuntimeError('Historical complete T data split differs')
    split_hash = sha(REPO / 'splits/dev.txt')
    if (split_hash != HETEROGENEOUS_REFERENCE['development_split_sha256'] or
            sha(reference_path / 'best.pt') != HETEROGENEOUS_REFERENCE['checkpoint_sha256']):
        raise RuntimeError('Reference checkpoint or split differs from case-refinement reference')
    root.mkdir(exist_ok=False)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {'complete_T_reference': {
            **{k: reference[k] for k in ('metrics', 'best_epoch', 'weights')},
            'case_refinement': HETEROGENEOUS_REFERENCE,
        }}
        plan = list(ABLATIONS)

        def publish(state, **kw):
            write(root / 'STATUS.json', {'state': state, 'completed': list(rows)[1:],
                                         'updated': time.time(), **kw})

        def summarize():
            write(root / 'COMPARISON.json', {'groups': rows})
            lines = ['# MLRF ablations in the complete T model', '',
                     '| Group | Best epoch | Weights | Raw Dice % | Raw HD95 | '
                     'Full Dice % | Full HD95 | Full zero overlap |',
                     '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
            for key, row in rows.items():
                raw = row['case_refinement']['without_case_refinement']
                full = row['case_refinement']['with_case_refinement']
                lines.append(f"| {key} | {row['best_epoch']} | {row['weights']} | {raw['dice_percent']:.4f} | "
                             f"{raw['hd95']:.4f} | {full['dice_percent']:.4f} | {full['hd95']:.4f} | "
                             f"{full['zero_overlap']} |")
            lines.extend(('', 'Compare the swap against no_gaussian. Single-seed development set.'))
            (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')

        write(root / 'PLAN.json', {
            'git': identity, 'after': str(dependency), 'groups': plan, 'config': base,
            'overrides': ABLATIONS, 'reference': str(reference_path),
            'fixed': 'Heterogeneous encoders; dual alignment; four scales; residual addition; complete text-conditioned TCSR; GN decoder; confirmed case refinement.',
            'selection': 'Select by direct-output development Dice. Evaluate the same checkpoint with and without case refinement, matching the reference protocol.',
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
                common = ['--paths', str(Path(args.paths).resolve()), '--output', str(output), '--gpu', args.gpu]
                cmd = [sys.executable, '-u', '-m', 'variants.scgr.train', '--recipe', 'T',
                       '--config', str(config), '--run', f'{args.campaign}_{group}', *common]
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
                case_log = root / f'{group}_case.log'
                checkpoint = output / 'best.pt'
                cmd = [sys.executable, '-u', 'scripts/evaluate_scgr_case_refinement.py',
                       '--checkpoint', str(checkpoint), *common]
                publish('evaluating_case_refinement', active=group, pending=pending, log=str(case_log))
                code = run_child(cmd, case_log, root, lambda child: child.terminate())
                if (root / 'STOP').exists():
                    publish('stopped_by_request', pending=plan[position:])
                    return
                if code != 0:
                    raise RuntimeError(f'{group} failed case evaluation; see {case_log}')
                case_metrics = load(output / 'CASE_REFINEMENT.json')
                if (case_metrics['mlrf_ablation'] != group or case_metrics['development_split_sha256'] != split_hash or
                        case_metrics['encoders'] != HETEROGENEOUS_REFERENCE['encoders']):
                    raise RuntimeError('Case evaluation differs from scheduled comparison')
                rows[group] = {**{k: result[k] for k in ('metrics', 'best_epoch', 'weights')},
                               'case_refinement': case_metrics}
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
