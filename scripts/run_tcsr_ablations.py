"""Run the V replication and the four single-factor TCSR controls."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, git_identity, local_paths, require_myenv, sha
from run_scgr_homogeneous import load, run_child, write


DESIGNS = {
    'V_repeat': ('V', 'Zero input, full visual channel modulation and spatial gate.'),
    'no_tccm': ('T', 'Set the channel multiplier to one; keep value and spatial paths.'),
    'no_lcsg': ('T', 'Set the spatial multiplier to 0.5; keep local-context value and text channel paths.'),
    'learnable_vector': ('T', 'Replace the frozen text embedding with one shared normalized learnable 512D vector.'),
    'mismatched_text': ('T', 'Replace the prompt with Benign hepatic cyst. using the same frozen encoder.'),
}
OPTIONAL_DESIGNS = {
    'synonym_text': ('T', 'Replace Pulmonary with Lung in the fixed BioMedCLIP task prompt.'),
}
ALL_DESIGNS = {**DESIGNS, **OPTIONAL_DESIGNS}


def reference_rows(output_root):
    rows = {}
    for recipe in ('T', 'V'):
        path = output_root / 'scgr_postfusion_20261116' / recipe
        result = load(path / 'RESULTS.json')
        identity = load(path / 'run_identity.json')
        original = load(REPO / 'configs' / f'scgr_{recipe}.json')
        if not result['complete'] or any(identity['config'][key] != value for key, value in original.items()):
            raise RuntimeError(f'Historical {recipe} reference protocol differs')
        for variable, filename in (('train_split_file', 'train.txt'), ('val_split_file', 'dev.txt')):
            if sha(identity['config'][variable]) != sha(REPO / 'splits' / filename):
                raise RuntimeError('Historical split differs from this campaign')
        rows[f'{recipe}_original'] = {'metrics': result['metrics'], 'best_epoch': result['best_epoch'],
                                     'weights': result['weights'], 'reference': str(path)}
    return rows


def run(args, output_root, root):
    source = git_identity()
    if source['dirty']:
        raise RuntimeError('Commit the ablation code before running')
    reference = reference_rows(output_root)
    plan = [args.only] if args.only else list(DESIGNS)
    dependency = args.after if args.after.is_absolute() else output_root / args.after
    configs = {r: load(REPO / 'configs' / f'scgr_{r}.json') for r in ('T', 'V')}
    write(root / 'PLAN.json', {
        'source': source, 'start_after': str(dependency), 'groups': plan,
        'controls': {k: ALL_DESIGNS[k][1] for k in plan}, 'reference': reference,
        'fixed': configs['T'], 'device': args.gpu, 'postprocessing': False,
        'selection': 'Best direct-output development Dice and same-epoch HD95.',
        'scope': 'Single-seed within-module comparison. Fixed prompt differences do not establish semantic alignment.',
    })
    results = {}

    def publish(state, **details):
        write(root / 'STATUS.json', {'state': state, 'updated': time.time(),
                                     'completed': list(results), **details})

    def summarize():
        write(root / 'COMPARISON.json', {'reference': reference, 'ablations': results})
        lines = ['# TCSR internal ablations', '',
                 '| Configuration | Best epoch | Weights | Dice % | HD95 | Precision % | Sensitivity % |',
                 '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
        for name, row in {**reference, **results}.items():
            m = row['metrics']
            lines.append(f"| {name} | {row['best_epoch']} | {row['weights']} | {100*m['dice']:.4f} | "
                         f"{m['hd95']:.4f} | {100*m['precision']:.4f} | {100*m['sensitivity']:.4f} |")
        lines.extend(('', 'Direct segmentation outputs on the reused development set.'))
        (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')

    summarize()
    publish('waiting_for_prior_experiments', dependency=str(dependency), pending=plan)
    while True:
        if (root / 'STOP').exists():
            publish('stopped_by_request', pending=plan)
            return
        status_file = dependency / 'STATUS.json'
        if status_file.is_file():
            previous = load(status_file)
            if previous['state'] == 'complete':
                if set(previous['completed']) != set(args.after_completed):
                    raise RuntimeError(f'Expected the dependency to finish {args.after_completed}')
                break
            if previous['state'] in ('needs_analysis', 'stopped_by_request',
                                         'waiting_for_dependency_resolution'):
                publish('waiting_for_dependency_resolution', pending=plan, upstream=previous)
                return
        time.sleep(15)

    for i, group in enumerate(plan):
        if (root / 'STOP').exists():
            publish('stopped_by_request', pending=plan[i:])
            return
        if git_identity() != source:
            raise RuntimeError('Ablation source changed after launch')
        recipe, description = ALL_DESIGNS[group]
        cfg = dict(configs[recipe])
        if group != 'V_repeat':
            cfg.update(tcsr_ablation=group, text_description=description)
        config_path = root / f'{group}.json'
        write(config_path, cfg)
        output = root / group
        log = root / f'{group}.log'
        command = [sys.executable, '-u', '-m', 'variants.scgr.train', '--recipe', recipe,
                   '--run', f'{args.campaign}_{group}', '--paths', str(Path(args.paths).resolve()),
                   '--config', str(config_path), '--output', str(output), '--gpu', args.gpu]
        publish('training', active=group, pending=plan[i+1:], log=str(log))
        code = run_child(command, log, root,
                         lambda child: (output / 'STOP').touch() if output.exists() else None)
        if (root / 'STOP').exists() or (output / 'STOP').exists():
            publish('stopped_by_request', pending=plan[i:])
            return
        if code != 0:
            raise RuntimeError(f'{group} failed; see {log}')
        result = load(output / 'RESULTS.json')
        if not result['complete'] or load(output / 'status.json')['state'] != 'complete':
            raise RuntimeError(f'{group} did not finish training')
        results[group] = {k: result[k] for k in ('metrics', 'best_epoch', 'weights')}
        results[group]['output'] = str(output)
        summarize()
    publish('complete')


def main(args):
    require_myenv()
    if Path(args.campaign).name != args.campaign or args.campaign in ('.', '..'):
        raise ValueError('Choose a simple campaign directory name')
    output_root = Path(local_paths(args.paths)['output_root'])
    root = output_root / args.campaign
    root.mkdir(exist_ok=False)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, output_root, root)
        except Exception as error:
            path = root / 'STATUS.json'
            prior = load(path) if path.exists() else {}
            write(path, {**prior, 'state': 'needs_analysis', 'error': str(error)})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    p.add_argument('--campaign', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--after', type=Path, default=Path('scgr_homogeneous_20261120'))
    p.add_argument('--after-completed', nargs='+', default=['mit_b1', 'convnextv2_nano'])
    p.add_argument('--only', choices=ALL_DESIGNS)
    main(p.parse_args())
