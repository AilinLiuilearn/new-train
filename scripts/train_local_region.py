"""Launch an explicit frozen GN protocol with one controlled fusion choice."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_staged


def training_argv(output, fusion_type='local_region_residual', use_gaussian=True):
    frozen = json.loads((Path(__file__).resolve().parents[1]/'configs/frozen_gn_training.json').read_text())
    frozen.pop('training_protocol')  # Derived/checked by run_staged, not a CLI setting.
    frozen.update(output=str(Path(output).resolve()), fusion_type=fusion_type, fusion_gaussian=use_gaussian)
    actions = {a.dest: a for a in run_staged.parser()._actions}
    argv = []
    for name, value in frozen.items():
        if value is None:
            continue
        action = actions[name]  # Unknown frozen fields fail, never silently disappear.
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append('--'+('' if value else 'no-')+name)
        elif isinstance(action, argparse._StoreTrueAction):
            if value: argv.append('--'+name)
        else:
            argv.extend(['--'+name, str(value)])
    return argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--fusion_type', choices=['add','local_region_residual'], default='local_region_residual')
    p.add_argument('--fusion_gaussian', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--print_command', action='store_true')
    a=p.parse_args()
    argv=training_argv(a.output,a.fusion_type,a.fusion_gaussian)
    if a.print_command:
        import shlex
        print(shlex.join([sys.executable,'-u','run_staged.py',*argv]))
        return
    run_staged.main(argv)


if __name__=='__main__': main()
