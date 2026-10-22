"""Bounded preregistered Copy-Paste pilots with one conditional design iteration.

No trainer duplication, no hyperparameter sweep, no source task-weight transfer.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.dual_state_maps import atomic_json


def selected_row(directory):
    rows=[json.loads(line) for line in (Path(directory)/'train.jsonl').read_text().splitlines() if line.strip()]
    return max(rows,key=lambda r:r['selection_dice'])


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True);p.add_argument('--paste_index',required=True)
    p.add_argument('--output',required=True);p.add_argument('--run_root',default='checkpoints_new/STAGED')
    p.add_argument('--reference_suite',default='checkpoints_new/AUGMENTATION/pilots-s2023-v1/status.json')
    a=p.parse_args();output=Path(a.output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    ref=json.loads(Path(a.reference_suite).read_text());baseline=selected_row(ref['runs']['none'])
    status={'pid':os.getpid(),'state':'running','started_unix':time.time(),'current':None,'history':[],
            'plan':{'v1':['dual_state','hard','random'],'v2_if_v1_not_above_all':['dual_state','hard'],
                    'single_v2_change':'assign whole targets by dominant D=1 vs D=2 FN pixel count; ties excluded',
                    'max_runs':5,'patience':5,'checkpoint':'best_only','seeds':[2023]},
            'baseline_joint':baseline['selection_dice']}
    atomic_json(output/'status.json',status)
    for version,selection,modes in [('v1','overlap',['dual_state','hard','random']),('v2','dominance',['dual_state','hard'])]:
        suite=output/version;status['current']=version;atomic_json(output/'status.json',status)
        command=[sys.executable,'-u','scripts/run_crop_pilots.py','--map_dir',a.map_dir,'--output',str(suite),
                 '--run_root',a.run_root,'--tag','cp-s2023-'+version,'--strategies',*modes,
                 '--paste_index',a.paste_index,'--paste_selection',selection,'--early_stopping_patience','5',
                 '--reference_suite',a.reference_suite]
        print(f'[ITERATION] start {version} selection={selection}',flush=True)
        with (output/(version+'.console.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        if result.returncode:
            status.update(state='failed',reason=f'{version} suite failed; no automatic retry')
            atomic_json(output/'status.json',status);raise RuntimeError(status['reason'])
        state=json.loads((suite/'status.json').read_text())
        metrics={mode:selected_row(directory) for mode,directory in state['runs'].items()}
        a3=metrics['dual_state']['selection_dice']
        controls=[metrics[mode]['selection_dice'] for mode in modes if mode!='dual_state']
        better=a3>max([baseline['selection_dice']]+controls)
        analysis={'version':version,'selection':selection,'selected_same_epoch_metrics':metrics,
                  'baseline_selected':baseline,'joint_delta_from_baseline':a3-baseline['selection_dice'],
                  'joint_delta_from_hard':a3-metrics['hard']['selection_dice'],
                  'observed_advantage_over_available_controls_and_baseline':better,
                  'warning':'Single-seed development protocol reuses test for selection; no independent-test or significance claim.'}
        if version=='v1':
            analysis['next_action']='stop: promising pilot, retain for verification' if better else 'one prespecified dominance-grouping iteration with matched hard control'
        else:
            analysis['next_action']='stop and review evidence; no further automatic algorithm or parameter search'
        atomic_json(output/(version+'-analysis.json'),analysis)
        lines=[f'# Copy-Paste {version} completed', '', '|mode|epoch|Full|Missing|Joint|','|---|---:|---:|---:|---:|']
        for mode,row in metrics.items():
            lines.append(f'|{mode}|{row["epoch"]}|{row["full_dice"]:.6f}|{row["missing_dice"]:.6f}|{row["selection_dice"]:.6f}|')
        lines+=['',analysis['next_action'],'',analysis['warning']]
        (output/(version+'-analysis.md')).write_text('\n'.join(lines)+'\n')
        status['history'].append({'version':version,'analysis':str(output/(version+'-analysis.json')),
                                  'better':better})
        atomic_json(output/'status.json',status)
        if better:break
    status.update(state='complete',current=None,finished_unix=time.time());atomic_json(output/'status.json',status)
    print('[ITERATION DONE] all allowed work complete; review saved analyses',flush=True)


if __name__=='__main__':run()
