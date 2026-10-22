"""Sequential prespecified crop controls, with predecessor wait and interpretation.

Each new run keeps the source 60-epoch schedule and stops after 5 unimproved epochs.
Writes status/comparison after each process. Failures stop the queue.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.dual_state_maps import atomic_json, sha256

NAMES={'none':'dual-add-joint-cropcontrol', 'foreground':'dual-add-joint-foregroundcrop',
       'hard':'dual-add-joint-hardcrop','dual_state':'dual-add-joint-dualstatecrop'}


def comparison(map_dir,runs):
    m=json.loads((Path(map_dir)/'manifest.json').read_text())
    source=Path(m['source']['path']).parent/'train.jsonl'
    rows=[]
    sources=[('historical_B0',source)]+[(mode,Path(directory)/'train.jsonl') for mode,directory in runs.items()]
    for mode,path in sources:
        if not path.exists():continue
        values=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not values:continue
        selected=max(values,key=lambda r:r['selection_dice'])
        keys=['epoch','weights','full_dice','missing_dice','joint_dice','full_hd95','missing_hd95',
              'full_sensitivity','missing_sensitivity','full_precision','missing_precision',
              'full_missed_positive_count','missing_missed_positive_count']
        rows.append({'strategy':mode,'observed_epochs':len(values),'selected':{k:selected.get(k) for k in keys},
                     'completed':(path.parent/'final_test.json').exists()})
    warnings=['Development protocol reuses test.txt for selection; these are not independent final-test results.',
              'Historical B0 has 24 recorded epochs and no new patience setting; do not claim an equal-budget prospective control.',
              'Compare A3 with A1/A2, not only B0. Region area/foreground composition differs; inspect audit summaries.',
              'Single seed; no statistical significance or SOTA claim. No v2 change is selected automatically.']
    lookup={r['strategy']:r for r in rows}
    deltas={}
    if 'dual_state' in lookup:
        a3=lookup['dual_state']['selected']['joint_dice']
        for mode,r in lookup.items():
            if mode!='dual_state':deltas[mode]=a3-r['selected']['joint_dice']
    return {'runs':rows,'A3_joint_delta':deltas,'warnings':warnings}


def interpretation(map_dir, runs, reference_status=None):
    current = comparison(map_dir, runs)
    reference = {}
    if reference_status:
        state = json.loads(Path(reference_status).read_text())
        reference = {r['strategy']: r for r in comparison(map_dir, state['runs'])['runs']}
    lookup = {r['strategy']: r for r in current['runs']}
    deltas = {}
    for mode in ('dual_state', 'hard'):
        if mode in lookup and mode in reference:
            deltas[mode] = lookup[mode]['selected']['joint_dice']-reference[mode]['selected']['joint_dice']
    conclusion = 'Awaiting completion of both prespecified v2 controls; no performance conclusion yet.'
    if all(mode in lookup and lookup[mode]['completed'] for mode in ('dual_state','hard')):
        a3 = lookup['dual_state']['selected']['joint_dice']
        a2 = lookup['hard']['selected']['joint_dice']
        baseline = reference.get('none', reference.get('historical_B0'))
        b0 = baseline['selected']['joint_dice'] if baseline else None
        if a3 <= a2:
            conclusion = 'No observed dual-state advantage over matched weighted-hard control; do not claim the grouping mechanism is effective.'
        elif b0 is not None and a3 <= b0:
            conclusion = 'A3 exceeds matched A2 but still does not beat B0; the augmentation is not supported as a baseline improvement.'
        else:
            conclusion = 'Single-seed development result is promising; uncertainty and independent validation remain untested.'
    return {'region_sampling': 'sqrt_area', 'single_change': 'component selection weights only',
            'same_strategy_joint_delta_from_v1': deltas, 'conclusion': conclusion,
            'current': current, 'reference': reference,
            'next_step': 'Review region size, foreground composition, recall and HD95 before proposing another single change. No automatic test-driven parameter sweep.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True)
    p.add_argument('--output',required=True,help='New suite directory for logs/status, not model weights')
    p.add_argument('--run_root',default='checkpoints_new/STAGED')
    p.add_argument('--tag',default='s2023-v1')
    p.add_argument('--strategies',nargs='+',choices=list(NAMES)+['random'],default=['foreground','hard','dual_state'])
    p.add_argument('--guidance_region_sampling',choices=['uniform','sqrt_area'],default='uniform')
    p.add_argument('--early_stopping_patience',type=int,default=5)
    p.add_argument('--wait_for_suite',help='Wait for this status.json to complete before using the GPU; failure stops this queue')
    p.add_argument('--reference_suite',help='Prior status.json for per-run quantitative interpretation')
    p.add_argument('--continue_existing',action='store_true',help='Continue a paused suite using its current last.pt')
    p.add_argument('--paste_index',help='Enable paired Copy-Paste controls instead of crop-guidance controls')
    p.add_argument('--paste_selection',choices=['overlap','dominance'],default='overlap')
    a=p.parse_args()
    if a.early_stopping_patience<0: p.error('early_stopping_patience must be nonnegative')
    output=Path(a.output).resolve()
    if output.exists() and any(output.iterdir()) and not a.continue_existing:
        raise FileExistsError('Use a new suite directory')
    output.mkdir(parents=True,exist_ok=True)
    if a.paste_index and any(mode not in ('dual_state','hard','random') for mode in a.strategies):
        p.error('Paste suite supports dual_state/hard/random')
    if not a.paste_index and 'random' in a.strategies:p.error('random target control requires paste_index')
    names=({'dual_state':'dual-add-joint-dualstatepaste','hard':'dual-add-joint-hardpaste',
            'random':'dual-add-joint-randompaste'} if a.paste_index else NAMES)
    runs={mode:str((Path(a.run_root)/(names[mode]+'-'+a.tag)).resolve()) for mode in a.strategies}
    if len(runs)!=len(a.strategies):raise ValueError('Duplicate strategy')
    if not a.continue_existing:
        for directory in runs.values():
            if Path(directory).exists() and any(Path(directory).iterdir()):raise FileExistsError(directory)
    status={'started_unix':time.time(),'pid':os.getpid(),'manifest_sha256':sha256(Path(a.map_dir)/'manifest.json'),
            'order':a.strategies,'runs':runs,'completed':[],'current':None,'state':'running',
            'region_sampling':a.guidance_region_sampling,'wait_for_suite':a.wait_for_suite,
            'paste_index':str(Path(a.paste_index).resolve()) if a.paste_index else None,
            'paste_selection':a.paste_selection,
            'early_stopping_patience':a.early_stopping_patience}
    if a.continue_existing:
        previous=json.loads((output/'status.json').read_text())
        if previous['state']!='paused' or previous['runs']!=runs or previous['order']!=a.strategies:
            raise ValueError('Only the exact paused suite may continue')
        if previous['manifest_sha256']!=status['manifest_sha256'] or previous['region_sampling']!=a.guidance_region_sampling:
            raise ValueError('Continuation cannot change cache or sampling')
        status.update(completed=previous['completed'],resumed_from_pid=previous['pid'])
    if a.guidance_region_sampling != 'uniform' and any(mode not in ('hard','dual_state') for mode in runs):
        raise ValueError('sqrt_area queue supports hard/dual_state only')
    if a.wait_for_suite:
        status['state']='waiting'
        atomic_json(output/'status.json',status)
        while True:
            prior=json.loads(Path(a.wait_for_suite).read_text())
            if prior['state']=='complete':break
            if prior['state']=='failed':
                status.update(state='failed',reason='predecessor failed')
                atomic_json(output/'status.json',status)
                raise RuntimeError('Predecessor queue failed; refusing to start')
            try: os.kill(prior['pid'],0)
            except ProcessLookupError:
                status.update(state='failed',reason='predecessor process disappeared')
                atomic_json(output/'status.json',status)
                raise RuntimeError('Predecessor status is stale; process is absent')
            print('[SUITE] waiting for predecessor to finish',flush=True)
            time.sleep(30)
        status['state']='running'
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode,directory in runs.items():
        if mode in status['completed']: continue
        status['current']=mode
        command=[sys.executable,'-u','scripts/launch_joint_aug.py',a.map_dir,directory,
                 '--crop_guidance',mode,'--guided_crop_prob','0.5','--early_stopping_patience',str(a.early_stopping_patience),'--device','cuda',
                 '--guidance_region_sampling',a.guidance_region_sampling]
        if a.paste_index:
            command=[sys.executable,'-u','scripts/launch_joint_aug.py',a.map_dir,directory,
                     '--crop_guidance','none','--copy_paste',mode,'--copy_paste_prob','0.3',
                     '--paste_index',a.paste_index,'--early_stopping_patience',str(a.early_stopping_patience),
                     '--paste_selection',a.paste_selection,
                     '--device','cuda','--no-save_last_checkpoint']
        if a.continue_existing and Path(directory).exists() and any(Path(directory).iterdir()):
            if not (Path(directory)/'last.pt').is_file():
                raise FileNotFoundError('No latest recovery checkpoint for paused run')
            command=[sys.executable,'-u','run_staged.py','--resume',str(Path(directory)/'last.pt'),
                     '--device','cuda','--no-save_last_checkpoint']
        status['checkpoint_policy']='best_only'
        status['command']=command
        atomic_json(output/'status.json',status)
        print(f'[SUITE] start {mode} output={directory}',flush=True)
        with (output/(mode+'.log')).open('a' if a.continue_existing else 'w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        atomic_json(output/'comparison.json',comparison(a.map_dir,runs))
        if a.reference_suite:
            report=interpretation(a.map_dir,runs,a.reference_suite)
            if a.paste_index:
                report.update(region_sampling='whole_gt_component_uniform; control size bins matched',
                              single_change='paired contextual Copy-Paste; dual vs pooled donor selection',
                              next_step='Inspect achieved paste rates, pasted foreground exposure, size bins and anatomical artifacts before proposing one further change.')
                lookup={r['strategy']:r for r in report['current']['runs']}
                if all(k in lookup and lookup[k]['completed'] for k in ('dual_state','random')):
                    if lookup['dual_state']['selected']['joint_dice']<=lookup['random']['selected']['joint_dice']:
                        report['conclusion']='No observed advantage over size-matched random-target Copy-Paste; no evidence for dual-state selection benefit.'
            atomic_json(output/'interpretation.json',report)
        if result.returncode:
            status.update(state='failed',returncode=result.returncode,finished_unix=time.time())
            atomic_json(output/'status.json',status)
            raise RuntimeError(f'{mode} failed; queue stopped. See {output/(mode+".log")}')
        status['completed'].append(mode)
        print(f'[SUITE] complete {mode}',flush=True)
    status.update(state='complete',current=None,finished_unix=time.time())
    atomic_json(output/'status.json',status)
    print(f'[SUITE] all done; comparison={output/"comparison.json"}',flush=True)


if __name__=='__main__':main()
