"""One explicit full-candidate run; no automatic follow-up or tuning queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,local_paths,git_identity,sha


def dump(path,obj):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit source before launch')
    preflight=REPO/'docs/BCSI_ADAPT_GPU_PREFLIGHT.json'
    checks=json.loads(preflight.read_text())
    if not checks['passed']:raise RuntimeError('GPU preflight failed')
    for path,expected in checks['source_sha256'].items():
        if sha(REPO/path)!=expected:raise RuntimeError('Preflight source changed: '+path)
    root=Path(local_paths(args.paths)['output_root'])/args.campaign
    root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PLAN.json').exists():raise RuntimeError('Existing campaign; choose a new run')
        (root/'logs').mkdir(exist_ok=True)
        command=[sys.executable,'-u','-m','variants.bcsi_adapt.train','--run',args.campaign+'_full',
                 '--paths',str(Path(args.paths).resolve()),'--output',str(root/'full'),'--gpu',args.gpu]
        dump(root/'PLAN.json',{'git':identity,'pid':os.getpid(),'command':command,'gpu':args.gpu,
                              'variant':'bcsi_adapt','groups':['full'],'selected_fraction':.5,
                              'preflight':checks,'next_experiments':[]})
        def status(state,**extra):
            dump(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),**extra})
        if (root/'STOP').exists():status('stopped_by_request');return
        status('running',active='full')
        with (root/'logs/full.log').open('w') as stream:
            child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
            dump(root/'CHILD.json',{'pid':child.pid,'group':'full'})
            while child.poll() is None:
                if (root/'STOP').exists() and (root/'full').exists():(root/'full/STOP').touch()
                time.sleep(10)
        out=root/'full'
        result=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
        state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        stopped=(root/'STOP').exists() or state.get('state')=='stopped_by_request'
        complete=child.returncode==0 and result.get('complete') and state.get('state')=='complete'
        final='stopped_by_request' if stopped else 'complete' if complete else 'needs_analysis'
        dump(root/'RESULTS.json',{'state':final,'result':result,'returncode':child.returncode})
        lines=['# BCSI-Adapt K50 full-model run','',f'State: {final}.']
        if result.get('metrics'):
            m=result['metrics']
            lines += [f"Best epoch: {result['best_epoch']} ({result['weights']}).",
                      f"Dice {100*m['dice']:.4f}%; paired HD95 {m['hd95']:.4f}."]
            if 'reference_M1' in result:
                r=result['reference_M1']['result']['metrics']
                lines += [f"Same-protocol M1: {100*r['dice']:.4f}%, HD95 {r['hd95']:.4f}.",
                          f"Difference: {result['delta_M1_dice_pp']:+.4f} percentage points Dice; "
                          f"{result['delta_M1_hd95']:+.4f} HD95."]
            lines += [f"Joint target Dice>=78.8%/HD95<17: {result['target_788_hd17_met']}."]
        lines += ['', 'One adapted paired-inference model, single seed and reused development set. '
                  'This does not reproduce full semi-supervised BCSI or isolate router and attention contributions. '
                  'No automatic next experiment; retained diagnostics support analysis, not a causal conclusion.']
        (root/'FINAL_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
        status(final,result=result,returncode=child.returncode,next_experiments=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'))
    p.add_argument('--campaign',required=True);p.add_argument('--gpu',default='0');args=p.parse_args()
    try:main(args)
    except Exception as error:
        import traceback;traceback.print_exc()
        root=Path(local_paths(args.paths)['output_root'])/args.campaign
        if root.exists():dump(root/'STATUS.json',{'state':'needs_analysis','error':repr(error)})
        raise SystemExit(1)
