"""Wait for all three reruns, then preflight -> joint training -> frozen correction."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,local_paths,git_identity
from variants.v31.train import dump


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit pipeline first')
    loc=local_paths(args.paths);root=Path(loc['output_root'])/args.run
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'PLAN.json').exists():raise RuntimeError('Existing pipeline; use a fresh name')
    wait=Path(args.wait_for).resolve()
    dump(root/'PLAN.json',{'pid':os.getpid(),'git':identity,'wait_for':str(wait),'init':args.init,
         'stages':['GPU preflight','joint','correction'],'ablation_jobs':[],
         'scope':'Full V3.1 pilot only; not matched-control proof or independent testing'})
    def status(state,**kwargs):dump(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),**kwargs})
    status('waiting_for_three_reruns')
    while True:
        if (root/'STOP').exists() or (wait/'STOP').exists():status('cancelled_by_request');return
        if (wait/'STATUS.json').exists():
            upstream=json.loads((wait/'STATUS.json').read_text())
            if upstream['state']=='complete':
                if len(upstream.get('results',[]))!=3 or any(r['state']!='complete' for r in upstream['results']):
                    raise RuntimeError('Upstream completion does not contain three successful trials')
                break
            if upstream['state'] in ['needs_analysis','stopped_by_request','failed']:
                status('blocked_by_upstream',upstream=upstream['state']);return
        time.sleep(30)
    def run(stage,cmd):
        if (root/'STOP').exists():status('cancelled_by_request');return False
        if git_identity()!=identity:raise RuntimeError('Pipeline code changed')
        status('running',stage=stage,command=cmd)
        with (root/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (root/'STOP').exists():
                    if stage=='preflight':child.terminate()
                    elif (root/stage).exists():(root/stage/'STOP').touch()
                time.sleep(10)
        if child.returncode!=0 or (root/'STOP').exists():
            status('cancelled_by_request' if (root/'STOP').exists() else 'needs_analysis',stage=stage,returncode=child.returncode);return False
        if stage!='preflight':
            result=json.loads((root/stage/'RESULTS.json').read_text()) if (root/stage/'RESULTS.json').exists() else {}
            if not result.get('complete'):status('needs_analysis',stage=stage);return False
        return True
    # The GPU UUID lock avoids racing another explicitly launched experiment.
    from runtime import gpu_lock_path
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    with open(gpu_lock_path(),'a') as gpu:
        while True:
            if (root/'STOP').exists():status('cancelled_by_request');return
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        cmd=[sys.executable,'-u','scripts/v31_preflight.py','--init',('generic' if args.init=='generic' else str(Path(args.init).resolve())),
             '--paths',str(Path(args.paths).resolve()),'--output',str(root/'PREFLIGHT.json')]
        if not run('preflight',cmd):return
    if not json.loads((root/'PREFLIGHT.json').read_text()).get('passed'):raise RuntimeError('Preflight failed')
    for stage in ['joint','correction']:
        init=('generic' if args.init=='generic' else str(Path(args.init).resolve())) if stage=='joint' else str(root/'joint/best.pt')
        cmd=[sys.executable,'-u','-m','variants.v31.train','--phase',stage,'--run',args.run+'_'+stage,
             '--paths',str(Path(args.paths).resolve()),'--output',str(root/stage),'--init',init,'--gpu',args.gpu]
        if not run(stage,cmd):return
    results={stage:json.loads((root/stage/'RESULTS.json').read_text()) for stage in ['joint','correction']}
    dump(root/'RESULTS.json',results)
    lines=['# V3.1 full pilot','','|Stage|Best epoch|Dice %|Same-epoch HD95|','|---|---:|---:|---:|']
    for stage,r in results.items():
        m=r['metrics'];lines.append(f"|{stage}|{r['best_epoch']}|{100*m['dice']:.4f}|{m['hd95']:.4f}|")
    lines+=['','Full candidate only; no matched continuation control or isolated component claims.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');status('complete',next_experiments=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for',required=True);p.add_argument('--init',required=True)
    p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--run',required=True);p.add_argument('--gpu',default='0')
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        import traceback;traceback.print_exc()
        root=Path(local_paths(args.paths)['output_root'])/args.run
        if root.exists():dump(root/'STATUS.json',{'state':'needs_analysis','error':repr(error)})
        raise SystemExit(1)
