"""Run exactly three authorized experiments, then summarize without automatic tuning."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,local_paths,git_identity,require_myenv


def write(path,obj):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit the campaign before running')
    locations=local_paths(args.paths)
    root=Path(locations['output_root'])/args.campaign
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'PLAN.json').exists():raise RuntimeError('Existing campaign; choose a new name')
    names=['m1','concat_single','tsbci'];rows=[]
    plan={'groups':names,'git':identity,'pid':os.getpid(),'gpu':args.gpu,'patience':3,
          'fixed':'seed2023,512,batch16,lr8e-5,original BCE+Dice,GN,original split and metric',
          'baseline':'CT pre-alignment only, native PET, existing concat1x1 compressor',
          'comparison_limit':'M1 vs concat also differs in PET alignment, so not an isolated fusion ablation',
          'next_after_three':[]}
    write(root/'PLAN.json',plan);(root/'logs').mkdir(exist_ok=True)
    def publish(state,**extra):
        write(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),'results':rows,**extra})
        lines=['# Three authorized reruns (patience=3)','','|Variant|Best epoch|Dice %|Paired HD95|Status|',
               '|---|---:|---:|---:|---|']
        for row in rows:
            r=row.get('result',{});m=r.get('metrics',{})
            lines.append('|'+row['variant']+'|'+str(r.get('best_epoch',''))+'|'+(f"{m['dice']*100:.4f}" if 'dice' in m else '')+'|'+(f"{m['hd95']:.4f}" if 'hd95' in m else '')+'|'+row['state']+'|')
        lines+=['','Same-checkpoint Dice/HD95 only. Reused development set; no independent test claim.',
                'Concat also removes PET alignment; M1 difference is not an isolated module effect.']
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    publish('queued')
    for variant in names:
        if (root/'STOP').exists():publish('stopped_by_request',pending=names[names.index(variant):]);return
        if git_identity()!=identity:raise RuntimeError('Code identity changed during campaign')
        out=root/variant
        cmd=[sys.executable,'-u','-m','variants.rerun.train','--variant',variant,'--run',args.campaign+'_'+variant,
             '--paths',str(Path(args.paths).resolve()),'--output',str(out),'--gpu',args.gpu]
        publish('running',active=variant,output=str(out),command=cmd)
        with (root/'logs'/f'{variant}.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            write(root/'CHILD.json',{'variant':variant,'pid':child.pid,'command':cmd})
            while child.poll() is None:
                if (root/'STOP').exists() and out.exists():
                    (out/'STOP').touch()
                time.sleep(10)
        result=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        state='complete' if child.returncode==0 and result.get('complete') and status.get('state')=='complete' else 'needs_analysis'
        if status.get('state')=='stopped_by_request' or (root/'STOP').exists():state='stopped_by_request'
        rows.append({'variant':variant,'state':state,'returncode':child.returncode,'result':result,'output':str(out)})
        if state!='complete':publish(state,active=variant);return
        publish('between_runs')
    publish('complete',next_experiments=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'))
    p.add_argument('--campaign',required=True);p.add_argument('--gpu',default='0')
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        import traceback;traceback.print_exc();raise SystemExit(1)
