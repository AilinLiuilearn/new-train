"""Exactly two user-authorized complete-model runs: visual, then fixed text."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,local_paths,git_identity,sha


def dump(p,x):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit code before launch')
    checks=json.loads((REPO/'docs/CCSCMI_GPU_PREFLIGHT.json').read_text())
    if not all(checks[name]['passed'] for name in ['visual','text']):raise RuntimeError('GPU preflight not passed')
    for name,expected in checks['source_sha256'].items():
        if sha(REPO/name)!=expected:raise RuntimeError('Preflight source changed: '+name)
    root=Path(local_paths(args.paths)['output_root'])/args.campaign
    root.mkdir(parents=True,exist_ok=True);lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'PLAN.json').exists():raise RuntimeError('Existing campaign')
    (root/'logs').mkdir(exist_ok=True);rows=[]
    dump(root/'PLAN.json',{'git':identity,'pid':os.getpid(),'groups':['visual','text'],'gpu':args.gpu,
        'preflight':checks,'initialization':'Generic encoders only; all task modules fresh; no V3.1 or segmentation checkpoint',
        'scope':'Two full candidates only. No new M1/shared-attention/semantic controls, so no isolated mechanism or text semantics claim',
        'next_experiments':[]})
    def status(state,**kw):
        dump(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),'results':rows,**kw})
        lines=['# CC-SCMI full-model visual/text trial','','|Group|Best epoch|Dice %|Paired HD95|Status|','|---|---:|---:|---:|---|']
        for row in rows:
            result=row['result'];m=result.get('metrics',{})
            lines.append('|'+row['group']+'|'+str(result.get('best_epoch',''))+'|'+(f"{100*m['dice']:.4f}" if m else '')+'|'+(f"{m['hd95']:.4f}" if m else '')+'|'+row['state']+'|')
        lines+=['','Reused dev, single seed. Two full candidates do not establish isolated core or semantic efficacy.']
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    status('queued')
    for name in ['visual','text']:
        if (root/'STOP').exists():status('stopped_by_request');return
        if git_identity()!=identity:raise RuntimeError('Source identity changed')
        out=root/name
        cmd=[sys.executable,'-u','-m','variants.ccscmi.train','--run',args.campaign+'_'+name,
             '--paths',str(Path(args.paths).resolve()),'--output',str(out),'--gpu',args.gpu]
        if name=='text':cmd.append('--text')
        status('running',active=name,command=cmd)
        with (root/'logs'/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            dump(root/'CHILD.json',{'pid':child.pid,'group':name})
            while child.poll() is None:
                if (root/'STOP').exists() and out.exists():(out/'STOP').touch()
                time.sleep(10)
        result=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
        s=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        state='complete' if child.returncode==0 and result.get('complete') and s.get('state')=='complete' else 'needs_analysis'
        if (root/'STOP').exists() or s.get('state')=='stopped_by_request':state='stopped_by_request'
        rows.append({'group':name,'state':state,'result':result,'returncode':child.returncode,'output':str(out)})
        if state!='complete':status(state,active=name);return
    status('complete',next_experiments=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--campaign',required=True);p.add_argument('--gpu',default='0')
    args=p.parse_args()
    try:main(args)
    except Exception as e:
        import traceback;traceback.print_exc()
        root=Path(local_paths(args.paths)['output_root'])/args.campaign
        if root.exists():dump(root/'STATUS.json',{'state':'needs_analysis','error':repr(e)})
        raise SystemExit(1)
