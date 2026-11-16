"""Exactly E1 -> E2 -> E4, each fresh; STOP or failure cancels the remainder."""
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

RECIPES=['E1','E2','E4']


def dump(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit code before launch')
    checks=json.loads((REPO/'docs/GT_VCL_GPU_PREFLIGHT.json').read_text())
    if not all(checks[r]['passed'] for r in RECIPES):raise RuntimeError('All three GPU preflights must pass')
    for file,expected in checks['source_sha256'].items():
        if sha(REPO/file)!=expected:raise RuntimeError('Preflight source changed: '+file)
    configs={r:json.loads((REPO/'configs'/('gt_vcl_'+r+'.json')).read_text()) for r in RECIPES}
    allowed={'recipe','calibration_enabled','semantic_enabled','campaign_loss'}
    common=[{k:v for k,v in configs[r].items() if k not in allowed} for r in RECIPES]
    if not common[0]==common[1]==common[2]:raise RuntimeError('Unintended extra configuration changes')
    root=Path(local_paths(args.paths)['output_root'])/args.campaign;root.mkdir(parents=True,exist_ok=True)
    rows=[]
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PLAN.json').exists():raise RuntimeError('Existing campaign, do not overwrite')
        (root/'logs').mkdir(exist_ok=True)
        dump(root/'PLAN.json',{'git':identity,'pid':os.getpid(),'groups':RECIPES,'gpu':args.gpu,
            'initialization':'Every recipe fresh from generic encoder pretraining; no previous task checkpoints',
            'preflight':checks,'fixed_config':common[0],
            'parents':{'E1':'M1 reference','E2':'E1','E4':'E2'},
            'changes':{'E1':'Shared CT/PET GN decoder and two GT losses, each weight1',
                       'E2':'Enable CT-reference calibration only',
                       'E4':'Enable cached text region regularizer only'},'next_after_three':[]})
        def publish(state,**extra):
            dump(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),'results':rows,**extra})
            lines=['# GT-VCL E1 -> E2 -> E4','','|Recipe|Best epoch|Weights|Fusion Dice %|Paired HD95|State|',
                   '|---|---:|---|---:|---:|---|']
            for row in rows:
                result=row['result'];m=result.get('metrics',{})
                lines.append('|'+row['recipe']+'|'+str(result.get('best_epoch',''))+'|'+str(result.get('weights',''))+'|'+
                             (f"{100*m['dice']:.4f}" if m else '')+'|'+(f"{m['hd95']:.4f}" if m else '')+'|'+row['state']+'|')
            lines+=['','Each recipe starts fresh; fusion Dice selects the reported checkpoint.',
                    'Reused development set, single seed. E4 compares the whole semantic regularizer, not medical language content alone.']
            (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        publish('queued')
        for index,recipe in enumerate(RECIPES):
            if (root/'STOP').exists():publish('stopped_by_request',pending=RECIPES[index:]);return
            if git_identity()!=identity:raise RuntimeError('Source identity changed during campaign')
            out=root/recipe
            command=[sys.executable,'-u','-m','variants.gt_vcl.train','--recipe',recipe,
                     '--run',args.campaign+'_'+recipe,'--paths',str(Path(args.paths).resolve()),
                     '--output',str(out),'--gpu',args.gpu]
            publish('running',active=recipe,pending=RECIPES[index+1:],command=command)
            with (root/'logs'/(recipe+'.log')).open('w') as stream:
                child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
                dump(root/'CHILD.json',{'pid':child.pid,'recipe':recipe})
                while child.poll() is None:
                    if (root/'STOP').exists() and out.exists():(out/'STOP').touch()
                    time.sleep(10)
            result=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
            state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            complete=child.returncode==0 and result.get('complete') and state.get('state')=='complete'
            stopped=(root/'STOP').exists() or state.get('state')=='stopped_by_request'
            final='stopped_by_request' if stopped else 'complete' if complete else 'needs_analysis'
            rows.append({'recipe':recipe,'state':final,'result':result,'returncode':child.returncode,'output':str(out)})
            if final!='complete':publish(final,active=recipe,pending=RECIPES[index+1:]);return
            publish('between_runs',pending=RECIPES[index+1:])
        comparisons={}
        for previous,current in [('E1','E2'),('E2','E4')]:
            a=next(r['result'] for r in rows if r['recipe']==previous)
            b=next(r['result'] for r in rows if r['recipe']==current)
            comparisons[current+'_minus_'+previous]={'dice_pp':100*(b['metrics']['dice']-a['metrics']['dice']),
                                                     'paired_hd95_difference':b['metrics']['hd95']-a['metrics']['hd95']}
        dump(root/'COMPARISON.json',{'groups':rows,'comparisons':comparisons,
            'scope':'Exploratory complete-recipe increments; no independent test/causal modality or text-content claim'})
        lines=['# Final comparison','','All values pair metrics from the same fusion-Dice-selected checkpoint.','']
        for name,values in comparisons.items():
            lines.append(f"{name}: Dice {values['dice_pp']:+.4f} pp; HD95 {values['paired_hd95_difference']:+.4f}.")
        lines+=['','E1 jointly changes supervision and an auxiliary decoder; E2 adds the CT-reference term; '
                'E4 adds text-region regularization and its trainable projections. Improvements are not guaranteed. '
                'Text content requires matched controls before attributing medical semantics. No further run is queued.']
        (root/'FINAL_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
        publish('complete',next_experiments=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'))
    p.add_argument('--campaign',required=True);p.add_argument('--gpu',default='0');args=p.parse_args()
    try:main(args)
    except Exception as error:
        import traceback;traceback.print_exc()
        root=Path(local_paths(args.paths)['output_root'])/args.campaign
        if root.exists():dump(root/'STATUS.json',{'state':'needs_analysis','error':repr(error)})
        raise SystemExit(1)
