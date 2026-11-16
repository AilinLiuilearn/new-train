"""Run exactly the authorized V -> T -> S pilots, independent fresh starts."""
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

GROUPS=['V','T','S']


def dump(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(args):
    require_myenv();identity=git_identity()
    if identity['dirty']:raise RuntimeError('Commit code before launch')
    checks=json.loads((REPO/'docs/scgr/GPU_PREFLIGHT.json').read_text())
    if not all(checks[r]['passed'] for r in GROUPS):raise RuntimeError('All GPU preflights must pass')
    for file,expected in checks['source_sha256'].items():
        if sha(REPO/file)!=expected:raise RuntimeError('Preflight source changed: '+file)
    configs={r:json.loads((REPO/'configs'/f'scgr_{r}.json').read_text()) for r in GROUPS}
    allowed={'recipe','text_enabled','score_enabled','campaign_loss','text_description'}
    common=[{k:v for k,v in configs[r].items() if k not in allowed} for r in GROUPS]
    if not common[0]==common[1]==common[2]:raise RuntimeError('Unintended configuration differences')
    root=Path(local_paths(args.paths)['output_root'])/args.campaign;root.mkdir(parents=True,exist_ok=True)
    rows=[]
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PLAN.json').exists():raise RuntimeError('Existing campaign, choose a new output')
        (root/'logs').mkdir(exist_ok=True)
        dump(root/'PLAN.json',{'git':identity,'pid':os.getpid(),'groups':GROUPS,'gpu':args.gpu,
            'fixed':common[0],'preflight':checks,'initialization':'Each group generic encoders only; common task parameters/RNG matched',
            'changes':{'V':'Complete visual refiner, zero text input, no score loss',
                       'T':'V with exact cached text input enabled only',
                       'S':'T with direct native-grid score BCE weight0.1 only'},
            'monitoring':'No continuous LLM polling; user query, exception or five completed epochs',
            'next_after_three':[]})
        def publish(state,**extra):
            dump(root/'STATUS.json',{'state':state,'pid':os.getpid(),'time_epoch':time.time(),'results':rows,**extra})
            lines=['# SCGR V -> T -> S','','|Group|Best epoch|Weights|Dice %|Paired HD95|State|',
                   '|---|---:|---|---:|---:|---|']
            for row in rows:
                r=row['result'];m=r.get('metrics',{})
                lines.append('|'+row['recipe']+'|'+str(r.get('best_epoch',''))+'|'+str(r.get('weights',''))+'|'+
                    (f"{100*m['dice']:.4f}" if m else '')+'|'+(f"{m['hd95']:.4f}" if m else '')+'|'+row['state']+'|')
            lines+=['','Fresh generic encoder pretraining for each group; one fusion validation per epoch; best-only.',
                    'Reused development set. These pilots do not isolate capacity or medical text semantics.']
            (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        publish('queued')
        for index,recipe in enumerate(GROUPS):
            if (root/'STOP').exists():publish('stopped_by_request',pending=GROUPS[index:]);return
            if git_identity()!=identity:raise RuntimeError('Source identity changed during campaign')
            out=root/recipe
            command=[sys.executable,'-u','-m','variants.scgr.train','--recipe',recipe,'--run',args.campaign+'_'+recipe,
                '--paths',str(Path(args.paths).resolve()),'--output',str(out),'--gpu',args.gpu]
            publish('running',active=recipe,pending=GROUPS[index+1:],command=command)
            with (root/'logs'/f'{recipe}.log').open('w') as stream:
                child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
                dump(root/'CHILD.json',{'pid':child.pid,'recipe':recipe})
                while child.poll() is None:
                    if (root/'STOP').exists() and out.exists():(out/'STOP').touch()
                    time.sleep(10)
            result=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
            status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            ok=child.returncode==0 and result.get('complete') and status.get('state')=='complete'
            stopped=(root/'STOP').exists() or status.get('state')=='stopped_by_request'
            final='stopped_by_request' if stopped else 'complete' if ok else 'needs_analysis'
            rows.append({'recipe':recipe,'state':final,'result':result,'returncode':child.returncode,'output':str(out)})
            if final!='complete':publish(final,active=recipe,pending=GROUPS[index+1:]);return
            if (root/'STOP').exists():publish('stopped_by_request',pending=GROUPS[index+1:]);return
            # Four fixed historical examples, after selection, not another
            # full validation or module-on/off comparison. No new selection.
            figure_command=[sys.executable,'-u','scripts/scgr_examples.py','--paths',str(Path(args.paths).resolve()),
                '--checkpoint',str(out/'best.pt'),'--output',str(out/'qualitative'),'--gpu',args.gpu]
            with (root/'logs'/f'{recipe}_examples.log').open('w') as stream:
                figure=subprocess.run(figure_command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
            if figure.returncode!=0:publish('needs_analysis',active=recipe,error='Fixed example export failed');return
            publish('between_runs',pending=GROUPS[index+1:])
        comparisons={};analysis=['# 三组结算','','全部指标来自各组同一Dice最佳权重；不拼不同轮次。','']
        for old,new in [('V','T'),('T','S')]:
            a=next(r['result'] for r in rows if r['recipe']==old);b=next(r['result'] for r in rows if r['recipe']==new)
            delta={'dice_pp':100*(b['metrics']['dice']-a['metrics']['dice']),
                   'hd95':b['metrics']['hd95']-a['metrics']['hd95']}
            ac=a['error_diagnostics']['counts'];bc=b['error_diagnostics']['counts']
            delta['error_pixel_changes']={k:bc[k]-ac[k] for k in ['fn_inner3','fn_deep','fp_outer3','fp_outer3_12','fp_far12','fp_empty_gt','fp_detached']}
            comparisons[new+'_minus_'+old]=delta
            analysis.append(f"{new}相对{old}：Dice {delta['dice_pp']:+.4f}个百分点，HD95 {delta['hd95']:+.4f}；错误像素变化{delta['error_pixel_changes']}。")
        analysis+=['','目标检查：']
        for row in rows:
            r=row['result'];m=r['metrics']
            analysis.append(f"{row['recipe']}：目标Dice>=78.8%且HD95<17：{r['target_788_hd17_met']}；相对当前协议M1 Dice差{r.get('delta_M1_dice_pp')}个百分点、HD95差{r.get('delta_M1_hd95')}。")
        analysis+=['','未改善时保留否定结果；门控描述统计不等于净修复。若V无收益，候选整体未获支持；T不优于V则本轮文本条件无增量；S不优于T则新增评分监督无收益或存在干扰。',
            '即使有改善，仍缺参数匹配普通残差/无评分门/常量向量对照，不能宣布论文创新或医学语义成立。没有自动调参、额外消融或后续训练。']
        dump(root/'COMPARISON.json',{'comparisons':comparisons,'groups':rows,'scope':'Exploratory recipe increments, not independent test'})
        (root/'FINAL_ANALYSIS.md').write_text('\n'.join(analysis)+'\n')
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
