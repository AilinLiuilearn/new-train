import json
from pathlib import Path
import traceback
from runtime import REPO,parser,local_paths
from variants.tsbci import train as engine
from utils.run_common import module_grad_norm
from .model import Model
from .losses import objective
from .metrics import ErrorMetrics

RECIPE=None
REFERENCE=None


def diagnostics(model,output,epoch,step,update):
    cap=output['capture'];scales={}
    for s,(before,after) in enumerate(zip(cap['fused'],cap['refined']),1):
        info=cap['refiner'][str(s)]
        row={k:engine.stats(v) for k,v in info.items()}
        row['input']=engine.stats(before);row['output']=engine.stats(after)
        row['relative_residual_rms']=float(info['delta'].detach().float().square().mean().sqrt()/(before.detach().float().square().mean().sqrt()+1e-8))
        if update['success']:
            block=model.refiner.blocks[s-1]
            row['AMP_safe_gradients']={k:module_grad_norm(v) for k,v in block.named_children()}
        scales[str(s)]=row
    engine.log('diagnostics.jsonl',{'epoch':epoch,'step':step,'AMP_gradient':update,'scales':scales,
        'text_enabled':model.cfg['text_enabled'],'score_supervision':model.cfg['score_enabled'],
        'scope':'Gate/feature descriptors, not evidence of semantic alignment or task benefit. No on/off validation pass.'})


def results(rows,complete=False):
    if not rows:return
    b=max(rows,key=lambda r:r['validation']['full']['dice']);m=b['validation']['full']
    result={'complete':complete,'recipe':RECIPE,'best_epoch':b['epoch'],'weights':b['weights'],
        'metrics':m,'error_diagnostics':b['validation'].get('auxiliary_diagnostics'),
        'historical_diagnostic_reference':{'dice':.7836444556934279,'hd95':17.48659619401712,
            'scope':'Historical patience5 reference, not the matched current-protocol main comparison'},
        'target_788_hd17_met':m['dice']>=.788 and m['hd95']<17,
        'scope':'Exploratory single-seed reused development set; three pilot recipes do not isolate capacity or medical text content',
        'selection':'Best full fusion Dice; HD95 and diagnostics from that same epoch','next_experiments':[]}
    if REFERENCE and REFERENCE.exists():
        ref=json.loads(REFERENCE.read_text());result['matched_protocol_M1']={'path':str(REFERENCE),'metrics':ref['metrics']}
        result['delta_M1_dice_pp']=100*(m['dice']-ref['metrics']['dice'])
        result['delta_M1_hd95']=m['hd95']-ref['metrics']['hd95']
    engine.dump('RESULTS.json',result)
    lines=['# SCGR '+str(RECIPE),'','|Epoch|Weights|Fusion Dice %|Paired HD95|Sensitivity %|Precision %|',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation']['full'];lines.append(f"|{r['epoch']}|{r['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|{100*v['sensitivity']:.4f}|{100*v['precision']:.4f}|")
    lines+=['','Fresh generic encoder initialization for every group; no task checkpoints or previous-group weights.',
            'V: no text/no score loss. T: text input only. S: T plus direct native-grid score BCE weight0.1.',
            'V/T scores learn via segmentation, not direct GT; shared head is not automatic calibration.',
            'Single full fusion validation per epoch. No independent test or medical semantic efficacy claim.']
    (engine.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=parser('scgr_V');p.add_argument('--recipe',choices=['V','T','S'],required=True);p.set_defaults(config=None)
    args=p.parse_args();RECIPE=args.recipe
    args.config=args.config or str(REPO/'configs'/f'scgr_{RECIPE}.json')
    try:
        cfg=json.loads(Path(args.config).read_text())
        if cfg['recipe']!=RECIPE:raise ValueError('Recipe/config mismatch')
        REFERENCE=Path(local_paths(args.paths)['output_root'])/cfg['comparison_m1_run']/'RESULTS.json'
        engine.main(args,variant='scgr_'+RECIPE,model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=objective,extra_metrics_factory=ErrorMetrics,
            protocol_extra={'loss':'Original BCE+Dice. Only S adds mean of eight native-grid support BCE terms weight0.1.',
                'recipe':RECIPE,'support':'Shared within-scale foreground head; native area-GT targets only in S; no correctness pseudo-labels.',
                'fixed_text':'Pulmonary tumor lesion.',
                'fixed_text_cache_sha256':engine.sha(REPO/'resources/scgr_text.json'),
                'inference':'Paired encoders -> align -> unchanged M1 -> four-scale refiner -> unchanged GN decoder; no GT/text tower.',
                'validation':'One full fusion traversal per epoch, spatial error counts on the same predictions',
                'scope':'Three explicitly authorized pilots; no automatic search or paper-level novelty claim'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
