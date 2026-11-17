"""Single coverage-scale repair relative completed ramp peak0.005 parent."""
import json
import traceback
from runtime import parser
from variants.tsbci import train as engine
from .model import Model
from .train import diagnostics
from . import train_tsbse as base
from .train_tsbse_ramp import epoch_start


def results(rows,complete=False):
    base.results(rows,complete)
    if not rows:return
    r=json.loads((engine.OUT/'RESULTS.json').read_text())
    reference={'run':'scgr_T_tsbse_ramp_20261117','dice':.786314135438005,
        'hd95':16.214441468768772,'best_epoch':11,'weights':'ema'}
    r.update(parent_ramp=reference,
        delta_parent_dice_pp=100*(r['metrics']['dice']-reference['dice']),
        delta_parent_hd95=r['metrics']['hd95']-reference['hd95'],
        single_change='Cover distance scale only:GT interior EDT instead of global radius; unchanged spill,peak0.005 anddelay3/ramp3.',
        target={'dice_min':.795,'hd95_max_exclusive':16.169107096291484},
        numerical_target_met=r['metrics']['dice']>=.795 and r['metrics']['hd95']<16.169107096291484,
        checkpoint_revalidation_required=True)
    engine.dump('RESULTS.json',r)


if __name__=='__main__':
    args=parser('scgr_T_tsbse_local_cover').parse_args()
    try:
        engine.main(args,variant='scgr_T_tsbse_local_cover',model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=base.objective,
            extra_metrics_factory=base.TSBSEErrorMetrics,epoch_start_fn=epoch_start,
            protocol_extra={'parent_run':'scgr_T_tsbse_ramp_20261117',
                'single_change':'Cover scale only. Parent is preserved ramp0.005, not the failed half-strength run.',
                'hypothesis':'Global radius gives low near-boundary FN weights in largerGT; local GT interior distance tests a geometry repair without new coefficients.',
                'loss':'Same logit risks,detached exactEDT,cover GT mean,spill original global scale/topK,common peak0.005,delay3/ramp3.',
                'initialization':'Generic encoder pretraining only; fresh tasks; all parameters train from first batch.',
                'scope':'No guaranteed efficacy or novelty; fixed64 diagnostic not evaluation subset. No new loss,head,numeric hyperparameter or task warm start.',
                'evaluation':'One complete fusion validation each epoch; same Dice-best paired metrics and error partitions.'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
