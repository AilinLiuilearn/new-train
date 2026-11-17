"""Single change from TSBSE: delay then ramp its coefficient; all parameters train."""
import json
import traceback
from runtime import parser
from variants.tsbci import train as engine
from .model import Model
from .train import diagnostics
from . import train_tsbse as parent


def coefficient(epoch,peak,delay,ramp):
    if epoch<1 or delay<0 or ramp<1:raise ValueError('Invalid epoch/schedule')
    return peak*min(1.,max(0.,(epoch-delay)/ramp))


def epoch_start(model,ema_model,epoch,cfg):
    weight=coefficient(epoch,cfg.tsbse_weight,cfg.tsbse_delay_epochs,cfg.tsbse_ramp_epochs)
    for m in (model,ema_model):
        m._tsbse_effective_weight=weight
        m._tsbse_epoch=epoch
    engine.log('loss_schedule.jsonl',{'epoch':epoch,'effective_weight':weight,'peak':cfg.tsbse_weight,
        'delay_epochs':cfg.tsbse_delay_epochs,'ramp_epochs':cfg.tsbse_ramp_epochs})


def results(rows,complete=False):
    parent.results(rows,complete)
    if not rows:return
    path=engine.OUT/'RESULTS.json';r=json.loads(path.read_text())
    reference={'dice':.7848713956352645,'hd95':16.601953308514492,'best_epoch':6}
    r.update(parent_TSBSE=reference,
        delta_parent_dice_pp=100*(r['metrics']['dice']-reference['dice']),
        delta_parent_hd95=r['metrics']['hd95']-reference['hd95'],
        single_change='Loss timing:0 first3 epochs,linear ramp epochs4-6 to unchanged peak0.005')
    engine.dump('RESULTS.json',r)


if __name__=='__main__':
    args=parser('scgr_T_tsbse_ramp').parse_args()
    try:
        engine.main(args,variant='scgr_T_tsbse_ramp',model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=parent.objective,
            extra_metrics_factory=parent.TSBSEErrorMetrics,epoch_start_fn=epoch_start,
            protocol_extra={'single_factor_change':'Only coefficient timing versus 3f7d413 TSBSE:0 epochs1-3,0.001667 epoch4,0.003333 epoch5,0.005 epoch6 onward.',
                'loss':'Same detached EDT + bounded weights + logit cover + target-budgeted background tail risk.',
                'initialization':'Generic encoder pretraining only; fresh M1/refiner/GN. All visual parameters trainable from first batch.',
                'scope':'One authorized rerun, no automatic successors. Numerical metric improvement not proof of loss novelty or causal attribution.',
                'evaluation':'One complete fusion validation each epoch with same metrics/extra error diagnostics.'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
