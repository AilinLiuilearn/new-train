"""Matched-harness originalT control, not a candidate third-loss model."""
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
    r.update(control_run=True,eligible_for_goal=False,third_loss_enabled=False,
        purpose='Current-harness originalT reference after matched replay demonstrated ordinary backward nondeterminism.',
        single_change='Auxiliary coefficient0, original BCE+Dice only; TSBS raw risks remain diagnostics.',
        goal_acceptance=False,
        scope='Diagnostic control, not a third-loss target candidate; a high control score cannot complete the goal.')
    engine.dump('RESULTS.json',r)
    path=engine.OUT/'RESULTS.md';text=path.read_text().replace('# OriginalT + revised TSBSE, see resolved config for audited weight', '# OriginalT matched-harness zero-auxiliary control')
    path.write_text(text+'\nDiagnostic control only; no third-loss efficacy or goal acceptance.\n')


if __name__=='__main__':
    args=parser('scgr_T_tsbse_control').parse_args()
    try:
        engine.main(args,variant='scgr_T_tsbse_control',model_factory=Model,results_fn=results,
            diagnostics_fn=diagnostics,model_objective_fn=base.objective,
            extra_metrics_factory=base.TSBSEErrorMetrics,epoch_start_fn=epoch_start,
            protocol_extra={'control_run':True,'eligible_for_goal':False,
                'loss':'Original BCE+Dice only; auxiliary coefficient exactly0 for all epochs.',
                'purpose':'Matched current-harness originalT reference, not a new loss design or target substitute.',
                'initialization':'Generic encoder pretraining only; fresh tasks and all visual parameters trainable.',
                'runtime':'Original AMP/attention/pooling/padding backends retained; no deterministic-kernel replacements.',
                'evaluation':'One complete original fusion traversal each epoch, unchanged selection/threshold/cohort.'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:engine.status('failed',error=repr(error))
        raise SystemExit(1)
