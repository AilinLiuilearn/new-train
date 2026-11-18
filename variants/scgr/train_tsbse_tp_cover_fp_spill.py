"""Run TP Cover plus FP-only Spill with the 0.005 delayed ramp."""
import json
import traceback

from runtime import parser
from variants.tsbci import train as engine

from . import train_tsbse as base
from .model import Model
from .train import diagnostics
from .train_tsbse_ramp import epoch_start
from .train_tsbse_tp_cover import PARENT, TPReferenceMetrics

VARIANT = 'scgr_T_tsbse_tp_cover_fp_spill'
COVER_ONLY = 'scgr_T_tsbse_tp_cover_20261118'
CHANGE = 'Select Spill Top-K among GT background pixels with logits > 0 and average selected risks.'


def results(rows, complete=False):
    base.results(rows, complete)
    if not rows:
        return
    result = json.loads((engine.OUT / 'RESULTS.json').read_text())
    best = next(row for row in rows if row['epoch'] == result['best_epoch'])
    result.update(
        ramp_reference=PARENT,
        delta_ramp_dice_pp=100 * (result['metrics']['dice'] - PARENT['dice']),
        delta_ramp_hd95=result['metrics']['hd95'] - PARENT['hd95'],
        parent_run=COVER_ONLY,
        single_change=CHANGE,
        best_epoch_loss_terms=best['validation']['losses'],
    )
    engine.dump('RESULTS.json', result)


if __name__ == '__main__':
    args = parser(VARIANT).parse_args()
    try:
        engine.main(
            args,
            variant=VARIANT,
            model_factory=Model,
            results_fn=results,
            diagnostics_fn=diagnostics,
            model_objective_fn=base.objective,
            extra_metrics_factory=TPReferenceMetrics,
            epoch_start_fn=epoch_start,
            protocol_extra={
                'parent_run': COVER_ONLY,
                'single_change': CHANGE,
                'loss': 'BCE + Dice and TSBSE with TP Cover and FP-only Spill. Peak 0.005 after a 3-epoch delay and 3-epoch ramp.',
                'scope': 'One authorized combination experiment on the reused development set.',
            },
        )
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
