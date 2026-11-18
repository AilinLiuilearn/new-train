"""Test pooled foreground reduction in Cover, relative to TP Cover plus FP-only Spill."""
import json
import traceback

from runtime import parser
from variants.tsbci import train as engine

from . import train_tsbse as base
from .model import Model
from .train import diagnostics
from .train_tsbse_ramp import epoch_start
from .train_tsbse_tp_cover import PARENT, TPReferenceMetrics

VARIANT = 'scgr_T_tsbse_pooled_cover'
PARENT_COMBINATION = {
    'run': 'scgr_T_tsbse_tp_cover_fp_spill_20261118',
    'commit': '8423440',
    'best_epoch': 6,
    'dice': 0.7866467726643975,
    'hd95': 15.994808941608042,
}
CHANGE = 'Pool weighted Cover risk across foreground pixels, preserving the original batch Cover logit-gradient L1.'


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
        parent_combination=PARENT_COMBINATION,
        delta_combination_dice_pp=100 * (result['metrics']['dice'] - PARENT_COMBINATION['dice']),
        delta_combination_hd95=result['metrics']['hd95'] - PARENT_COMBINATION['hd95'],
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
                'parent': PARENT_COMBINATION,
                'single_change': CHANGE,
                'loss': 'BCE + Dice, TP Cover with pooled foreground reduction, and the original FP-only Spill. Same 0.005 delayed ramp.',
                'scope': 'One authorized Cover allocation experiment on the reused development set.',
            },
        )
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
