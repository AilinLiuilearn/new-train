"""Compare a 0.01 TSBSE peak against the original 0.005 ramp."""
import json
import traceback

from runtime import parser
from variants.tsbci import train as engine

from . import train_tsbse as base
from .model import Model
from .train import diagnostics
from .train_tsbse_ramp import epoch_start


PARENT = {
    'run': 'scgr_T_tsbse_ramp_20261117',
    'commit': '345c4bc',
    'best_epoch': 11,
    'weights': 'ema',
    'dice': 0.786314135438005,
    'hd95': 16.214441468768772,
}
TARGET = {'dice_min': 0.795, 'hd95_max_exclusive': 16.169107096291484}
CHANGE = 'Common TSBSE peak 0.005 to 0.01, with the same 3-epoch delay and ramp.'


def results(rows, complete=False):
    base.results(rows, complete)
    if not rows:
        return
    result = json.loads((engine.OUT / 'RESULTS.json').read_text())
    metrics = result['metrics']
    result.update(
        parent_ramp=PARENT,
        delta_parent_dice_pp=100 * (metrics['dice'] - PARENT['dice']),
        delta_parent_hd95=metrics['hd95'] - PARENT['hd95'],
        single_change=CHANGE,
        target=TARGET,
        numerical_target_met=(
            metrics['dice'] >= TARGET['dice_min']
            and metrics['hd95'] < TARGET['hd95_max_exclusive']
        ),
    )
    engine.dump('RESULTS.json', result)


if __name__ == '__main__':
    args = parser('scgr_T_tsbse_ramp_double').parse_args()
    try:
        engine.main(
            args,
            variant='scgr_T_tsbse_ramp_double',
            model_factory=Model,
            results_fn=results,
            diagnostics_fn=diagnostics,
            model_objective_fn=base.objective,
            extra_metrics_factory=base.TSBSEErrorMetrics,
            epoch_start_fn=epoch_start,
            protocol_extra={
                'parent_run': PARENT['run'],
                'single_change': CHANGE,
                'loss': 'BCE + Dice and TSBSE with global cover scale and the original spill risk.',
                'scope': 'One user-authorized peak coefficient trial on the reused development set.',
            },
        )
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
