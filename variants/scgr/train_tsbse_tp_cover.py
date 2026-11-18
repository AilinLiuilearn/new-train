"""Test the TP Cover reference against the 0.005 delayed-ramp parent."""
import json
import traceback

import torch

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
CHANGE = 'Cover measures distance to GT intersection with logits > 0, with image-diagonal fallback.'


class TPReferenceMetrics(base.TSBSEErrorMetrics):
    @torch.no_grad()
    def update(self, logits, target):
        super().update(logits, target)
        pred = logits.float().sigmoid() > .5
        gt = target > .5
        tp_per_slice = (gt & pred).flatten(1).sum(1)
        fn_per_slice = (gt & ~pred).flatten(1).sum(1)
        fn_with_tp = int(fn_per_slice[tp_per_slice > 0].sum())
        self.counts['fn_with_overlap'] += fn_with_tp
        self.counts['fn_without_overlap'] += int(fn_per_slice.sum()) - fn_with_tp


def results(rows, complete=False):
    base.results(rows, complete)
    if not rows:
        return
    result = json.loads((engine.OUT / 'RESULTS.json').read_text())
    best = next(row for row in rows if row['epoch'] == result['best_epoch'])
    result.update(
        parent_ramp=PARENT,
        delta_parent_dice_pp=100 * (result['metrics']['dice'] - PARENT['dice']),
        delta_parent_hd95=result['metrics']['hd95'] - PARENT['hd95'],
        single_change=CHANGE,
        best_epoch_loss_terms=best['validation']['losses'],
    )
    engine.dump('RESULTS.json', result)


if __name__ == '__main__':
    args = parser('scgr_T_tsbse_tp_cover').parse_args()
    try:
        engine.main(
            args,
            variant='scgr_T_tsbse_tp_cover',
            model_factory=Model,
            results_fn=results,
            diagnostics_fn=diagnostics,
            model_objective_fn=base.objective,
            extra_metrics_factory=TPReferenceMetrics,
            epoch_start_fn=epoch_start,
            protocol_extra={
                'parent': PARENT,
                'single_change': CHANGE,
                'loss': 'BCE + Dice and TSBSE. Peak 0.005, 3-epoch delay, then 3-epoch ramp. Original global scale and spill risk.',
                'scope': 'One authorized experiment on the reused development set.',
            },
        )
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
