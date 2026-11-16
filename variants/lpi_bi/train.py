"""One complete LPI-BI experiment, using the original engine and evaluation."""
import json
from pathlib import Path
import traceback

import torch

from runtime import REPO, parser, local_paths
from variants.tsbci import train as engine
from utils.run_common import module_grad_norm
from .losses import localization_loss
from .model import Model

ROLE_WEIGHT = 0.1
REFERENCE = None


def objective(output, target, criterion):
    seg, parts = criterion(output['logits'].float(), target)
    loc, details = localization_loss(output['localization_logits'], target)
    total = seg + ROLE_WEIGHT * loc
    return total, {'total': float(total.detach()), 'seg_total': float(seg.detach()),
                   'bce': float(parts['loss_bce']), 'dice_loss': float(parts['loss_dice']),
                   **{key: float(value) for key, value in details.items()}}


def diagnostics(model, output, epoch, step, update):
    capture = output['capture']
    scales = {}
    for i, info in enumerate(capture['interaction']):
        block = model.front.blocks[i]
        row = {name: engine.stats(capture[name][i]) for name in
               ['ct_aligned', 'pet_aligned', 'ct_interacted', 'pet_interacted', 'fused']}
        row['localization_and_channel'] = {name: engine.stats(value) for name, value in info.items()
                                           if not name.startswith('to_')}
        row['attention'] = {name: {key: float(value) for key, value in info[name].items()}
                            for name in ['to_ct', 'to_pet']}
        for name in ['ct', 'pet']:
            x = capture[name + '_aligned'][i].detach().float()
            delta = info[name + '_delta'].float()
            row[name + '_relative_delta_rms'] = float(
                delta.square().mean().sqrt() / (x.square().mean().sqrt() + 1e-8))
        if update['success']:
            row['unscaled_clipped_gradient_norms'] = {
                'ct_head': module_grad_norm(block.ct_condition.head),
                'pet_head': module_grad_norm(block.pet_condition.head),
                'ct_mlp': module_grad_norm(block.ct_condition.mlp),
                'pet_mlp': module_grad_norm(block.pet_condition.mlp),
                **{direction + '_' + part: module_grad_norm(getattr(getattr(block, direction), part))
                   for direction in ['to_ct', 'to_pet'] for part in ['q', 'k', 'v', 'out']},
                **{direction + '_beta': float(getattr(block, direction).bias_logit.grad.abs())
                   if getattr(block, direction).bias_logit.grad is not None else 0.
                   for direction in ['to_ct', 'to_pet']},
            }
        scales[str(i + 1)] = row
    engine.log('diagnostics.jsonl', {'epoch': epoch, 'step': step, 'AMP_gradient': update,
                                    'scales': scales,
                                    'scope': 'Training batch snapshots, not validation contribution'})


def results(rows, complete=False):
    if not rows:
        return
    best = max(rows, key=lambda row: row['validation']['full']['dice'])
    metrics = best['validation']['full']
    baseline_path = REFERENCE
    # The comparison is optional on other machines; never substitute a different historical result.
    reference = json.loads(baseline_path.read_text()) if baseline_path and baseline_path.is_file() else None
    result = {'complete': complete, 'best_epoch': best['epoch'], 'weights': best['weights'],
              'metrics': metrics, 'selection': 'Maximum validation Dice; other metrics from that epoch',
              'target_788_hd17_met': metrics['dice'] >= .788 and metrics['hd95'] < 17,
              'target_789_hd17_met': metrics['dice'] >= .789 and metrics['hd95'] < 17,
              'scope': 'Full visual candidate only, reused dev, seed2023; no isolated mechanism/novelty claim'}
    if reference:
        result['reference_M1'] = {'path': str(baseline_path), 'result': reference}
        result['delta_M1_dice_pp'] = 100 * (metrics['dice'] - reference['metrics']['dice'])
        result['delta_M1_hd95'] = metrics['hd95'] - reference['metrics']['hd95']
    if complete:
        met = result['target_788_hd17_met']
        result['assessment'] = ('Exploratory metric target reached; mechanism still requires matched controls'
                                if met else 'This full-model run did not meet the requested joint metric target')
        result['next_experiments'] = []
    engine.dump('RESULTS.json', result)
    lines = ['# LPI-BI V1 + original M1 + GN (no text)', '',
             '|Epoch|Weights|Dice %|Paired HD95|Sensitivity %|Precision %|',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        m = row['validation']['full']
        lines.append(f"|{row['epoch']}|{row['weights']}|{100*m['dice']:.4f}|{m['hd95']:.4f}|"
                     f"{100*m['sensitivity']:.4f}|{100*m['precision']:.4f}|")
    if reference:
        m = reference['metrics']
        lines += ['', f"Same-protocol M1 reference: Dice {100*m['dice']:.4f}%, HD95 {m['hd95']:.4f}."]
    lines += ['', result.get('assessment', 'In progress; no final performance claim.'),
              'Single-seed reused development set. One full validation per epoch; no automatic next run.']
    (engine.OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    args = parser('lpi_bi').parse_args()
    try:
        cfg = json.loads(Path(args.config).read_text())
        if cfg['localization_weight'] != ROLE_WEIGHT:
            raise ValueError('V1 fixed localization weight is 0.1')
        if cfg.get('comparison_m1_run'):
            REFERENCE = Path(local_paths(args.paths)['output_root']) / cfg['comparison_m1_run'] / 'RESULTS.json'
        engine.main(args, variant='lpi_bi', model_factory=Model, objective_fn=objective,
                    results_fn=results, diagnostics_fn=diagnostics,
                    protocol_extra={
                        'loss': 'Original BCE+Dice + 0.1*mean(eight occupancy BCE+squared-Dice heads)',
                        'auxiliary_empty_policy': 'No auxiliary Dice on empty GT; BCE retained; fixed eight-head denominator',
                        'gradient_contract': 'No detach; all visual parameters trainable, original task modules fresh',
                        'scope': 'One full visual LPI-BI candidate; no text, no automatic ablations or next run',
                    })
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
