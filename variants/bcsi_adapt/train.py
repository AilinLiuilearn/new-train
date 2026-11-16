"""50% BCSI-Adapt full model; unchanged original BCE+Dice training engine."""
import json
import math
from pathlib import Path
import traceback

import torch

from runtime import parser, local_paths
from variants.tsbci import train as engine
from utils.run_common import module_grad_norm
from .model import Model
from .metrics import SmallTargetMetrics

REFERENCE = None
PREVIOUS_SELECTION = {}


def selection_statistics(indices, width, key):
    selected = torch.zeros(indices.shape[0], width, device=indices.device, dtype=torch.bool)
    selected.scatter_(1, indices, True)
    intersection = selected.float() @ selected.float().T
    union = 2 * indices.shape[1] - intersection
    non_diagonal = ~torch.eye(len(indices), device=indices.device, dtype=torch.bool)
    previous = PREVIOUS_SELECTION.get(key)
    current = selected.detach().cpu()
    overlap_previous = None
    if previous is not None:
        overlap_previous = float((current.float() @ previous.float().T).mean() / indices.shape[1])
    PREVIOUS_SELECTION[key] = current
    return {'unique_selected_sets_in_batch': int(torch.unique(selected, dim=0).shape[0]),
            'union_channel_count': int(selected.any(0).sum()),
            'mean_pairwise_jaccard': float((intersection / union)[non_diagonal].mean())
                                     if non_diagonal.any() else None,
            'mean_overlap_previous_snapshot': overlap_previous,
            'channel_selection_frequency': selected.float().mean(0).tolist(),
            'scope': 'Batch/snapshot variation, not matched-patient temporal stability'}


def diagnostics(model, output, epoch, step, update):
    capture, scales = output['capture'], {}
    for i, info in enumerate(capture['interaction']):
        block = model.front.blocks[i]
        row = {name: engine.stats(capture[name][i]) for name in
               ['ct_aligned', 'pet_aligned', 'ct_interacted', 'pet_interacted', 'fused']}
        row['attention'] = {}
        for modality in ['ct', 'pet']:
            x = capture[modality + '_aligned'][i].detach().float()
            delta = info[modality + '_delta'].float()
            indices = info[modality + '_indices']
            selected = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool).scatter(1, indices, True)
            row[modality + '_relative_delta_rms'] = float(delta.square().mean().sqrt()
                                                         / (x.square().mean().sqrt() + 1e-8))
            row[modality + '_selection'] = selection_statistics(indices, x.shape[1], (i, modality))
            row[modality + '_first_sample_indices'] = indices[0].tolist()
            row[modality + '_score'] = engine.stats(info[modality + '_scores'])
            att = info[modality + '_attention']
            row['attention'][modality] = {k: float(v) for k, v in att.items() if k != 'attention'}
            matrix = att['attention'][0]
            row['attention'][modality]['first_sample_submatrix_16x16'] = matrix[:16, :16].cpu().tolist()
            row['attention'][modality]['first_sample_top1_source_slots'] = matrix.argmax(-1).cpu().tolist()
            row['attention'][modality]['first_sample_top1_weights'] = matrix.amax(-1).cpu().tolist()
            score = info[modality + '_scores']
            if update['success'] and score.grad is not None:
                grad = score.grad.detach().float() / update['backward_scale']
                row[modality + '_router_score_gradient'] = {
                    'selected_abs_mean': float(grad[selected].abs().mean()),
                    'unselected_abs_mean': float(grad[~selected].abs().mean()),
                    'finite': bool(torch.isfinite(grad).all())}
        row['gradient_groups'] = {name: module_grad_norm(module)
                                  for name, module in block.named_children()} if update['success'] else {}
        row['same_index_interpretation'] = 'No CT/PET channel-ID semantic correspondence assumed'
        scales[str(i+1)] = row
    engine.log('diagnostics.jsonl', {'epoch': epoch, 'step': step, 'AMP_gradient': update,
                                    'scales': scales, 'scope': 'Training snapshots only, not task efficacy'})


def results(rows, complete=False):
    if not rows:
        return
    best = max(rows, key=lambda r: r['validation']['full']['dice'])
    metrics = best['validation']['full']
    result = {'complete': complete, 'best_epoch': best['epoch'], 'weights': best['weights'],
              'metrics': metrics, 'selection': 'Maximum validation Dice; other metrics from the same epoch',
              'target_788_hd17_met': metrics['dice'] >= .788 and metrics['hd95'] < 17,
              'target_789_hd17_met': metrics['dice'] >= .789 and metrics['hd95'] < 17,
              'scope': '50% BCSI-inspired paired inference adaptation, full model only, reused dev, seed2023',
              'mechanism_claim': 'A single full candidate does not isolate router/attention contributions',
              'next_experiments': []}
    result['small_target_diagnostics'] = best['validation'].get('auxiliary_diagnostics')
    if REFERENCE is not None and REFERENCE.is_file():
        reference = json.loads(REFERENCE.read_text())
        result['reference_M1'] = {'path': str(REFERENCE), 'result': reference}
        result['delta_M1_dice_pp'] = 100 * (metrics['dice'] - reference['metrics']['dice'])
        result['delta_M1_hd95'] = metrics['hd95'] - reference['metrics']['hd95']
    engine.dump('RESULTS.json', result)
    lines = ['# BCSI-Adapt K50 + M1 + GN', '', '|Epoch|Weights|Dice %|Paired HD95|Sensitivity %|Precision %|',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        m = row['validation']['full']
        lines.append(f"|{row['epoch']}|{row['weights']}|{100*m['dice']:.4f}|{m['hd95']:.4f}|"
                     f"{100*m['sensitivity']:.4f}|{100*m['precision']:.4f}|")
    lines += ['', 'One complete-model run only. No text, localization head, auxiliary loss, or automatic follow-up.',
              'Direct residual initialization changes the initial prediction; no identity-at-initialization claim.']
    (engine.OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    args = parser('bcsi_adapt').parse_args()
    try:
        cfg = json.loads(Path(args.config).read_text())
        if cfg.get('comparison_m1_run'):
            REFERENCE = Path(local_paths(args.paths)['output_root']) / cfg['comparison_m1_run'] / 'RESULTS.json'
        engine.main(args, variant='bcsi_adapt', model_factory=Model, results_fn=results,
                    diagnostics_fn=diagnostics,
                    extra_metrics_factory=SmallTargetMetrics,
                    protocol_extra={'loss': 'Original BCE+Dice only, no auxiliary losses',
                                    'routing': '50% hard Top-K forward, selected-token ST softmax surrogate backward',
                                    'residual': 'Direct scatter addition, no alpha/output gate or zero-initialization exit',
                                    'scope': 'One complete paired CT/PET BCSI-inspired candidate; not full BCSI replication'})
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)
