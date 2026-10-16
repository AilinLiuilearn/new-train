"""Frozen-matrix evaluation and off/identity/shuffle diagnostic controls."""
import argparse
import os
import json
from configs.base import str2bool
from run_ct_full_matrix_add import CTFullMatrixConfig, _loaders
from models.ct_full_matrix_add_model import build_matrix_model
from tasks.ct_full_matrix_seg import CTFullMatrixSegTask
from utils.run_common import seed_everything, eval_missing_rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--use_ema', type=str2bool, default=None)
    p.add_argument('--read_mode', choices=('normal','off','identity','shuffle'), default='normal')
    p.add_argument('--missing_rates', type=float, nargs='+', default=[0.,1.])
    p.add_argument('--compare_on_off', type=str2bool, default=False)
    args = p.parse_args()
    ckpt = CTFullMatrixSegTask.load_state_dicts(args.checkpoint)
    if ckpt.get('experiment_type') != 'ct_full_kv_matrix_add':
        raise ValueError('requires a CT-key/matrix-value checkpoint')
    if any(not 0 <= r <= 1 for r in args.missing_rates): raise ValueError('invalid missing rate')
    if not {0., 1.}.issubset(args.missing_rates):
        raise ValueError('include missing rates 0 and 1 for Full/Missing comparison')
    saved = dict(ckpt['config']); saved.pop('checkpoint_dir',None)
    saved.update(pretrained=False, ct_pretrained_path=None, pet_pretrained_path=None,
                 ema_enabled=False, aug_mode='none')
    if args.root: saved['root'] = args.root
    cfg = CTFullMatrixConfig(args=saved); seed_everything(cfg.random_state)
    task = CTFullMatrixSegTask(build_matrix_model(cfg),cfg)
    ema = ckpt['eval_weights']=='ema' if args.use_ema is None else args.use_ema
    state = ckpt.get('model_ema') if ema else ckpt['model']
    if state is None: raise ValueError('EMA state not available')
    task.model.load_state_dict(state, strict=True)
    if args.compare_on_off and not task.model.matrix_enabled:
        raise ValueError('checkpoint has matrix_enabled=False; cannot diagnose active compensation')
    task.model.matrix_read_mode = args.read_mode
    task.model.eval()
    _, _, loader = _loaders(cfg)
    out = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                         'matrix_eval_'+args.read_mode)
    os.makedirs(out, exist_ok=True)
    modes = ('normal', 'off') if args.compare_on_off else (args.read_mode,)
    comparison = {}
    for mode in modes:
        task.model.matrix_read_mode = mode
        seed_everything(cfg.random_state)
        directory = os.path.join(out, mode) if args.compare_on_off else out
        os.makedirs(directory, exist_ok=True)
        results, joint = eval_missing_rates(task, task.model, loader, cfg.random_state,
            directory, 'final_test', rates=args.missing_rates,
            weights_tag='ema' if ema else 'raw')
        comparison[mode] = dict(joint_dice=joint, results=results)
    if args.compare_on_off:
        from utils.matrix_region_diagnostics import region_correction_report
        comparison['region_statistics'] = region_correction_report(
            task.model, loader, task.device, os.path.join(out, 'missing_region_correction.csv'))
        comparison['on_minus_off'] = [dict(missing_rate=a['missing_rate'],
            dice=a['dice']-b['dice'], hd95=a['hd95']-b['hd95'])
            for a,b in zip(comparison['normal']['results'], comparison['off']['results'])]
        with open(os.path.join(out, 'comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        print('[on minus off]', comparison['on_minus_off'], flush=True)


if __name__ == '__main__': main()
