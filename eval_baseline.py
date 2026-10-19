"""Re-evaluate a saved baseline with its recorded raw/EMA selection."""
import argparse
import json
from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_dual_model
from tasks.mdt_seg import MDTSegTeacher
from run_baseline import loaders, evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root')
    args = p.parse_args()
    ckpt = MDTSegTeacher.load_state_dicts(args.checkpoint)
    saved = dict(ckpt['config'])
    saved.update(pretrained=False, ema_enabled=False, mixed_precision=False)
    if args.root:
        saved['root'] = args.root
    cfg = SegMDTConfig(args=saved)
    task = MDTSegTeacher(build_dual_model(cfg), cfg)
    tag = ckpt['eval_weights']
    task.model.load_state_dict(ckpt['model_ema'] if tag == 'ema' else ckpt['model'], strict=True)
    _, _, test = loaders(cfg)
    print(json.dumps({'weights': tag, 'metrics': evaluate(task, test, task.model)}, indent=2))
