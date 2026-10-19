"""Evaluate the saved heterogeneous baseline, honoring selected raw/EMA weights."""

import argparse
import os

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_dual_model
from run_full_missing_baseline import _loaders, assert_protocol
from tasks.mdt_seg import MDTSegTeacher
from utils.run_common import eval_missing_rates, seed_everything


def select_checkpoint_weights(checkpoint, requested="auto"):
    tag = checkpoint.get("eval_weights") if requested == "auto" else requested
    if tag not in ("raw", "ema"):
        raise ValueError("checkpoint must record eval_weights=raw/ema, or specify --weights")
    key = "model_ema" if tag == "ema" else "model"
    if checkpoint.get(key) is None:
        raise ValueError(f"checkpoint does not contain {tag} weights")
    return checkpoint[key], tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default=None, help="Override the saved dataset root")
    parser.add_argument("--weights", choices=("auto", "raw", "ema"), default="auto")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    checkpoint = MDTSegTeacher.load_state_dicts(args.checkpoint)
    saved = dict(checkpoint["config"])
    if saved.get("experiment_type") != "clean_full_missing_hetero_lr4e5":
        raise ValueError("expected this branch's heterogeneous baseline checkpoint")
    saved.pop("checkpoint_dir", None)
    if args.root is not None:
        saved["root"] = args.root
    # Complete checkpoints carry all network weights; no pretrained files or EMA copy needed.
    saved.update(pretrained=False, ema_enabled=False, mixed_precision=False)
    cfg = SegMDTConfig(args=saved)
    assert_protocol(cfg)
    state, tag = select_checkpoint_weights(checkpoint, args.weights)
    seed_everything(cfg.random_state)
    task = MDTSegTeacher(build_dual_model(cfg), cfg)
    task.model.load_state_dict(state, strict=True)
    _, _, test_loader = _loaders(cfg)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[eval] epoch={checkpoint['epoch']} weights={tag} hd95={cfg.hd95_mode}", flush=True)
    eval_missing_rates(
        task, task.model, test_loader, cfg.random_state, output_dir, "reeval_test",
        rates=list(cfg.final_test_missing_rates), weights_tag=tag,
    )


if __name__ == "__main__":
    main()
