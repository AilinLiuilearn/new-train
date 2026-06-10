# -*- coding: utf-8 -*-
"""Evaluate a saved MDT segmentation checkpoint on the PCLT20K test split."""

import argparse
import os
import sys

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from run_mdt_seg import _build_loaders, _prepare_env
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.vis_teacher import save_segmentation_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ckpt.best_dice.pth.tar on the test set.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/mit-b1_sum_attn_noema_baseline/MDT/2026-06-09_10-52-26",
        help="Experiment directory containing config_args.json and checkpoint files.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to <exp_dir>/ckpt.best_dice.pth.tar.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers for evaluation.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index inside CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--threshold", type=float, default=None, help="Override eval threshold.")
    parser.add_argument("--print_every", type=int, default=20, help="Print progress every N test batches.")
    parser.add_argument("--save_vis", action="store_true", help="Save qualitative diagnostics with ADC-MAC visual panels.")
    parser.add_argument("--vis_samples", type=int, default=8, help="Number of test samples to visualize.")
    parser.add_argument("--vis_dir", type=str, default=None, help="Visualization output directory. Defaults to <exp_dir>/vis_test_best_dice_adc_mac.")
    return parser.parse_args()


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sys.path.insert(0, os.getcwd())
    sys.modules.pop("datasets", None)

    args = parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    ckpt_path = os.path.abspath(args.ckpt) if args.ckpt else os.path.join(exp_dir, "ckpt.best_dice.pth.tar")
    config_path = os.path.join(exp_dir, "config_args.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config_args.json not found: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    config = SegMDTConfig.from_json(config_path)
    config.gpus = [int(args.gpu)]
    config.batch_size = int(args.batch_size)
    config.num_workers = int(args.num_workers)
    config.pin_memory = True
    config.mixed_precision = False
    config.vis_every_epoch = False
    config.task = "MDT_Teacher"
    if args.threshold is not None:
        config.eval_threshold = float(args.threshold)

    _prepare_env(config)

    print(f"[TEST] exp_dir: {exp_dir}", flush=True)
    print(f"[TEST] checkpoint: {ckpt_path}", flush=True)
    print(f"[TEST] root: {config.root}", flush=True)
    print(f"[TEST] split mode cipa_aligned: {config.cipa_aligned}", flush=True)
    print(f"[TEST] aug_mode: {config.aug_mode}", flush=True)
    print(f"[TEST] norm_mode: {config.norm_mode}", flush=True)
    print(f"[TEST] batch_size: {config.batch_size}", flush=True)
    print(f"[TEST] num_workers: {config.num_workers}", flush=True)
    print(f"[TEST] eval_threshold: {getattr(config, 'eval_threshold', 0.5)}", flush=True)

    _, _, test_loader = _build_loaders(config)
    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key, model in task.networks.items():
        if key in ckpt:
            task.load_model_state_dict(model, ckpt[key], strict=False)

    print(f"[TEST] loaded checkpoint epoch: {ckpt.get('epoch', 'NA')}", flush=True)

    if args.save_vis:
        vis_dir = args.vis_dir or os.path.join(exp_dir, "vis_test_best_dice_adc_mac")
        print(f"[VIS] saving {args.vis_samples} diagnostics to: {vis_dir}", flush=True)
        save_segmentation_diagnostics(
            task=task,
            loader=test_loader,
            out_dir=vis_dir,
            num_samples=max(1, int(args.vis_samples)),
            threshold=getattr(config, "eval_threshold", 0.5),
        )

    print(f"[TEST] start evaluating test.txt: batches={len(test_loader)}, samples={len(test_loader.dataset)}", flush=True)

    model = task.networks["model"]
    model.eval()
    threshold = getattr(config, "eval_threshold", 0.5)
    meter = SegmentationMetricsCIPA(threshold=threshold).to(task.device)
    meter.reset()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            ct = batch["ct"].float().to(task.device, non_blocking=True)
            pet = batch["pet"].float().to(task.device, non_blocking=True)
            mask = batch["mask"].float().to(task.device, non_blocking=True)
            outputs = model(ct, pet, target_size=mask.shape[-2:])
            pred = task._select_main_pred(outputs)
            loss_seg, _ = task.loss_seg(pred, mask)
            bs = ct.size(0)
            total_loss += loss_seg.item() * bs
            total_samples += bs
            meter.update(pred, mask)
            if batch_idx == 1 or batch_idx % max(1, int(args.print_every)) == 0 or batch_idx == len(test_loader):
                current = meter.compute()
                print(
                    "[TEST progress] {}/{} batches, {}/{} samples | Dice={:.4f} IoU={:.4f} HD95={:.2f}".format(
                        batch_idx,
                        len(test_loader),
                        total_samples,
                        len(test_loader.dataset),
                        current["dice"],
                        current["iou"],
                        current["hd95"],
                    ),
                    flush=True,
                )

    metrics = meter.compute()
    metrics["total_loss"] = total_loss / max(total_samples, 1)
    print(
        "[TEST FINAL best_dice] "
        "Dice={:.4f} IoU={:.4f} Acc={:.4f} Acc_pixel={:.4f} Sen={:.4f} Spe={:.4f} Pre={:.4f} F1={:.4f} HD95={:.2f} loss={:.4f}".format(
            metrics["dice"],
            metrics["iou"],
            metrics["acc"],
            metrics["acc_pixel"],
            metrics["sensitivity"],
            metrics["specificity"],
            metrics["precision"],
            metrics["f1"],
            metrics["hd95"],
            metrics["total_loss"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
