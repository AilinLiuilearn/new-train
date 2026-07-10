# -*- coding: utf-8 -*-
"""Diagnostic ON/OFF evaluation for AC-HMTR under the same checkpoint."""

import argparse
import hashlib
import json
import os
from copy import deepcopy

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from run_mdt_seg import _build_loaders, _prepare_env, _unwrap_model
from tasks.mdt_seg import MDTSegTeacher
from utils.vis_teacher import save_segmentation_diagnostics


def _parse_args():
    parser = argparse.ArgumentParser("AC-HMTR ON/OFF diagnostic", add_help=True)
    parser.add_argument("--diagnostic_checkpoint", type=str, required=True)
    args, remaining = parser.parse_known_args()
    old_argv = list(os.sys.argv)
    try:
        os.sys.argv = [old_argv[0]] + remaining
        config = SegMDTConfig.parse_arguments()
    finally:
        os.sys.argv = old_argv
    config.diagnostic_checkpoint = args.diagnostic_checkpoint
    return config


def _checksum_model(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_checkpoint_into_task(task, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = int(checkpoint.get("epoch", checkpoint.get("epoch_idx", 0)))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Checkpoint keys: {sorted(checkpoint.keys())}")
    for key, network in task.networks.items():
        if key in checkpoint:
            msg = task.load_model_state_dict(network, checkpoint[key], strict=False)
            print(f"[{key}] missing keys: {len(msg.missing_keys)}")
            print(f"[{key}] unexpected keys: {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"[{key}] missing sample: {msg.missing_keys[:8]}")
            if msg.unexpected_keys:
                print(f"[{key}] unexpected sample: {msg.unexpected_keys[:8]}")
        else:
            print(f"[{key}] not found in checkpoint")
    return checkpoint, epoch


def _metric_row(metrics):
    return {
        "dice": float(metrics.get("dice", 0.0)),
        "iou": float(metrics.get("iou", 0.0)),
        "acc": float(metrics.get("acc", 0.0)),
        "hd95": float(metrics.get("hd95", 0.0)),
        "total_loss": float(metrics.get("total_loss", metrics.get("loss", 0.0))),
    }


def _print_table(checkpoint_path, on_metrics, off_metrics):
    delta = {
        k: on_metrics[k] - off_metrics[k]
        for k in ("dice", "iou", "acc", "hd95", "total_loss")
    }
    print("\n========================================================")
    print("Same Checkpoint AC-HMTR ON/OFF Test")
    print("========================================================")
    print(f"\nCheckpoint:\n{checkpoint_path}")
    print("\n100% Missing PET\n")
    print("Metric        AC-HMTR ON     AC-HMTR OFF     Delta(ON-OFF)")
    print(f"Dice          {on_metrics['dice']:.4f}         {off_metrics['dice']:.4f}         {delta['dice']:+.4f}")
    print(f"IoU           {on_metrics['iou']:.4f}         {off_metrics['iou']:.4f}         {delta['iou']:+.4f}")
    print(f"ACC           {on_metrics['acc']:.4f}         {off_metrics['acc']:.4f}         {delta['acc']:+.4f}")
    print(f"HD95          {on_metrics['hd95']:.4f}         {off_metrics['hd95']:.4f}         {delta['hd95']:+.4f}")
    print(f"Loss          {on_metrics['total_loss']:.4f}         {off_metrics['total_loss']:.4f}         {delta['total_loss']:+.4f}")
    print("========================================================")
    return delta


def _diagnosis(on_metrics, off_metrics):
    dice_delta = on_metrics["dice"] - off_metrics["dice"]
    iou_delta = on_metrics["iou"] - off_metrics["iou"]
    acc_delta = on_metrics["acc"] - off_metrics["acc"]
    hd95_delta = on_metrics["hd95"] - off_metrics["hd95"]
    overlap_gain = (dice_delta > 0.2) or (iou_delta > 0.2)
    overlap_same = abs(dice_delta) <= 0.2 and abs(iou_delta) <= 0.2 and abs(acc_delta) <= 0.2
    hd95_worse = hd95_delta > 0.2
    hd95_better = hd95_delta < -0.2
    if overlap_same and hd95_worse:
        return "AC-HMTR residual provides little overlap gain but introduces spatial localization errors."
    if overlap_same and abs(hd95_delta) <= 0.2:
        return "AC-HMTR residual has little effective contribution."
    if overlap_gain and hd95_worse:
        return "AC-HMTR improves region overlap but harms spatial robustness."
    if overlap_gain and hd95_better:
        return "AC-HMTR provides effective missing-PET compensation."
    return "AC-HMTR ON/OFF differences are mixed; inspect the raw metrics and visualizations." 


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    config = _parse_args()
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    g0 = _prepare_env(config)
    print(f"GPU={g0}")
    print(f"model_arch={config.model_arch}")
    print(f"missing_mode={getattr(config, 'missing_mode', None)}")
    print(f"diagnostic_checkpoint={config.diagnostic_checkpoint}")
    print(f"eval_random_seed={getattr(config, 'eval_random_seed', 2026)}")

    _, _, test_loader = _build_loaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    if getattr(test_loader, "shuffle", False):
        raise RuntimeError("test_loader must be deterministic and shuffle=False")

    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    model = _unwrap_model(task.networks["model"])
    print(f"Model object id: {id(model)}")
    init_checksum = _checksum_model(model)
    print(f"Parameter checksum before load: {init_checksum}")

    checkpoint, checkpoint_epoch = _load_checkpoint_into_task(task, config.diagnostic_checkpoint)
    loaded_checksum = _checksum_model(model)
    print(f"Parameter checksum after load: {loaded_checksum}")

    original_missing_mode = getattr(model, "missing_mode", "ct")

    with torch.no_grad():
        model.missing_mode = "ac_hmtr"
        metrics_on = task.evaluate(
            test_loader,
            eval_mode="fixed_missing",
            random_seed=2026,
            tag="ac_hmtr_on_100_missing",
        )
        on_metrics = _metric_row(metrics_on)
        save_segmentation_diagnostics(
            task=task,
            loader=test_loader,
            out_dir=os.path.join(config.checkpoint_dir, "diagnostics_ac_hmtr_on"),
            num_samples=min(8, len(test_loader.dataset)),
            threshold=getattr(config, "eval_threshold", 0.5),
            eval_mode="fixed_missing",
            random_pet_drop_prob=0.0,
            random_seed=2026,
            mode="fixed_missing",
        )

        model.missing_mode = "ct"
        metrics_off = task.evaluate(
            test_loader,
            eval_mode="fixed_missing",
            random_seed=2026,
            tag="ac_hmtr_off_100_missing",
        )
        off_metrics = _metric_row(metrics_off)
        save_segmentation_diagnostics(
            task=task,
            loader=test_loader,
            out_dir=os.path.join(config.checkpoint_dir, "diagnostics_ac_hmtr_off"),
            num_samples=min(8, len(test_loader.dataset)),
            threshold=getattr(config, "eval_threshold", 0.5),
            eval_mode="fixed_missing",
            random_pet_drop_prob=0.0,
            random_seed=2026,
            mode="fixed_missing",
        )

    model.missing_mode = original_missing_mode
    restored_checksum = _checksum_model(model)
    print(f"Parameter checksum after restore: {restored_checksum}")
    if loaded_checksum != restored_checksum:
        raise RuntimeError("Model parameters changed during ON/OFF diagnostics")

    delta = _print_table(config.diagnostic_checkpoint, on_metrics, off_metrics)
    diagnosis = _diagnosis(on_metrics, off_metrics)
    print(f"\nDiagnosis:\n{diagnosis}\n")

    results = {
        "checkpoint": config.diagnostic_checkpoint,
        "checkpoint_epoch": int(checkpoint_epoch),
        "test_samples": int(len(test_loader.dataset)),
        "model_object_id": int(id(model)),
        "missing_mode_on": "ac_hmtr",
        "missing_mode_off": "ct",
        "ac_hmtr_on": on_metrics,
        "ac_hmtr_off": off_metrics,
        "delta_on_minus_off": delta,
        "parameter_checksum_before": init_checksum,
        "parameter_checksum_after": restored_checksum,
    }
    out_json = os.path.join(config.checkpoint_dir, "ac_hmtr_on_off_results.json")
    _save_json(out_json, results)
    print(f"Results JSON: {out_json}")


if __name__ == "__main__":
    main()
