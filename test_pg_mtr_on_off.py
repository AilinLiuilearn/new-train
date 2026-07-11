# -*- coding: utf-8 -*-
"""
Same-checkpoint PG-MTR ON/OFF diagnostic under 100% Missing PET.

Strict controls:
- same checkpoint loaded once
- same model instance
- same test loader
- same seed
- no retraining / backward / optimizer step
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from eval_mdt_missing_with_metrics import (
    _set_seed,
    _load_checkpoint,
    _build_test_loader,
)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _parameter_checksum(model: torch.nn.Module) -> Dict[str, float]:
    total = 0.0
    sq_total = 0.0
    numel = 0
    with torch.no_grad():
        for p in model.parameters():
            x = p.detach().float()
            total += float(x.sum().cpu())
            sq_total += float((x * x).sum().cpu())
            numel += int(x.numel())
    return {"sum": total, "sq_sum": sq_total, "numel": numel}


def _to_float_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu()) if v.numel() == 1 else v.detach().cpu().tolist()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _evaluate_fixed_missing(task, loader, tag: str, seed: int) -> Dict[str, Any]:
    metrics = task.evaluate(
        loader,
        eval_mode="fixed_missing",
        random_pet_drop_prob=1.0,
        random_seed=int(seed),
        tag=tag,
        force_missing_pet=True,
    )
    return _to_float_dict(metrics)


def _print_table(metrics_on: Dict[str, Any], metrics_off: Dict[str, Any]) -> None:
    specs = [
        ("Dice", "dice", "higher"),
        ("IoU", "iou", "higher"),
        ("ACC", "acc", "higher"),
        ("HD95", "hd95", "lower"),
        ("Loss", "total_loss", "lower"),
    ]
    print("\n" + "=" * 86)
    print("PG-MTR SAME-CHECKPOINT ON/OFF @ 100% MISSING PET")
    print("=" * 86)
    print(f'{"Metric":<14}{"PG-MTR ON":>18}{"PG-MTR OFF":>18}{"Delta(ON-OFF)":>20}{"Preferred":>14}')
    print("-" * 86)
    for name, key, preferred in specs:
        if key not in metrics_on or key not in metrics_off:
            continue
        on = float(metrics_on[key])
        off = float(metrics_off[key])
        print(f"{name:<14}{on:>18.6f}{off:>18.6f}{on-off:>20.6f}{preferred:>14}")
    print("=" * 86)
    print("Dice/IoU/ACC: Delta > 0 means ON is better")
    print("HD95/Loss:    Delta < 0 means ON is better")
    print("=" * 86)


def main() -> None:
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(
        description="Same-checkpoint PG-MTR ON/OFF diagnostic.",
        parents=parents,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)

    seed = int(getattr(config, "eval_random_seed", 2026))
    _set_seed(seed)

    print("[1/5] Build model/task once")
    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)

    print("[2/5] Load checkpoint once")
    _load_checkpoint(task, config.ckpt_path)

    print("[3/5] Build test loader once")
    test_loader = _build_test_loader(config)

    wrapped_model = task.networks["model"]
    model = _unwrap(wrapped_model)

    if not hasattr(model, "missing_mode"):
        raise AttributeError("Model has no missing_mode attribute.")
    if not hasattr(model, "pg_mtr"):
        raise AttributeError("Model has no pg_mtr module.")

    original_missing_mode = str(model.missing_mode)
    original_training_state = bool(wrapped_model.training)

    checksum_before = _parameter_checksum(model)

    print("[4/5] Evaluate same model instance twice")
    try:
        model.missing_mode = "pg_mtr"
        print("\n[A] PG-MTR ON: C1, C2, C3+R3, C4+R4")
        metrics_on = _evaluate_fixed_missing(
            task, test_loader, "pg_mtr_on_100pct_missing", seed
        )

        model.missing_mode = "ct"
        print("\n[B] PG-MTR OFF: C1, C2, C3, C4")
        metrics_off = _evaluate_fixed_missing(
            task, test_loader, "pg_mtr_off_100pct_missing", seed
        )
    finally:
        model.missing_mode = original_missing_mode
        wrapped_model.train(original_training_state)

    checksum_after = _parameter_checksum(model)
    checksum_diff = {
        "sum_abs_diff": abs(checksum_after["sum"] - checksum_before["sum"]),
        "sq_sum_abs_diff": abs(checksum_after["sq_sum"] - checksum_before["sq_sum"]),
        "numel_diff": int(checksum_after["numel"] - checksum_before["numel"]),
    }

    print("[5/5] Parameter integrity check")
    print(json.dumps(checksum_diff, indent=2))

    if (
        checksum_diff["sum_abs_diff"] > 1e-8
        or checksum_diff["sq_sum_abs_diff"] > 1e-8
        or checksum_diff["numel_diff"] != 0
    ):
        raise RuntimeError("Model parameters changed during evaluation.")

    _print_table(metrics_on, metrics_off)

    keys = ["dice", "iou", "acc", "hd95", "total_loss"]
    delta = {
        k: float(metrics_on[k]) - float(metrics_off[k])
        for k in keys
        if k in metrics_on and k in metrics_off
    }

    payload = {
        "experiment": "same_checkpoint_pg_mtr_on_off",
        "checkpoint": os.path.abspath(config.ckpt_path),
        "seed": seed,
        "eval_mode": "fixed_missing",
        "missing_rate": 1.0,
        "same_checkpoint_loaded_once": True,
        "same_model_instance": True,
        "same_test_loader": True,
        "no_retraining": True,
        "pg_mtr_on": {"missing_mode": "pg_mtr", "metrics": metrics_on},
        "pg_mtr_off": {"missing_mode": "ct", "metrics": metrics_off},
        "delta_on_minus_off": delta,
        "parameter_checksum": {
            "before": checksum_before,
            "after": checksum_after,
            "diff": checksum_diff,
        },
    }

    output_path = os.path.join(config.output_dir, "pg_mtr_on_off_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
