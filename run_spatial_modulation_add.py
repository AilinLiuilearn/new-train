"""Exact-half mixed Add experiment; only new model/task/config are selected.

Loader, BCE+Dice, seeded masks, scheduler, metrics and test-rate evaluation
are reused from the clean baseline. Original baseline runners are untouched.
"""

import argparse
import copy
import csv
import json
import os
import time

import torch

from configs.base import str2bool
from models.spatial_modulation_bank import EXPERIMENT_TYPE
from configs.seg_mdt import SegMDTConfig
from models.spatial_modulation_add_model import build_matrix_model, MATRIX_DEFAULTS
from tasks.spatial_modulation_seg import SpatialModulationSegTask
from utils.spatial_modulation_runtime import _loaders, _checkpoint_paths
from utils.optimization import get_cosine_scheduler
from utils.run_common import (
    seed_everything,
    build_balanced_pet_available,
    optimizer_step_succeeded,
    eval_missing_rates,
)
from utils.train_logger import init_train_log, append_epoch_log, CSV_HEADER
from utils.spatial_modulation_diagnostics import ModulationCSVLogger as MatrixCSVLogger


class SpatialModulationConfig(SegMDTConfig):
    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        p.set_defaults(use_deep_supervision=False)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser(
            "EMA CT keys / learnable affine Values", add_help=False
        )
        for name, default in MATRIX_DEFAULTS.items():
            kind = str2bool if isinstance(default, bool) else type(default)
            p.add_argument("--matrix_" + name, type=kind, default=default)
        p.add_argument("--resume", type=str, default=None)
        p.add_argument("--log_interval", type=int, default=100)
        return p


def _assert_mixed(cfg, train_loader):
    if cfg.batch_size < 2 or cfg.batch_size % 2:
        raise ValueError("mixed mode requires even batch_size >= 2")
    if not bool(getattr(train_loader, "drop_last", False)):
        raise ValueError("mixed mode requires drop_last=True")
    if cfg.accumulation_steps != 1:
        raise ValueError("matrix runner requires accumulation_steps=1")


def main():
    cfg = SpatialModulationConfig.parse_arguments()
    if cfg.log_interval < 1:
        raise ValueError("log_interval must be positive")
    if (
        cfg.train_batch_mode != "mixed"
        or cfg.optimizer != "adamw"
        or cfg.accumulation_steps != 1
    ):
        raise ValueError(
            "matrix experiment requires --train_batch_mode mixed --optimizer adamw --accumulation_steps 1"
        )
    if cfg.matrix_mod_loss_weight < 0:
        raise ValueError("modulation loss weight must be nonnegative")
    cfg.experiment_type = EXPERIMENT_TYPE
    cfg.model_arch = "spatial_modulation_add_model"
    cfg.baseline_reference = "exp/ct-full-spatial-modulation-bank@e697de2"
    cfg.fusion_type, cfg.pet_missing_rate = "AddFusion", 0.5
    print(
        f"[protocol] norm_mode={cfg.norm_mode} hd95_mode={cfg.hd95_mode} "
        "alignment=CT_and_PET_Conv1x1_BN_ReLU fusion=AddFusion",
        flush=True,
    )
    if not {0.0, 1.0}.issubset(cfg.final_test_missing_rates) or any(
        rate < 0 or rate > 1 for rate in cfg.final_test_missing_rates
    ):
        raise ValueError(
            "final_test_missing_rates must be in [0,1] and include 0 and 1 for Joint"
        )
    seed_everything(cfg.random_state)
    print("[setup] checking data splits and building loaders...", flush=True)
    train_loader, val_loader, test_loader = _loaders(cfg)
    _assert_mixed(cfg, train_loader)
    cfg.steps_per_epoch = len(train_loader)
    print(
        f"[setup] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader)} batch_size={cfg.batch_size} "
        f"num_workers={cfg.num_workers}",
        flush=True,
    )
    # Restore complete checkpoint weights without needing original pretrained files.
    build_cfg = copy.deepcopy(cfg)
    if cfg.resume:
        build_cfg.pretrained = False
        build_cfg.ct_pretrained_path = build_cfg.pet_pretrained_path = None
    task = SpatialModulationSegTask(build_matrix_model(build_cfg), cfg)
    print(
        f'[setup] device={task.device} AMP={bool(cfg.mixed_precision) and task.device.type == "cuda"} '
        "first_epoch_collection_only=True "
        f"log_interval={cfg.log_interval}",
        flush=True,
    )
    task.loader_generators = {
        name: loader.generator
        for name, loader in (
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        )
        if getattr(loader, "generator", None) is not None
    }
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )
    paths = _checkpoint_paths(cfg.checkpoint_dir)
    start_epoch, best, best_epoch = 1, -1.0, 0
    if cfg.resume:
        ckpt = task.resume(cfg.resume)
        start_epoch, best, best_epoch = (
            int(ckpt["epoch"]) + 1,
            float(ckpt["best"]),
            int(ckpt["best_epoch"]),
        )
        # A resumed run must keep the original best candidate for final selection.
        old_best = os.path.join(
            os.path.dirname(os.path.abspath(cfg.resume)), "ckpt.best_joint.pth.tar"
        )
        if not os.path.isfile(paths["best_joint"]):
            if not os.path.isfile(old_best):
                raise ValueError(
                    "resume requires original ckpt.best_joint.pth.tar alongside last checkpoint"
                )
            import shutil

            shutil.copy2(old_best, paths["best_joint"])
    elif os.path.exists(paths["last"]):
        raise ValueError(
            "checkpoint directory already contains a run; use a new --hash or --resume"
        )

    with open(os.path.join(cfg.checkpoint_dir, "config_args.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    extra_headers = [
        "train_full_loss",
        "train_missing_loss",
        "train_full_samples",
        "train_missing_samples",
        "joint_dice",
        "val_full_dice",
        "val_missing_dice",
        "best_joint",
        "modulation_initialized_scales",
        "modulation_loss",
        "finite_grad_count",
        "nonfinite_grad_count",
        "val_full_hd95",
        "val_missing_hd95",
        "skipped_updates",
        "epoch_seconds",
    ]
    log_path = os.path.join(cfg.checkpoint_dir, "train_log.csv")
    if cfg.resume and os.path.exists(log_path):
        with open(log_path, newline="") as f:
            if next(csv.reader(f)) != CSV_HEADER + extra_headers:
                raise ValueError("resume log schema does not match this runner")
    else:
        init_train_log(log_path, extra_headers)
    matrix_logger = MatrixCSVLogger(cfg.checkpoint_dir, start_epoch=start_epoch)

    for epoch in range(start_epoch, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        sums = dict(full=0.0, missing=0.0, total=0.0, grad=0.0)
        finite_grad_count = nonfinite_grad_count = 0
        batches, samples, skipped = 0, 0, 0
        begin = time.time()
        print(
            f"[epoch {epoch}/{cfg.epochs}] train start; waiting for first batch...",
            flush=True,
        )
        for batch in train_loader:
            if batch["ct"].shape[0] != cfg.batch_size:
                raise ValueError("expected fixed full-size batches with drop_last=True")
            state = build_balanced_pet_available(
                cfg.batch_size, task.global_batch_step, cfg.random_state, task.device
            )
            task.optimizer.zero_grad(set_to_none=True)
            task.capture_matrix_reads = (
                batches + 1
            ) % cfg.log_interval == 0 or batches + 1 == len(train_loader)
            with torch.autocast(
                device_type=task.device.type,
                enabled=bool(cfg.mixed_precision) and task.device.type == "cuda",
            ):
                loss, _, _, stats = task.train_step_mixed(batch, state)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grad = (
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_parameters(), cfg.grad_clip
                )
                if cfg.grad_clip > 0
                else 0.0
            )
            flags = [
                torch.isfinite(p.grad).all()
                for p in task.model.parameters()
                if p.grad is not None
            ]
            finite_gradients = bool(torch.stack(flags).all()) if flags else False
            if task.scaler.is_enabled():
                # GradScaler owns overflow detection/update of its scale.
                succeeded = optimizer_step_succeeded(task)
            else:
                succeeded = finite_gradients and optimizer_step_succeeded(task)
            task.after_optimizer_step(succeeded)
            if succeeded:
                task.scheduler.step()
                task.update_ema()
            else:
                skipped += 1
            for key, stat in (("full", "loss_full"), ("missing", "loss_missing")):
                sums[key] += float(stats[stat])
            sums["total"] += float(loss.detach())
            if torch.isfinite(torch.as_tensor(grad)):
                sums["grad"] += float(grad)
                finite_grad_count += 1
            else:
                nonfinite_grad_count += 1
            batches += 1
            samples += cfg.batch_size // 2
            task.global_batch_step += 1
            if batches % cfg.log_interval == 0:
                elapsed = time.time() - begin
                remaining = elapsed / batches * (len(train_loader) - batches)
                sv_now = sum(bool(s.bank.initialized) for s in task.model.matrix.scales)
                print(
                    f"[train epoch {epoch}] batch {batches}/{len(train_loader)} "
                    f'loss={float(stats["loss_total"]):.4f} '
                    f'Full={float(stats["loss_full"]):.4f} '
                    f'Missing={float(stats["loss_missing"]):.4f} '
                    f'lr={task.optimizer.param_groups[0]["lr"]:.3e} '
                    f"unified_initialized={sv_now}/4 "
                    f"elapsed={elapsed:.0f}s ETA={remaining:.0f}s skipped={skipped}",
                    flush=True,
                )
                matrix_logger.snapshot(task, epoch, batches, event="interval")
        if batches == 0:
            raise ValueError("training loader has no complete mixed batch")
        matrix_logger.snapshot(task, epoch, batches, event="epoch_end")
        print(
            f"[epoch {epoch}] train complete in {time.time() - begin:.1f}s; validation start",
            flush=True,
        )
        active = task.eval_model()
        full = task.evaluate(
            val_loader, eval_mode="full", tag=f"val_full epoch {epoch}", model=active
        )
        missing = task.evaluate(
            val_loader,
            eval_mode="missing",
            tag=f"val_missing epoch {epoch}",
            model=active,
        )
        joint = 0.5 * (full["dice"] + missing["dice"])
        improved = joint > best
        if improved:
            best, best_epoch, task.no_improve = joint, epoch, 0
        else:
            task.no_improve += 1
        val = {"full": full, "missing": missing, "joint_dice": joint}
        if improved:
            print(f"[checkpoint epoch {epoch}] saving best_joint...", flush=True)
            task.save_checkpoint(paths["best_joint"], epoch, best, best_epoch, val)
        print(f"[checkpoint epoch {epoch}] saving last...", flush=True)
        task.save_checkpoint(paths["last"], epoch, best, best_epoch, val)
        print(f"[checkpoint epoch {epoch}] save complete", flush=True)
        diag = task.model.matrix.diagnostics()
        with open(os.path.join(cfg.checkpoint_dir, "matrix_log.jsonl"), "a") as f:
            f.write(json.dumps(dict(epoch=epoch, **diag)) + "\n")
        sv = sum(bool(s.bank.initialized) for s in task.model.matrix.scales)
        joint_metrics = {
            key: 0.5 * (full[key] + missing[key])
            for key in ("total_loss", "dice", "iou", "acc", "acc_pixel", "hd95")
        }
        append_epoch_log(
            log_path,
            epoch,
            sums["total"] / batches,
            joint_metrics,
            lr=task.optimizer.param_groups[0]["lr"],
            grad_norm=sums["grad"] / max(1, finite_grad_count),
            extra_metrics=dict(
                train_full_loss=sums["full"] / batches,
                train_missing_loss=sums["missing"] / batches,
                train_full_samples=samples,
                train_missing_samples=samples,
                joint_dice=joint,
                val_full_dice=full["dice"],
                val_missing_dice=missing["dice"],
                best_joint=best,
                modulation_initialized_scales=sv,
                modulation_loss=task.mod_loss_sum / max(1, task.mod_loss_batches),
                finite_grad_count=finite_grad_count,
                nonfinite_grad_count=nonfinite_grad_count,
                val_full_hd95=full["hd95"],
                val_missing_hd95=missing["hd95"],
                skipped_updates=skipped,
                epoch_seconds=time.time() - begin,
            ),
        )
        print(
            f'[epoch {epoch}] Full={full["dice"]:.4f} Missing={missing["dice"]:.4f} '
            f"Joint={joint:.4f} unified_initialized={sv}/4 retrieval_loss={task.mod_loss_sum/max(1,task.mod_loss_batches):.5f}",
            flush=True,
        )
        if task.no_improve >= cfg.early_stop_patience:
            break

    ckpt = task.load_state_dicts(paths["best_joint"])
    weights_tag = ckpt["eval_weights"]
    # Honor the weights actually used to select this checkpoint, including
    # checkpoints selected before model EMA activation.
    state = ckpt["model_ema"] if weights_tag == "ema" else ckpt["model"]
    task.model.load_state_dict(state, strict=True)
    task.model.eval()
    print(
        f'[final] evaluating best epoch {ckpt["best_epoch"]}, weights={weights_tag}, '
        f"missing_rates={cfg.final_test_missing_rates}",
        flush=True,
    )
    eval_missing_rates(
        task,
        task.model,
        test_loader,
        cfg.random_state,
        cfg.checkpoint_dir,
        "final_test",
        rates=list(cfg.final_test_missing_rates),
        weights_tag=weights_tag,
    )


if __name__ == "__main__":
    main()
