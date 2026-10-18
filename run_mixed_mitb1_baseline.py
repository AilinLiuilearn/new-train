# -*- coding: utf-8 -*-
"""Dual independent MiT-B1, aligned Add, exact-half Full/Missing baseline.

No compensation, experimental fusion, teacher losses or deep supervision.
Uses the reference branch's model, data, BCE+Dice, metrics and mixed helpers.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.components.add_fusion import AddFusion
from run_full_missing_baseline import _loaders, _assert_mixed, _checkpoint_paths
from models.build_mdt_seg import build_dual_model
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (
    build_balanced_pet_available,
    count_parameters,
    eval_missing_rates,
    module_grad_norm,
    optimizer_step_succeeded,
    seed_everything,
)
from utils.train_logger import append_epoch_log, init_train_log


class MixedMiTB1Config(SegMDTConfig):
    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        p.set_defaults(
            ct_backbone="mit_b1",
            pet_backbone="mit_b1",
            ct_pretrained_path="/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1",
            pet_pretrained_path="/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1",
        )
        return p

    @staticmethod
    def data_parser():
        p = SegMDTConfig.data_parser()
        p.set_defaults(norm_mode="imagenet")
        return p

    @staticmethod
    def train_parser():
        p = SegMDTConfig.train_parser()
        p.set_defaults(
            learning_rate=4e-5,
            train_batch_mode="mixed",
            epochs=60,
            early_stop_patience=60,
            ema_enabled=True,
            ema_decay=0.999,
            ema_start_epoch=3,
            ema_decay_warmup=True,
        )
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser("Mixed MiT-B1 baseline", add_help=False)
        p.add_argument("--log_interval", type=int, default=100)
        return p


class MixedMiTB1Task(MDTSegTeacher):
    def train_step_mixed(self, batch, pet_available):
        state = torch.as_tensor(pet_available)
        if state.numel() != len(batch["ct"]) or int(state.eq(1).sum()) != int(
            state.eq(0).sum()
        ):
            raise ValueError("requires exact-half Full/Missing state per batch")
        return super().train_step_mixed(batch, pet_available)

    def evaluate(self, loader, eval_mode="full", tag="val", model=None):
        total = len(loader) if hasattr(loader, "__len__") else "?"
        interval = int(getattr(self.config, "log_interval", 100))
        print(f"[{tag}] start mode={eval_mode} batches={total}", flush=True)
        start = time.time()

        def batches():
            for index, batch in enumerate(loader, 1):
                yield batch
                if index % interval == 0:
                    print(
                        f"[{tag}] batch {index}/{total} elapsed={time.time()-start:.0f}s",
                        flush=True,
                    )

        result = super().evaluate(batches(), eval_mode, tag, model)
        print(
            f'[{tag}] Dice={result["dice"]:.4f} HD95={result["hd95"]:.4f}', flush=True
        )
        return result


def assert_protocol(cfg, model=None):
    if cfg.train_batch_mode != "mixed" or cfg.optimizer != "adamw":
        raise ValueError("this baseline requires mixed batches and AdamW")
    if cfg.accumulation_steps != 1 or cfg.use_deep_supervision or cfg.deep_supervision:
        raise ValueError("requires one update per batch and no deep supervision")
    if cfg.batch_size < 2 or cfg.batch_size % 2:
        raise ValueError("requires an even batch_size >= 2")
    if any(
        str(getattr(cfg, name)).lower().replace("-", "_") != "mit_b1"
        for name in ("ct_backbone", "pet_backbone")
    ):
        raise ValueError("both independent encoders must be MiT-B1")
    if cfg.log_interval < 1:
        raise ValueError("log_interval must be positive")
    rates = cfg.final_test_missing_rates
    if not {0.0, 1.0}.issubset(rates) or any(not 0 <= r <= 1 for r in rates):
        raise ValueError("final missing rates must include 0 and 1 and lie in [0,1]")
    if model is not None:
        if not isinstance(model.fusion, AddFusion) or model.enc_ct is model.enc_pet:
            raise RuntimeError("requires AddFusion and independent encoders")
        if not hasattr(model, "ct_align") or not hasattr(model, "pet_align"):
            raise RuntimeError("both channel aligners are required")


def main():
    print(
        "[INFO] aligned MiT-B1 + MiT-B1 Add; exact-half Full/Missing; no compensation",
        flush=True,
    )
    cfg = MixedMiTB1Config.parse_arguments()
    assert_protocol(cfg)
    cfg.experiment_type = "mixed_mitb1_add_baseline"
    cfg.baseline_reference = "exp/full-add-local-reassembly@81ff8cf"
    cfg.fusion_type, cfg.pet_missing_rate = "AddFusion", 0.5
    if any(
        os.path.exists(os.path.join(cfg.checkpoint_dir, name))
        for name in ("ckpt.last.pth.tar", "ckpt.best_joint.pth.tar", "train_log.csv")
    ):
        raise FileExistsError("existing run directory; use a new --hash")
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, "config_args.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    _assert_mixed(cfg, train_loader)
    if not len(train_loader):
        raise ValueError("no complete training batch; check batch_size and drop_last")
    print(
        f"[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader)}",
        flush=True,
    )

    task = MixedMiTB1Task(build_dual_model(cfg), cfg)
    assert_protocol(cfg, task.model)
    total_params, trainable_params = count_parameters(task.model)
    print(
        f"[INFO] params_total={total_params} params_trainable={trainable_params}",
        flush=True,
    )
    if task.ema is not None:
        print(
            f"[INFO] ema_enabled=True decay={task.ema.decay} warmup={task.ema.warmup} "
            f"start_epoch={task.ema_start_epoch}",
            flush=True,
        )
    else:
        print("[INFO] ema_enabled=False eval_uses_raw=True", flush=True)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        "train_full_loss",
        "train_missing_loss",
        "train_mixed_loss",
        "train_full_samples",
        "train_missing_samples",
        "mixed_train_batches",
        "val_full_loss",
        "val_full_dice",
        "val_full_iou",
        "val_full_acc",
        "val_full_acc_pixel",
        "val_full_hd95",
        "val_missing_loss",
        "val_missing_dice",
        "val_missing_iou",
        "val_missing_acc",
        "val_missing_acc_pixel",
        "val_missing_hd95",
        "joint_dice",
        "best_joint",
        "best_joint_epoch",
        "grad_mixed_enc_ct",
        "grad_mixed_enc_pet",
        "grad_mixed_ct_align",
        "grad_mixed_pet_align",
        "grad_mixed_decoder",
        "ema_enabled",
        "ema_updates",
        "skipped_updates",
        "epoch_time",
    ]
    init_train_log(
        os.path.join(cfg.checkpoint_dir, "train_log.csv"), extra_headers=extra_headers
    )

    best_joint = -1.0
    best_joint_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        full_loss_sum = missing_loss_sum = mixed_loss_sum = 0.0
        full_sample_count = missing_sample_count = 0
        mixed_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {
            "enc_ct": [],
            "enc_pet": [],
            "ct_align": [],
            "pet_align": [],
            "decoder": [],
        }
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            actual_batch_size = batch["ct"].shape[0]
            if actual_batch_size != int(cfg.batch_size):
                raise ValueError(
                    f"mixed mode requires fixed batch size {int(cfg.batch_size)} "
                    f"(drop_last=True), got {actual_batch_size}"
                )
            pet_available = build_balanced_pet_available(
                int(cfg.batch_size),
                global_batch_step,
                cfg.random_state,
                task.device,
            )
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=amp_enabled and torch.cuda.is_available()
            ):
                loss, _, _, train_stats = task.train_step_mixed(
                    batch, pet_available=pet_available
                )
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads["enc_ct"].append(module_grad_norm(task.model.enc_ct))
            grads["enc_pet"].append(module_grad_norm(task.model.enc_pet))
            grads["ct_align"].append(module_grad_norm(task.model.ct_align))
            grads["pet_align"].append(module_grad_norm(task.model.pet_align))
            grads["decoder"].append(module_grad_norm(task.model.decoder))
            total_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    task.trainable_parameters(), float(cfg.grad_clip)
                )
                if float(cfg.grad_clip) > 0
                else 0.0
            )
            if torch.isfinite(torch.as_tensor(total_grad_norm)):
                grad_norm_accum += float(total_grad_norm)
                grad_norm_steps += 1
            finite = torch.stack(
                [
                    torch.isfinite(p.grad).all()
                    for p in task.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            # Let GradScaler update its overflow scale; non-AMP also rejects nonfinite updates.
            succeeded = (
                optimizer_step_succeeded(task)
                if task.scaler.is_enabled()
                else bool(finite) and optimizer_step_succeeded(task)
            )
            if succeeded:
                task.scheduler.step()
                task.update_ema()
            else:
                skipped_update_count += 1
            if (batch_idx + 1) % cfg.log_interval == 0:
                print(
                    f"[BATCH {batch_idx + 1}] mode=mixed "
                    f'num_full={int(train_stats["num_full"])} '
                    f'num_missing={int(train_stats["num_missing"])} '
                    f"loss={float(loss.detach()):.6f}",
                    flush=True,
                )
            full_loss_sum += float(train_stats["loss_full"]) * int(
                train_stats["num_full"]
            )
            missing_loss_sum += float(train_stats["loss_missing"]) * int(
                train_stats["num_missing"]
            )
            full_sample_count += int(train_stats["num_full"])
            missing_sample_count += int(train_stats["num_missing"])
            mixed_loss_sum += float(loss.detach())
            mixed_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_full = task.evaluate(
            val_loader, eval_mode="full", tag="val_full", model=task.eval_model()
        )
        val_missing = task.evaluate(
            val_loader, eval_mode="missing", tag="val_missing", model=task.eval_model()
        )
        joint_dice = 0.5 * val_full["dice"] + 0.5 * val_missing["dice"]

        if joint_dice > best_joint:
            best_joint = joint_dice
            best_joint_epoch = epoch
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            improved = False
        if improved:
            task.save_checkpoint(
                paths["best_joint"],
                epoch,
                best_joint,
                best_joint_epoch,
                {
                    "full": {k: val_full[k] for k in ("dice", "iou", "hd95", "acc")},
                    "missing": {
                        k: val_missing[k] for k in ("dice", "iou", "hd95", "acc")
                    },
                    "joint_dice": joint_dice,
                },
            )
        task.save_checkpoint(
            paths["last"],
            epoch,
            best_joint,
            best_joint_epoch,
            {
                "full": val_full["dice"],
                "missing": val_missing["dice"],
                "joint_dice": joint_dice,
            },
        )

        epoch_full_loss = full_loss_sum / max(1, full_sample_count)
        epoch_missing_loss = missing_loss_sum / max(1, missing_sample_count)
        train_mixed_loss = mixed_loss_sum / max(1, mixed_n)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        val_joint = {
            "total_loss": 0.5 * val_full["total_loss"]
            + 0.5 * val_missing["total_loss"],
            "dice": joint_dice,
            "iou": 0.5 * val_full["iou"] + 0.5 * val_missing["iou"],
            "acc": 0.5 * val_full["acc"] + 0.5 * val_missing["acc"],
            "acc_pixel": 0.5 * val_full.get("acc_pixel", 0.0)
            + 0.5 * val_missing.get("acc_pixel", 0.0),
            "hd95": 0.5 * val_full["hd95"] + 0.5 * val_missing["hd95"],
        }
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, "train_log.csv"),
            epoch,
            train_mixed_loss,
            val_joint,
            lr=task.optimizer.param_groups[0]["lr"],
            grad_norm=avg_grad_norm,
            extra_metrics={
                "train_full_loss": epoch_full_loss,
                "train_missing_loss": epoch_missing_loss,
                "train_mixed_loss": train_mixed_loss,
                "train_full_samples": float(full_sample_count),
                "train_missing_samples": float(missing_sample_count),
                "mixed_train_batches": float(mixed_n),
                "val_full_loss": val_full["total_loss"],
                "val_full_dice": val_full["dice"],
                "val_full_iou": val_full["iou"],
                "val_full_acc": val_full["acc"],
                "val_full_acc_pixel": val_full.get("acc_pixel", 0.0),
                "val_full_hd95": val_full["hd95"],
                "val_missing_loss": val_missing["total_loss"],
                "val_missing_dice": val_missing["dice"],
                "val_missing_iou": val_missing["iou"],
                "val_missing_acc": val_missing["acc"],
                "val_missing_acc_pixel": val_missing.get("acc_pixel", 0.0),
                "val_missing_hd95": val_missing["hd95"],
                "joint_dice": joint_dice,
                "best_joint": best_joint,
                "best_joint_epoch": best_joint_epoch,
                "grad_mixed_enc_ct": (
                    float(np.mean(grads["enc_ct"])) if grads["enc_ct"] else 0.0
                ),
                "grad_mixed_enc_pet": (
                    float(np.mean(grads["enc_pet"])) if grads["enc_pet"] else 0.0
                ),
                "grad_mixed_ct_align": (
                    float(np.mean(grads["ct_align"])) if grads["ct_align"] else 0.0
                ),
                "grad_mixed_pet_align": (
                    float(np.mean(grads["pet_align"])) if grads["pet_align"] else 0.0
                ),
                "grad_mixed_decoder": (
                    float(np.mean(grads["decoder"])) if grads["decoder"] else 0.0
                ),
                "ema_enabled": 1.0 if task.ema is not None else 0.0,
                "ema_updates": float(task.ema.updates) if task.ema is not None else 0.0,
                "skipped_updates": float(skipped_update_count),
                "epoch_time": time.time() - epoch_start,
            },
        )
        print(
            f'[EPOCH {epoch}] full_dice={val_full["dice"]:.4f} '
            f'missing_dice={val_missing["dice"]:.4f} joint_dice={joint_dice:.4f} '
            f"best_joint={best_joint:.4f} "
            f'lr={task.optimizer.param_groups[0]["lr"]:.8f}',
            flush=True,
        )
        if no_improve >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs", flush=True)
            break

    if not os.path.isfile(paths["best_joint"]):
        raise RuntimeError("best_joint checkpoint was not created")
    ckpt = MDTSegTeacher.load_state_dicts(paths["best_joint"])
    use_ema = ckpt["eval_weights"] == "ema"
    if use_ema and ckpt.get("model_ema") is None:
        raise RuntimeError("selected EMA weights are absent from checkpoint")
    weights_tag = "ema" if use_ema else "raw"
    # Reuse the training model: no pretrained re-read or third model allocation.
    eval_model = task.model
    eval_model.load_state_dict(
        ckpt["model_ema"] if use_ema else ckpt["model"], strict=True
    )
    eval_model.eval()
    print(
        f'[FINAL] evaluating best_joint checkpoint at epoch {ckpt["best_epoch"]} '
        f"with weights={weights_tag}",
        flush=True,
    )
    eval_missing_rates(
        task,
        eval_model,
        test_loader,
        cfg.random_state,
        cfg.checkpoint_dir,
        "final_test",
        rates=list(cfg.final_test_missing_rates),
        weights_tag=weights_tag,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
