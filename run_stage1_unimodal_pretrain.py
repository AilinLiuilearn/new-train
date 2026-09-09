# -*- coding: utf-8 -*-
"""Stage-1 unimodal task-specific pretraining runner.

This script trains either a CT expert or a PET expert on the full paired
train split, but only one modality participates in the segmentation forward
path. It is the pretraining stage of the SimMLM-style recipe:

    Stage-1  -> unimodal task-specific pretraining
    Stage-1.5 -> optional PSPI bank bootstrap
    Stage-2  -> cooperative PSPI training

Two values are supported for --stage1_modality:
    ct   -> CTOnlySegmentationModel
    pet  -> PETOnlySegmentationModel

The same dataset, augmentation, optimizer, scheduler, AMP, gradient clipping,
early stopping and metric code are reused. No prototype bank, no Full/Missing
alternation, no modality dropout, and no extra auxiliary loss are introduced.
"""

import argparse
import csv
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.unimodal_pretrain import CTOnlySegmentationModel, PETOnlySegmentationModel
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.train_logger import append_epoch_log, init_train_log


def _seed(cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class UnimodalTeacher:
    def __init__(self, model, cfg):
        self.model = model
        self.config = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(
            smooth=cfg.loss_smooth,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
        )
        self.metrics = SegmentationMetricsCIPA()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch, modality="ct"):
        ct = batch["ct"].to(self.device, non_blocking=True)
        pet = batch["pet"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True).float()
        if modality == "ct":
            outputs = self.model(ct=ct, pet=None, target_size=mask.shape[-2:])
        elif modality == "pet":
            outputs = self.model(pet=pet, target_size=mask.shape[-2:])
        else:
            raise ValueError("modality must be 'ct' or 'pet'")
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        seg_loss, loss_stats = self.criterion(logits, mask)
        stats = {
            "loss_total": seg_loss.detach(),
            "loss_seg": loss_stats.get("loss_dice", seg_loss.detach()),
            "loss_seg_total": seg_loss.detach(),
            "prototype_loss": seg_loss.new_zeros(()),
            "prototype_loss_weighted": seg_loss.new_zeros(()),
            "prototype_loss_num_terms": 0,
            "loss_boundary": torch.tensor(0.0, device=seg_loss.device),
        }
        return seg_loss, logits, outputs, stats

    @torch.no_grad()
    def evaluate(self, loader, modality="ct"):
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        sample_count = 0
        self.metrics.reset()
        for batch in loader:
            ct = batch["ct"].to(self.device, non_blocking=True)
            pet = batch["pet"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True).float()
            batch_size = ct.shape[0]
            if modality == "ct":
                outputs = self.model(ct=ct, pet=None, target_size=mask.shape[-2:])
            elif modality == "pet":
                outputs = self.model(pet=pet, target_size=mask.shape[-2:])
            else:
                raise ValueError("modality must be 'ct' or 'pet'")
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss, _ = self.criterion(logits, mask)
            self.metrics.update(logits, mask)
            total_loss += float(loss) * batch_size
            sample_count += batch_size
        out = self.metrics.compute()
        out["total_loss"] = total_loss / max(1, sample_count)
        self.model.train(was_training)
        return out

    def save_checkpoint(self, path, epoch, best_dice=None, best_epoch=None, val_metrics=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "stage": "unimodal_pretrain",
            "modality": getattr(self.config, "stage1_modality", None),
            "epoch": epoch,
            "global_batch_step": self.global_batch_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "random_state": getattr(self.config, "random_state", None),
            "seed": getattr(self.config, "random_state", None),
            "config": vars(self.config),
            "random_state_python": random.getstate(),
            "random_state_numpy": np.random.get_state(),
            "random_state_torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["random_state_cuda"] = torch.cuda.get_rng_state_all()
        if getattr(self.config, "stage1_modality", "ct") == "ct":
            payload["encoder"] = self.model.enc_ct.state_dict()
            payload["ct_align"] = self.model.ct_align.state_dict()
            payload["decoder"] = self.model.decoder.state_dict()
        else:
            payload["encoder"] = self.model.enc_pet.state_dict()
            payload["decoder"] = self.model.decoder.state_dict()
        torch.save(payload, path)


def _build_model(cfg):
    if cfg.stage1_modality == "ct":
        model = CTOnlySegmentationModel(
            ct_backbone=cfg.ct_backbone,
            ct_pretrained_path=cfg.ct_pretrained_path,
            in_channels=3,
            out_channels=1,
            decoder_channels=cfg.decoder_channels,
            use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision),
        )
    elif cfg.stage1_modality == "pet":
        model = PETOnlySegmentationModel(
            pet_backbone=cfg.pet_backbone,
            pet_pretrained_path=cfg.pet_pretrained_path,
            in_channels=3,
            out_channels=1,
            decoder_channels=cfg.decoder_channels,
            use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision),
        )
    else:
        raise ValueError("stage1_modality must be 'ct' or 'pet'")
    print(
        f"[STAGE1] modality={cfg.stage1_modality} "
        f"ct_backbone={getattr(cfg, 'ct_backbone', None)} "
        f"pet_backbone={getattr(cfg, 'pet_backbone', None)} "
        f"decoder=UNetStyleDecoder "
        f"deep_supervision={bool(cfg.use_deep_supervision or cfg.deep_supervision)}",
        flush=True,
    )
    return {"model": model}


def _parse_arguments():
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(
        "Stage-1 unimodal pretraining",
        add_help=True,
        parents=parents,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--stage1_modality",
        type=str,
        required=True,
        choices=("ct", "pet"),
        help="Train either the CT or PET unimodal expert.",
    )
    cfg = SegMDTConfig()
    parser.parse_args(namespace=cfg)
    cfg._ensure_hash()
    return cfg


def _loaders(cfg):
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root,
        cfg.image_size_2d,
        cfg.batch_size,
        cfg.num_workers,
        cfg.random_state,
        cfg.pin_memory,
        cfg.aug_mode,
        cfg.norm_mode,
        cfg.train_split_file,
        cfg.val_split_file,
        cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )


def _assert_stage1_protocol(cfg):
    assert cfg.accumulation_steps == 1
    assert float(cfg.train_pet_drop_prob) == 0.0
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert float(cfg.boundary_loss_weight) == 0.0
    assert str(cfg.optimizer).lower() == "adamw"
    assert cfg.stage1_modality in ("ct", "pet")


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _checkpoint_paths(checkpoint_dir, modality):
    if modality == "ct":
        return {
            "best": os.path.join(checkpoint_dir, "ckpt.best_ct.pth.tar"),
            "last": os.path.join(checkpoint_dir, "ckpt.last.pth.tar"),
        }
    return {
        "best": os.path.join(checkpoint_dir, "ckpt.best_pet.pth.tar"),
        "last": os.path.join(checkpoint_dir, "ckpt.last.pth.tar"),
    }


def main():
    print("[INFO] starting Stage-1 unimodal pretraining", flush=True)
    cfg = _parse_arguments()
    _assert_stage1_protocol(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, "config_args.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _ = _loaders(cfg)
    print(
        f"[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}",
        flush=True,
    )

    task = UnimodalTeacher(_build_model(cfg)["model"], cfg)
    total_params, trainable_params = _count_parameters(task.model)
    print(
        f"[INFO] params_total={total_params} params_trainable={trainable_params}",
        flush=True,
    )
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    metric_prefix = cfg.stage1_modality
    extra_headers = [
        f"train_{metric_prefix}_loss",
        "train_batches",
        f"val_{metric_prefix}_loss",
        f"val_{metric_prefix}_dice",
        f"val_{metric_prefix}_iou",
        f"val_{metric_prefix}_acc",
        f"val_{metric_prefix}_acc_pixel",
        f"val_{metric_prefix}_hd95",
        f"best_{metric_prefix}_dice",
        f"best_{metric_prefix}_epoch",
        "grad_encoder",
        "grad_decoder",
        "epoch_time",
    ]
    train_log_path = os.path.join(cfg.checkpoint_dir, "train_log.csv")
    init_train_log(train_log_path, extra_headers=extra_headers)

    paths = _checkpoint_paths(cfg.checkpoint_dir, cfg.stage1_modality)
    best_dice = -1.0
    best_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        train_loss_sum = 0.0
        train_batch_count = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        encoder_grads = []
        decoder_grads = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step(batch, modality=cfg.stage1_modality)
            if not torch.isfinite(loss):
                raise RuntimeError("loss became non-finite")
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            if cfg.stage1_modality == "ct":
                encoder_grads.append(float(sum((p.grad.detach().float().pow(2).sum() for p in task.model.enc_ct.parameters() if p.grad is not None), torch.tensor(0.0)).sqrt().item()) if any(p.grad is not None for p in task.model.enc_ct.parameters()) else 0.0)
            else:
                encoder_grads.append(float(sum((p.grad.detach().float().pow(2).sum() for p in task.model.enc_pet.parameters() if p.grad is not None), torch.tensor(0.0)).sqrt().item()) if any(p.grad is not None for p in task.model.enc_pet.parameters()) else 0.0)
            decoder_grads.append(float(sum((p.grad.detach().float().pow(2).sum() for p in task.model.decoder.parameters() if p.grad is not None), torch.tensor(0.0)).sqrt().item()) if any(p.grad is not None for p in task.model.decoder.parameters()) else 0.0)

            if float(cfg.grad_clip) > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    task.trainable_parameters(),
                    float(cfg.grad_clip),
                )
                grad_norm_sum += float(total_grad_norm)
                grad_norm_count += 1

            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()

            task.scheduler.step()
            train_loss_sum += float(loss.detach())
            train_batch_count += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[BATCH {batch_idx + 1}] modality={cfg.stage1_modality} "
                    f"loss={float(loss.detach()):.6f}",
                    flush=True,
                )

        val_metrics = task.evaluate(val_loader, modality=cfg.stage1_modality)
        improved = val_metrics["dice"] > best_dice
        if improved:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if improved:
            task.save_checkpoint(paths["best"], epoch, best_dice, best_epoch, val_metrics)
        task.save_checkpoint(paths["last"], epoch, best_dice, best_epoch, val_metrics)

        train_loss = train_loss_sum / max(1, train_batch_count)
        avg_grad_norm = grad_norm_sum / max(1, grad_norm_count)
        extra_metrics = {
            f"train_{metric_prefix}_loss": train_loss,
            "train_batches": train_batch_count,
            f"val_{metric_prefix}_loss": val_metrics["total_loss"],
            f"val_{metric_prefix}_dice": val_metrics["dice"],
            f"val_{metric_prefix}_iou": val_metrics["iou"],
            f"val_{metric_prefix}_acc": val_metrics["acc"],
            f"val_{metric_prefix}_acc_pixel": val_metrics.get("acc_pixel", 0.0),
            f"val_{metric_prefix}_hd95": val_metrics["hd95"],
            f"best_{metric_prefix}_dice": best_dice,
            f"best_{metric_prefix}_epoch": best_epoch,
            "grad_encoder": float(np.mean(encoder_grads)) if encoder_grads else 0.0,
            "grad_decoder": float(np.mean(decoder_grads)) if decoder_grads else 0.0,
            "epoch_time": time.time() - epoch_start,
        }
        append_epoch_log(
            train_log_path,
            epoch,
            train_loss,
            val_metrics,
            lr=task.optimizer.param_groups[0]["lr"],
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(
            f"[EPOCH {epoch}] modality={cfg.stage1_modality} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"best_dice={best_dice:.4f} "
            f"lr={task.optimizer.param_groups[0]['lr']:.8f}",
            flush=True,
        )
        if no_improve >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs", flush=True)
            break

    if not os.path.isfile(paths["best"]):
        raise RuntimeError("best Stage-1 checkpoint was not created")
    print("done", flush=True)


if __name__ == "__main__":
    main()
