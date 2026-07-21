# -*- coding: utf-8 -*-
"""CT-only lower-bound experiment for e1-joint-baseline-rebuild.

Only the model is changed relative to run_mdt_seg.py:

    CT -> ConvNeXtV2-Nano -> four-stage channel alignment
       -> original UNetStyleDecoder -> segmentation logits

The PET encoder and AddFusion are not instantiated.  The original data loader,
BCEDiceLoss, AdamW optimizer, cosine schedule, AMP, gradient clipping, early
stopping, metrics and checkpoint format are reused.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from configs.seg_mdt import SegMDTConfig
from models.baseline_blocks import (
    UNetStyleDecoder,
    _check_tensor,
    _check_tensor_list,
)
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_shared_add_baseline import StageChannelAlign
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log


# In the original dual-modal baseline, MiT-B1 defines the feature channels that
# enter the shared decoder.  Keeping these values preserves the exact decoder
# structure and capacity after PET is removed.
DECODER_INPUT_CHANNELS = (64, 128, 320, 512)


class CTOnlySegmentationModel(nn.Module):
    """ConvNeXtV2-Nano CT encoder plus the baseline's shared decoder."""

    def __init__(
        self,
        ct_backbone="convnextv2_nano",
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)

        # CT is the only encoder instantiated in this experiment.
        self.enc_ct = create_feature_backbone(
            ct_backbone,
            in_channels=in_channels,
        )
        load_local_weights_safe(
            self.enc_ct,
            ct_pretrained_path,
            name="CT_Encoder",
        )

        ct_channels = list(self.enc_ct.feature_info.channels())
        decoder_input_channels = list(DECODER_INPUT_CHANNELS)

        # This is the same StageChannelAlign used by the joint baseline before
        # SUM fusion.  Here its outputs go directly to the decoder skips.
        self.ct_align = StageChannelAlign(
            ct_channels,
            decoder_input_channels,
        )
        self.decoder = UNetStyleDecoder(
            decoder_input_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list("ct_feats", ct_feats)
        aligned_ct = self.ct_align(ct_feats)
        _check_tensor_list("aligned_ct", aligned_ct)
        return aligned_ct

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode="missing",
    ):
        # These arguments are retained only for compatibility with
        # MDTSegTeacher.  They never affect this CT-only forward pass.
        del pet, pet_available, forward_mode

        if target_size is None:
            target_size = ct.shape[-2:]

        out = self.decoder(self._encode_ct(ct), target_size)
        _check_tensor("logits", out["logits"])
        out["pred"] = out["logits"]
        out["aux"] = {}
        return out


def build_ct_only_model(cfg):
    model = CTOnlySegmentationModel(
        ct_backbone=cfg.ct_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path,
        in_channels=3,
        out_channels=1,
        decoder_channels=cfg.decoder_channels,
        use_deep_supervision=bool(
            cfg.use_deep_supervision or cfg.deep_supervision
        ),
    )
    print(
        f"[ct_only_baseline] ct={cfg.ct_backbone} "
        f"decoder_input_channels={list(DECODER_INPUT_CHANNELS)} "
        "fusion=none pet_encoder=none shared_decoder=UNetStyleDecoder "
        f"deep_supervision={bool(cfg.use_deep_supervision or cfg.deep_supervision)}",
        flush=True,
    )
    return {"model": model}


def _parse_arguments():
    """Use the branch's original config and add only run-control arguments."""
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(
        "CT-only lower-bound segmentation",
        add_help=True,
        parents=parents,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate a saved CT-only checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint used by --eval_only. Defaults to ckpt.best_ct.pth.tar in checkpoint_dir.",
    )
    cfg = SegMDTConfig()
    parser.parse_args(namespace=cfg)
    cfg._ensure_hash()
    return cfg


def _seed(cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned

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


def _assert_ct_only_protocol(cfg):
    normalized_backbone = str(cfg.ct_backbone).lower().replace("-", "_")
    assert normalized_backbone == "convnextv2_nano", (
        "This lower-bound experiment is fixed to ConvNeXtV2-Nano, got "
        f"{cfg.ct_backbone!r}."
    )
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert float(cfg.boundary_loss_weight) == 0.0
    assert str(cfg.optimizer).lower() == "adamw"


def module_grad_norm(module):
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return float(total.sqrt().item()) if total is not None else 0.0


def _count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _checkpoint_paths(checkpoint_dir):
    return {
        "best_ct": os.path.join(checkpoint_dir, "ckpt.best_ct.pth.tar"),
        "last": os.path.join(checkpoint_dir, "ckpt.last.pth.tar"),
    }


def _torch_load(path, map_location="cpu"):
    # PyTorch 2.6 changed the default of weights_only.  This checkpoint also
    # contains RNG states, so request the historical full-checkpoint behavior.
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_ct_checkpoint(task, path, epoch, best_ct, best_ct_epoch, val_ct):
    # Reuse the original checkpoint writer and schema.  For a CT-only model,
    # Full and Missing are the same prediction, so all legacy score fields are
    # filled with the same CT validation value.
    task.save_checkpoint(
        path,
        epoch,
        best_joint=best_ct,
        best_full=best_ct,
        best_missing=best_ct,
        best_joint_epoch=best_ct_epoch,
        val_full=val_ct,
        val_missing=val_ct,
        joint_dice=val_ct["dice"],
    )


def _save_test_metrics(checkpoint_dir, checkpoint_path, metrics):
    result = {
        "model": "ct_only_convnextv2_nano",
        "checkpoint": os.path.abspath(checkpoint_path),
        **{key: float(value) for key, value in metrics.items()},
    }

    json_path = os.path.join(checkpoint_dir, "final_test_ct_only.json")
    csv_path = os.path.join(checkpoint_dir, "final_test_ct_only.csv")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print("[TEST CT-ONLY]", json.dumps(result, indent=2), flush=True)
    print(f"[INFO] test metrics saved to {json_path}", flush=True)


def _evaluate_test(task, test_loader, checkpoint_path, checkpoint_dir):
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    task.model.load_state_dict(checkpoint["model"], strict=True)
    task.model.to(task.device)
    task.model.eval()

    # fixed_missing is used deliberately: MDTSegTeacher then passes pet=None.
    # The CT-only model never receives or encodes a PET tensor.
    test_ct = task.evaluate(
        test_loader,
        eval_mode="fixed_missing",
        tag="test_ct_only",
    )
    _save_test_metrics(
        checkpoint_dir,
        checkpoint_path,
        test_ct,
    )
    return test_ct


def main():
    print("[INFO] starting CT-only lower-bound experiment", flush=True)
    cfg = _parse_arguments()
    _assert_ct_only_protocol(cfg)
    _seed(cfg)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(cfg.checkpoint_dir, "config_args.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vars(cfg), file, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(
        f"[INFO] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    task = MDTSegTeacher(build_ct_only_model(cfg), cfg)
    total_params, trainable_params = _count_parameters(task.model)
    print(
        f"[INFO] params_total={total_params} "
        f"params_trainable={trainable_params}",
        flush=True,
    )

    paths = _checkpoint_paths(cfg.checkpoint_dir)
    if cfg.eval_only:
        checkpoint_path = cfg.checkpoint_path or paths["best_ct"]
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        _evaluate_test(
            task,
            test_loader,
            checkpoint_path,
            cfg.checkpoint_dir,
        )
        return

    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        "train_ct_loss",
        "train_batches",
        "val_ct_loss",
        "val_ct_dice",
        "val_ct_iou",
        "val_ct_acc",
        "val_ct_acc_pixel",
        "val_ct_hd95",
        "best_ct_dice",
        "best_ct_epoch",
        "grad_enc_ct",
        "grad_ct_align",
        "grad_decoder",
        "epoch_time",
    ]
    train_log_path = os.path.join(cfg.checkpoint_dir, "train_log.csv")
    init_train_log(train_log_path, extra_headers=extra_headers)

    best_ct = -1.0
    best_ct_epoch = 0
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
        grads = {
            "enc_ct": [],
            "ct_align": [],
            "decoder": [],
        }
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)

            # Every batch performs exactly one CT-only forward.  Passing the
            # missing route keeps PET on CPU and out of the model.
            with torch.cuda.amp.autocast(
                enabled=amp_enabled and torch.cuda.is_available()
            ):
                loss, _, _, _ = task.train_step(
                    batch,
                    forward_mode="missing",
                )

            if not torch.isfinite(loss):
                raise RuntimeError("loss became non-finite")

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            grads["enc_ct"].append(module_grad_norm(task.model.enc_ct))
            grads["ct_align"].append(module_grad_norm(task.model.ct_align))
            grads["decoder"].append(module_grad_norm(task.model.decoder))

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
                    f"[BATCH {batch_idx + 1}] route=ct_only "
                    f"loss={float(loss.detach()):.6f}",
                    flush=True,
                )

        val_ct = task.evaluate(
            val_loader,
            eval_mode="fixed_missing",
            tag="val_ct_only",
        )

        improved = val_ct["dice"] > best_ct
        if improved:
            best_ct = val_ct["dice"]
            best_ct_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if improved:
            _save_ct_checkpoint(
                task,
                paths["best_ct"],
                epoch,
                best_ct,
                best_ct_epoch,
                val_ct,
            )
        _save_ct_checkpoint(
            task,
            paths["last"],
            epoch,
            best_ct,
            best_ct_epoch,
            val_ct,
        )

        train_loss = train_loss_sum / max(1, train_batch_count)
        avg_grad_norm = grad_norm_sum / max(1, grad_norm_count)
        extra_metrics = {
            "train_ct_loss": train_loss,
            "train_batches": train_batch_count,
            "val_ct_loss": val_ct["total_loss"],
            "val_ct_dice": val_ct["dice"],
            "val_ct_iou": val_ct["iou"],
            "val_ct_acc": val_ct["acc"],
            "val_ct_acc_pixel": val_ct.get("acc_pixel", 0.0),
            "val_ct_hd95": val_ct["hd95"],
            "best_ct_dice": best_ct,
            "best_ct_epoch": best_ct_epoch,
            "grad_enc_ct": float(np.mean(grads["enc_ct"]))
            if grads["enc_ct"]
            else 0.0,
            "grad_ct_align": float(np.mean(grads["ct_align"]))
            if grads["ct_align"]
            else 0.0,
            "grad_decoder": float(np.mean(grads["decoder"]))
            if grads["decoder"]
            else 0.0,
            "epoch_time": time.time() - epoch_start,
        }
        append_epoch_log(
            train_log_path,
            epoch,
            train_loss,
            val_ct,
            lr=task.optimizer.param_groups[0]["lr"],
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(
            f"[EPOCH {epoch}] val_ct_dice={val_ct['dice']:.4f} "
            f"val_ct_iou={val_ct['iou']:.4f} "
            f"val_ct_hd95={val_ct['hd95']:.4f} "
            f"best_ct_dice={best_ct:.4f} "
            f"lr={task.optimizer.param_groups[0]['lr']:.8f}",
            flush=True,
        )

        if no_improve >= patience:
            print(
                f"[EARLY STOP] no improvement for {patience} epochs",
                flush=True,
            )
            break

    if not os.path.isfile(paths["best_ct"]):
        raise RuntimeError("best CT-only checkpoint was not created")

    _evaluate_test(
        task,
        test_loader,
        paths["best_ct"],
        cfg.checkpoint_dir,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
