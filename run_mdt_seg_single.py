# -*- coding: utf-8 -*-
"""Single-modality CT/PET segmentation baseline with one encoder and light UNet decoder."""

import math
import os
import sys

import torch
import torch.nn as nn

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import (
    LightConcatUNetDecoder,
    create_feature_backbone,
    load_local_weights_safe,
)
from run_mdt_seg import _build_loaders, _prepare_env, _save_config
from tasks.mdt_seg import MDTSegTeacher
from utils.model_profile import print_baseline_profile
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log
from utils.vis_teacher import save_segmentation_diagnostics


class SingleModalitySegConfig(SegMDTConfig):
    @staticmethod
    def model_parser():
        parser = SegMDTConfig.model_parser()
        parser.add_argument(
            "--single_modality",
            type=str,
            default="ct",
            choices=("ct", "pet"),
            help="Single modality baseline: use only CT or PET with one encoder.",
        )
        return parser


class SingleModalityUNet(nn.Module):
    def __init__(self, backbone="mit-b1", pretrained_path=None, modality="ct", out_channels=1):
        super().__init__()
        if modality not in ("ct", "pet"):
            raise ValueError(f"Unsupported single modality: {modality}. Use 'ct' or 'pet'.")

        self.modality = modality
        self.encoder = create_feature_backbone(backbone, in_channels=3)

        if pretrained_path:
            load_local_weights_safe(
                self.encoder,
                pretrained_path,
                name=f"Single_{modality.upper()}_Encoder",
            )

        enc_channels = self.encoder.feature_info.channels()
        self.decoder = LightConcatUNetDecoder(enc_channels, out_channels=out_channels)

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        x = ct if self.modality == "ct" else pet
        x = self._to_3ch(x)
        feats = self.encoder(x)

        if target_size is None:
            target_size = x.shape[-2:]

        return self.decoder(feats, target_size)

    def set_epoch(self, epoch):
        return None

    def get_fusion_visuals(self):
        return {}


def build_single_modality_teacher(config):
    model = SingleModalityUNet(
        backbone=getattr(config, "backbone", "mit-b1"),
        pretrained_path=getattr(config, "pretrained_path", None),
        modality=getattr(config, "single_modality", "ct"),
        out_channels=1,
    )
    return {"model": model}


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sys.path.insert(0, os.getcwd())
    sys.modules.pop("datasets", None)

    config = SingleModalitySegConfig.parse_arguments()
    config.task = "MDT_Teacher"
    config.decoder_type = "light"
    config.fusion_type = "single_" + config.single_modality

    g0 = _prepare_env(config)

    print(
        f"GPU={g0} backbone={config.backbone} "
        f"single_modality={config.single_modality} decoder=light"
    )
    print(f"lr={config.learning_rate} wd={config.weight_decay} bs={config.batch_size}")

    _save_config(config)
    train_loader, val_loader, test_loader = _build_loaders(config)

    networks = build_single_modality_teacher(config)

    print("\n" + "=" * 30 + " MODEL PROFILE " + "=" * 30)
    print_baseline_profile(networks, config)
    print("=" * 75 + "\n")

    task = MDTSegTeacher(networks, config)

    spe = len(train_loader)
    accum_iter = max(1, int(getattr(config, "accumulation_steps", 1)))
    updates_per_epoch = math.ceil(spe / accum_iter)

    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        config.epochs,
        warmup_steps=config.cosine_warmup * updates_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=updates_per_epoch,
        flat_ratio=getattr(config, "lr_flat_ratio", 0.3),
    )

    log_path = os.path.join(config.checkpoint_dir, "train_log.csv")
    init_train_log(log_path)

    grad_clip = getattr(config, "grad_clip", 0.5)
    clip_params = [p for net in task.networks.values() for p in net.parameters()]

    best_dice = -1.0
    best_hd95 = float("inf")
    no_improve = 0
    patience = getattr(config, "early_stop_patience", 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tn = 0.0, 0

        if hasattr(task.networks.get("model"), "set_epoch"):
            task.networks["model"].set_epoch(epoch)

        task.set_epoch(epoch)
        task.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            stepped = False

            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                loss, _, _, loss_dict = task.train_step(batch)
                loss = loss / accum_iter

            if task.scaler:
                task.scaler.scale(loss).backward()

                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        task.scaler.unscale_(task.optimizer)
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)

                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    task.optimizer.zero_grad()
                    stepped = True
            else:
                loss.backward()

                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)

                    task.optimizer.step()
                    task.optimizer.zero_grad()
                    stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            tloss += loss.item() * accum_iter
            tn += 1

            if (i + 1) % 50 == 0:
                curr_lr = task.optimizer.param_groups[0]["lr"]
                print(
                    f"  Ep{epoch}[{i + 1}/{spe}] "
                    f"loss={loss.item() * accum_iter:.4f} "
                    f"seg={loss_dict['loss_seg'].item():.4f} "
                    f"lr={curr_lr:.6f}"
                )

        val_m = task.evaluate(val_loader)
        append_epoch_log(log_path, epoch, tloss / max(tn, 1), val_m)

        print(
            "Epoch {} loss={:.4f} Dice={:.4f} IoU={:.4f} HD95={:.2f}".format(
                epoch,
                tloss / max(tn, 1),
                val_m["dice"],
                val_m["iou"],
                val_m["hd95"],
            )
        )

        if getattr(config, "vis_every_epoch", False):
            save_segmentation_diagnostics(
                task=task,
                loader=val_loader,
                out_dir=os.path.join(config.checkpoint_dir, "vis_epochs", f"epoch_{epoch:03d}"),
                num_samples=max(1, int(getattr(config, "vis_epoch_samples", 2))),
                threshold=getattr(config, "eval_threshold", 0.5),
            )

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            no_improve = 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, "ckpt.best.pth.tar"), epoch)
            task.save_checkpoint(os.path.join(config.checkpoint_dir, "ckpt.best_dice.pth.tar"), epoch)
        else:
            no_improve += 1

        if val_m["hd95"] < best_hd95:
            best_hd95 = val_m["hd95"]
            task.save_checkpoint(os.path.join(config.checkpoint_dir, "ckpt.best_hd95.pth.tar"), epoch)

        if patience > 0 and no_improve >= patience:
            print("Early stop at epoch", epoch)
            break

    task.save_checkpoint(os.path.join(config.checkpoint_dir, "ckpt.last.pth.tar"), epoch)

    def _load_checkpoint(path):
        ckpt = torch.load(path, map_location="cpu")

        for k, v in task.networks.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k], strict=False)

    best_dice_path = os.path.join(config.checkpoint_dir, "ckpt.best_dice.pth.tar")
    best_hd95_path = os.path.join(config.checkpoint_dir, "ckpt.best_hd95.pth.tar")

    _load_checkpoint(best_dice_path)
    test_m_dice = task.evaluate(test_loader)

    print(
        "\n=== TEST(best_dice) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ===".format(
            test_m_dice["dice"],
            test_m_dice["iou"],
            test_m_dice["acc"],
            test_m_dice["hd95"],
        )
    )

    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, "vis_best_dice"),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, "eval_threshold", 0.5),
    )

    _load_checkpoint(best_hd95_path)
    test_m_hd95 = task.evaluate(test_loader)

    print(
        "=== TEST(best_hd95) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ===".format(
            test_m_hd95["dice"],
            test_m_hd95["iou"],
            test_m_hd95["acc"],
            test_m_hd95["hd95"],
        )
    )

    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, "vis_best_hd95"),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, "eval_threshold", 0.5),
    )


if __name__ == "__main__":
    main()
