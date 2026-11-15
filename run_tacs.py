"""Train the isolated Full-only PET-CT TACS fusion experiment."""
import json
import os
import time

import torch

from configs.tacs import TACSConfig
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.full_tacs import build_tacs_model
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import seed_everything, optimizer_step_succeeded
from utils.train_logger import init_train_log, append_epoch_log


def loaders(cfg):
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir, ct_only=False,
        allow_val_equals_test=cfg.allow_val_equals_test,
    )


def main():
    cfg = TACSConfig.parse_arguments()
    if (cfg.ct_backbone, cfg.pet_backbone) != ("convnextv2_nano", "mit_b1"):
        raise ValueError("TACS comparison fixes CT=ConvNeXtV2-Nano and PET=MiT-B1")
    directory = cfg.checkpoint_dir
    if os.path.exists(os.path.join(directory, "config_args.json")):
        raise FileExistsError("existing run: choose a new --hash")
    seed_everything(cfg.random_state)
    cfg.experiment_type = "full_tacs_channel_spatial_text"
    cfg.fusion_type = "TACSFusion"
    cfg.save()
    with open(os.path.join(directory, "config_args.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train, val, test = loaders(cfg)
    if not len(train):
        raise ValueError("empty training loader")
    task = MDTSegTeacher(build_tacs_model(cfg), cfg)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train),
        steps_per_epoch=len(train), min_lr=cfg.cosine_min_lr,
        flat_ratio=cfg.lr_flat_ratio,
    )

    log = os.path.join(directory, "train_log.csv")
    diag_headers = []
    for i in range(1, 5):
        diag_headers.extend((
            f"tacs_delta_ratio_s{i}",
            f"tacs_offset_abs_s{i}",
            f"tacs_selected_channels_s{i}",
        ))
    init_train_log(
        log,
        extra_headers=["val_full_dice", "ema_updates", "skipped_updates", "epoch_time"] + diag_headers,
    )

    best, best_epoch, stale = -1.0, 0, 0
    best_path = os.path.join(directory, "ckpt.best.pth.tar")
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        task.model.train()
        task.begin_epoch(epoch)
        task.model.fusion.reset_diagnostics()
        loss_sum = samples = skipped = 0
        grad_sum = 0.0
        for batch in train:
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=bool(cfg.mixed_precision) and task.device.type == "cuda"
            ):
                loss, _ = task.train_step(batch)
            task.scaler.scale(loss).backward()
            task.scaler.unscale_(task.optimizer)
            if cfg.grad_clip > 0:
                grad_sum += float(torch.nn.utils.clip_grad_norm_(
                    task.trainable_parameters(), cfg.grad_clip
                ))
            if optimizer_step_succeeded(task):
                task.scheduler.step()
                task.update_ema()
            else:
                skipped += 1
            task.global_batch_step += 1
            n = len(batch["ct"])
            samples += n
            loss_sum += float(loss.detach()) * n

        diagnostics = task.model.fusion.pop_diagnostics()
        result = task.evaluate(val, "full", model=task.eval_model())
        score = result["dice"]
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            task.save_checkpoint(best_path, epoch, best, best_epoch, result)
        else:
            stale += 1
        task.save_checkpoint(
            os.path.join(directory, "ckpt.last.pth.tar"),
            epoch, best, best_epoch, result,
        )
        append_epoch_log(
            log, epoch, loss_sum / max(1, samples), result,
            lr=task.optimizer.param_groups[0]["lr"],
            grad_norm=grad_sum / max(1, len(train)),
            extra_metrics={
                "val_full_dice": score,
                "ema_updates": task.ema.updates if task.ema else 0,
                "skipped_updates": skipped,
                "epoch_time": time.time() - start,
                **diagnostics,
            },
        )
        delta_summary = [
            round(diagnostics[f"tacs_delta_ratio_s{i}"], 5) for i in range(1, 5)
        ]
        print(
            f"[epoch {epoch}] full_dice={score:.4f} best={best:.4f} "
            f"lr={task.optimizer.param_groups[0]['lr']:.8f} "
            f"tacs_delta={delta_summary}",
            flush=True,
        )
        if stale >= cfg.early_stop_patience:
            break

    checkpoint = task.load_state_dicts(best_path)
    tag = checkpoint["eval_weights"]
    task.model.load_state_dict(
        checkpoint["model_ema"] if tag == "ema" else checkpoint["model"]
    )
    final_metrics = task.evaluate(test, "full", model=task.model)
    final = {
        "experiment_type": cfg.experiment_type,
        "epoch": checkpoint["epoch"],
        "weights": tag,
        "selection": "full",
        "metrics": final_metrics,
        "tacs_channel_ratio": cfg.tacs_channel_ratio,
        "tacs_use_text": cfg.tacs_use_text,
        "tacs_text_embedding_path": cfg.tacs_text_embedding_path,
    }
    with open(os.path.join(directory, "final_test.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(final, flush=True)


if __name__ == "__main__":
    main()
