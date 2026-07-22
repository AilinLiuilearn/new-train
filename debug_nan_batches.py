import os
import torch
import numpy as np

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def tensor_stats(x):
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.float().mean().item()),
        "nan": bool(torch.isnan(x).any().item()),
        "inf": bool(torch.isinf(x).any().item()),
    }


def main():
    print("[INFO] start debug scan")
    print("[INFO] torch:", torch.__version__)
    print("[INFO] numpy:", np.__version__)
    print("[INFO] cwd:", os.getcwd())

    cfg = SegMDTConfig.parse_arguments()

    train_loader, _, _ = get_pclt20k_loaders_cipa_aligned(
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

    networks = build_mdt_seg_teacher(cfg)
    task = MDTSegTeacher(networks, cfg)
    model = task.model

    model.train()

    print(f"[INFO] num_batches={len(train_loader)}")
    print(f"[INFO] device={task.device}")

    for batch_idx, batch in enumerate(train_loader):
        route = "full" if batch_idx % 2 == 0 else "missing"

        ct = batch["ct"]
        mask = batch["mask"]
        pet = batch["pet"] if route == "full" else None

        print("\n" + "=" * 80)
        print(f"[BATCH {batch_idx}] route={route}")

        ct_stat = tensor_stats(ct)
        mask_stat = tensor_stats(mask)
        print("[CT  ]", ct_stat)
        print("[MASK]", mask_stat)

        if route == "full":
            pet_stat = tensor_stats(pet)
            print("[PET ]", pet_stat)

        try:
            with torch.no_grad():
                loss, logits, outputs, stats = task.train_step(batch, forward_mode=route)

            logits_stat = tensor_stats(logits)
            print("[LOGITS]", logits_stat)
            print("[LOSS ]", float(loss.item()))
            print("[STATS]", {k: float(v.item()) if torch.is_tensor(v) else v for k, v in stats.items()})

            if not torch.isfinite(loss):
                print(f"[ALERT] loss is non-finite at batch {batch_idx}")
                break
            if logits_stat["nan"] or logits_stat["inf"]:
                print(f"[ALERT] logits have invalid values at batch {batch_idx}")
                break

        except Exception as e:
            print(f"[ERROR] batch {batch_idx} failed")
            print(f"[ERROR] {type(e).__name__}: {e}")
            break

        if batch_idx >= 80:
            print("[STOP] reached scan limit 80 batches")
            break

    print("[DONE]")


if __name__ == "__main__":
    main()