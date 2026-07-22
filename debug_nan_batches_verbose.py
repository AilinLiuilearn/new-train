import os
import torch
import numpy as np

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import PCLT20KSegDataset, _records_from_ids, _read_list
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


def sample_stats(sample):
    out = {}
    for k in ("ct", "pet", "mask"):
        if k in sample and sample[k] is not None:
            out[k] = tensor_stats(sample[k])
    out["image_id"] = sample.get("image_id", None)
    out["case_id"] = sample.get("case_id", None)
    out["slice_id"] = sample.get("slice_id", None)
    out["idx"] = sample.get("idx", None)
    return out


def main():
    print("[INFO] start verbose debug scan")
    print("[INFO] torch:", torch.__version__)
    print("[INFO] numpy:", np.__version__)
    print("[INFO] cwd:", os.getcwd())

    cfg = SegMDTConfig.parse_arguments()

    # 强制用单进程 dataloader，方便定位错误
    train_ids = _read_list(os.path.join(cfg.root, cfg.train_split_file))
    if train_ids is None:
        raise FileNotFoundError("train split file not found")

    train_records = _records_from_ids(cfg.root, train_ids)
    ds = PCLT20KSegDataset(
        train_records,
        image_size=cfg.image_size_2d,
        train=True,
        random_state=cfg.random_state,
        aug_mode=cfg.aug_mode,
        norm_mode=cfg.norm_mode,
    )

    networks = build_mdt_seg_teacher(cfg)
    task = MDTSegTeacher(networks, cfg)
    model = task.model
    model.train()

    print(f"[INFO] dataset_size={len(ds)}")
    print(f"[INFO] batch_size={cfg.batch_size}")
    print(f"[INFO] device={task.device}")

    # 先逐样本扫一遍前 200 个样本，看看单样本是否已经异常
    limit = min(len(ds), 200)
    print(f"[INFO] scanning first {limit} samples individually")

    for idx in range(limit):
        try:
            sample = ds[idx]
            stats = sample_stats(sample)
            bad = False
            for key in ("ct", "pet", "mask"):
                if key in stats and stats[key]["nan"] or stats[key]["inf"]:
                    bad = True
            if bad:
                print("\n" + "=" * 80)
                print(f"[BAD SAMPLE] idx={idx}")
                print(stats)
                print("[STOP] found a bad raw sample before batching")
                return
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"[ERROR] sample idx={idx} failed")
            print(f"[ERROR] {type(e).__name__}: {e}")
            return

    print("[INFO] no bad raw sample found in first 200 samples")

    # 然后再跑 batch 级前向
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    for batch_idx, batch in enumerate(loader):
        route = "full" if batch_idx % 2 == 0 else "missing"

        ct = batch["ct"]
        mask = batch["mask"]
        pet = batch["pet"] if route == "full" else None

        print("\n" + "=" * 80)
        print(f"[BATCH {batch_idx}] route={route}")

        print("[CT  ]", tensor_stats(ct))
        print("[MASK]", tensor_stats(mask))
        if route == "full":
            print("[PET ]", tensor_stats(pet))

        print("[IDX ]", batch.get("idx", None))
        print("[CID ]", batch.get("case_id", None))
        print("[IID ]", batch.get("image_id", None))
        print("[SID ]", batch.get("slice_id", None))

        try:
            with torch.no_grad():
                loss, logits, outputs, stats = task.train_step(batch, forward_mode=route)
            print("[OK] loss =", float(loss.item()))
            print("[OK] logits =", tuple(logits.shape))
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"[MODEL ERROR] batch {batch_idx} route={route}")
            print(f"[MODEL ERROR] {type(e).__name__}: {e}")
            print("[DETAIL] ct stats:", tensor_stats(ct))
            print("[DETAIL] mask stats:", tensor_stats(mask))
            if route == "full":
                print("[DETAIL] pet stats:", tensor_stats(pet))
            print("[DETAIL] idx:", batch.get("idx", None))
            print("[DETAIL] case_id:", batch.get("case_id", None))
            print("[DETAIL] image_id:", batch.get("image_id", None))
            print("[DETAIL] slice_id:", batch.get("slice_id", None))
            break

        if batch_idx >= 80:
            print("[STOP] reached batch scan limit 80")
            break

    print("[DONE]")


if __name__ == "__main__":
    main()