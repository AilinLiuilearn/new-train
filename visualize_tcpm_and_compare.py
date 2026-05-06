# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher


def parse_args():
    p = argparse.ArgumentParser("Visualize BiomedCLIP-TCPM internals and old qualitative results")
    p.add_argument("--mode", choices=("tcpm", "old_vis", "both"), default="both")
    p.add_argument("--exp_dir", type=str, default="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/2026-05-05_22-53-31")
    p.add_argument("--old_vis_dir", type=str, default="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/2026-05-05_18-08-10/vis_epochs")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(exp_dir, gpu):
    cfg_path = os.path.join(exp_dir, "config_args.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            data = json.load(f)
        cfg = SegMDTConfig(args=data)
    else:
        cfg = SegMDTConfig()
    cfg.gpus = [str(gpu)]
    cfg.task = "MDT_Teacher"
    cfg.mixed_precision = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.vis_every_epoch = False
    return cfg


def load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, "datasets", "pclt20k_seg.py")
    spec = importlib.util.spec_from_file_location("local_pclt20k_seg", dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_loader(cfg, split):
    dataset_mod = load_dataset_module()
    if getattr(cfg, "cipa_aligned", False):
        loaders = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            cfg.root, cfg.image_size_2d, 1, 0, cfg.random_state, pin_memory=False
        )
    else:
        loaders = dataset_mod.get_pclt20k_loaders(
            cfg.root, cfg.image_size_2d, 1, 0, val_ratio=cfg.val_ratio,
            random_state=cfg.random_state, use_case_split=getattr(cfg, "use_case_split", True), pin_memory=False
        )
    return dict(train=loaders[0], val=loaders[1], test=loaders[2])[split]


def find_ckpt(exp_dir, explicit=None):
    if explicit:
        return explicit
    for name in ("ckpt.best_dice.pth.tar", "ckpt.best.pth.tar", "ckpt.last.pth.tar", "ckpt.best_hd95.pth.tar"):
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_model(cfg, ckpt_path, device):
    nets = build_mdt_seg_teacher(cfg)
    model = nets["model"].to(device)
    if ckpt_path is None:
        print("[WARN] No checkpoint found. Visualizing randomly initialized current model.")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("ema_model", ckpt.get("model", ckpt))
        msg = model.load_state_dict(state, strict=False)
        print(f"[+] Loaded checkpoint: {ckpt_path}")
        print(f"[+] Load status: {msg}")
    model.eval()
    return model


def norm01(x):
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    return x / (np.nanmax(x) + 1e-8)


def tensor_map(t, sample_idx=0):
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().float().cpu()
    if t.dim() >= 4:
        arr = t[min(sample_idx, t.shape[0] - 1)].mean(0).numpy()
    elif t.dim() == 3:
        arr = t[min(sample_idx, t.shape[0] - 1)].numpy()
    elif t.dim() == 2:
        arr = t[min(sample_idx, t.shape[0] - 1)][None, :].numpy()
    else:
        arr = t.numpy()
    return norm01(arr)


def to_gray(x):
    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = arr.mean(0)
    else:
        arr = arr.squeeze(0)
    return norm01(arr)


def overlay(gray, mask, color=(0, 1, 0), alpha=0.65):
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = (mask > 0.5)[..., None].astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * (1 - alpha * m) + c * alpha * m, 0, 1)


def make_tcpm_debugger(module, name):
    original_forward = module.forward

    def debug_forward(pet_feature, ct_feature, text_code):
        b, _, h, w = pet_feature.shape
        both = torch.cat([pet_feature, ct_feature], dim=1)
        weight = module.se_fc(module.pool(both).flatten(1))
        idx = torch.topk(weight, module.topk, dim=1).indices[:, :, None, None].expand(-1, -1, h, w)
        selected = both.gather(1, idx)
        img = module.pick_conv(selected)
        text_logits = module.text_fc(text_code)
        tidx = torch.topk(text_logits, 2 * module.dim, dim=1).indices
        shuffled = img[torch.arange(b, device=img.device)[:, None], tidx, :, :]
        q = module.out_conv(shuffled)
        ref = module.out_conv(both)
        att_out = module.attn(module.n1(q), module.n2(ref))
        out = att_out + module.ffn(module.n3(att_out))
        module._tcpm_debug = {
            "pet_before": pet_feature.detach(),
            "ct_before": ct_feature.detach(),
            "concat_before": both.detach(),
            "channel_weight": weight.detach(),
            "selected_before_text": selected.detach(),
            "text_logits": text_logits.detach(),
            "q_text_guided": q.detach(),
            "cross_modal_ref": ref.detach(),
            "cross_attn_out": att_out.detach(),
            "tcpm_after": out.detach(),
        }
        return out, ref

    module.forward = debug_forward
    module._original_forward = original_forward
    module._debug_name = name


def patch_tcpm_modules(model):
    if not hasattr(model, "tcpm_blocks"):
        raise RuntimeError("Current model has no tcpm_blocks. Please check build_mdt_seg.py.")
    for i, m in enumerate(model.tcpm_blocks, 1):
        make_tcpm_debugger(m, f"stage{i}")


def save_tcpm_visuals(model, batch, out_dir, sample_idx, threshold, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ct = batch["ct"].float().to(device)
    pet = batch["pet"].float().to(device)
    mask = batch["mask"].float().to(device)
    with torch.no_grad():
        logit = model(ct, pet, target_size=mask.shape[-2:])
        prob = torch.sigmoid(logit)

    ct_img, pet_img = to_gray(ct), to_gray(pet)
    gt = mask[0].detach().float().cpu().squeeze().numpy()
    pred = prob[0].detach().float().cpu().squeeze().numpy()
    pred_bin = pred > threshold

    panels = [
        ("CT", ct_img, "gray"),
        ("PET", pet_img, "inferno"),
        ("GT on CT", overlay(ct_img, gt, (1, 0, 0)), None),
        ("Prediction prob", norm01(pred), "jet"),
        ("Pred on CT", overlay(ct_img, pred_bin, (0, 1, 0)), None),
    ]

    for i, m in enumerate(model.tcpm_blocks, 1):
        dbg = getattr(m, "_tcpm_debug", {})
        if not dbg:
            continue
        panels.extend([
            (f"S{i} PET before", tensor_map(dbg["pet_before"]), "inferno"),
            (f"S{i} CT before", tensor_map(dbg["ct_before"]), "gray"),
            (f"S{i} selected pre-text", tensor_map(dbg["selected_before_text"]), "magma"),
            (f"S{i} text logits", tensor_map(dbg["text_logits"]), "viridis"),
            (f"S{i} Q text-guided", tensor_map(dbg["q_text_guided"]), "jet"),
            (f"S{i} cross-modal ref", tensor_map(dbg["cross_modal_ref"]), "cividis"),
            (f"S{i} cross-attn out", tensor_map(dbg["cross_attn_out"]), "plasma"),
            (f"S{i} TCPM after skip", tensor_map(dbg["tcpm_after"]), "turbo"),
            (f"S{i} after-before diff", norm01(np.abs(tensor_map(dbg["tcpm_after"]) - tensor_map(dbg["cross_modal_ref"]))), "hot"),
        ])

    ncols = 5
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"tcpm_internal_sample_{sample_idx:03d}.png")
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    print(f"[+] saved {save_path}")


def run_tcpm(args, out_root):
    gpu = int(args.gpus.split()[0]) if isinstance(args.gpus, str) else int(args.gpus[0])
    device = torch.device(args.device or (f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.exp_dir, gpu)
    loader = build_loader(cfg, args.split)
    ckpt = find_ckpt(args.exp_dir, args.ckpt)
    if ckpt is None:
        print(f"[WARN] No ckpt found under {args.exp_dir}. Use --ckpt /path/to/ckpt.best_dice.pth.tar if available.")
    model = load_model(cfg, ckpt, device)
    patch_tcpm_modules(model)

    saved = 0
    for batch in loader:
        save_tcpm_visuals(model, batch, os.path.join(out_root, "tcpm_internal"), saved, args.threshold, device)
        saved += 1
        if saved >= args.num_samples:
            break


def run_old_vis(args, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    old_dir = Path(args.old_vis_dir)
    out_dir = Path(out_root) / "old_vis_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(old_dir.glob("**/*.png"))
    if not pngs:
        print(f"[WARN] No png found in old vis dir: {old_dir}")
        return

    selected = pngs[:args.num_samples]
    for i, p in enumerate(selected):
        dst = out_dir / f"old_{i:03d}_{p.parent.name}_{p.name}"
        shutil.copy2(p, dst)

    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in selected]
    ncols = min(2, len(imgs))
    nrows = int(np.ceil(len(imgs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, img, p in zip(axes.flat, imgs, selected):
        ax.imshow(img)
        ax.set_title(str(p.relative_to(old_dir)), fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(imgs):]:
        ax.axis("off")
    plt.tight_layout()
    save_path = out_dir / "old_vis_contact_sheet.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[+] saved old visualization summary to {save_path}")


def main():
    args = parse_args()
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_root = args.out_dir or os.path.join(args.exp_dir, "tcpm_visualization")
    os.makedirs(out_root, exist_ok=True)
    if args.mode in ("tcpm", "both"):
        run_tcpm(args, out_root)
    if args.mode in ("old_vis", "both"):
        run_old_vis(args, out_root)


if __name__ == "__main__":
    main()
