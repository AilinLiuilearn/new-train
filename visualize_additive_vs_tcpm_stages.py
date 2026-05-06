# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher, LightUNetDecoder, load_local_weights_safe


def parse_args():
    p = argparse.ArgumentParser("Same-sample stage visualization: additive fusion vs TCPM fusion")
    p.add_argument("--tcpm_exp", type=str, default="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/2026-05-05_22-53-31")
    p.add_argument("--add_exp", type=str, default="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/2026-05-05_18-08-10")
    p.add_argument("--tcpm_ckpt", type=str, default=None)
    p.add_argument("--add_ckpt", type=str, default=None)
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
            cfg = SegMDTConfig(args=json.load(f))
    else:
        cfg = SegMDTConfig()
    cfg.gpus = [str(gpu)]
    cfg.task = "MDT_Teacher"
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.mixed_precision = False
    cfg.vis_every_epoch = False
    return cfg


def load_dataset_module():
    dataset_path = os.path.join(os.getcwd(), "datasets", "pclt20k_seg.py")
    spec = importlib.util.spec_from_file_location("local_pclt20k_seg", dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_loader(cfg, split):
    dataset_mod = load_dataset_module()
    if getattr(cfg, "cipa_aligned", False):
        loaders = dataset_mod.get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, 1, 0, cfg.random_state, pin_memory=False)
    else:
        loaders = dataset_mod.get_pclt20k_loaders(
            cfg.root, cfg.image_size_2d, 1, 0, val_ratio=cfg.val_ratio,
            random_state=cfg.random_state, use_case_split=getattr(cfg, "use_case_split", True), pin_memory=False
        )
    return dict(train=loaders[0], val=loaders[1], test=loaders[2])[split]


def auto_ckpt(exp_dir, explicit):
    if explicit:
        return explicit
    for name in ("ckpt.best_dice.pth.tar", "ckpt.best.pth.tar", "ckpt.last.pth.tar", "ckpt.best_hd95.pth.tar"):
        p = os.path.join(exp_dir, name)
        if os.path.exists(p):
            return p
    return None


class AdditivePVTB1LightUNet(nn.Module):
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1):
        super().__init__()
        self.enc_ct = timm.create_model("pvt_v2_b1", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels)
        self.enc_pet = timm.create_model("pvt_v2_b1", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name="Add_CT_Encoder")
            load_local_weights_safe(self.enc_pet, pretrained_path, name="Add_PET_Encoder")
        self.decoder = LightUNetDecoder(self.enc_ct.feature_info.channels())
        if out_channels != 1:
            self.decoder.out_head = nn.Conv2d(self.decoder.out_head.in_channels, out_channels, 1)
        self.stage_debug = {}

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        fused = [c + p for c, p in zip(ct_feats, pet_feats)]
        self.stage_debug = {
            f"stage{i + 1}": {"ct_before": c.detach(), "pet_before": p.detach(), "add_after": f.detach(), "add_diff": (f - 0.5 * (c + p)).detach()}
            for i, (c, p, f) in enumerate(zip(ct_feats, pet_feats, fused))
        }
        if target_size is None:
            target_size = ct.shape[-2:]
        return self.decoder(fused, target_size)


def load_state(model, ckpt_path, label):
    if ckpt_path is None:
        print(f"[WARN] {label}: no checkpoint found, using random weights.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("ema_model", ckpt.get("model", ckpt))
    msg = model.load_state_dict(state, strict=False)
    print(f"[+] {label}: loaded {ckpt_path}")
    print(f"[+] {label}: {msg}")


def build_tcpm_model(cfg, ckpt_path, device):
    model = build_mdt_seg_teacher(cfg)["model"].to(device)
    load_state(model, ckpt_path, "TCPM")
    patch_tcpm(model)
    model.eval()
    return model


def build_add_model(cfg, ckpt_path, device):
    model = AdditivePVTB1LightUNet(pretrained_path=getattr(cfg, "pretrained_path", None), in_channels=3, out_channels=1).to(device)
    load_state(model, ckpt_path, "Additive")
    model.eval()
    return model


def patch_tcpm(model):
    for m in getattr(model, "tcpm_blocks", []):
        def make_forward(module):
            def forward(pet_feature, ct_feature, text_code):
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
                module.stage_debug = {
                    "ct_before": ct_feature.detach(), "pet_before": pet_feature.detach(), "selected_pre_text": selected.detach(),
                    "text_logits": text_logits.detach(), "q_text_guided": q.detach(), "cross_modal_ref": ref.detach(),
                    "cross_attn_out": att_out.detach(), "tcpm_after": out.detach(), "tcpm_diff": (out - ref).detach(),
                }
                return out, ref
            return forward
        m.forward = make_forward(m)


def norm01(x):
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    return x / (np.nanmax(x) + 1e-8)


def fmap(t):
    t = t.detach().float().cpu()
    if t.dim() == 4:
        arr = t[0].mean(0).numpy()
    elif t.dim() == 3:
        arr = t[0].numpy()
    elif t.dim() == 2:
        arr = t[0][None, :].numpy()
    else:
        arr = t.numpy()
    return norm01(arr)


def gray(x):
    a = x.detach().float().cpu()[0].numpy()
    a = a.mean(0) if a.shape[0] == 3 else a.squeeze(0)
    return norm01(a)


def overlay(g, m, color, alpha=0.65):
    rgb = np.stack([g, g, g], -1)
    mm = (m > 0.5)[..., None].astype(np.float32)
    cc = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * (1 - alpha * mm) + cc * alpha * mm, 0, 1)


def draw_panels(panels, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = 6
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.7 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    print(f"[+] saved {save_path}")


def visualize_one(add_model, tcpm_model, batch, out_dir, sample_id, threshold, device):
    os.makedirs(out_dir, exist_ok=True)
    ct = batch["ct"].float().to(device)
    pet = batch["pet"].float().to(device)
    mask = batch["mask"].float().to(device)
    with torch.no_grad():
        add_logit = add_model(ct, pet, target_size=mask.shape[-2:])
        tcpm_logit = tcpm_model(ct, pet, target_size=mask.shape[-2:])
    add_prob = torch.sigmoid(add_logit)[0, 0].detach().cpu().numpy()
    tcpm_prob = torch.sigmoid(tcpm_logit)[0, 0].detach().cpu().numpy()
    gt = mask[0].detach().cpu().squeeze().numpy()
    ct_img, pet_img = gray(ct), gray(pet)

    panels = [
        ("Same sample CT", ct_img, "gray"), ("Same sample PET", pet_img, "inferno"),
        ("GT on CT", overlay(ct_img, gt, (1, 0, 0)), None),
        ("Additive prob", norm01(add_prob), "jet"), ("TCPM prob", norm01(tcpm_prob), "jet"),
        ("Prob diff |TCPM-Add|", norm01(np.abs(tcpm_prob - add_prob)), "hot"),
        ("Add pred", overlay(ct_img, add_prob > threshold, (0, 1, 0)), None),
        ("TCPM pred", overlay(ct_img, tcpm_prob > threshold, (0, 1, 0)), None),
    ]

    for i, tcpm_block in enumerate(tcpm_model.tcpm_blocks, 1):
        add_dbg = add_model.stage_debug[f"stage{i}"]
        tcp_dbg = tcpm_block.stage_debug
        panels.extend([
            (f"S{i} Add CT before", fmap(add_dbg["ct_before"]), "gray"),
            (f"S{i} Add PET before", fmap(add_dbg["pet_before"]), "inferno"),
            (f"S{i} Add fused CT+PET", fmap(add_dbg["add_after"]), "viridis"),
            (f"S{i} TCPM selected pre-text", fmap(tcp_dbg["selected_pre_text"]), "magma"),
            (f"S{i} TCPM text logits", fmap(tcp_dbg["text_logits"]), "viridis"),
            (f"S{i} TCPM Q text-guided", fmap(tcp_dbg["q_text_guided"]), "jet"),
            (f"S{i} TCPM cross ref", fmap(tcp_dbg["cross_modal_ref"]), "cividis"),
            (f"S{i} TCPM attn out", fmap(tcp_dbg["cross_attn_out"]), "plasma"),
            (f"S{i} TCPM after skip", fmap(tcp_dbg["tcpm_after"]), "turbo"),
            (f"S{i} |TCPM-Add fused|", norm01(np.abs(fmap(tcp_dbg["tcpm_after"]) - fmap(add_dbg["add_after"]))), "hot"),
        ])

    draw_panels(panels, os.path.join(out_dir, f"same_sample_add_vs_tcpm_{sample_id:03d}.png"))


def main():
    args = parse_args()
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gpu = int(args.gpus.split()[0])
    device = torch.device(args.device or (f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"))
    tcpm_cfg = load_config(args.tcpm_exp, gpu)
    add_cfg = load_config(args.add_exp, gpu)
    loader = build_loader(tcpm_cfg, args.split)
    tcpm_ckpt = auto_ckpt(args.tcpm_exp, args.tcpm_ckpt)
    add_ckpt = auto_ckpt(args.add_exp, args.add_ckpt)
    out_dir = args.out_dir or os.path.join(args.tcpm_exp, "same_sample_additive_vs_tcpm")
    add_model = build_add_model(add_cfg, add_ckpt, device)
    tcpm_model = build_tcpm_model(tcpm_cfg, tcpm_ckpt, device)
    for i, batch in enumerate(loader):
        visualize_one(add_model, tcpm_model, batch, out_dir, i, args.threshold, device)
        if i + 1 >= args.num_samples:
            break


if __name__ == "__main__":
    main()
