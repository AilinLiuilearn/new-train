#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download ConvNeXt pretrained weights (atto / femto / pico / nano) to local.

Usage:
    python download_convnext_pretrained.py          # download all
    python download_convnext_pretrained.py atto      # download one
"""

import os
import sys
import json

SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")

MODELS = {
    "convnext_atto": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth",
        "timm_name": "convnext_atto",
        "channels": [40, 80, 160, 320],
    },
    "convnext_femto": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth",
        "timm_name": "convnext_femto",
        "channels": [48, 96, 192, 384],
    },
    "convnext_pico": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth",
        "timm_name": "convnext_pico",
        "channels": [64, 128, 256, 512],
    },
    "convnext_nano": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth",
        "timm_name": "convnext_nano",
        "channels": [80, 160, 320, 640],
    },
}


def download_one(name, info):
    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, "pytorch_model.bin")
    config_path = os.path.join(save_dir, "config.json")

    cfg = {
        "model_name": name,
        "timm_name": info["timm_name"],
        "encoder_channels": info["channels"],
        "source_url": info["url"],
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[+] Saved config: {config_path}")

    if os.path.exists(weight_path):
        size_mb = os.path.getsize(weight_path) / 1e6
        print(f"[=] {name}: weights already exist ({size_mb:.1f} MB), skip download.")
        return

    print(f"[*] Downloading {name} from {info['url']} ...")
    import torch
    state_dict = torch.hub.load_state_dict_from_url(
        info["url"], map_location="cpu", model_dir=save_dir, check_hash=False
    )
    torch.save(state_dict, weight_path)
    size_mb = os.path.getsize(weight_path) / 1e6
    print(f"[+] Saved: {weight_path} ({size_mb:.1f} MB)")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for name in targets:
        key = name if name.startswith("convnext_") else f"convnext_{name}"
        if key not in MODELS:
            print(f"[-] Unknown model: {name}, choices: {list(MODELS.keys())}")
            continue
        download_one(key, MODELS[key])
    print("\nDone!")


if __name__ == "__main__":
    main()
