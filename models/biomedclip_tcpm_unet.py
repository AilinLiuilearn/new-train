# -*- coding: utf-8 -*-
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import open_clip
except ImportError as e:
    open_clip = None
    OPEN_CLIP_ERROR = e

LOCAL_BIOMEDCLIP_DIR = "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model"
LOCAL_BIOMEDBERT_TEXT_TOWER_DIR = "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower"
BIOMEDCLIP_MODEL = "biomedclip_local"
BIOMEDCLIP_WEIGHTS = os.path.join(LOCAL_BIOMEDCLIP_DIR, "open_clip_pytorch_model.bin")
PROMPT = (
    "The image contains PET and CT scans of the lung, providing complementary information. "
    "Focus on accurately identifying abnormal tumor regions and preserving clear lesion boundaries."
)


def _register_local_biomedclip_config():
    src_config = os.path.join(LOCAL_BIOMEDCLIP_DIR, "open_clip_config.json")
    if not os.path.exists(src_config):
        raise FileNotFoundError(f"BiomedCLIP config not found: {src_config}")
    if not os.path.exists(BIOMEDCLIP_WEIGHTS):
        raise FileNotFoundError(f"BiomedCLIP weights not found: {BIOMEDCLIP_WEIGHTS}")
    if not os.path.isdir(LOCAL_BIOMEDBERT_TEXT_TOWER_DIR):
        raise FileNotFoundError(f"BiomedBERT text tower not found: {LOCAL_BIOMEDBERT_TEXT_TOWER_DIR}")

    with open(src_config, "r") as f:
        cfg = json.load(f)
    cfg["model_cfg"]["text_cfg"]["hf_model_name"] = LOCAL_BIOMEDBERT_TEXT_TOWER_DIR
    cfg["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = LOCAL_BIOMEDBERT_TEXT_TOWER_DIR

    cfg_dir = os.path.join(tempfile.gettempdir(), "open_clip_local_biomedclip")
    os.makedirs(cfg_dir, exist_ok=True)
    dst_config = os.path.join(cfg_dir, f"{BIOMEDCLIP_MODEL}.json")
    with open(dst_config, "w") as f:
        json.dump(cfg["model_cfg"], f)
    open_clip.add_model_config(cfg_dir)


class BiomedCLIPTextEncoder(nn.Module):
    def __init__(self, prompt=PROMPT, freeze=True):
        super().__init__()
        if open_clip is None:
            raise ImportError("Install open_clip_torch first: pip install open_clip_torch") from OPEN_CLIP_ERROR
        _register_local_biomedclip_config()
        self.model, _, _ = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL,
            pretrained=BIOMEDCLIP_WEIGHTS,
            pretrained_hf=False,
        )
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
        self.register_buffer("tokens", self.tokenizer([prompt]), persistent=False)
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, batch_size, device):
        tokens = self.tokens.to(device).expand(batch_size, -1)
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            text_code = self.model.encode_text(tokens).float()
            text_code = F.normalize(text_code, dim=-1)
        if training:
            self.model.train()
        return text_code


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class FeedForward(nn.Module):
    def __init__(self, c, expansion=2.0):
        super().__init__()
        h = int(c * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(c, h * 2, 1),
            nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2),
        )
        self.out = nn.Conv2d(h, c, 1)

    def forward(self, x):
        x1, x2 = self.net(x).chunk(2, dim=1)
        return self.out(F.gelu(x1) * x2)


class TopmCrossAttentionRestormerPrivileged(nn.Module):
    def __init__(self, c, heads=4, keep_ratio=0.9):
        super().__init__()
        assert c % heads == 0, f"channels {c} must be divisible by heads {heads}"
        self.heads = heads
        self.keep_ratio = keep_ratio
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Sequential(nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 3, padding=1, groups=c))
        self.kv = nn.Sequential(nn.Conv2d(c, c * 2, 1), nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2))
        self.proj = nn.Conv2d(c, c, 1)
        self.scale = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x_q, x_kv):
        _, _, h, w = x_q.shape
        q = self.q(x_q)
        k, v = self.kv(x_kv).chunk(2, dim=1)
        q = rearrange(q, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        k = rearrange(k, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        v = rearrange(v, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        keep = max(1, int(q.shape[2] * self.keep_ratio))
        idx = torch.topk(attn, keep, dim=-1).indices
        mask = torch.zeros_like(attn, dtype=torch.bool).scatter_(-1, idx, True)
        attn = torch.where(mask, attn, torch.full_like(attn, torch.finfo(attn.dtype).min))
        out = F.softmax(attn, dim=-1) @ v
        out = rearrange(out * self.scale, "b hd c (h w) -> b (hd c) h w", hd=self.heads, h=h, w=w)
        return self.proj(out)


class MULTI_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim, num_heads=4, lin_ch=512, topk_ratio=0.5):
        super().__init__()
        self.dim = ch_dim
        self.topk = max(1, int(2 * ch_dim * topk_ratio))
        hidden = max(1, 2 * ch_dim // 8)
        self.text_fc = nn.Sequential(nn.Linear(lin_ch, lin_ch), nn.ReLU(True), nn.Linear(lin_ch, 2 * ch_dim))
        self.se_fc = nn.Sequential(nn.Linear(2 * ch_dim, hidden), nn.ReLU(True), nn.Linear(hidden, 2 * ch_dim), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pick_conv = nn.Conv2d(self.topk, 2 * ch_dim, 1)
        self.out_conv = nn.Conv2d(2 * ch_dim, ch_dim, 1)
        self.n1, self.n2, self.n3 = LayerNorm2d(ch_dim), LayerNorm2d(ch_dim), LayerNorm2d(ch_dim)
        self.attn = TopmCrossAttentionRestormerPrivileged(ch_dim, num_heads)
        self.ffn = FeedForward(ch_dim)

    def forward(self, pet_feature, ct_feature, text_code):
        b, _, h, w = pet_feature.shape
        both = torch.cat([pet_feature, ct_feature], dim=1)
        weight = self.se_fc(self.pool(both).flatten(1))
        idx = torch.topk(weight, self.topk, dim=1).indices[:, :, None, None].expand(-1, -1, h, w)
        img = self.pick_conv(both.gather(1, idx))
        tidx = torch.topk(self.text_fc(text_code), 2 * self.dim, dim=1).indices
        img = img[torch.arange(b, device=img.device)[:, None], tidx, :, :]
        q = self.out_conv(img)
        ref = self.out_conv(both)
        att = self.attn(self.n1(q), self.n2(ref))
        return att + self.ffn(self.n3(att)), ref


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class PETCTBiomedCLIPTCPMUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32, freeze_text_encoder=True):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        heads = [1, 2, 4, 8]
        self.text_encoder = BiomedCLIPTextEncoder(freeze=freeze_text_encoder)
        self.pet_e1, self.pet_e2, self.pet_e3, self.pet_e4 = ConvBlock(in_channels, ch[0]), ConvBlock(ch[0], ch[1]), ConvBlock(ch[1], ch[2]), ConvBlock(ch[2], ch[3])
        self.ct_e1, self.ct_e2, self.ct_e3, self.ct_e4 = ConvBlock(in_channels, ch[0]), ConvBlock(ch[0], ch[1]), ConvBlock(ch[1], ch[2]), ConvBlock(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)
        self.tcpm1 = MULTI_shuffle_high_text(ch[0], heads[0])
        self.tcpm2 = MULTI_shuffle_high_text(ch[1], heads[1])
        self.tcpm3 = MULTI_shuffle_high_text(ch[2], heads[2])
        self.tcpm4 = MULTI_shuffle_high_text(ch[3], heads[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.dec3 = ConvBlock(ch[2] * 2, ch[2])
        self.dec2 = ConvBlock(ch[1] * 2, ch[1])
        self.dec1 = ConvBlock(ch[0] * 2, ch[0])
        self.out = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, pet, ct, text_code=None):
        b = pet.shape[0]
        if text_code is None:
            text_code = self.text_encoder(b, pet.device)
        else:
            text_code = F.normalize(text_code.to(pet.device).float(), dim=-1)

        p1, c1 = self.pet_e1(pet), self.ct_e1(ct)
        p2, c2 = self.pet_e2(self.pool(p1)), self.ct_e2(self.pool(c1))
        p3, c3 = self.pet_e3(self.pool(p2)), self.ct_e3(self.pool(c2))
        p4, c4 = self.pet_e4(self.pool(p3)), self.ct_e4(self.pool(c3))

        s1, _ = self.tcpm1(p1, c1, text_code)
        s2, _ = self.tcpm2(p2, c2, text_code)
        s3, _ = self.tcpm3(p3, c3, text_code)
        x, _ = self.tcpm4(p4, c4, text_code)

        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return self.out(x)


def build_biomedclip_tcpm_unet(config=None):
    config = config or object()
    return {"model": PETCTBiomedCLIPTCPMUNet(
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        base=getattr(config, "base", 32),
        freeze_text_encoder=getattr(config, "freeze_text_encoder", True),
    )}
