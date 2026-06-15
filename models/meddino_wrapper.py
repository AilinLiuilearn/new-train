import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct, SimpleFeatureInfo


class _PlaceholderMedDINOv3(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        self.feature_info = SimpleFeatureInfo(channels)
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=2)
        self.stage1 = ConvBNAct(c1, c1, kernel_size=3, stride=2)
        self.stage2 = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.stage3 = ConvBNAct(c2, c3, kernel_size=3, stride=2)
        self.stage4 = ConvBNAct(c3, c4, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p1, p2, p3, p4]


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _MedDINOBlock(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn.proj = nn.Linear(dim, dim)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        hidden = dim * mlp_ratio
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.ls2 = _LayerScale(dim)
        self.num_heads = 12

    def _attn_forward(self, x):
        b, n, c = x.shape
        qkv = self.attn.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.attn.proj(x)

    def forward(self, x):
        x = x + self.ls1(self._attn_forward(self.norm1(x)))
        x = x + self.ls2(self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x)))))
        return x


class _RealMedDINOv3Backbone(nn.Module):
    def __init__(self, out_channels=(64, 128, 256, 512), dim=768, patch_size=16, depth=12):
        super().__init__()
        self.feature_info = SimpleFeatureInfo(out_channels)
        self.patch_size = patch_size
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([_MedDINOBlock(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.ModuleList([nn.Conv2d(dim, c, 1, bias=False) for c in out_channels])
        self.stage_indices = (1, 4, 7, 10)

    def forward(self, x):
        input_hw = x.shape[-2:]
        x = self.patch_embed.proj(x)
        hp, wp = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        storage = self.storage_tokens.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, storage, tokens], dim=1)
        outs = []
        for idx, block in enumerate(self.blocks):
            tokens = block(tokens)
            if idx in self.stage_indices:
                patch_tokens = self.norm(tokens[:, 5:, :])
                feat = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.dim, hp, wp)
                outs.append(feat)
        while len(outs) < 4:
            patch_tokens = self.norm(tokens[:, 5:, :])
            outs.append(patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.dim, hp, wp))
        target_sizes = [
            (max(1, input_hw[0] // 4), max(1, input_hw[1] // 4)),
            (max(1, input_hw[0] // 8), max(1, input_hw[1] // 8)),
            (max(1, input_hw[0] // 16), max(1, input_hw[1] // 16)),
            (max(1, input_hw[0] // 32), max(1, input_hw[1] // 32)),
        ]
        feats = []
        for feat, proj, size in zip(outs[:4], self.proj, target_sizes):
            feat = proj(feat)
            if feat.shape[-2:] != size:
                feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            feats.append(feat)
        return feats


def _unwrap_checkpoint(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ('teacher', 'state_dict', 'model', 'module'):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def _strip_backbone_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ('module.', 'model.', 'teacher.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if new_key.startswith('backbone.'):
            new_key = new_key[len('backbone.'):]
        if new_key.startswith('rope_embed.'):
            continue
        cleaned[new_key] = value
    return cleaned


class FrozenMedDINOv3Encoder(nn.Module):
    """Frozen MedDINOv3 anatomical prior wrapper with placeholder fallback."""

    DEFAULT_CKPT = '/root/autodl-tmp/mkd-main/new-train/pretrained/MedDinov3/model.pth'

    def __init__(self, ckpt_path=None, use_placeholder_if_missing=True, out_channels=(64, 128, 256, 512)):
        super().__init__()
        self.ckpt_path = ckpt_path or self.DEFAULT_CKPT
        self.use_placeholder_if_missing = bool(use_placeholder_if_missing)
        self.out_channels = list(out_channels)
        self.is_placeholder = True
        self.encoder = None
        self._last_feature_shapes = None

        if self.ckpt_path and os.path.exists(str(self.ckpt_path)):
            try:
                self.encoder = self._build_real_encoder(str(self.ckpt_path), out_channels)
                self.is_placeholder = False
                print('[+] MedDINOv3 real checkpoint loaded')
                print(f'    checkpoint: {self.ckpt_path}')
                print(f'    output channels: {self.out_channels}')
            except Exception as exc:
                print(f'[-] MedDINOv3 real checkpoint load failed: {exc}')
                if not self.use_placeholder_if_missing:
                    raise
                self.encoder = self._build_placeholder(out_channels)
        else:
            msg = f'MedDINOv3 checkpoint not found: {self.ckpt_path}'
            if not self.use_placeholder_if_missing:
                raise FileNotFoundError(msg)
            print(f'[-] {msg}. Use placeholder prior.')
            self.encoder = self._build_placeholder(out_channels)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _build_placeholder(self, out_channels):
        self.is_placeholder = True
        return _PlaceholderMedDINOv3(in_channels=3, channels=out_channels)

    def _build_real_encoder(self, ckpt_path, out_channels):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = _strip_backbone_prefix(_unwrap_checkpoint(ckpt))
        patch_weight = state_dict.get('patch_embed.proj.weight')
        if patch_weight is None:
            raise KeyError('patch_embed.proj.weight not found in checkpoint')
        dim = int(patch_weight.shape[0])
        patch_size = int(patch_weight.shape[-1])
        block_ids = [int(k.split('.')[1]) for k in state_dict if k.startswith('blocks.') and len(k.split('.')) > 2 and k.split('.')[1].isdigit()]
        depth = max(block_ids) + 1 if block_ids else 12
        encoder = _RealMedDINOv3Backbone(out_channels=out_channels, dim=dim, patch_size=patch_size, depth=depth)
        model_state = encoder.state_dict()
        loadable = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        missing, unexpected = encoder.load_state_dict(loadable, strict=False)
        print(f'    MedDINOv3 loaded tensors: {len(loadable)} / {len(model_state)}')
        print(f'    MedDINOv3 missing keys: {len(missing)}, unexpected/skipped: {len(state_dict) - len(loadable)}')
        return encoder

    @staticmethod
    def _to_3ch(ct):
        if ct.shape[1] == 1:
            return ct.repeat(1, 3, 1, 1)
        return ct

    def forward(self, ct):
        with torch.no_grad():
            ct = self._to_3ch(ct)
            feats = self.encoder(ct)
            safe_feats = []
            for i, feat in enumerate(feats):
                feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
                feat = torch.clamp(feat, -1e4, 1e4)
                if not torch.isfinite(feat).all():
                    raise RuntimeError(f'[NaN/Inf] prior_feats[{i}] contains invalid values')
                safe_feats.append(feat)
            shapes = [tuple(f.shape) for f in safe_feats]
            if self._last_feature_shapes != shapes:
                mode = 'placeholder' if self.is_placeholder else 'real'
                print(f'[MedDINOv3:{mode}] feature shapes: {shapes}')
                self._last_feature_shapes = shapes
            return safe_feats
