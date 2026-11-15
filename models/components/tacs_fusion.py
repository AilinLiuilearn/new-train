"""Text-guided Adaptive Channel-Spatial Sparse Fusion (TACS).

TACS combines:
1) BCSI-inspired Top-K channel routing, adapted from labeled/unlabeled
   streams to paired CT/PET features;
2) DSCF-inspired modality-specific deformable sampling on a sparse 2-D grid;
3) bidirectional cross-modal attention inside the selected channel subspace.

The original CT+PET addition is an identity-preserving residual path. The
sparse correction head is zero-initialized, so initialization is exactly
CT + PET. Text is an optional, precomputed frozen embedding and only biases
the channel router; it does not directly generate spatial locations.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE_CHANNELS = (64, 128, 320, 512)
STAGE_FACTORS = (4, 8, 16, 32)


def _load_text_embedding(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("embedding", payload.get("text_embedding"))
    if payload is None:
        raise ValueError("Text embedding file must contain 'embedding' or 'text_embedding'")
    embedding = torch.as_tensor(payload, dtype=torch.float32).flatten()
    if embedding.numel() < 1 or not torch.isfinite(embedding).all():
        raise ValueError("Text embedding must be a non-empty finite vector")
    return F.normalize(embedding, dim=0)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ChannelRouter(nn.Module):
    """Select a shared sparse channel subset relevant to either modality.

    Separate CT/PET scoring MLPs accommodate heterogeneous encoders. Their
    scores are combined by smooth-max (log-mean-exp), so a channel can be
    selected when it is useful to either modality. A fixed text embedding,
    when provided, contributes a small learned semantic bias.
    """

    def __init__(self, channels, selected_channels, text_dim=None):
        super().__init__()
        hidden = max(8, channels // 4)
        self.channels = int(channels)
        self.selected_channels = int(selected_channels)
        self.ct_score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.pet_score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.text_proj = nn.Linear(text_dim, channels, bias=False) if text_dim else None
        if self.text_proj is not None:
            nn.init.zeros_(self.text_proj.weight)
            self.text_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("text_scale", None)

    def forward(self, ct, pet, text_embedding=None):
        ct_desc = F.adaptive_avg_pool2d(ct, 1).flatten(1)
        pet_desc = F.adaptive_avg_pool2d(pet, 1).flatten(1)
        score_ct = self.ct_score(ct_desc)
        score_pet = self.pet_score(pet_desc)
        scores = torch.logaddexp(score_ct, score_pet) - math.log(2.0)

        if self.text_proj is not None:
            if text_embedding is None:
                raise ValueError("This TACS router requires a text embedding")
            text = text_embedding.to(device=ct.device, dtype=ct.dtype).reshape(1, -1)
            if text.shape[1] != self.text_proj.in_features:
                raise ValueError(
                    f"text embedding dim mismatch: got {text.shape[1]}, "
                    f"expected {self.text_proj.in_features}"
                )
            scores = scores + self.text_scale * self.text_proj(text).expand(ct.shape[0], -1)

        indices = scores.topk(self.selected_channels, dim=1, largest=True, sorted=False).indices
        selected_scores = scores.gather(1, indices)
        # The factor 2 makes the initial gate close to 1 for scores near zero.
        selected_weights = 2.0 * torch.sigmoid(selected_scores)
        return indices, selected_weights, scores


class OffsetNet(nn.Module):
    """Lightweight modality-specific offset predictor."""

    def __init__(self, channels, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=stride,
                padding=1, groups=channels, bias=False,
            ),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1, bias=True),
        )
        # Start from the regular reference grid; offsets are learned gradually.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _reference_grid(batch, grid_size, device, dtype):
    ys = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).reshape(1, grid_size * grid_size, 2)
    return grid.expand(batch, -1, -1)


def _sample_points(feature, points):
    """BCHW feature + BNC normalized (x,y) points -> BNC sampled features."""
    grid = points.unsqueeze(2)
    sampled = F.grid_sample(
        feature, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


class TACSStage(nn.Module):
    """One scale of channel selection + sparse spatial cross-modal interaction."""

    def __init__(
        self,
        channels,
        selected_channels,
        grid_size,
        stride,
        text_dim=None,
        embed_dim=32,
        num_heads=4,
        max_offset_cells=0.25,
    ):
        super().__init__()
        self.channels = int(channels)
        self.selected_channels = int(selected_channels)
        self.grid_size = int(grid_size)
        self.max_offset_cells = float(max_offset_cells)
        self.use_channel_router = self.selected_channels < self.channels

        self.router = (
            ChannelRouter(channels, selected_channels, text_dim=text_dim)
            if self.use_channel_router else None
        )
        self.offset_ct = OffsetNet(selected_channels, stride)
        self.offset_pet = OffsetNet(selected_channels, stride)

        self.ct_query = nn.Linear(selected_channels, embed_dim)
        self.pet_query = nn.Linear(selected_channels, embed_dim)
        self.ct_key_value = nn.Linear(selected_channels, embed_dim)
        self.pet_key_value = nn.Linear(selected_channels, embed_dim)
        self.position = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.output_projection = nn.Linear(embed_dim, selected_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _select_channels(self, ct, pet, text_embedding):
        batch, channels, height, width = ct.shape
        if self.router is None:
            indices = torch.arange(channels, device=ct.device).view(1, channels)
            indices = indices.expand(batch, -1)
            weights = torch.ones(batch, channels, device=ct.device, dtype=ct.dtype)
        else:
            indices, weights, _ = self.router(ct, pet, text_embedding)
        gather_index = indices[:, :, None, None].expand(-1, -1, height, width)
        ct_selected = ct.gather(1, gather_index) * weights[:, :, None, None]
        pet_selected = pet.gather(1, gather_index) * weights[:, :, None, None]
        return indices, gather_index, ct_selected, pet_selected

    def _deformed_grid(self, selected, offset_net, reference):
        batch = selected.shape[0]
        raw = offset_net(selected)
        if raw.shape[-2:] != (self.grid_size, self.grid_size):
            raise RuntimeError(
                f"offset grid mismatch: got {tuple(raw.shape[-2:])}, "
                f"expected {(self.grid_size, self.grid_size)}"
            )
        # Offset channels are normalized (dx, dy), bounded to a fraction of
        # one reference-grid cell to respect paired CT/PET spatial alignment.
        scale = self.max_offset_cells * (2.0 / max(1, self.grid_size - 1))
        offset = torch.tanh(raw).permute(0, 2, 3, 1) * scale
        return (reference.reshape(1, self.grid_size, self.grid_size, 2) + offset).clamp(-1.0, 1.0)

    def forward(self, ct, pet, text_embedding=None):
        if ct.ndim != 4 or pet.ndim != 4 or ct.shape != pet.shape:
            raise ValueError(
                f"TACSStage expects matching BCHW CT/PET tensors, "
                f"got {tuple(ct.shape)} and {tuple(pet.shape)}"
            )
        batch, channels, height, width = ct.shape
        if channels != self.channels:
            raise ValueError(f"TACSStage expected {self.channels} channels, got {channels}")
        if min(height, width) < self.grid_size:
            raise ValueError(
                f"feature map {(height, width)} is smaller than grid {self.grid_size}"
            )

        original_dtype = ct.dtype
        # grid_sample and attention are evaluated in FP32 for stable offsets
        # and similarity scores; the residual is cast back to the feature dtype.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            ct32, pet32 = ct.float(), pet.float()
            indices, gather_index, ct_selected, pet_selected = self._select_channels(
                ct32, pet32, text_embedding
            )

            reference = _reference_grid(batch, self.grid_size, ct.device, torch.float32)
            grid_ct = self._deformed_grid(ct_selected, self.offset_ct, reference)
            grid_pet = self._deformed_grid(pet_selected, self.offset_pet, reference)
            points_ct = grid_ct.reshape(batch, -1, 2)
            points_pet = grid_pet.reshape(batch, -1, 2)
            union_points = torch.cat((points_ct, points_pet), dim=1)

            # Queries stay on the regular reference grid; keys/values are
            # sampled at the union of CT- and PET-selected deformed positions.
            ct_query_features = _sample_points(ct_selected, reference)
            pet_query_features = _sample_points(pet_selected, reference)
            ct_key_features = _sample_points(ct_selected, union_points)
            pet_key_features = _sample_points(pet_selected, union_points)

            position_query = self.position(reference)
            position_union = self.position(union_points)
            query_ct = self.ct_query(ct_query_features) + position_query
            query_pet = self.pet_query(pet_query_features) + position_query
            key_ct = self.ct_key_value(ct_key_features) + position_union
            key_pet = self.pet_key_value(pet_key_features) + position_union

            # Bidirectional interaction: CT queries retrieve PET context and
            # PET queries retrieve CT context, both within the selected channels.
            context_ct = self.cross_attention(query_ct, key_pet, key_pet, need_weights=False)[0]
            context_pet = self.cross_attention(query_pet, key_ct, key_ct, need_weights=False)[0]
            context = 0.5 * (context_ct + context_pet)

            delta_tokens = self.output_projection(context)
            delta_grid = delta_tokens.transpose(1, 2).reshape(
                batch, self.selected_channels, self.grid_size, self.grid_size
            )
            delta_selected = F.interpolate(
                delta_grid, size=(height, width), mode="bilinear", align_corners=False
            )

            # Reinsert only the selected channel corrections. Unselected
            # channels retain the original additive CT+PET representation.
            delta_full = torch.zeros_like(ct32).scatter(1, gather_index, delta_selected)
            base = ct32 + pet32
            fused = base + delta_full

            delta_ratio = (
                delta_full.flatten(1).norm(dim=1)
                / base.flatten(1).norm(dim=1).clamp_min(1e-6)
            ).mean()
            reference_2d = reference.reshape(1, self.grid_size, self.grid_size, 2)
            offset_abs = 0.5 * (
                (grid_ct - reference_2d).abs().mean()
                + (grid_pet - reference_2d).abs().mean()
            )

        if not torch.isfinite(fused).all():
            raise RuntimeError("[NaN/Inf] TACSStage output")
        return fused.to(dtype=original_dtype), {
            "delta_ratio": float(delta_ratio.detach()),
            "offset_abs": float(offset_abs.detach()),
            "selected_channels": self.selected_channels,
        }


class TACSFusion(nn.Module):
    """Four-scale Full-path TACS fusion; Missing bypasses this module upstream."""

    def __init__(
        self,
        image_size=512,
        channel_ratio=0.25,
        text_embedding_path=None,
        use_text=False,
        grid_size=8,
    ):
        super().__init__()
        if type(image_size) is not int or image_size < 64 or image_size % 32:
            raise ValueError("image_size must be an integer >=64 and divisible by 32")
        if not (0.125 <= float(channel_ratio) <= 1.0):
            raise ValueError("channel_ratio must be in [0.125, 1.0]")
        if use_text and not text_embedding_path:
            raise ValueError("use_text=True requires --tacs_text_embedding_path")
        if use_text and float(channel_ratio) >= 1.0:
            raise ValueError("text guidance is only meaningful when channel selection is enabled")

        text_embedding = _load_text_embedding(text_embedding_path) if use_text else None
        self.register_buffer("text_embedding", text_embedding, persistent=True)
        self.image_size = image_size
        self.channel_ratio = float(channel_ratio)
        self.use_text = bool(use_text)
        self.stage_sizes = tuple(image_size // factor for factor in STAGE_FACTORS)
        self.grid_sizes = tuple(min(int(grid_size), size) for size in self.stage_sizes)
        self.strides = tuple(max(1, size // grid) for size, grid in zip(self.stage_sizes, self.grid_sizes))

        stages = []
        self.selected_channels = []
        for channels, grid, stride in zip(STAGE_CHANNELS, self.grid_sizes, self.strides):
            selected = channels if self.channel_ratio >= 1.0 else max(1, int(round(channels * self.channel_ratio)))
            self.selected_channels.append(selected)
            stages.append(TACSStage(
                channels=channels,
                selected_channels=selected,
                grid_size=grid,
                stride=stride,
                text_dim=(int(text_embedding.numel()) if text_embedding is not None else None),
            ))
        self.stages = nn.ModuleList(stages)
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._delta_sums = [0.0] * 4
        self._offset_sums = [0.0] * 4
        self._sample_count = 0

    def pop_diagnostics(self):
        denom = max(1, self._sample_count)
        result = {}
        for i in range(4):
            result[f"tacs_delta_ratio_s{i+1}"] = self._delta_sums[i] / denom
            result[f"tacs_offset_abs_s{i+1}"] = self._offset_sums[i] / denom
            result[f"tacs_selected_channels_s{i+1}"] = float(self.selected_channels[i])
        self.reset_diagnostics()
        return result

    def forward(self, ct_feats, pet_feats):
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError("TACSFusion requires four CT/PET scales in S1..S4 order")
        outputs = []
        batch_size = ct_feats[0].shape[0]
        for i, (ct, pet, stage, expected_channels, expected_size) in enumerate(
            zip(ct_feats, pet_feats, self.stages, STAGE_CHANNELS, self.stage_sizes)
        ):
            if ct.shape[1] != expected_channels or pet.shape[1] != expected_channels:
                raise ValueError(
                    f"S{i+1} expected {expected_channels} channels, "
                    f"got CT={ct.shape[1]} PET={pet.shape[1]}"
                )
            if ct.shape[-2:] != pet.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode="bilinear", align_corners=False)
            if ct.shape[-2:] != (expected_size, expected_size):
                raise ValueError(
                    f"S{i+1} expected {(expected_size, expected_size)}, got {tuple(ct.shape[-2:])}; "
                    "check image_size_2d"
                )
            fused, stats = stage(ct, pet, self.text_embedding)
            outputs.append(fused)
            if self.training:
                self._delta_sums[i] += stats["delta_ratio"] * batch_size
                self._offset_sums[i] += stats["offset_abs"] * batch_size
        if self.training:
            self._sample_count += batch_size
        return outputs
