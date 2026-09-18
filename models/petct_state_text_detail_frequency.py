"""PET/CT 模块二：CT解剖细节 + PET傅里叶频带调制 + 单次输出残差。

放置位置: models/petct_state_text_detail_frequency.py；仅依赖 torch，
导出文本另需 transformers。建议 Python>=3.10、PyTorch>=2.0。

从 models/petct_state_text_detail_region.py 迁移了文本编码、双文本缓存、
输入校验、有效行筛选与诊断接口；旧区域分支文件已删除。

每尺度同结构、参数独立；Full/Missing 共用同一套参数。
C, P: [B,d,H,W]，进入 fusion 的原始特征（P 的来源由上游决定）。

  q_ct  = GELU(LayerNorm(Linear_ct(t_ct)))      # [1,d,1,1]
  q_pet = GELU(LayerNorm(Linear_pet(t_pet)))    # [1,d,1,1]
  q_pet_cond = q_pet + missing * E_missing      # 只进PET频带门控条件
  A_ct = sigmoid(gate_ct(GMP(C) + q_ct))        # [B,d,1,1]
  D_ct = C - avgpool3x3(C)
  E_ct = A_ct * detail_pw(GELU(detail_dw(D_ct)))
  P_hat = rfft2(P); M_low = exp(-rho^2/(2 sigma^2))  # cycles/pixel坐标
  P_low = irfft2(P_hat * M_low); P_high = P - P_low
  A_low = sigmoid(gate_low(GMP(P_low) + q_pet_cond))
  A_high = sigmoid(gate_high(GMP(P_high) + q_pet_cond))
  P_band = A_low * P_low + A_high * P_high
  E_pet = pet_pw(GELU(P_band))
  delta = out_proj(cat([E_ct, E_pet]))          # 2C->C，零初始化
  F = (C + P) + delta                           # 唯一输出残差

无效行（pet_valid=False）直接返回 C，不运行内部计算。
sigma 是频率标准差（cycles/pixel），配置常量而非 Parameter。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_CT_PROMPT = 'A CT image showing anatomical structures and tumor boundaries in the lungs.'
DEFAULT_PET_PROMPT = 'A PET image showing metabolically active tumor regions in the lungs.'
DEFAULT_CHANNELS = (64, 128, 320, 512)
TEXT_CACHE_FORMAT_VERSION = 2
EXTRA_STATE_VERSION = 6
EXTRA_STATE_ARCHITECTURE = 'state_text_detail_frequency_v1'


def _sentence_vector(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError('text_feature must be a torch.Tensor')
    value = value.detach().float().cpu().clone()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError('Expected one cached sentence vector [D] or [1,D], not tokens')
    if not torch.isfinite(value).all():
        raise ValueError('text_feature contains NaN/Inf')
    return value.contiguous()


def save_text_pair_cache(path: str | Path, ct_vector: Tensor, pet_vector: Tensor,
                         metadata: dict[str, Any]) -> None:
    """保存双文本缓存（format_version=2），拒绝覆盖已有文件。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing text cache: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    for key in ('ct_prompt', 'pet_prompt', 'backend', 'model_path'):
        if not metadata.get(key):
            raise ValueError(f'metadata requires {key}')
    torch.save({
        'format_version': TEXT_CACHE_FORMAT_VERSION,
        'ct_text_feature': _sentence_vector(ct_vector),
        'pet_text_feature': _sentence_vector(pet_vector),
        'metadata': dict(metadata),
    }, path)


def encode_fixed_text(model_path: str | Path, *, backend: str = 'clip',
                      prompt: str, max_length: int = 30,
                      device: str = 'cpu') -> tuple[Tensor, dict[str, Any]]:
    """离线冻结编码（单句）。双文本由调用方对两句各调一次。"""
    if backend not in ('clip', 'biomedbert'):
        raise ValueError('backend must be clip or biomedbert')
    if not str(prompt).strip() or max_length < 3:
        raise ValueError('Nonempty prompt and max_length >= 3 are required')
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'Local text model directory not found: {root}')
    try:
        from transformers import AutoTokenizer, AutoModel, CLIPTextModel
    except ImportError as exc:
        raise ImportError('Text export needs transformers and a local pretrained checkpoint') from exc
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                              trust_remote_code=False)
    raw = tokenizer(prompt, add_special_tokens=True, truncation=False)['input_ids']
    if len(raw) > max_length:
        raise ValueError(f'Prompt needs {len(raw)} tokens; increase max_length={max_length}')
    cls = CLIPTextModel if backend == 'clip' else AutoModel
    model = cls.from_pretrained(str(root), local_files_only=True,
                                trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    encoded = tokenizer(prompt, padding='max_length', max_length=max_length,
                        truncation=False, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()
               if k in ('input_ids', 'attention_mask', 'token_type_ids')}
    if backend == 'clip':
        encoded.pop('token_type_ids', None)
    with torch.no_grad():
        output = model(**encoded)
        if backend == 'clip':
            vector = output.pooler_output
            pooling = 'CLIP EOS pooler_output; no text_projection'
        else:
            mask = encoded['attention_mask'].unsqueeze(-1).to(output.last_hidden_state.dtype)
            vector = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooling = 'masked mean last_hidden_state; includes non-padding special tokens'
    vector = _sentence_vector(vector)
    return vector, {'prompt': prompt, 'backend': backend, 'pooling': pooling,
                    'model_path': str(root.resolve()), 'max_length': max_length,
                    'text_dim': vector.shape[1], 'normalized': False}


def encode_text_pair(model_path: str | Path, *, ct_prompt: str = DEFAULT_CT_PROMPT,
                     pet_prompt: str = DEFAULT_PET_PROMPT, backend: str = 'clip',
                     max_length: int = 30, device: str = 'cpu'
                     ) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """用同一冻结编码器分别编码 CT/PET 两句，返回向量与合并元信息。"""
    ct_vector, ct_meta = encode_fixed_text(model_path, backend=backend, prompt=ct_prompt,
                                           max_length=max_length, device=device)
    pet_vector, pet_meta = encode_fixed_text(model_path, backend=backend, prompt=pet_prompt,
                                             max_length=max_length, device=device)
    if ct_vector.shape != pet_vector.shape:
        raise ValueError(f'CT/PET text dims differ: {tuple(ct_vector.shape)} vs {tuple(pet_vector.shape)}')
    metadata = {
        'ct_prompt': ct_prompt, 'pet_prompt': pet_prompt,
        'backend': backend, 'model_path': ct_meta['model_path'],
        'max_length': max_length, 'text_dim': ct_vector.shape[1],
        'ct_pooling': ct_meta['pooling'], 'pet_pooling': pet_meta['pooling'],
        'normalized': False,
    }
    return ct_vector, pet_vector, metadata


def _binary_rows(value: Any, batch: int, device: torch.device, name: str) -> Tensor:
    if value is None:
        raise ValueError(f'{name} is required: 1=True, 0=False')
    result = torch.as_tensor(value, device=device)
    if result.ndim == 0:
        result = result.expand(batch)
    elif result.shape[0] == batch and result.numel() == batch:
        result = result.reshape(batch)
    else:
        raise ValueError(f'{name} must be scalar or one value per sample; got {tuple(result.shape)}')
    if not torch.all((result == 0) | (result == 1)):
        raise ValueError(f'{name} must contain only 0 or 1')
    return result.bool()


def band_masks(height: int, width: int, sigma: float, device: torch.device) -> Tensor:
    """固定高斯低通掩码 [1,1,H,W//2+1]，cycles/pixel 坐标，不做归一化。"""
    if not math.isfinite(sigma) or not 0.0 < sigma <= 0.5:
        raise ValueError(f'frequency sigma must be finite in (0, 0.5], got {sigma!r}')
    fy = torch.fft.fftfreq(height, d=1.0)
    fx = torch.fft.rfftfreq(width, d=1.0)
    rho2 = fy[:, None] ** 2 + fx[None, :] ** 2
    mask = torch.exp(-rho2 / (2.0 * sigma * sigma))
    return mask.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


class _ScaleDetailFrequencyFusion(nn.Module):
    def __init__(self, channels: int, text_dim: int, reduction: int, frequency_sigma: float):
        super().__init__()
        if not math.isfinite(frequency_sigma) or not 0.0 < frequency_sigma <= 0.5:
            raise ValueError(f'frequency_sigma must be finite in (0, 0.5], got {frequency_sigma!r}')
        hidden = max(channels // reduction, 1)
        self.frequency_sigma = float(frequency_sigma)
        self.missing_prompt = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.proj_ct = nn.Sequential(nn.Linear(text_dim, channels),
                                     nn.LayerNorm(channels), nn.GELU())
        self.proj_pet = nn.Sequential(nn.Linear(text_dim, channels),
                                      nn.LayerNorm(channels), nn.GELU())
        self.gate_ct = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                     nn.GELU(),
                                     nn.Conv2d(hidden, channels, 1))
        self.gate_low = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                      nn.GELU(),
                                      nn.Conv2d(hidden, channels, 1))
        self.gate_high = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                       nn.GELU(),
                                       nn.Conv2d(hidden, channels, 1))
        self.detail_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.detail_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.pet_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(2 * channels, channels, 1, bias=True)
        nn.init.zeros_(self.missing_prompt)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self._text_used = True

    def text_params(self):
        for module in (self.proj_ct, self.proj_pet, self.gate_ct,
                       self.gate_low, self.gate_high):
            for param in module.parameters():
                yield param
        yield self.missing_prompt

    def forward(self, ct: Tensor, pet: Tensor, missing: Tensor,
                text_ct: Tensor, text_pet: Tensor,
                *, use_state: bool, use_text: bool,
                diagnostics: bool,
                _fft_calls: list | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        param_dtype = self.detail_dw.weight.dtype
        a_ct = a_low = a_high = None
        e_ct_band = e_pet_band = None
        if use_text:
            qc = self.proj_ct(text_ct.to(param_dtype)).unsqueeze(-1).unsqueeze(-1)
            qp = self.proj_pet(text_pet.to(param_dtype)).unsqueeze(-1).unsqueeze(-1)
            if use_state:
                qp = qp + missing.reshape(-1, 1, 1, 1).to(qp.dtype) * self.missing_prompt.to(qp.dtype)
            with torch.autocast(device_type=ct.device.type, enabled=False):
                gc_in = F.adaptive_max_pool2d(ct.float(), 1) + qc.float()
                a_ct = torch.sigmoid(self.gate_ct(gc_in)).to(ct.dtype)
        else:
            a_ct = ct.new_ones((ct.shape[0], ct.shape[1], 1, 1))
        detail_in = (ct - F.avg_pool2d(ct, kernel_size=3, stride=1, padding=1,
                                       count_include_pad=False)).to(param_dtype)
        b_raw = self.detail_dw(detail_in)
        b_raw = self.detail_pw(F.gelu(b_raw)).to(ct.dtype)
        e_ct = a_ct * b_raw
        if use_text:
            _, _, h, w = pet.shape
            with torch.autocast(device_type=pet.device.type, enabled=False):
                if _fft_calls is not None:
                    _fft_calls.append(1)
                spec = torch.fft.rfft2(pet.float(), dim=(-2, -1), norm='ortho')
                mask = band_masks(h, w, self.frequency_sigma, pet.device)
                p_low = torch.fft.irfft2(spec * mask, s=(h, w), dim=(-2, -1), norm='ortho')
                p_high = pet.float() - p_low
                gl_in = F.adaptive_max_pool2d(p_low, 1) + qp.float()
                gh_in = F.adaptive_max_pool2d(p_high, 1) + qp.float()
                a_low = torch.sigmoid(self.gate_low(gl_in))
                a_high = torch.sigmoid(self.gate_high(gh_in))
                p_band = a_low * p_low + a_high * p_high
            a_low = a_low.to(pet.dtype)
            a_high = a_high.to(pet.dtype)
        else:
            p_band = pet.float()
            a_low = pet.new_ones((pet.shape[0], pet.shape[1], 1, 1))
            a_high = pet.new_ones((pet.shape[0], pet.shape[1], 1, 1))
        e_pet = self.pet_pw(F.gelu(p_band.to(param_dtype))).to(pet.dtype)
        if diagnostics:
            e_ct_band = e_ct.detach()
            e_pet_band = e_pet.detach()
        delta = self.out_proj(torch.cat([e_ct.to(param_dtype), e_pet.to(param_dtype)], dim=1))
        base = ct + pet
        out = base + delta.to(base.dtype)
        info: dict[str, Tensor] = {}
        if diagnostics:
            info['a_ct'] = a_ct.detach()
            info['a_low'] = a_low.detach()
            info['a_high'] = a_high.detach()
            info['e_ct'] = e_ct_band
            info['e_pet'] = e_pet_band
            info['base'] = base.detach()
            info['delta'] = delta.detach()
        return out, info


class StateTextDetailFrequencyFusion(nn.Module):
    """CT解剖细节 + PET傅里叶频带调制 + 单次输出残差。"""

    def __init__(self, channels: Sequence[int] = DEFAULT_CHANNELS, *,
                 ct_text_feature: Tensor | None = None,
                 pet_text_feature: Tensor | None = None,
                 text_dim: int = 512,
                 text_metadata: dict[str, Any] | None = None,
                 reduction: int = 16, frequency_sigma: float = 0.15,
                 enabled: bool = True, use_state: bool = True, use_text: bool = True,
                 diag_enabled: bool = False, diag_interval: int = 50):
        super().__init__()
        if not channels or any(not isinstance(c, int) or c < 2 for c in channels):
            raise ValueError('channels must be a nonempty sequence of integers >= 2')
        if not isinstance(reduction, int) or reduction < 1:
            raise ValueError('reduction must be a positive integer')
        if not math.isfinite(float(frequency_sigma)) or not 0.0 < float(frequency_sigma) <= 0.5:
            raise ValueError('frequency_sigma must be finite in (0, 0.5]')
        if ct_text_feature is None or pet_text_feature is None:
            if use_text and enabled:
                raise ValueError('Text enabled: supply real cached ct/pet text features')
            if not isinstance(text_dim, int) or text_dim < 1:
                raise ValueError('text_dim must be a positive integer')
            ct_vector = torch.zeros(1, text_dim)
            pet_vector = torch.zeros(1, text_dim)
        else:
            ct_vector = _sentence_vector(ct_text_feature)
            pet_vector = _sentence_vector(pet_text_feature)
            if ct_vector.shape != pet_vector.shape:
                raise ValueError('CT/PET text features must share [1,D]')
        self.channels = tuple(channels)
        self.reduction = reduction
        self.frequency_sigma = float(frequency_sigma)
        self.enabled, self.use_state = bool(enabled), bool(use_state)
        self.use_text = bool(use_text)
        self.diag_enabled = bool(diag_enabled)
        self.diag_interval = int(diag_interval)
        if self.diag_interval < 1:
            raise ValueError('diag_interval must be >= 1')
        self.register_buffer('ct_text_feature', ct_vector)
        self.register_buffer('pet_text_feature', pet_vector)
        self.register_buffer('text_ready', torch.tensor(ct_text_feature is not None
                                                        and pet_text_feature is not None))
        self.text_metadata = dict(text_metadata or {'ct_prompt': DEFAULT_CT_PROMPT,
                                                    'pet_prompt': DEFAULT_PET_PROMPT,
                                                    'backend': 'provided-vector'})
        self.scales = nn.ModuleList([_ScaleDetailFrequencyFusion(c, ct_vector.shape[1], reduction,
                                                                 self.frequency_sigma)
                                     for c in channels])
        if not self.use_text:
            for block in self.scales:
                for param in block.text_params():
                    param.requires_grad_(False)
        elif not self.use_state:
            for block in self.scales:
                block.missing_prompt.requires_grad_(False)
        self._diag_stats: dict[str, Any] = {'forwards': 0, 'scales': {}}
        self._diag_forward_count = 0

    @classmethod
    def from_text_cache(cls, path: str | Path, *,
                        channels: Sequence[int] = DEFAULT_CHANNELS,
                        **kwargs: Any) -> 'StateTextDetailFrequencyFusion':
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(cache, dict) or cache.get('format_version') != TEXT_CACHE_FORMAT_VERSION:
            raise ValueError(
                'Dual text cache must be format_version=2 (save_text_pair_cache or '
                '--encode-text-pair).')
        metadata = cache.get('metadata')
        if not isinstance(metadata, dict):
            raise ValueError('Text pair cache requires metadata')
        return cls(channels, ct_text_feature=cache['ct_text_feature'],
                   pet_text_feature=cache['pet_text_feature'],
                   text_metadata=metadata, **kwargs)

    def get_extra_state(self) -> dict[str, Any]:
        return {'version': EXTRA_STATE_VERSION, 'architecture': EXTRA_STATE_ARCHITECTURE,
                'text_metadata': self.text_metadata,
                'config': {'channels': self.channels, 'reduction': self.reduction,
                           'frequency_sigma': self.frequency_sigma,
                           'enabled': self.enabled, 'use_state': self.use_state,
                           'use_text': self.use_text,
                           'diag_enabled': self.diag_enabled,
                           'diag_interval': self.diag_interval}}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get('version') != EXTRA_STATE_VERSION:
            raise ValueError(
                'Unsupported fusion checkpoint metadata: expected '
                f'{EXTRA_STATE_ARCHITECTURE} version={EXTRA_STATE_VERSION}, got '
                f'version={state.get("version") if isinstance(state, dict) else type(state)}; '
                'detail-region-v1, competitive-v1 and older AFA weights cannot convert; '
                'evaluate them on their branches.')
        if state.get('architecture') != EXTRA_STATE_ARCHITECTURE:
            raise ValueError(
                f"Fusion architecture mismatch: {state.get('architecture')!r} != "
                f'{EXTRA_STATE_ARCHITECTURE!r}')
        current = self.get_extra_state()['config']
        incoming = dict(state.get('config') or {})
        for key in ('diag_enabled', 'diag_interval'):
            incoming.pop(key, None)
            current.pop(key, None)
        if incoming != current:
            raise ValueError(
                'Fusion checkpoint config differs (channels/reduction/frequency_sigma/'
                'enabled/use_text/use_state); construct the same architecture/flags')
        self.text_metadata = dict(state['text_metadata'])

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
                pet_available: Any, *, pet_valid: Any,
                return_diagnostics: bool = False) -> list[Tensor] | tuple[list[Tensor], list[dict[str, Tensor]]]:
        if len(ct_feats) != len(self.channels) or len(pet_feats) != len(self.channels):
            raise ValueError(f'Expected {len(self.channels)} CT and PET scales')
        first = ct_feats[0]
        if first.ndim != 4 or first.shape[0] < 1:
            raise ValueError('Features must be nonempty [B,C,H,W] tensors')
        batch = first.shape[0]
        state = _binary_rows(pet_available, batch, first.device, 'pet_available')
        valid = _binary_rows(pet_valid, batch, first.device, 'pet_valid')
        if torch.any(state & ~valid):
            raise ValueError('A real-PET sample cannot have pet_valid=False')
        if self.enabled and self.use_text and not bool(self.text_ready):
            raise RuntimeError('Text modulation enabled without a real text pair cache')
        if self.ct_text_feature.device != first.device:
            raise ValueError('Move fusion and features to the same device before forward')
        rows = valid.nonzero(as_tuple=False).flatten()
        diag_on = (bool(self.diag_enabled) and self.training
                   and (self._diag_forward_count % self.diag_interval == 0))
        if bool(self.diag_enabled):
            self._diag_forward_count += 1
        outputs, infos = [], []
        for i, (ct, pet, channels, block) in enumerate(zip(ct_feats, pet_feats, self.channels, self.scales)):
            if ct.ndim != 4 or pet.ndim != 4 or ct.shape[:2] != (batch, channels) or pet.shape[:2] != (batch, channels):
                raise ValueError(f'Scale {i}: expected CT/PET [B={batch},C={channels},H,W]')
            if min(ct.shape[2:] + pet.shape[2:]) < 1:
                raise ValueError(f'Scale {i}: empty spatial dimension')
            if ct.device != first.device or pet.device != ct.device or ct.dtype != first.dtype or pet.dtype != ct.dtype:
                raise ValueError('All CT/PET scales must share device and floating dtype')
            if not ct.is_floating_point():
                raise TypeError('Feature tensors must be floating point')
            info = {'valid_rows': rows.detach()}
            if rows.numel() == 0:
                out = ct
            else:
                c, p = ct.index_select(0, rows), pet.index_select(0, rows)
                if p.shape[-2:] != c.shape[-2:]:
                    p = F.interpolate(p, size=c.shape[-2:], mode='bilinear', align_corners=False)
                if not self.enabled:
                    fused = c + p
                else:
                    full = state.index_select(0, rows)
                    fused, more = block(c, p, (~full).to(c.dtype),
                                        self.ct_text_feature, self.pet_text_feature,
                                        use_state=self.use_state, use_text=self.use_text,
                                        diagnostics=(return_diagnostics or diag_on))
                    info.update(more)
                    if diag_on:
                        self._accumulate_diag(i, state, rows, more.get('a_ct'),
                                              more.get('a_low'), more.get('a_high'),
                                              more.get('e_ct'), more.get('e_pet'),
                                              more.get('base'), more.get('delta'))
                out = ct.index_copy(0, rows, fused.to(ct.dtype))
            outputs.append(out)
            infos.append(info)
        if diag_on:
            self._diag_stats['forwards'] = int(self._diag_stats.get('forwards', 0)) + 1
        return (outputs, infos) if return_diagnostics else outputs

    @staticmethod
    def _sample_rms(x: Tensor) -> Tensor:
        flat = x.detach().float().reshape(x.shape[0], -1)
        return flat.pow(2).mean(1).sqrt()

    def _accumulate_diag(self, scale_idx, state, rows, a_ct, a_low, a_high,
                         e_ct, e_pet, base, delta) -> None:
        entry = self._diag_stats['scales'].setdefault(
            f'scale{scale_idx + 1}',
            {'full': self._empty_group(), 'missing': self._empty_group()},
        )
        row_state = state.index_select(0, rows).tolist()
        groups = {
            'full': [k for k, s in enumerate(row_state) if s],
            'missing': [k for k, s in enumerate(row_state) if not s],
        }
        for tag, positions in groups.items():
            group = entry[tag]
            if not positions:
                continue
            group['count'] += len(positions)
            pos = torch.tensor(positions, device=rows.device)
            if self.use_text:
                group['a_ct_sum'] += float(a_ct.detach().float().index_select(0, pos).mean().item()) * len(positions)
                group['a_low_sum'] += float(a_low.detach().float().index_select(0, pos).mean().item()) * len(positions)
                group['a_high_sum'] += float(a_high.detach().float().index_select(0, pos).mean().item()) * len(positions)
            group['e_ct_rms_sum'] += float(self._sample_rms(e_ct.index_select(0, pos)).mean().item()) * len(positions)
            group['e_pet_rms_sum'] += float(self._sample_rms(e_pet.index_select(0, pos)).mean().item()) * len(positions)
            group['base_rms_sum'] += float(self._sample_rms(base.index_select(0, pos)).mean().item()) * len(positions)
            group['delta_rms_sum'] += float(self._sample_rms(delta.index_select(0, pos)).mean().item()) * len(positions)

    @staticmethod
    def _empty_group() -> dict[str, Any]:
        return {'count': 0, 'a_ct_sum': 0.0, 'a_low_sum': 0.0, 'a_high_sum': 0.0,
                'e_ct_rms_sum': 0.0, 'e_pet_rms_sum': 0.0,
                'base_rms_sum': 0.0, 'delta_rms_sum': 0.0}

    def reset_diag_stats(self) -> None:
        self._diag_stats = {'forwards': 0, 'scales': {}}
        self._diag_forward_count = 0

    def pop_diag_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {'forwards': int(self._diag_stats.get('forwards', 0)), 'scales': {}}
        for scale, groups in self._diag_stats.get('scales', {}).items():
            out['scales'][scale] = {}
            for tag, g in groups.items():
                n = int(g['count'])
                if n == 0:
                    out['scales'][scale][tag] = {'count': 0, 'a_ct_mean': None,
                                                 'a_low_mean': None, 'a_high_mean': None,
                                                 'e_ct_rms': None, 'e_pet_rms': None,
                                                 'base_rms': None, 'delta_rms': None}
                    continue
                out['scales'][scale][tag] = {
                    'count': n,
                    'a_ct_mean': g['a_ct_sum'] / n if self.use_text else 'n/a',
                    'a_low_mean': g['a_low_sum'] / n if self.use_text else 'n/a',
                    'a_high_mean': g['a_high_sum'] / n if self.use_text else 'n/a',
                    'e_ct_rms': g['e_ct_rms_sum'] / n,
                    'e_pet_rms': g['e_pet_rms_sum'] / n,
                    'base_rms': g['base_rms_sum'] / n,
                    'delta_rms': g['delta_rms_sum'] / n,
                }
        self.reset_diag_stats()
        return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('detail-frequency text-pair cache')
    p.add_argument('--encode-text-pair', type=str, default=None)
    p.add_argument('--model-path', type=str, default=None)
    p.add_argument('--ct-prompt', type=str, default=DEFAULT_CT_PROMPT)
    p.add_argument('--pet-prompt', type=str, default=DEFAULT_PET_PROMPT)
    p.add_argument('--backend', type=str, default='clip', choices=('clip', 'biomedbert'))
    p.add_argument('--max-length', type=int, default=30)
    return p


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    if not args.encode_text_pair or not args.model_path:
        raise SystemExit('Use --encode-text-pair OUT --model-path DIR [--ct-prompt .. --pet-prompt ..]')
    ct_v, pet_v, meta = encode_text_pair(args.model_path, ct_prompt=args.ct_prompt,
                                         pet_prompt=args.pet_prompt, backend=args.backend,
                                         max_length=args.max_length)
    save_text_pair_cache(args.encode_text_pair, ct_v, pet_v, meta)
    print(f'saved dual text cache to {args.encode_text_pair} dim={meta["text_dim"]}')
