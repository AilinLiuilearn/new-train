"""Standalone Module-2 for PET/CT segmentation. Put this file under models/.

Contract (S1 -> S4):
    ct_feats:  four aligned CT tensors, normally C=(64,128,320,512).
    pet_feats: ALREADY assembled real/Module-1-compensated PET tensors.
    pet_available: [B], 1=real PET, 0=imputed PET (NOT a feature-zero mask).
    bank_ready: bool or [B]. Required if any sample is imputed.
    return: list of four fused tensors, same shapes/dtypes as CT inputs.

Mixed-batch example:
    fusion = ProbabilisticPETFusion.from_text_cache('pet_state_text.pt').to(device)
    fused = fusion(ct_feats, pet_for_fusion, pet_available,
                   bank_ready=model.module1.bank_ready)
    decoded = model.decoder(fused, target_size)

Core-only use (no tokenizer/Transformers/text files needed):
    fusion = ProbabilisticPETFusion(text_enabled=False)

Design:
    delta = R(mu + std * noise) in train; R(mu) in eval
    alpha = sigmoid(one_visual_logit + log(clamp(1 / mean(std), .2, 1)))
    E = X + alpha * delta
    text is used ONLY for PET; two tokens, width 32, one head, no FFN/residual
    PET gain = 1 + .5 * (2*sigmoid(DGNet-inspired logit) - 1)
    fused = E_CT + modulated_E_PET + low_rank_bilinear_interaction

Cold Missing rows are excluded from the PET Gaussian/text branches entirely;
the PET input is ignored for these rows. No encoder, prototype update, loss,
optimizer, or scheduler is called here. Real PET must not be supplied in place
of compensated PET on Missing rows: no fusion module can infer this mistake.

Normalization adaptation: per-pixel channel LayerNorm replaces the borrowed
Gaussian head's BatchNorm to avoid adding cross-sample Full/Missing statistics.
This does not modify any existing backbone/decoder normalization.

Sources (independent implementation of the agreed equations):
  https://github.com/zitalk/UMFNet/blob/main/models/UMFNet.py
  https://github.com/iLearn-Lab/MM26-DGNet/blob/main/model/dgnet/pwm.py
  https://aclanthology.org/P18-1209/
  https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/hf_model.py

Python >=3.10, PyTorch >=2.1. Optional cache export: transformers, safetensors.
This script never downloads a model. Local-only load errors are not replaced
with random embeddings. Sigma is a latent-dispersion proxy, not a calibrated
segmentation error probability. No new loss is implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORMAT_VERSION = 1
PROMPTS = (
    'Focus on patient-specific metabolic activity in real PET features.',
    'Focus on lesion-related metabolic information represented in personalized imputed PET features.',
)
DEFAULT_TEXT_TOWER = '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower'
DEFAULT_BIOMEDCLIP = '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model'
State = Union[bool, int, Tensor, Sequence[int]]


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'{name} contains NaN/Inf; no silent nan_to_num is applied')


def _embeddings(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != 2 or value.shape[1] < 1:
        raise ValueError('text_embeddings must be [2,D], ordered [real PET, imputed PET]')
    value = value.detach().to(device='cpu', dtype=torch.float32).contiguous().clone()
    _finite('text_embeddings', value)
    if (value.norm(dim=-1) == 0).any():
        raise ValueError('text embeddings must be nonzero')
    return value


def save_text_cache(path: Union[str, Path], embeddings: Tensor,
                    metadata: Optional[dict] = None) -> None:
    value = _embeddings(embeddings)
    metadata = json.loads(json.dumps(metadata or {}))
    torch.save({'format_version': FORMAT_VERSION, 'prompts': list(PROMPTS),
                'embeddings': value, 'metadata': metadata}, path)


def load_text_cache(path: Union[str, Path]) -> tuple[Tensor, dict]:
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if cache.get('format_version') != FORMAT_VERSION or cache.get('prompts') != list(PROMPTS):
        raise ValueError('Text cache format/prompt content/order mismatch. Rebuild the two-PET-prompt cache.')
    return _embeddings(cache['embeddings']), cache.get('metadata', {})


class _BiomedCLIPTextTower(nn.Module):
    """Text-only recreation of official HFTextEncoder for the BiomedCLIP config.

    Loads the trained text transformer AND trained 768->640->512 projection.
    Omitting that projection would not reproduce BiomedCLIP embeddings.
    No visual tower or open_clip import is needed for these two fixed prompts.
    """
    def __init__(self, hf_config: Any, text_config: dict, output_dim: int):
        super().__init__()
        from transformers import AutoModel
        if hf_config.model_type != 'bert':
            raise ValueError('This local BiomedCLIP loader supports its BERT text tower only')
        self.pooler_type = text_config.get('hf_pooler_type')
        if self.pooler_type not in ('cls_last_hidden_state_pooler', 'mean_pooler'):
            raise ValueError(f'Unsupported BiomedCLIP pooler: {self.pooler_type}')
        self.transformer = AutoModel.from_config(hf_config, add_pooling_layer=False)
        width = int(hf_config.hidden_size)
        projection = text_config.get('hf_proj_type')
        if projection == 'mlp':
            hidden = (width + output_dim) // 2
            self.proj = nn.Sequential(nn.Linear(width, hidden, bias=False), nn.GELU(),
                                      nn.Linear(hidden, output_dim, bias=False))
        elif projection == 'linear':
            self.proj = nn.Linear(width, output_dim, bias=False)
        elif projection is None and width == output_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(f'Unsupported BiomedCLIP projection: {projection}')

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooler_type == 'cls_last_hidden_state_pooler':
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.proj(pooled)


@torch.no_grad()
def encode_local_pet_prompts(
    text_tower_path: Union[str, Path] = DEFAULT_TEXT_TOWER,
    biomedclip_path: Union[str, Path] = DEFAULT_BIOMEDCLIP,
    backend: str = 'biomedclip', device: str = 'cpu',
) -> tuple[Tensor, dict]:
    """Run a frozen local text encoder ONCE, returning [real, imputed] vectors.

    biomedbert: HF config/tokenizer/model weights, masked-mean hidden features.
    biomedclip: local BERT config/tokenizer + official open_clip_config.json and
      full OpenCLIP checkpoint; strict text.* loading includes trained projection.
    The encoder is local to this function and is not kept in Module-2.
    """
    if backend not in ('biomedbert', 'biomedclip'):
        raise ValueError('backend must be biomedbert or biomedclip')
    tower = Path(text_tower_path).expanduser().resolve()
    if not tower.is_dir():
        raise FileNotFoundError(f'Local text tower directory not found: {tower}')
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tower), local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(str(tower), local_files_only=True, trust_remote_code=False)
    if config.model_type != 'bert':
        raise ValueError(f'Expected the local BiomedBERT tower (model_type=bert), got {config.model_type}')
    metadata = {'backend': backend, 'text_tower_path': str(tower)}
    if backend == 'biomedclip':
        root = Path(biomedclip_path).expanduser().resolve()
        cfg = json.loads((root / 'open_clip_config.json').read_text(encoding='utf-8'))['model_cfg']
        text_cfg = cfg['text_cfg']
        encoder = _BiomedCLIPTextTower(config, text_cfg, int(cfg['embed_dim']))
        candidates = ('open_clip_model.safetensors', 'open_clip_pytorch_model.bin', 'model.safetensors')
        checkpoint = next((root / name for name in candidates if (root / name).is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(f'Expected an official full OpenCLIP checkpoint under {root}: {candidates}')
        if checkpoint.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(checkpoint), device='cpu')
        else:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        text_state = {key[len('text.'):]: value for key, value in state.items() if key.startswith('text.')}
        if not text_state:
            raise ValueError('BiomedCLIP checkpoint must contain text.* weights; no random fallback is allowed')
        try:
            encoder.load_state_dict(text_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError('Local BiomedCLIP config/checkpoint mismatch; no partial loading is allowed') from exc
        del state, text_state
        max_length = int(text_cfg['context_length'])
        metadata.update(checkpoint=str(checkpoint), biomedclip_path=str(root),
                        pooler=text_cfg['hf_pooler_type'], projection=text_cfg['hf_proj_type'])
    else:
        encoder, loading = AutoModel.from_pretrained(
            str(tower), config=config, local_files_only=True, trust_remote_code=False,
            add_pooling_layer=False, output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise RuntimeError(f'Incomplete local BiomedBERT weights: {loading}')
        max_length = min(256, int(config.max_position_embeddings))
        metadata.update(pooler='masked_mean', projection='none')
    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError('Local tokenizer/config pad_token_id mismatch')
    if max_length > config.max_position_embeddings:
        raise ValueError('Text context_length exceeds local transformer position capacity')
    encoder = encoder.float().to(device).eval().requires_grad_(False)
    encoded = tokenizer(list(PROMPTS), padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
    if backend == 'biomedclip':
        embeddings = encoder(encoded['input_ids'], encoded['attention_mask'])
    else:
        hidden = encoder(**encoded).last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    embeddings = F.normalize(embeddings.float(), dim=-1).cpu()
    metadata['dimension'] = int(embeddings.shape[1])
    metadata['embedding_sha256'] = hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()
    return _embeddings(embeddings), metadata


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each sample/pixel, in float32."""
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y.float(), (x.shape[1],), self.weight.float(), self.bias.float(), 1e-5)
        return y.permute(0, 3, 1, 2).to(x.dtype)


def _conv_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))


class GaussianResidual(nn.Module):
    def __init__(self, channels: int, latent: int, reliability_min: float,
                 sigma_init: float, logvar_bounds: Sequence[float]):
        super().__init__()
        self.reliability_min = reliability_min
        self.logvar_bounds = tuple(logvar_bounds)
        self.proj = nn.Conv2d(channels, latent, 1, bias=False)
        self.local_norm = ChannelLayerNorm(latent)
        self.local = nn.Conv2d(latent, latent, 3, padding=1, groups=latent)
        self.mu_norm = ChannelLayerNorm(latent)
        self.lv_norm = ChannelLayerNorm(latent)
        self.mu_head = _conv_mlp(latent)
        self.logvar_head = _conv_mlp(latent)
        self.residual_out = nn.Conv2d(latent, channels, 1, bias=False)
        # One visual logit = the difference of the former two softmax logits.
        self.gate = nn.Conv2d(2 * channels, 1, 1)
        nn.init.zeros_(self.logvar_head[2].weight)
        nn.init.constant_(self.logvar_head[2].bias, 2 * math.log(sigma_init))
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: Tensor, diagnostics: bool = False) -> tuple[Tensor, dict]:
        h = self.proj(x)
        h = h + F.gelu(self.local(self.local_norm(h)))
        mu = self.mu_head(self.mu_norm(h))
        logvar = self.logvar_head(self.lv_norm(h))
        # exp, sampling, log and sigmoid use float32 under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logvar = logvar.float().clamp(*self.logvar_bounds)
            std = (0.5 * logvar).exp()
            z = mu.float()
            if self.training:
                z = z + std * torch.randn_like(std)
            reliability = (std.mean(1, keepdim=True) + 1e-6).reciprocal()
            reliability = reliability.clamp(self.reliability_min, 1.0)
        delta = self.residual_out(z.to(h.dtype)).to(x.dtype)
        score = self.gate(torch.cat((x, delta), dim=1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            alpha = torch.sigmoid(score.float() + reliability.log())
        enhanced = x + alpha.to(x.dtype) * delta
        stats = {}
        if diagnostics:
            stats = {'sigma_mean': std.detach().mean(),
                     'reliability_mean': reliability.detach().mean(),
                     'reliability_upper_fraction': (reliability.detach() >= 1).float().mean(),
                     'alpha_mean': alpha.detach().mean(),
                     'residual_rms': delta.detach().float().square().mean().sqrt()}
        return enhanced, stats


class PETTextConditioner(nn.Module):
    """Two fixed states share one learned token, one head and one 32-d layer."""
    def __init__(self, embeddings: Tensor, width: int):
        super().__init__()
        self.width = width
        self.register_buffer('fixed_embeddings', _embeddings(embeddings), persistent=True)
        self.input_proj = nn.Linear(embeddings.shape[1], width, bias=False)
        self.token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.token, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 1, dropout=0.0, bias=False, batch_first=True)

    def forward(self, pet_available: Tensor) -> Tensor:
        # Compute only two state vectors once per forward, shared across scales.
        fixed = self.input_proj(self.fixed_embeddings).unsqueeze(1)
        tokens = torch.cat((fixed, self.token.expand(2, -1, -1)), dim=1)
        tokens = self.norm(tokens)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        table = attended[:, 1]  # No residual/FFN, use learned-token position.
        return table[(~pet_available.bool()).long()]


class PETPriorModulation(nn.Module):
    """DGNet-inspired channel gate; text affects PET exactly here, once."""
    def __init__(self, channels: int, text_width: int, strength: float):
        super().__init__()
        self.strength = strength
        self.text_proj = nn.Linear(text_width, channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        hidden = max(8, channels // 16)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
        nn.init.zeros_(self.mlp[2].weight)

    def forward(self, pet: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        prior = self.text_proj(context).to(pet.dtype)[:, :, None, None]
        response = self.local(pet * prior)
        logit = self.mlp(F.adaptive_max_pool2d(response, 1)) + self.mlp(prior)
        with torch.autocast(device_type=pet.device.type, enabled=False):
            gain = 1.0 + self.strength * (2.0 * torch.sigmoid(logit.float()) - 1.0)
        return pet * gain.to(pet.dtype), gain


class LowRankInteraction(nn.Module):
    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.ct_proj = nn.Conv2d(channels, rank, 1, bias=False)
        self.pet_proj = nn.Conv2d(channels, rank, 1, bias=False)
        self.out = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        left, right = self.ct_proj(ct), self.pet_proj(pet)
        # Avoid overflowing the low-rank multiplication itself in float16.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            product = left.float() * right.float()
            interaction = F.conv2d(product, self.out.weight.float())
        return interaction.to(ct.dtype)


def _binary_state(value: State, batch: int, device: torch.device, name: str,
                  allow_scalar: bool = False) -> Tensor:
    t = torch.as_tensor(value, device=device)
    if t.numel() == 1 and allow_scalar:
        t = t.reshape(1).expand(batch)
    elif t.ndim not in (1, 4) or (t.ndim == 4 and tuple(t.shape[1:]) != (1, 1, 1)):
        raise ValueError(f'{name} must be [B] (or [B,1,1,1])')
    if t.numel() != batch or not torch.all((t == 0) | (t == 1)):
        raise ValueError(f'{name} must contain exactly B={batch} binary values (0/1)')
    return t.reshape(batch).bool()


class ProbabilisticPETFusion(nn.Module):
    """Four-scale Module-2 replacing the AddFusion boundary, not Module-1.

    Configure all parameters BEFORE creating the optimizer. Text disabled at
    construction creates no text modules and requires only PyTorch. A model
    constructed with text can temporarily bypass it with use_text=False.
    Missing rows REQUIRE bank_ready even with text disabled. Passing None as
    the third positional argument is deliberately rejected; old AddFusion call
    sites must pass their actual source state instead.
    """
    def __init__(
        self, channels: Sequence[int] = (64, 128, 320, 512),
        latent_dims: Optional[Sequence[int]] = None,
        interaction_ranks: Optional[Sequence[int]] = None,
        text_enabled: bool = False, text_embeddings: Optional[Tensor] = None,
        text_width: int = 32, text_strength: float = 0.5,
        reliability_min: float = 0.2, sigma_init: float = 1.5,
        logvar_bounds: Sequence[float] = (-10.0, 10.0),
        check_finite: bool = False, text_metadata: Optional[dict] = None,
    ):
        super().__init__()
        channels = tuple(channels)
        if not channels or any(type(c) is not int or c < 1 for c in channels):
            raise ValueError('channels must contain positive integers in shallow-to-deep order')
        latent_dims = tuple(latent_dims) if latent_dims is not None else tuple(max(32, c // 4) for c in channels)
        interaction_ranks = tuple(interaction_ranks) if interaction_ranks is not None else tuple(min(c, max(16, c // 8)) for c in channels)
        for name, dims in (('latent_dims', latent_dims), ('interaction_ranks', interaction_ranks)):
            if len(dims) != len(channels) or any(type(d) is not int or d < 1 for d in dims):
                raise ValueError(f'{name} must match channels and contain positive integers')
        if type(text_width) is not int or text_width < 2:
            raise ValueError('text_width must be an integer >=2')
        if not 0 <= text_strength < 1:
            raise ValueError('text_strength must be in [0,1)')
        if not 0 < reliability_min <= 1:
            raise ValueError('reliability_min must be in (0,1]')
        if len(logvar_bounds) != 2 or not all(math.isfinite(v) for v in logvar_bounds) or logvar_bounds[0] >= logvar_bounds[1]:
            raise ValueError('logvar_bounds must be two finite increasing values')
        if not math.isfinite(sigma_init) or sigma_init <= 0 or not logvar_bounds[0] <= 2 * math.log(sigma_init) <= logvar_bounds[1]:
            raise ValueError('sigma_init must be positive with log-variance inside logvar_bounds')
        if text_enabled and text_embeddings is None:
            raise ValueError('Text enabled requires real cached embeddings; use from_text_cache/from_local_text_encoder')
        self.channels = channels
        self.text_enabled = bool(text_enabled)
        self.check_finite = bool(check_finite)
        self.text_metadata = json.loads(json.dumps(text_metadata or {}))
        self.config = dict(channels=list(channels), latent_dims=list(latent_dims),
                           interaction_ranks=list(interaction_ranks), text_enabled=self.text_enabled,
                           text_width=text_width, text_strength=float(text_strength),
                           reliability_min=float(reliability_min), sigma_init=float(sigma_init),
                           logvar_bounds=list(logvar_bounds), check_finite=self.check_finite)
        self.ct_blocks = nn.ModuleList([GaussianResidual(c, d, reliability_min, sigma_init, logvar_bounds)
                                       for c, d in zip(channels, latent_dims)])
        self.pet_blocks = nn.ModuleList([GaussianResidual(c, d, reliability_min, sigma_init, logvar_bounds)
                                        for c, d in zip(channels, latent_dims)])
        self.interactions = nn.ModuleList([LowRankInteraction(c, k) for c, k in zip(channels, interaction_ranks)])
        self.text_conditioner = PETTextConditioner(_embeddings(text_embeddings), text_width) if self.text_enabled else None
        self.pet_modulators = nn.ModuleList([PETPriorModulation(c, text_width, text_strength) for c in channels]
                                           if self.text_enabled else [])

    @classmethod
    def from_text_cache(cls, path: Union[str, Path], **kwargs) -> 'ProbabilisticPETFusion':
        embeddings, metadata = load_text_cache(path)
        return cls(text_enabled=True, text_embeddings=embeddings, text_metadata=metadata, **kwargs)

    @classmethod
    def from_local_text_encoder(cls, text_tower_path: Union[str, Path] = DEFAULT_TEXT_TOWER,
                                biomedclip_path: Union[str, Path] = DEFAULT_BIOMEDCLIP,
                                backend: str = 'biomedclip', encoder_device: str = 'cpu',
                                **kwargs) -> 'ProbabilisticPETFusion':
        embeddings, metadata = encode_local_pet_prompts(text_tower_path, biomedclip_path, backend, encoder_device)
        return cls(text_enabled=True, text_embeddings=embeddings, text_metadata=metadata, **kwargs)

    def get_extra_state(self) -> dict:
        return {'format_version': FORMAT_VERSION, 'config': self.config,
                'prompts': list(PROMPTS), 'text_metadata': self.text_metadata}

    def set_extra_state(self, state: dict) -> None:
        if state.get('format_version') != FORMAT_VERSION or state.get('config') != self.config or state.get('prompts') != list(PROMPTS):
            raise RuntimeError('Module-2 checkpoint configuration mismatch. Reconstruct with saved config; do not silently change gates/prompts.')
        self.text_metadata = state.get('text_metadata', {})

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        torch.save({'format_version': FORMAT_VERSION, 'config': self.config,
                    'state_dict': self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: str = 'cpu') -> 'ProbabilisticPETFusion':
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != FORMAT_VERSION:
            raise ValueError('Unsupported Module-2 checkpoint format')
        state = payload['state_dict']
        embeddings = state.get('text_conditioner.fixed_embeddings')
        model = cls(**payload['config'], text_embeddings=embeddings)
        model.load_state_dict(state, strict=True)
        return model.to(device)

    def parameter_report(self) -> dict[str, int]:
        count = lambda module: sum(p.numel() for p in module.parameters()) if module is not None else 0
        text_core = 0
        if self.text_conditioner is not None:
            text_core = (count(self.text_conditioner.attn) + count(self.text_conditioner.norm)
                         + self.text_conditioner.token.numel())
        return {'trainable_total': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'ct_gaussian': count(self.ct_blocks), 'pet_gaussian': count(self.pet_blocks),
                'low_rank_interaction': count(self.interactions),
                'text_attention_core': text_core,
                'text_conditioner_total': count(self.text_conditioner),
                'pet_modulation_total': count(self.pet_modulators)}

    def forward(
        self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
        pet_available: Optional[State] = None, *, bank_ready: Optional[State] = None,
        use_text: Optional[bool] = None, return_diagnostics: bool = False,
    ):
        if not isinstance(ct_feats, (list, tuple)) or not isinstance(pet_feats, (list, tuple)):
            raise TypeError('ct_feats and pet_feats must be lists/tuples of NCHW tensors')
        if len(ct_feats) != len(self.channels) or len(pet_feats) != len(self.channels):
            raise ValueError('Both feature lists must match the configured scale count')
        ref = ct_feats[0]
        if not isinstance(ref, Tensor) or ref.ndim != 4 or ref.shape[0] < 1:
            raise ValueError('CT features must be nonempty NCHW tensors')
        batch, device = ref.shape[0], ref.device
        for scale, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, self.channels)):
            for label, x in (('CT', ct), ('PET', pet)):
                if not isinstance(x, Tensor) or x.ndim != 4 or not x.is_floating_point():
                    raise ValueError(f'S{scale+1} {label} must be a floating NCHW tensor')
                if x.shape[0] != batch or x.shape[1] != channels or min(x.shape[-2:]) < 1:
                    raise ValueError(f'S{scale+1} {label} shape mismatch: {tuple(x.shape)}')
                if x.device != device or x.dtype != ref.dtype:
                    raise ValueError('All CT/PET scales must share device and floating dtype')
            if ct.shape != pet.shape:
                raise ValueError(f'S{scale+1} CT/PET shapes must already be aligned; no silent interpolation')
        if next(self.parameters()).device != device:
            raise ValueError('Move Module-2 to the feature device before forward')
        if pet_available is None:
            raise ValueError('pet_available is required: 1=real PET, 0=imputed PET; replace old AddFusion(..., None) calls')
        real = _binary_state(pet_available, batch, device, 'pet_available')
        if bank_ready is None:
            if (~real).any():
                raise ValueError('bank_ready is required when any PET is imputed (cold-start contract)')
            ready = torch.ones_like(real)
        else:
            ready = _binary_state(bank_ready, batch, device, 'bank_ready', allow_scalar=True)
        active = real | ready
        rows = torch.nonzero(active, as_tuple=False).flatten()
        text_on = self.text_enabled if use_text is None else bool(use_text)
        if text_on and not self.text_enabled:
            raise ValueError('Cannot enable unconstructed text layers; construct from a cache before creating the optimizer')
        context = self.text_conditioner(real[rows]) if text_on and rows.numel() else None
        outputs, scale_stats = [], []
        for i, (ct, pet) in enumerate(zip(ct_feats, pet_feats)):
            if self.check_finite:
                _finite(f'S{i+1} CT', ct)
                if rows.numel(): _finite(f'S{i+1} active PET', pet[rows])
            ec, ct_stats = self.ct_blocks[i](ct, return_diagnostics)
            ep = torch.zeros_like(ct)
            pet_stats, gain = {}, None
            if rows.numel():
                selected, pet_stats = self.pet_blocks[i](pet[rows], return_diagnostics)
                if context is not None:
                    selected, gain = self.pet_modulators[i](selected, context)
                ep = ep.index_copy(0, rows, selected)
            interaction = self.interactions[i](ec, ep)
            fused = ec + ep + interaction
            if self.check_finite: _finite(f'S{i+1} fused', fused)
            outputs.append(fused)
            if return_diagnostics:
                scale_stats.append({'ct': ct_stats, 'pet': pet_stats,
                                    'pet_gain_mean': gain.detach().mean() if gain is not None else ct.new_tensor(1.),
                                    'interaction_rms': interaction.detach().float().square().mean().sqrt()})
        if return_diagnostics:
            return outputs, {'scales': scale_stats, 'pet_active': active.detach(),
                             'pet_available': real.detach(), 'text_enabled': text_on}
        return outputs


def _smoke(device: str, text_cache: Optional[str], full_resolution: bool, batch_size: int) -> None:
    torch.manual_seed(2023)
    if device == 'cpu': torch.set_num_threads(min(4, torch.get_num_threads()))
    model = (ProbabilisticPETFusion.from_text_cache(text_cache, check_finite=True)
             if text_cache else ProbabilisticPETFusion(check_finite=True)).to(device)
    sizes = (128, 64, 32, 16) if full_resolution else (16, 8, 4, 2)
    ct = [torch.randn(batch_size, c, s, s, device=device, requires_grad=True)
          for c, s in zip(model.channels, sizes)]
    pet = [torch.randn_like(x, requires_grad=True) for x in ct]
    state = torch.arange(batch_size, device=device).remainder(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for x in ct + pet: x.grad = None
        outputs = model(ct, pet, state, bank_ready=(step > 0))
        if step == 0:
            for x, y, z in zip(ct, pet, outputs):
                expected = x + torch.where(state[:, None, None, None].bool(), y, torch.zeros_like(y))
                torch.testing.assert_close(z, expected, rtol=0, atol=0)
        loss = sum(x.float().square().mean() for x in outputs)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None: _finite(name + '.grad', param.grad)
        optimizer.step()
        print(f'step={step+1} ready={step>0} loss={loss.item():.6f}')
    model.eval()
    with torch.no_grad():
        first = model(ct, pet, state, bank_ready=True)
        second = model(ct, pet, state, bank_ready=True)
        for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)
    print(json.dumps({'parameters': model.parameter_report(), 'shapes': [list(x.shape) for x in first],
                      'device': device, 'text': text_cache is not None, 'smoke': 'PASS'}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare-text-cache', metavar='PATH')
    parser.add_argument('--text-backend', choices=('biomedbert', 'biomedclip'), default='biomedclip')
    parser.add_argument('--text-tower-path', default=DEFAULT_TEXT_TOWER)
    parser.add_argument('--biomedclip-path', default=DEFAULT_BIOMEDCLIP)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--text-cache')
    parser.add_argument('--full-resolution', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.prepare_text_cache:
        embeddings, metadata = encode_local_pet_prompts(args.text_tower_path, args.biomedclip_path,
                                                        args.text_backend, args.device)
        save_text_cache(args.prepare_text_cache, embeddings, metadata)
        print(f'Saved {tuple(embeddings.shape)} fixed PET embeddings to {args.prepare_text_cache}')
    if args.smoke_test:
        if args.batch_size < 1: parser.error('--batch-size must be positive')
        _smoke(args.device, args.text_cache, args.full_resolution, args.batch_size)
    if not args.prepare_text_cache and not args.smoke_test: parser.print_help()


if __name__ == '__main__':
    main()
