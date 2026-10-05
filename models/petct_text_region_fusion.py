"""PET text enhanced, CT queried regional fusion (2D, standalone PyTorch).

Designed for aligned ConvNeXt-CT / MiT-PET feature pyramids. This is a proposed
adaptation, not an implementation of the original CIPA Mamba blocks.

Full:     B_CT = Conv1x1([C, L_CT]); F = B_CT + D(PET, CT).
CT-only:  F = B_CT (explicit ablation).
Missing:  F = B_CT + external D_hat (retrieval is NOT implemented here).

No post-attention sigmoid gate, learnable residual gamma, Mamba, or bank.
Frozen text is cached in a persistent buffer; no text encoder in forward.
Only torch is required unless load_offline_pet_text() is explicitly called.
"""
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PET_PROMPT = 'A PET image showing bright tumor regions in the lungs.'


@torch.no_grad()
def load_offline_pet_text(clip_path: str, prompt: str = PET_PROMPT) -> Tensor:
    """Return CPU float32 [1,D] from a local Hugging Face CLIP directory.

    Uses CLIPTextModel.pooler_output, matching the target repo's CLIP loader.
    A standalone OpenAI .pt file is not this format. Never accesses the network.
    The caller caches the returned tensor in the fusion module, then this
    temporary CPU model is released. On resume inject the checkpoint buffer
    instead, so the CLIP directory is unnecessary for restoring a trained model.
    """
    path = Path(clip_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Offline Hugging Face CLIP directory missing: {path}')
    try:
        from transformers import AutoTokenizer, CLIPTextModel
    except ImportError as exc:
        raise ImportError('Install transformers to encode text once, or pass text_embeddings.') from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(str(path), local_files_only=True).cpu().float().eval()
    encoder.requires_grad_(False)
    inputs = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True)
    output = encoder(**inputs).pooler_output.detach().float().cpu()
    if output.ndim != 2 or output.shape[0] != 1 or not torch.isfinite(output).all():
        raise RuntimeError('Invalid offline CLIP pooled embedding')
    return output


def _block(cin: int, cout: int, kernel: int = 1, stride: int = 1, groups: int = 1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride=stride,
                                  padding=kernel//2, groups=groups, bias=False),
                         nn.GroupNorm(1, cout), nn.GELU())


class PETTextChannelEnhancement(nn.Module):
    """GMP(P) * projected fixed text -> sigmoid channel weights; P + P*s."""
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, channels)
        self.image_norm = nn.LayerNorm(channels)
        self.text_norm = nn.LayerNorm(channels)
        hidden = max(channels//4, 8)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels))
        # Small nonzero final weights allow gradients into text_proj from step 1.
        # sigmoid(-4) ~= .018, so this starts near (but not exactly at) identity.
        nn.init.normal_(self.mlp[-1].weight, std=.01)
        nn.init.constant_(self.mlp[-1].bias, -4.)

    def forward(self, pet: Tensor, text: Tensor, return_weights: bool = False):
        pooled = pet.amax(dim=(-2, -1))
        projected = self.text_proj(text)
        h = self.image_norm(pooled) * self.text_norm(projected)
        weights = self.mlp(h).sigmoid().to(pet.dtype)[:, :, None, None]
        enhanced = pet + pet * weights
        return (enhanced, weights) if return_weights else enhanced


class RegionalCrossAttention(nn.Module):
    """Each CT pixel reads up to 9 PET tokens around its matching 4x4 region.

    CT queries are grouped by region. Nine regional K/V vectors are shared by
    the region's 16 queries, without duplicating V for every pixel. Attention
    logits: [B, region_chunk, heads, 16, 9], never [HW,HW]. Borders are masked.
    Query padding is cropped away; the module accepts non-divisible H/W.
    """
    def __init__(self, dim: int, num_heads: int, region_size: int = 4,
                 region_chunk_size: int = 128, checkpoint_attention: bool = False):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError('dim must be positive and divisible by positive num_heads')
        if region_size != 4 or region_chunk_size < 1:
            raise ValueError('This implementation uses region_size=4 and positive chunk size')
        self.dim, self.num_heads = dim, num_heads
        self.region_size, self.region_chunk_size = region_size, region_chunk_size
        self.checkpoint_attention = bool(checkpoint_attention)
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

    @staticmethod
    def _read(q: Tensor, k: Tensor, v: Tensor, valid: Tensor) -> Tensor:
        # Explicit FP32 scores/softmax for AMP stability; values keep AMP dtype.
        with torch.autocast(device_type=q.device.type, enabled=False):
            score = torch.matmul(q.float(), k.float().transpose(-2,-1)) * (q.shape[-1] ** -.5)
            score = score.masked_fill(~valid, float('-inf'))
            weights = score.softmax(dim=-1)
        return torch.matmul(weights.to(v.dtype), v)

    def forward(self, local_ct: Tensor, pet_regions: Tensor) -> Tensor:
        b, d, h, w = local_ct.shape
        r, heads = self.region_size, self.num_heads
        gh, gw = (h+r-1)//r, (w+r-1)//r
        if pet_regions.shape != (b,d,gh,gw):
            raise ValueError(f'PET regions must be {(b,d,gh,gw)}, got {tuple(pet_regions.shape)}')
        q = F.pad(self.q(local_ct), (0,gw*r-w,0,gh*r-h))
        # B,D,GH,4,GW,4 -> B,R,heads,16,head_dim
        q = q.reshape(b,d,gh,r,gw,r).permute(0,2,4,3,5,1)
        q = q.reshape(b,gh*gw,r*r,heads,d//heads).permute(0,1,3,2,4)

        def neighbors(x):
            x = F.unfold(x, kernel_size=3, padding=1)
            x = x.reshape(b,heads,d//heads,9,gh*gw)
            return x.permute(0,4,1,3,2)

        k, v = neighbors(self.k(pet_regions)), neighbors(self.v(pet_regions))
        valid = F.unfold(local_ct.new_ones((1,1,gh,gw)), 3, padding=1)
        valid = valid.transpose(1,2).bool()[:, :, None, None, :]
        outputs = []
        for start in range(0, gh*gw, self.region_chunk_size):
            end = start + self.region_chunk_size
            args = (q[:,start:end], k[:,start:end], v[:,start:end], valid[:,start:end])
            if self.training and torch.is_grad_enabled() and self.checkpoint_attention:
                out = checkpoint(self._read, *args, use_reentrant=False)
            else:
                out = self._read(*args)
            outputs.append(out)
        out = torch.cat(outputs, dim=1).permute(0,1,3,2,4)
        out = out.reshape(b,gh,gw,r,r,d).permute(0,5,1,3,2,4)
        return out.reshape(b,d,gh*r,gw*r)[:, :, :h, :w].contiguous()


class _FusionStage(nn.Module):
    def __init__(self, channels, dim, heads, text_dim, use_text, chunk, checkpoint_attention):
        super().__init__()
        self.ct_local = nn.Sequential(_block(channels, dim), _block(dim, dim, 3, groups=dim))
        self.ct_base = nn.Conv2d(channels+dim, channels, 1, bias=True)
        # CT baseline starts as C. Local CT still receives gradients via queries.
        with torch.no_grad():
            self.ct_base.weight.zero_(); self.ct_base.bias.zero_()
            self.ct_base.weight[:, :channels, 0, 0].copy_(torch.eye(channels))
        self.text_gate = PETTextChannelEnhancement(channels, text_dim) if use_text else None
        # Two stride-2 layers: the official CIPA region correspondence is 4x4.
        # This light stem adapts it; it is not a copy of CIPA's whole stem.
        self.pet_stem = nn.Sequential(_block(channels, dim),
                                      _block(dim, dim, 3, stride=2, groups=dim),
                                      _block(dim, dim, 3, stride=2, groups=dim))
        self.pet_context = _block(dim, dim, 3, groups=dim)
        self.attention = RegionalCrossAttention(dim, heads, 4, chunk, checkpoint_attention)
        self.pet_out = nn.Conv2d(dim, channels, 1, bias=False)

    def base(self, ct):
        local = self.ct_local(ct)
        return self.ct_base(torch.cat([ct, local], dim=1)), local

    def delta(self, local, pet, text):
        if self.text_gate is not None:
            pet = self.text_gate(pet, text)
        h, w = pet.shape[-2:]
        # Replicate padding only at bottom/right, same 4x4 grouping as CT queries.
        pet = F.pad(pet, (0,(-w)%4,0,(-h)%4), mode='replicate')
        region = self.pet_stem(pet)
        region = region + self.pet_context(region)
        return self.pet_out(self.attention(local, region))


class PETCTTextRegionFusion(nn.Module):
    """Multi-scale fusion. CT/PET must already share per-scale channels and grids.

    forward(ct_features, pet_features, state='full') -> list of fused tensors.
    forward(ct_features, state='ct_only') -> CT-only ablation.
    forward(ct_features, state='missing', retrieved_deltas=D_hat) -> CT+D_hat.
    return_components=True -> (fused, {'ct_base': list, 'pet_delta': list}).

    D_hat is AFTER the learned PET output projection, shape [B,C_l,H_l,W_l].
    Component tensors retain gradients; detach explicitly when writing a bank.
    This module does not cache per-batch feature tensors.
    """
    def __init__(self, channels: Sequence[int] = (64,128,320,512),
                 inner_channels: Sequence[int] = (64,128,160,256),
                 num_heads: int = 4, use_text: bool = True,
                 text_embeddings: Optional[Tensor] = None, clip_path: Optional[str] = None,
                 prompt: str = PET_PROMPT, region_chunk_size: int = 128,
                 checkpoint_attention: bool = True):
        super().__init__()
        self.channels = tuple(channels)
        self.inner_channels = tuple(inner_channels)
        if not self.channels or len(self.channels) != len(self.inner_channels):
            raise ValueError('channels and inner_channels must have equal nonzero lengths')
        if any(not isinstance(c,int) or c < 1 for c in self.channels+self.inner_channels):
            raise ValueError('All channel counts must be positive integers')
        self.use_text, self.num_heads, self.prompt = bool(use_text), num_heads, prompt
        if self.use_text:
            if text_embeddings is None:
                if clip_path is None:
                    raise ValueError('use_text=True needs cached text_embeddings or an offline clip_path')
                text_embeddings = load_offline_pet_text(clip_path, prompt)
            if text_embeddings.ndim == 1:
                text_embeddings = text_embeddings[None]
            if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 1 or text_embeddings.shape[1] < 2:
                raise ValueError('Provide only the PET text vector [1,D]; select row 1 from old [CT,PET] cache')
            if not torch.isfinite(text_embeddings).all():
                raise ValueError('text_embeddings contains NaN/Inf')
            text = text_embeddings.detach().to(device='cpu', dtype=torch.float32).clone()
        else:
            text = torch.empty(0,0)
        self.register_buffer('text_embeddings', text, persistent=True)
        self.stages = nn.ModuleList([
            _FusionStage(c,d,num_heads,text.shape[-1],self.use_text,region_chunk_size,checkpoint_attention)
            for c,d in zip(self.channels,self.inner_channels)])

    def get_extra_state(self):
        return dict(version=1, channels=self.channels, inner_channels=self.inner_channels,
                    num_heads=self.num_heads, region_size=4, neighborhood=3,
                    use_text=self.use_text, text_dim=self.text_embeddings.shape[-1], prompt=self.prompt)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError('Fusion checkpoint architecture/text contract does not match constructor')

    def _validate(self, values, name, reference=None):
        if not isinstance(values,(list,tuple)) or len(values) != len(self.channels):
            raise ValueError(f'{name} must have {len(self.channels)} scale tensors')
        batch, device = None, None
        for i,(x,c) in enumerate(zip(values,self.channels)):
            if not isinstance(x,Tensor) or x.ndim != 4 or x.shape[1] != c or min(x.shape) < 1:
                raise ValueError(f'{name}[{i}] must be nonempty [B,{c},H,W]')
            if not x.is_floating_point():
                raise TypeError(f'{name}[{i}] must be floating point')
            if batch is None:
                batch,device = x.shape[0],x.device
            if x.shape[0] != batch or x.device != device:
                raise ValueError(f'{name} scales must share batch and device')
            if reference is not None and (x.shape != reference[i].shape or x.device != reference[i].device):
                raise ValueError(f'{name}[{i}] must match CT shape and device exactly; no hidden resizing')

    def forward(self, ct_features, pet_features=None, *, state='full',
                retrieved_deltas=None, return_components=False):
        if state not in ('full','ct_only','missing'):
            raise ValueError(f'Unknown fusion state: {state!r}')
        self._validate(ct_features, 'ct_features')
        if state == 'full':
            self._validate(pet_features, 'pet_features', ct_features)
            if retrieved_deltas is not None:
                raise ValueError('Full uses real PET, not retrieved_deltas')
        else:
            if pet_features is not None:
                raise ValueError('Do not supply PET in ct_only/missing; avoid hidden PET leakage')
            if state == 'missing':
                self._validate(retrieved_deltas, 'retrieved_deltas', ct_features)
            elif retrieved_deltas is not None:
                raise ValueError('ct_only does not accept a delta')
        fused, bases, deltas = [], [], []
        for i,(ct,stage) in enumerate(zip(ct_features,self.stages)):
            base,local = stage.base(ct)
            if state == 'full':
                delta = stage.delta(local, pet_features[i].to(ct.dtype), self.text_embeddings)
            elif state == 'missing':
                delta = retrieved_deltas[i]
            else:
                delta = torch.zeros_like(base)
            delta = delta.to(base.dtype)
            # Keep repository feature dtype; mixed precision computation stays inside blocks.
            base,delta = base.to(ct.dtype),delta.to(ct.dtype)
            fused.append(base+delta); bases.append(base); deltas.append(delta)
        if return_components:
            return fused, {'ct_base':bases, 'pet_delta':deltas}
        return fused
