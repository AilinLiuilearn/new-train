"""State-guided, scale-shared CT / real-PET / imputed-PET expert fusion.

Standalone runtime dependency: PyTorch >= 2.1. No repository imports, downloads,
text encoder construction, losses or prototype-bank updates in forward().

Input: four aligned CT maps and four PET maps [B,C_l,H_l,W_l]. PET means
real encoder features for Full, RAW retrieved Module-1 features for Missing.
Output: (four decoder-ready fused maps of the same shapes, detached diagnostics).
Missing with bank_ready=False is an exact CT-only bypass, including beta bias.

This is a task-specific adaptation, not an official TG-ECNet implementation.
References: https://github.com/LeeX54946/TG-ECNet
Text tower loader reuses OpenCLIP HFTextEncoder, including the learned projection:
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/hf_model.py
Verified integration target: AilinLiuilearn/new-train,
branch e1-api-masked-baseline-PSPI-module1-clean (see README_MODULE2.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROMPTS = (
    "Focus on anatomical structures and fine boundaries in CT features.",
    "Focus on patient-specific metabolic activity in real PET features.",
    "Focus on metabolic information in imputed PET features.",
    "Fuse CT features with real PET features.",
    "Fuse CT features with imputed PET features.",
)
GROUPS = ('ct', 'real', 'imputed')
FORMAT_VERSION = 1


def channel_norm(x: Tensor) -> Tensor:
    """Per-pixel channel LayerNorm; no sample coupling or running statistics."""
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), eps=1e-5).permute(0, 3, 1, 2)


def _check_embeddings(embeddings: Tensor) -> Tensor:
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2 or embeddings.shape[0] != 5 or embeddings.shape[1] < 1:
        raise ValueError('text_embeddings must be [5,D], ordered as PROMPTS')
    value = embeddings.detach().float().cpu().contiguous().clone()
    if not torch.isfinite(value).all() or (value.norm(dim=-1) == 0).any():
        raise ValueError('Text embeddings must be finite and nonzero')
    return value


def save_text_cache(path, embeddings: Tensor, metadata: Optional[dict] = None) -> None:
    """Save once before training. No text encoder is registered in the model."""
    value = _check_embeddings(embeddings)
    meta = json.loads(json.dumps(metadata or {}))  # JSON-safe checkpoint metadata
    torch.save({'version': FORMAT_VERSION, 'prompts': list(PROMPTS),
                'embeddings': value, 'metadata': meta}, path)


def load_text_cache(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != FORMAT_VERSION or payload.get('prompts') != list(PROMPTS):
        raise ValueError('Text cache version / prompt content / prompt order mismatch; rebuild cache')
    return _check_embeddings(payload['embeddings']), payload.get('metadata', {})


@torch.no_grad()
def encode_local_text_prompts(text_tower_path, biomedclip_path=None,
                              backend='biomedclip', device='cpu'):
    """Strict local-only five-prompt encoding.

    biomedclip: local HF config/tokenizer + official full OpenCLIP checkpoint.
      Loads ONLY text.* weights into HFTextEncoder, with the trained projection.
    biomedbert: local HF pretrained model, masked mean pooling; a different
      embedding space, explicitly recorded in metadata. Never a silent fallback.
    Dependencies are imported only here. The training module needs only PyTorch.
    """
    tower_path = Path(text_tower_path).expanduser().resolve()
    if not tower_path.is_dir():
        raise FileNotFoundError(f'Local text tower directory not found: {tower_path}')
    if backend not in ('biomedclip', 'biomedbert'):
        raise ValueError('backend must be biomedclip or biomedbert')
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tower_path), local_files_only=True)
    meta = {'backend': backend, 'text_tower_path': str(tower_path)}
    if backend == 'biomedclip':
        if biomedclip_path is None:
            raise ValueError('biomedclip backend requires biomedclip_path')
        model_path = Path(biomedclip_path).expanduser().resolve()
        cfg_path = model_path / 'open_clip_config.json'
        if not cfg_path.is_file():
            raise FileNotFoundError(f'Missing official model configuration: {cfg_path}')
        model_cfg = json.loads(cfg_path.read_text())['model_cfg']
        text_cfg = model_cfg['text_cfg']
        config = AutoConfig.from_pretrained(str(tower_path), local_files_only=True)
        from open_clip.hf_model import HFTextEncoder
        # Providing config and pretrained=False prevents HFTextEncoder from
        # fetching a hub config/model. Local checkpoint supplies all weights.
        encoder = HFTextEncoder(
            model_name_or_path=str(tower_path), config=config,
            output_dim=int(model_cfg['embed_dim']),
            pooler_type=text_cfg['hf_pooler_type'],
            proj_type=text_cfg['hf_proj_type'], pretrained=False,
        )
        # Pooler compatibility across OpenCLIP 2.x + Transformers 4.x:
        # with pooler_type != 'cls_pooler' (BiomedCLIP uses
        # cls_last_hidden_state_pooler), the official text.* checkpoint and
        # OpenCLIP's config-less pretrained=False path omit the UNUSED HF
        # transformer.pooler, but Transformers 4.38 BertModel.forward checks
        # `self.pooler is not None` and crashes if the attribute is deleted.
        # Keep an UNREGISTERED empty BertPooler for attribute presence only:
        # it is never called by the CLS pooling path and never enters
        # strict state_dict (registered keys match the checkpoint exactly).
        if text_cfg['hf_pooler_type'] != 'cls_pooler' and config.model_type in ('bert','roberta','xlm-roberta'):
            if hasattr(encoder.transformer,'pooler'):
                from transformers.models.bert.modeling_bert import BertPooler
                empty_pooler = BertPooler(config)
                with torch.no_grad():
                    empty_pooler.dense.weight.zero_()
                    empty_pooler.dense.bias.zero_()
                encoder.transformer._modules.pop('pooler', None)
                object.__setattr__(encoder.transformer, 'pooler', empty_pooler)
        checkpoint_path = model_path / 'open_clip_pytorch_model.bin'
        if checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        else:
            checkpoint_path = model_path / 'open_clip_model.safetensors'
            if not checkpoint_path.is_file():
                raise FileNotFoundError('Expected open_clip_pytorch_model.bin or open_clip_model.safetensors in '+str(model_path))
            from safetensors.torch import load_file
            state = load_file(str(checkpoint_path), device='cpu')
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k,v in state.items()}
        text_state = {k[len('text.'):]: v for k,v in state.items() if k.startswith('text.')}
        if not text_state:
            raise ValueError('Official full BiomedCLIP checkpoint must contain text.* weights')
        # Official checkpoints carry a registered position_ids buffer that the
        # current HFTextEncoder/transformers layer does not register as a
        # parameter or persistent buffer. Drop ONLY this exact buffer key;
        # every other text.* weight must match the encoder exactly.
        text_state = {k: v for k, v in text_state.items()
                      if k != 'transformer.embeddings.position_ids'}
        try:
            encoder.load_state_dict(text_state, strict=True)
        except RuntimeError as error:
            raise RuntimeError('BiomedCLIP text weights/config mismatch; use matching OpenCLIP and Transformers versions. No partial/random fallback.') from error
        del state, text_state
        max_length = int(text_cfg['context_length'])
        meta.update(biomedclip_path=str(model_path), checkpoint=str(checkpoint_path),
                    pooler=text_cfg['hf_pooler_type'], projection=text_cfg['hf_proj_type'])
    else:
        encoder = AutoModel.from_pretrained(str(tower_path), local_files_only=True)
        max_length = min(256, int(encoder.config.max_position_embeddings))
        meta.update(pooler='masked_mean', projection='none')
    encoder = encoder.to(device).eval().requires_grad_(False)
    tokens = tokenizer(list(PROMPTS), padding='max_length', truncation=True,
                       max_length=max_length, return_tensors='pt').to(device)
    if backend == 'biomedclip':
        # OpenCLIP masks pad tokens internally; no version-specific forward kwarg.
        if tokenizer.pad_token_id != config.pad_token_id:
            raise ValueError('Tokenizer/model pad_token_id mismatch')
        embeddings = encoder(tokens['input_ids'])
    else:
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    embeddings = F.normalize(embeddings.float(), dim=-1).cpu()
    meta['dimension'] = embeddings.shape[1]
    meta['embedding_sha256'] = hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()
    return _check_embeddings(embeddings), meta


class SpatialPETPersonalization(nn.Module):
    """Direct gamma(CT.detach()) * prior + beta(CT.detach()); ordinary init."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 4)
        self.generator = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, 2*channels, 1),
        )

    def forward(self, ct: Tensor, prior: Tensor) -> Tensor:
        gamma, beta = self.generator(channel_norm(ct.detach())).chunk(2, dim=1)
        return gamma * prior + beta


class SharedExpert(nn.Module):
    """Same expert object is reused at all four scales; no batch normalization."""
    def __init__(self, width: int, text_dim: int):
        super().__init__()
        self.text_film = nn.Linear(text_dim, 2*width) if text_dim else None
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x: Tensor, group_text: Optional[Tensor]) -> Tensor:
        x = channel_norm(x)
        if self.text_film is not None:
            scale, shift = self.text_film(group_text).chunk(2, dim=-1)
            x = (1 + scale[None,:,None,None]) * x + shift[None,:,None,None]
        return self.body(x)


class ScaleRouter(nn.Module):
    """Joint fixed+learnable state prompt multiplies visual GAP/GMP descriptor.

    Attention operates on TWO tokens (GAP and GMP), not a singleton token.
    Group-constrained top-1 CT + top-1 active PET, joint two-score softmax.
    """
    def __init__(self, width: int, n: int, text_dim: int, heads: int, noise_std: float):
        super().__init__()
        self.n, self.width, self.noise_std = n, width, noise_std
        self.interaction = nn.Conv2d(3*width, width, 1)
        self.state_projection = nn.Linear(text_dim, width, bias=False) if text_dim else None
        self.learned_prompt = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.learned_prompt, std=.02)
        self.prompt_gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,width),
                                         nn.GELU(), nn.Linear(width,2*width))
        self.token_type = nn.Parameter(torch.empty(1,2,width))
        nn.init.normal_(self.token_type, std=.02)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width,3*n))

    def forward(self, ct: Tensor, pet: Tensor, state: int, state_text: Optional[Tensor]):
        z = self.interaction(torch.cat((ct, pet, ct*pet),dim=1))
        v = torch.cat((z.mean((2,3)), z.amax((2,3))),dim=1)
        prompt = self.learned_prompt[state]
        if self.state_projection is not None:
            prompt = prompt + self.state_projection(state_text)
        gate = 2 * torch.sigmoid(self.prompt_gate(prompt))
        tokens = (v * gate).reshape(v.shape[0],2,self.width) + self.token_type
        normed = self.attention_norm(tokens)
        attended, _ = self.attention(normed,normed,normed,need_weights=False)
        logits = self.readout((tokens+attended).flatten(1)).float()
        if self.training and self.noise_std:
            logits = logits + self.noise_std*torch.randn_like(logits)
        pet_offset = (1 if state == 0 else 2)*self.n
        ct_scores = logits[:, :self.n]
        pet_scores = logits[:, pet_offset:pet_offset+self.n]
        ct_id, pet_id = ct_scores.argmax(1), pet_scores.argmax(1)
        selected = torch.stack((ct_scores.gather(1,ct_id[:,None])[:,0],
                                pet_scores.gather(1,pet_id[:,None])[:,0]),dim=1)
        # Do not separately softmax single selected scores: that would make
        # each weight 1 and eliminate useful router gradients (even for n=1).
        weights = selected.softmax(-1)
        return ct_id, pet_id, weights


class StateGuidedExpertFusion(nn.Module):
    """Drop-in feature-fusion API, placed after Module-1 and before decoder.

    mode='full': pet_feats contains real PET; bank_ready does not gate Full.
    mode='missing': pet_feats contains unscaled retrieved prior, never real PET.
    mode='auto': pet_available[B] must be binary; each row of pet_feats already
      contains its appropriate real-PET / retrieved-prior feature. The caller
      must perform this selection without leaking real PET into missing rows.

    Default use_text=False needs no text packages/cache. For text use
    from_text_cache(...). Experts/group is positive int: 1, 2, 3, ... .
    The same n applies to all three groups, all scales; only TWO experts are
    executed per sample per ready scale. No route/reconstruction auxiliary loss.
    """
    def __init__(self, channels: Sequence[int] = (64,128,320,512), width: int = 32,
                 experts_per_group: int = 2, use_text: bool = False,
                 text_embeddings: Optional[Tensor] = None, text_metadata: Optional[dict] = None,
                 personalization: bool = True, attention_heads: int = 2,
                 router_noise_std: float = .1):
        super().__init__()
        if len(channels) != 4 or any(type(c) is not int or c < 2 for c in channels):
            raise ValueError('channels must contain four integer channel counts >= 2')
        if type(experts_per_group) is not int or experts_per_group < 1:
            raise ValueError('experts_per_group must be a positive integer')
        if type(width) is not int or type(attention_heads) is not int or width < 2 or attention_heads < 1 or width % attention_heads:
            raise ValueError('width >= 2 must be divisible by positive attention_heads')
        if not math.isfinite(router_noise_std) or router_noise_std < 0:
            raise ValueError('router_noise_std must be finite and nonnegative')
        if use_text:
            value = _check_embeddings(text_embeddings)
        else:
            if text_embeddings is not None:
                raise ValueError('use_text=False must not receive text embeddings')
            value = torch.empty(0,0)
        self.register_buffer('text_embeddings',value,persistent=True)
        self.text_metadata = json.loads(json.dumps(text_metadata or {}))
        self.config = dict(channels=list(channels), width=width,
            experts_per_group=experts_per_group, use_text=bool(use_text),
            personalization=bool(personalization), attention_heads=attention_heads,
            router_noise_std=float(router_noise_std))
        self.channels, self.n, self.width = tuple(channels), experts_per_group, width
        self.use_text, self.personalization = bool(use_text), bool(personalization)
        text_dim = value.shape[1]
        self.personalizers = nn.ModuleList(
            [SpatialPETPersonalization(c) for c in channels] if personalization else [])
        self.ct_adapters = nn.ModuleList(nn.Conv2d(c,width,1) for c in channels)
        self.pet_adapters = nn.ModuleList(nn.Conv2d(c,width,1) for c in channels)
        self.experts = nn.ModuleDict({g: nn.ModuleList(SharedExpert(width,text_dim)
            for _ in range(experts_per_group)) for g in GROUPS})
        self.routers = nn.ModuleList(ScaleRouter(width,experts_per_group,text_dim,
            attention_heads,router_noise_std) for _ in channels)
        self.output_projections = nn.ModuleList(nn.Conv2d(width,c,1,bias=False) for c in channels)
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)

    @classmethod
    def from_text_cache(cls, path, **kwargs):
        embeddings, metadata = load_text_cache(path)
        return cls(use_text=True,text_embeddings=embeddings,text_metadata=metadata,**kwargs)

    def get_extra_state(self):
        return {'version': FORMAT_VERSION, 'config': self.config.copy(),
                'prompts': list(PROMPTS), 'text_metadata': self.text_metadata}

    def set_extra_state(self, state):
        if state.get('version') != FORMAT_VERSION or state.get('config') != self.config or state.get('prompts') != list(PROMPTS):
            raise RuntimeError('Module-2 checkpoint configuration mismatch; restore matching architecture/configuration')
        self.text_metadata = state.get('text_metadata',{})

    def save_checkpoint(self, path):
        torch.save({'module2_state_dict':self.state_dict()},path)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        state = torch.load(path,map_location='cpu',weights_only=True)['module2_state_dict']
        extra = state['_extra_state']
        config = dict(extra['config'])
        model = cls(**config,text_embeddings=state['text_embeddings'] if config['use_text'] else None,
                    text_metadata=extra.get('text_metadata',{}))
        model.load_state_dict(state,strict=True)
        return model.to(device)

    def _dispatch(self, group: str, features: Tensor, selected: Tensor) -> Tensor:
        result = torch.zeros_like(features)
        group_text = self.text_embeddings[GROUPS.index(group)] if self.use_text else None
        for expert_id, expert in enumerate(self.experts[group]):
            indices = (selected == expert_id).nonzero(as_tuple=False).flatten()
            if indices.numel():
                values = expert(features.index_select(0,indices), group_text)
                result = result.index_copy(0,indices,values.to(result.dtype))
        return result

    def _forward_state(self, ct_feats, pet_feats, state: int, ready: bool,
                       base_pet_feats=None):
        # Weighted fusion: F^l = a_C^l C^l + a_P^l P_base^l + O_l(a_C^l E_C + a_P^l E_P),
        # with a_* = 2 * softmax weights so [0.5,0.5] restores plain addition.
        # pet_feats feeds router + PET expert (Missing: RAW prior, personalized
        # inside); base_pet_feats feeds ONLY the main fusion path (Missing:
        # alpha-scaled RAW prior). base_pet NEVER enters the personalizer.
        batch = ct_feats[0].shape[0]
        if state == 1 and not ready:
            return list(ct_feats), {
                'selected_experts': torch.full((batch,4,2),-1,device=ct_feats[0].device,dtype=torch.long),
                'route_weights': ct_feats[0].new_zeros((batch,4,2)),
                'modality_scales': ct_feats[0].new_zeros((batch,4,2)),
                'active': torch.zeros(batch,4,device=ct_feats[0].device,dtype=torch.bool),
            }
        if base_pet_feats is None:
            base_pet_feats = pet_feats
        if len(base_pet_feats) != 4:
            raise ValueError('Expected four base PET scales')
        for base, ct, channels in zip(base_pet_feats, ct_feats, self.channels):
            if (base.ndim != 4 or base.shape[:2] != (batch, channels)
                    or base.shape[2:] != ct.shape[2:] or base.device != ct.device
                    or not base.is_floating_point()):
                raise ValueError('base PET shape/channel/device/dtype must match CT')
        outputs, ids, all_weights, all_scales = [], [], [], []
        group = 'real' if state == 0 else 'imputed'
        state_text = self.text_embeddings[3+state] if self.use_text else None
        for scale, (ct, pet) in enumerate(zip(ct_feats, pet_feats)):
            raw_pet = pet
            base_pet = base_pet_feats[scale]
            if state == 1 and self.personalization:
                expert_pet = self.personalizers[scale](ct, raw_pet)
            else:
                expert_pet = raw_pet
            c, p = self.ct_adapters[scale](ct), self.pet_adapters[scale](expert_pet)
            ct_id, pet_id, weights = self.routers[scale](c,p,state,state_text)
            ce, pe = self._dispatch('ct',c,ct_id), self._dispatch(group,p,pet_id)
            w = weights.to(dtype=ce.dtype)
            w_ct = w[:,0,None,None,None]
            w_pet = w[:,1,None,None,None]
            a_ct = 2.0 * w_ct
            a_pet = 2.0 * w_pet
            expert_residual = a_ct * ce + a_pet * pe
            base_dtype = torch.promote_types(ct.dtype, base_pet.dtype)
            weighted_base = (a_ct.to(dtype=base_dtype) * ct.to(dtype=base_dtype)
                             + a_pet.to(dtype=base_dtype) * base_pet.to(dtype=base_dtype))
            projected_residual = self.output_projections[scale](expert_residual).to(dtype=base_dtype)
            outputs.append(weighted_base + projected_residual)
            ids.append(torch.stack((ct_id,pet_id+(1 if state==0 else 2)*self.n),dim=-1))
            all_weights.append(weights.detach())
            all_scales.append((2.0 * weights).detach())
        return outputs, {'selected_experts':torch.stack(ids,dim=1).detach(),
            'route_weights':torch.stack(all_weights,dim=1),
            'modality_scales':torch.stack(all_scales,dim=1),
            'active':torch.ones(batch,4,device=ct_feats[0].device,dtype=torch.bool)}

    def forward(self, ct_feats, pet_feats=None, *,
                base_pet_feats=None, mode='full',
                bank_ready=False, pet_available=None):
        if mode not in ('full','missing','auto'):
            raise ValueError('mode must be full, missing or auto')
        if isinstance(bank_ready, Tensor):
            if bank_ready.numel()!=1: raise ValueError('bank_ready must be a scalar')
            bank_ready = bool(bank_ready.item())
        elif not isinstance(bank_ready,bool):
            raise ValueError('bank_ready must be bool or scalar Tensor')
        if len(ct_feats)!=4:
            raise ValueError('Expected four CT scales')
        ref = ct_feats[0]
        if not isinstance(ref,Tensor) or ref.ndim!=4 or ref.shape[0]<1:
            raise ValueError('Expected nonempty [B,C,H,W] features')
        for c,channels in zip(ct_feats,self.channels):
            if c.ndim!=4 or c.shape[:2]!=(ref.shape[0],channels) or min(c.shape[2:])<1 or c.device!=ref.device or not c.is_floating_point():
                raise ValueError('CT shape/channel/device/dtype mismatch')
        if mode=='auto':
            if pet_available is None:
                raise ValueError('auto requires pet_available[B]')
            available = torch.as_tensor(pet_available,device=ref.device)
            if available.shape!=(ref.shape[0],) or not ((available==0)|(available==1)).all():
                raise ValueError('pet_available must be binary [B]')
            available = available.bool()
            needs_pet = bool(available.any()) or bank_ready
        else:
            if pet_available is not None:
                raise ValueError('pet_available is only accepted in auto mode')
            needs_pet = mode=='full' or bank_ready
        if pet_feats is None:
            if needs_pet: raise ValueError('PET features required for Full or ready Missing')
        else:
            if len(pet_feats)!=4: raise ValueError('Expected four PET scales')
            for c,p in zip(ct_feats,pet_feats):
                if p.shape!=c.shape or p.device!=c.device or not p.is_floating_point():
                    raise ValueError('PET must match already-aligned CT shape/device and be floating point')
        if base_pet_feats is not None:
            if len(base_pet_feats)!=4: raise ValueError('Expected four base PET scales')
            for refbase,c in zip(base_pet_feats,ct_feats):
                if refbase.shape!=c.shape or refbase.device!=c.device or not refbase.is_floating_point():
                    raise ValueError('base PET must match CT shape/device and be floating point')
        if mode!='auto':
            out,aux=self._forward_state(ct_feats,pet_feats,0 if mode=='full' else 1,
                                        bank_ready,base_pet_feats=base_pet_feats)
        elif bool(available.all()) or not bool(available.any()):
            out,aux=self._forward_state(ct_feats,pet_feats,0 if bool(available.all()) else 1,
                                        bank_ready,base_pet_feats=base_pet_feats)
        else:
            # Encoders may emit different floating dtypes under autocast.
            # Match normal CT+PET type promotion without downcasting real PET.
            out=[torch.zeros_like(c,dtype=torch.promote_types(c.dtype,p.dtype))
                 for c,p in zip(ct_feats,pet_feats)]
            aux={}
            for state,selected in ((0,available),(1,~available)):
                indices=selected.nonzero(as_tuple=False).flatten()
                c=[x.index_select(0,indices) for x in ct_feats]
                p=[x.index_select(0,indices) for x in pet_feats]
                if base_pet_feats is None:
                    base=[x.index_select(0,indices) for x in p]
                else:
                    base=[x.index_select(0,indices) for x in base_pet_feats]
                values,diagnostics=self._forward_state(c,p,state,bank_ready,base_pet_feats=base)
                out=[dst.index_copy(0,indices,v.to(dst.dtype)) for dst,v in zip(out,values)]
                for key,value in diagnostics.items():
                    if key not in aux: aux[key]=value.new_zeros((ref.shape[0],*value.shape[1:]))
                    aux[key]=aux[key].index_copy(0,indices,value)
        ids=aux['selected_experts']
        aux['expert_counts']=torch.stack([(ids==i).sum(dim=(0,2)) for i in range(3*self.n)],dim=1)
        aux['bank_ready']=bank_ready
        return out,aux


def _main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-text',type=Path,help='Encode five local prompts and save cache')
    parser.add_argument('--text-tower-path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower')
    parser.add_argument('--biomedclip-path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model')
    parser.add_argument('--backend',choices=('biomedclip','biomedbert'),default='biomedclip')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--experts-per-group',type=int,default=2)
    parser.add_argument('--text-cache',type=Path)
    args=parser.parse_args()
    if args.cache_text:
        embeddings,meta=encode_local_text_prompts(args.text_tower_path,args.biomedclip_path,args.backend,args.device)
        save_text_cache(args.cache_text,embeddings,meta)
        print(json.dumps({'cache':str(args.cache_text),'shape':list(embeddings.shape),'metadata':meta},indent=2))
    if args.smoke_test:
        torch.manual_seed(2023)
        torch.set_num_threads(1)
        kwargs=dict(experts_per_group=args.experts_per_group)
        model=(StateGuidedExpertFusion.from_text_cache(args.text_cache,**kwargs)
               if args.text_cache else StateGuidedExpertFusion(**kwargs)).to(args.device).eval()
        c=[torch.randn(2,ch,h,h,device=args.device) for ch,h in zip(model.channels,(16,8,4,2))]
        p=[torch.randn_like(x) for x in c]
        for mode,ready in [('full',False),('missing',False),('missing',True)]:
            out,aux=model(c,p,mode=mode,bank_ready=ready)
            assert all(torch.isfinite(x).all() for x in out)
            print(mode,'ready=',ready,'shapes=',[list(x.shape) for x in out])
        print('trainable_parameters=',sum(p.numel() for p in model.parameters() if p.requires_grad))
    if not args.cache_text and not args.smoke_test: parser.print_help()


if __name__=='__main__':
    _main()
