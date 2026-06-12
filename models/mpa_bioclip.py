import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    open_clip = None


class MPABioCLIPBlock(nn.Module):
    """Minimal MPA-BioCLIP block with cross-modal spatial gating.

    The block keeps the original projection/down-up prompt/adapters while adding:
    1) PET->CT and CT->PET lightweight spatial gates for visual interaction.
    2) Text-visual consensus gating so text prompts are injected mainly into
       high-consensus lesion-like regions instead of being broadcast globally.
    """

    def __init__(self, ct_channels, pet_channels, out_channels, text_feat, mlp_ratio=0.25, dropout=0.1):
        super().__init__()
        hidden = max(1, int(out_channels * mlp_ratio))
        gate_hidden = max(1, out_channels // 4)
        text_dim = int(text_feat.shape[-1])

        self.ct_proj = nn.Sequential(
            nn.Conv2d(ct_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pet_proj = nn.Sequential(
            nn.Conv2d(pet_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pet_guide_ct = nn.Sequential(
            nn.Conv2d(out_channels, gate_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ct_guide_pet = nn.Sequential(
            nn.Conv2d(out_channels, gate_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.ct_down = nn.Linear(out_channels, hidden)
        self.pet_down = nn.Linear(out_channels, hidden)
        self.up = nn.Linear(hidden, out_channels)
        self.act = nn.GELU()

        self.register_buffer('text_embed', text_feat.float())
        self.text_proj = nn.Linear(text_dim, out_channels) if text_dim != out_channels else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.consensus_temp = nn.Parameter(torch.tensor(5.0))
        self.consensus_bias = nn.Parameter(torch.tensor(0.1))

        self.gamma_ct = nn.Parameter(torch.ones(1))
        self.beta_ct = nn.Parameter(torch.zeros(1))
        self.gamma_pet = nn.Parameter(torch.ones(1))
        self.beta_pet = nn.Parameter(torch.zeros(1))
        self.s = nn.Parameter(torch.ones(1))

        self.adapter_ct = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        self.adapter_pet = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        self.cache_visuals = False
        self._last_visuals = {}

    def forward(self, ct_feat, pet_feat):
        if self.cache_visuals:
            self._last_visuals = {
                'ct_encoder': ct_feat.detach().cpu(),
                'pet_encoder': pet_feat.detach().cpu(),
            }

        ct = self.ct_proj(ct_feat)
        pet = self.pet_proj(pet_feat)
        if pet.shape[-2:] != ct.shape[-2:]:
            pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)

        pet_to_ct_gate = self.pet_guide_ct(pet)
        ct_to_pet_gate = self.ct_guide_pet(ct)
        ct_guided = ct * pet_to_ct_gate
        pet_guided = pet * ct_to_pet_gate

        b, c, h, w = ct_guided.shape
        ct_t = ct_guided.flatten(2).transpose(1, 2)
        pet_t = pet_guided.flatten(2).transpose(1, 2)

        p_vis = self.act(self.up(self.ct_down(ct_t) + self.pet_down(pet_t)))
        text_embed = self.text_embed.to(device=ct.device, dtype=ct.dtype)
        p_text = self.text_proj(text_embed)
        p_text = F.normalize(p_text, dim=-1)

        p_vis_norm = F.normalize(p_vis, dim=-1)
        p_text_norm = p_text.unsqueeze(1)
        consensus = (p_vis_norm * p_text_norm).sum(dim=-1, keepdim=True)

        temp = self.consensus_temp.clamp(1.0, 10.0)
        consensus_gate = torch.sigmoid((consensus - self.consensus_bias) * temp)
        p_text_spatial = consensus_gate * p_text
        p_final = p_vis + torch.sigmoid(self.alpha) * p_text_spatial

        p_ct = self.gamma_ct * p_final + self.beta_ct + self.s * p_final
        p_pet = self.gamma_pet * p_final + self.beta_pet + self.s * p_final

        ct_p = ct_t + p_ct
        pet_p = pet_t + p_pet
        ct_out = ct_p + self.adapter_ct(ct_p)
        pet_out = pet_p + self.adapter_pet(pet_p)

        ct_out = ct_out.transpose(1, 2).reshape(b, c, h, w)
        pet_out = pet_out.transpose(1, 2).reshape(b, c, h, w)
        if self.cache_visuals:
            self._last_visuals.update({
                'ct_projected': ct.detach().cpu(),
                'pet_projected': pet.detach().cpu(),
                'pet_to_ct_gate': pet_to_ct_gate.detach().cpu(),
                'ct_to_pet_gate': ct_to_pet_gate.detach().cpu(),
                'ct_text_modulated': ct_out.detach().cpu(),
                'pet_text_modulated': pet_out.detach().cpu(),
                'consensus': consensus.transpose(1, 2).reshape(b, 1, h, w).detach().cpu(),
                'consensus_gate': consensus_gate.transpose(1, 2).reshape(b, 1, h, w).detach().cpu(),
            })
        return ct_out, pet_out


class MPABioCLIPSumFusion(nn.Module):
    """Four-stage MPA-BioCLIP fusion for the heterogeneous main model.

    Data flow per stage:
        CT encoder feature + PET encoder feature
        -> MPABioCLIPBlock(text prompt injection + modality alignment)
        -> ct_aligned + pet_aligned
        -> fused stage feature for LightConcatUNetDecoder.
    """

    def __init__(self, ct_channels, pet_channels, out_channels, text_feat, mlp_ratio=0.25):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels)):
            raise ValueError('ct_channels, pet_channels and out_channels must have the same length.')
        self.blocks = nn.ModuleList([
            MPABioCLIPBlock(ct_ch, pet_ch, out_ch, text_feat, mlp_ratio=mlp_ratio)
            for ct_ch, pet_ch, out_ch in zip(ct_channels, pet_channels, out_channels)
        ])
        self.cache_visuals = False

    def set_visuals(self, enabled):
        self.cache_visuals = bool(enabled)
        for block in self.blocks:
            block.cache_visuals = self.cache_visuals
            if self.cache_visuals:
                block._last_visuals = {}

    def forward(self, ct_feats, pet_feats):
        fused = []
        for block, ct_feat, pet_feat in zip(self.blocks, ct_feats, pet_feats):
            ct_aligned, pet_aligned = block(ct_feat, pet_feat)
            fused.append(ct_aligned + pet_aligned)
        return fused

    def get_fusion_visuals(self):
        visuals = {}
        for idx, block in enumerate(self.blocks, start=1):
            if block._last_visuals:
                visuals[f'mpa_s{idx}'] = dict(block._last_visuals)
        return visuals


def _find_local_weight_file(model_dir):
    for name in ('open_clip_pytorch_model.bin', 'pytorch_model.bin', 'model.safetensors'):
        path = model_dir / name
        if path.exists():
            return path
    # HuggingFace snapshot downloads may leave blob files without the original name.
    for path in sorted(model_dir.rglob('*')):
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            return path
    return None


def _build_local_tokenizer(text_tower_dir, context_length):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('Local BiomedCLIP tokenization requires transformers.AutoTokenizer.') from exc

    hf_tokenizer = AutoTokenizer.from_pretrained(str(text_tower_dir), local_files_only=True)

    def tokenize(texts):
        encoded = hf_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=context_length,
            return_tensors='pt',
        )
        return encoded['input_ids']

    return tokenize


def _ensure_local_hf_text_config(text_tower_dir):
    """Create the missing local HuggingFace BERT config needed by open_clip HFTextEncoder."""
    config_path = text_tower_dir / 'config.json'
    if config_path.exists():
        return

    vocab_path = text_tower_dir / 'vocab.txt'
    vocab_size = 30522
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab_size = sum(1 for _ in f)

    bert_cfg = {
        'architectures': ['BertModel'],
        'attention_probs_dropout_prob': 0.1,
        'classifier_dropout': None,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'hidden_size': 768,
        'initializer_range': 0.02,
        'intermediate_size': 3072,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512,
        'model_type': 'bert',
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'pad_token_id': 0,
        'position_embedding_type': 'absolute',
        'transformers_version': '4.0.0',
        'type_vocab_size': 2,
        'use_cache': True,
        'vocab_size': vocab_size,
    }
    with open(config_path, 'w') as f:
        json.dump(bert_cfg, f, indent=2)


def _register_local_open_clip_config(model_dir, text_tower_dir):
    config_path = model_dir / 'open_clip_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing local BiomedCLIP config: {config_path}')

    with open(config_path, 'r') as f:
        local_cfg = json.load(f)
    model_cfg = local_cfg.get('model_cfg', local_cfg)
    text_cfg = model_cfg.get('text_cfg', {})

    # Force HuggingFace text tower/tokenizer to use the local text tower directory instead of Hub.
    text_cfg['hf_model_name'] = str(text_tower_dir)
    text_cfg['hf_tokenizer_name'] = str(text_tower_dir)
    model_cfg['text_cfg'] = text_cfg

    local_name = 'local_biomedclip_pubmedbert_256_vit_base_patch16_224'

    # Most open_clip versions keep model definitions in factory._MODEL_CONFIGS.
    # Registering here avoids using the hf-hub: model id, so no Hub request is made.
    try:
        from open_clip.factory import _MODEL_CONFIGS
        _MODEL_CONFIGS[local_name] = model_cfg
        return local_name
    except Exception:
        pass

    # Some newer versions expose add_model_config(config_path_or_dir), not
    # add_model_config(name, cfg). In that case create a small local config file
    # and let open_clip register it by path.
    try:
        local_config_path = model_dir / 'mpa_bioclip_local_open_clip_config.json'
        with open(local_config_path, 'w') as f:
            json.dump({'model_cfg': model_cfg}, f, indent=2)
        open_clip.add_model_config(str(local_config_path))
        return local_config_path.stem
    except Exception as exc:
        raise RuntimeError('This open_clip version does not expose a usable local model config registration API.') from exc


def get_bioclip_text_feature(
    model_path,
    text='focal abnormal metabolic lung lesion on PET-CT scan',
    device=None,
    text_tower_path='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower',
):
    """Load local BiomedCLIP and encode text once as a frozen prompt feature [1, D].

    This function is intentionally offline-first: it reads open_clip_config.json,
    tokenizer files, and weights from model_path instead of resolving the hf-hub
    model id microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.
    """
    if open_clip is None:
        raise ImportError('MPA-BioCLIP requires open_clip. Please install open_clip_torch first.')

    model_dir = Path(model_path).expanduser().resolve()
    text_tower_dir = Path(text_tower_path).expanduser().resolve() if text_tower_path else model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f'Local BiomedCLIP directory does not exist: {model_dir}')
    if not text_tower_dir.exists():
        raise FileNotFoundError(f'Local BiomedBERT text tower directory does not exist: {text_tower_dir}')

    weight_path = _find_local_weight_file(model_dir)
    if weight_path is None:
        raise FileNotFoundError(
            f'No local BiomedCLIP weight file found under {model_dir}. '
            'Expected open_clip_pytorch_model.bin, pytorch_model.bin, model.safetensors, '
            'or a large downloaded blob file.'
        )

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    _ensure_local_hf_text_config(text_tower_dir)
    model_name = _register_local_open_clip_config(model_dir, text_tower_dir)
    with open(model_dir / 'open_clip_config.json', 'r') as f:
        cfg = json.load(f)
    context_length = int(cfg.get('model_cfg', {}).get('text_cfg', {}).get('context_length', 256))

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weight_path),
        device=device,
    )
    tokenizer = _build_local_tokenizer(text_tower_dir, context_length)
    model.eval()

    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = F.normalize(text_feat, dim=-1)
    return text_feat.detach().cpu()
