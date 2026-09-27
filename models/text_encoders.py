# -*- coding: utf-8 -*-
"""Switchable offline text encoders for the asymmetric fusion.

Supported ``--text_encoder`` values (all frozen, CPU, no_grad; only the two
float32 prompt embeddings are returned):

- ``clip``: CLIP ViT-B/32 text tower pooler output, dim 512 (previous default).
- ``biomedclip``: BiomedCLIP text tower (HF BiomedBERT + MLP proj), dim 512.
  Weights come from the local OpenCLIP checkpoint; only the text tower and
  its projection are used.
- ``biomedbert``: PubMed BiomedBERT ``[CLS]`` hidden state, dim 768.

``beit3`` is intentionally unsupported: transformers 4.38 ships no BEiT-3
modeling code, so its MultiWay text tower cannot be run from the downloaded
checkpoint without a custom implementation.

Returns ``(embeddings, dim)`` with ``embeddings`` shaped ``[2, dim]`` (CT
first, PET second), detached float32 on CPU.
"""
from pathlib import Path

import torch
from torch import Tensor

TEXT_ENCODERS = ('clip', 'biomedclip', 'biomedbert')
BIOMED_CONTEXT_LENGTH = 256


def _prompts(prompts):
    if prompts is None:
        from models.full_petct_asymmetric_fusion import CT_PROMPT, PET_PROMPT
        return [CT_PROMPT, PET_PROMPT]
    if len(prompts) != 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError('Provide two nonempty prompts, CT then PET.')
    return list(prompts)


def _finite_check(name, embeddings):
    if not isinstance(embeddings, Tensor) or embeddings.shape[0] != 2:
        raise ValueError(f'{name}: expected [2, dim] embeddings.')
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError(f'{name}: embeddings must be finite floating point values.')
    return embeddings.detach().to(device='cpu', dtype=torch.float32).clone()


def _clip_embeddings(path, prompts):
    from transformers import AutoTokenizer, CLIPTextModel
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'Offline Hugging Face CLIP directory not found: {path}')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(str(path), local_files_only=True)
    encoder.to(device='cpu', dtype=torch.float32).requires_grad_(False).eval()
    tokens = tokenizer(list(prompts), padding=True, truncation=True,
                       max_length=encoder.config.max_position_embeddings,
                       return_tensors='pt')
    with torch.no_grad():
        return encoder(**tokens).pooler_output.detach().float().cpu().clone()


def _biomedbert_tokenizer(vocab_path):
    from transformers import BertTokenizer
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f'BiomedBERT vocab not found: {vocab_path}')
    return BertTokenizer(str(vocab_path), do_lower_case=True)


def _biomedbert_model():
    from transformers import BertConfig, BertModel
    cfg = BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                     intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2)
    model = BertModel(cfg)
    model.to(device='cpu', dtype=torch.float32).requires_grad_(False).eval()
    return model


def _biomedclip_embeddings(root, prompts, vocab_path):
    root = Path(root)
    ckpt = root / 'open_clip_pytorch_model.bin'
    if not ckpt.is_file():
        raise FileNotFoundError(f'BiomedCLIP checkpoint not found: {ckpt}')
    tokenizer = _biomedbert_tokenizer(vocab_path)
    tokens = tokenizer(list(prompts), padding='max_length', max_length=BIOMED_CONTEXT_LENGTH,
                       truncation=True, return_tensors='pt')
    sd = torch.load(str(ckpt), map_location='cpu')
    bert = _biomedbert_model()
    stripped = {k[len('text.transformer.'):]: v for k, v in sd.items()
                if k.startswith('text.transformer.')}
    bert.load_state_dict(stripped, strict=False)
    proj = torch.nn.Sequential(torch.nn.Linear(768, 640, bias=False), torch.nn.GELU(),
                               torch.nn.Linear(640, 512, bias=False))
    with torch.no_grad():
        proj[0].weight.copy_(sd['text.proj.0.weight'])
        proj[2].weight.copy_(sd['text.proj.2.weight'])
    proj.requires_grad_(False).eval()
    with torch.no_grad():
        cls_hidden = bert(**tokens).last_hidden_state[:, 0]
        return proj(cls_hidden).detach().float().cpu().clone()


def _biomedbert_embeddings(path, prompts, vocab_path):
    path = Path(path)
    ckpt = path / 'pytorch_model.bin'
    if not ckpt.is_file():
        raise FileNotFoundError(f'BiomedBERT checkpoint not found: {ckpt}')
    tokenizer = _biomedbert_tokenizer(vocab_path)
    tokens = tokenizer(list(prompts), padding='max_length', max_length=BIOMED_CONTEXT_LENGTH,
                       truncation=True, return_tensors='pt')
    sd = torch.load(str(ckpt), map_location='cpu')
    bert = _biomedbert_model()
    stripped = {k[5:] if k.startswith('bert.') else k: v for k, v in sd.items()}
    bert.load_state_dict(stripped, strict=False)
    with torch.no_grad():
        return bert(**tokens).last_hidden_state[:, 0].detach().float().cpu().clone()


def default_encoder_dir(encoder, pretrained_root='pretrained'):
    root = Path(pretrained_root)
    return {'clip': root / 'clip-vit-base-patch32',
            'biomedclip': root / 'biomedclip_model',
            'biomedbert': root / 'biomedbert_text_tower'}.get(encoder, root / encoder)


def load_text_embeddings(encoder='clip', prompts=None, encoder_path=None,
                         vocab_path=None, pretrained_root='pretrained'):
    """Load frozen prompt embeddings for ``encoder``.

    Returns ``(embeddings [2, dim], dim)``. ``encoder_path`` overrides the
    default directory; ``vocab_path`` overrides the BiomedBERT vocab file.
    """
    if encoder == 'beit3':
        raise NotImplementedError(
            'beit3 text encoding is not supported: the installed transformers '
            '(4.38) ships no BEiT-3 modeling code, so its MultiWay text tower '
            'cannot be run. Use clip, biomedclip or biomedbert.')
    if encoder not in TEXT_ENCODERS:
        raise ValueError(f'Unsupported text_encoder={encoder!r}; choose from {TEXT_ENCODERS}.')
    prompts = _prompts(prompts)
    if encoder_path is None:
        encoder_path = default_encoder_dir(encoder, pretrained_root)
    if vocab_path is None:
        vocab_path = Path(pretrained_root) / 'biomedbert_vocab' / 'vocab.txt'
    if encoder == 'clip':
        embeddings = _clip_embeddings(encoder_path, prompts)
    elif encoder == 'biomedclip':
        embeddings = _biomedclip_embeddings(encoder_path, prompts, vocab_path)
    else:
        embeddings = _biomedbert_embeddings(encoder_path, prompts, vocab_path)
    embeddings = _finite_check(f'text_encoder={encoder}', embeddings)
    return embeddings, int(embeddings.shape[1])
