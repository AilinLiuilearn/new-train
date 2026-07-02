"""BioMedCLIP text encoder for fixed modality prior texts (encode once, cache in model)."""

import json
import os

import torch
import torch.nn.functional as F

CT_PRIOR_TEXT = (
    "CT shows anatomical structure in PET-CT images. Lung tumors usually appear as "
    "soft-tissue lesions against low-intensity aerated lung. CT provides lesion shape, "
    "location, lung tissue context, and boundary information."
)

PET_PRIOR_TEXT = (
    "PET shows metabolic activity in PET-CT images. Lung tumors usually present increased "
    "tracer uptake with high metabolic intensity. PET helps localize active tumor regions, "
    "but high uptake may also appear in physiological or inflammatory regions."
)

DEFAULT_MODALITY_PRIOR_TEXTS = (CT_PRIOR_TEXT, PET_PRIOR_TEXT)

DEFAULT_BIOMEDBERT_LOCAL = (
    "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower"
)


def _resolve_local_biomedbert_path(model_dir: str, biomedbert_path: str = None) -> str:
    if biomedbert_path and os.path.isdir(biomedbert_path):
        return biomedbert_path

    candidates = [
        os.path.join(os.path.dirname(model_dir), "biomedbert_text_tower"),
        DEFAULT_BIOMEDBERT_LOCAL,
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            return path
    return biomedbert_path or DEFAULT_BIOMEDBERT_LOCAL


def _make_local_hf_tokenizer(hf_tokenizer, context_length: int):
    from open_clip.tokenizer import get_clean_fn

    class _LocalHFTokenizer:
        """Standalone HF tokenizer wrapper for offline local BioMedBERT."""

        def __init__(self, tokenizer, ctx_len):
            self.tokenizer = tokenizer
            self.context_length = ctx_len
            self.clean_fn = get_clean_fn("whitespace")
            self.strip_sep_token = False

        def __call__(self, texts, context_length=None):
            if isinstance(texts, str):
                texts = [texts]

            context_length = context_length or self.context_length
            texts = [self.clean_fn(text) for text in texts]
            input_ids = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=context_length,
                padding="max_length",
                truncation=True,
            ).input_ids

            if self.strip_sep_token:
                input_ids = torch.where(
                    input_ids == self.tokenizer.sep_token_id,
                    torch.zeros_like(input_ids),
                    input_ids,
                )
            return input_ids

    return _LocalHFTokenizer(hf_tokenizer, context_length)


def _load_biomedclip_text_model(
    model_dir: str,
    device: torch.device,
    biomedbert_path: str = None,
):
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS
    from transformers import AutoTokenizer

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    config_path = os.path.join(model_dir, "open_clip_config.json")
    weight_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"BioMedCLIP config not found: {config_path}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"BioMedCLIP weights not found: {weight_path}")

    bert_local = _resolve_local_biomedbert_path(model_dir, biomedbert_path=biomedbert_path)

    with open(config_path, "r", encoding="utf-8") as f:
        clip_cfg = json.load(f)
    model_cfg = dict(clip_cfg["model_cfg"])
    preprocess_cfg = clip_cfg.get("preprocess_cfg", {})

    text_cfg = dict(model_cfg.get("text_cfg", {}))
    text_cfg["hf_model_name"] = bert_local
    text_cfg["hf_tokenizer_name"] = bert_local
    text_cfg["hf_model_pretrained"] = False
    model_cfg["text_cfg"] = text_cfg

    model_name = "biomedclip_local"
    _MODEL_CONFIGS[model_name] = model_cfg

    context_length = int(text_cfg.get("context_length", 256))
    hf_tokenizer = AutoTokenizer.from_pretrained(bert_local, local_files_only=True)
    tokenizer = _make_local_hf_tokenizer(hf_tokenizer, context_length)

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=weight_path,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer, context_length


@torch.no_grad()
def biomedclip_encode_text(
    texts,
    model_dir: str,
    device=None,
    normalize: bool = True,
    biomedbert_path: str = None,
) -> torch.Tensor:
    """Encode a list of texts with local BioMedCLIP; returns [N, 512] float tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model, tokenizer, context_length = _load_biomedclip_text_model(
        model_dir,
        device,
        biomedbert_path=biomedbert_path,
    )
    token_ids = tokenizer(list(texts), context_length=context_length).to(device)
    text_features = model.encode_text(token_ids)
    if normalize:
        text_features = F.normalize(text_features, dim=-1)
    return text_features.detach().cpu().float()


def encode_modality_prior_texts(
    model_dir: str,
    device=None,
    texts=None,
    biomedbert_path: str = None,
) -> torch.Tensor:
    """Encode CT/PET modality prior texts; returns [2, 512]."""
    texts = texts or DEFAULT_MODALITY_PRIOR_TEXTS
    if len(texts) != 2:
        raise ValueError(f"Expected 2 modality prior texts, got {len(texts)}")
    return biomedclip_encode_text(
        texts,
        model_dir=model_dir,
        device=device,
        normalize=True,
        biomedbert_path=biomedbert_path,
    )
