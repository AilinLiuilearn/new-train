"""Generate the fixed TCSR text vector from local BioMedCLIP files."""
import argparse
import json
from pathlib import Path

import torch


PROMPT = 'Pulmonary tumor lesion.'


def encode_anchor(biomedclip_dir, text_config_dir):
    from open_clip.hf_model import HFTextEncoder
    from transformers import AutoTokenizer

    root = Path(biomedclip_dir)
    config = json.loads((root / 'open_clip_config.json').read_text())['model_cfg']
    tower_config = config['text_cfg']
    model = HFTextEncoder(str(text_config_dir), output_dim=config['embed_dim'],
                          pooler_type=tower_config['hf_pooler_type'],
                          proj_type=tower_config['hf_proj_type'], pretrained=False)
    weights = torch.load(str(root / 'open_clip_pytorch_model.bin'), map_location='cpu',
                         weights_only=True, mmap=True)
    state = {key[len('text.'):]: value for key, value in weights.items() if key.startswith('text.')}
    model.transformer.embeddings.register_buffer(
        'position_ids', state['transformer.embeddings.position_ids'].clone(), persistent=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, use_fast=True)
    tokens = tokenizer([PROMPT], padding='max_length', max_length=256, truncation=True,
                       return_tensors='pt')['input_ids']
    if bool((tokens == tokenizer.unk_token_id).any()):
        raise ValueError('The fixed TCSR prompt contains an unknown token')
    with torch.inference_mode():
        anchor = torch.nn.functional.normalize(model(tokens).float(), dim=-1)[0]
    if anchor.shape != (512,) or not torch.isfinite(anchor).all():
        raise ValueError('Expected a finite 512-dimensional BioMedCLIP anchor')
    return anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biomedclip-dir', type=Path, required=True)
    parser.add_argument('--text-config-dir', type=Path, required=True,
                        help='Local 30,522-token BiomedBERT text-tower config directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    anchor = encode_anchor(args.biomedclip_dir, args.text_config_dir)
    with args.output.open('x') as file:
        json.dump({'prompts': {'fusion': PROMPT}, 'embeddings': [anchor.tolist()]}, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
