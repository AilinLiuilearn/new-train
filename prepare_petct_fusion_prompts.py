#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from models.text_guided_ct_anchor_fusion import save_local_biomedclip_prompt_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare cached BioMedCLIP prompt embeddings for PET-CT fusion.')
    parser.add_argument('--biomedclip_dir', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model')
    parser.add_argument('--text_tower_dir', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower')
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model/petct_fusion_prompts.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    biomedclip_dir = Path(args.biomedclip_dir).expanduser().resolve()
    text_tower_dir = Path(args.text_tower_dir).expanduser().resolve()
    if not biomedclip_dir.is_dir():
        raise FileNotFoundError(f'BioMedCLIP directory missing: {biomedclip_dir}')
    if not text_tower_dir.is_dir():
        raise FileNotFoundError(f'BioMedBERT text tower directory missing: {text_tower_dir}')

    out = save_local_biomedclip_prompt_embeddings(
        output_path=args.output_path,
        biomedclip_dir=biomedclip_dir,
        text_tower_dir=text_tower_dir,
        device=args.device,
    )
    print(out)


if __name__ == '__main__':
    main()
