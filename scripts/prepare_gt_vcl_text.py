"""Encode the exact user-approved prompts with the trained local BioMedCLIP tower."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,local_paths,sha

PROMPTS={
    'CT':'A CT image showing the structure, shape, and boundary of a lung tumor.',
    'PET':'A PET image showing the location and metabolic activity of a lung tumor.',
}


def main(args):
    require_myenv()
    import torch
    from transformers import AutoTokenizer
    from open_clip.hf_model import HFTextEncoder
    from torch.nn import functional as F
    torch.set_num_threads(4)
    root=Path(local_paths(args.paths)['pretrained_root'])/'biomedclip_model'
    ck=root/'open_clip_pytorch_model.bin'
    cfg=json.loads((root/'open_clip_config.json').read_text())['model_cfg']
    text_cfg=cfg['text_cfg']
    # The bundled CLIP directory has a stale 28,895-vocab Bert config, while its
    # trained text embedding and tokenizer both have 30,522 entries. Use the
    # matching local text-tower CONFIG only; all weights below come from CLIP.
    config_root=root.parent/'biomedbert_text_tower'
    model=HFTextEncoder(str(config_root),output_dim=cfg['embed_dim'],
                        pooler_type=text_cfg['hf_pooler_type'],proj_type=text_cfg['hf_proj_type'],
                        pretrained=False)
    weights=torch.load(str(ck),map_location='cpu',weights_only=True,mmap=True)
    selected={k[len('text.'):]:v for k,v in weights.items() if k.startswith('text.')}
    # Version bridge: the trained checkpoint stores position_ids, whereas
    # Transformers4.37 marks the same deterministic buffer non-persistent.
    position=selected['transformer.embeddings.position_ids']
    model.transformer.embeddings.register_buffer('position_ids',position.clone(),persistent=True)
    model.load_state_dict(selected,strict=True)
    model.eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,use_fast=True)
    def encode(strings):
        tokens=tokenizer(strings,padding='max_length',max_length=256,truncation=True,return_tensors='pt')
        assert not (tokens['input_ids']==tokenizer.unk_token_id).any()
        with torch.inference_mode():return F.normalize(model(tokens['input_ids']).float(),dim=-1)
    old=json.loads((REPO/'resources/biomedclip_roles.json').read_text())
    reproduced=encode([old['prompts']['CT'],old['prompts']['PET']])
    error=float((reproduced-torch.tensor(old['embeddings'])).abs().max())
    if error>2e-6:raise RuntimeError(f'Existing official text cache parity failed: {error}')
    vectors=encode(list(PROMPTS.values()))
    repeated=encode(list(PROMPTS.values()))
    assert torch.equal(vectors,repeated) and torch.isfinite(vectors).all()
    record={'model':old['model'],'prompts':PROMPTS,'embeddings':vectors.tolist(),
            'checkpoint_sha256':sha(ck),'strict_text_tensors':len(selected),'frozen':True,
            'tokenizer_sha256':sha(root/'tokenizer.json'),
            'representation':'Official OpenCLIP HFTextEncoder plus trained global MLP projection; normalized512',
            'context_length':256,'old_cache_max_abs_error':error,
            'matching_config_sha256':sha(config_root/'config.json'),
            'vocabulary_size':model.config.vocab_size,
            'source':'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'scope':'Fixed foreground anchors only; semantic efficacy requires matched controls'}
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='embeddings'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'))
    p.add_argument('--output',default=str(REPO/'resources/gt_vcl_text.json'))
    main(p.parse_args())
