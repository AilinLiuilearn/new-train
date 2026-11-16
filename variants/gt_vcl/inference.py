"""Deployment removes the shared auxiliary decoder and all text machinery."""
import torch
from variants.tsbci.model import SHCIPETCT


def load_model(path,device='cuda'):
    ck=torch.load(str(path),map_location='cpu',weights_only=False)
    if ck['format']!='gt_vcl_inference_v1':raise ValueError('Wrong GT-VCL checkpoint')
    model=SHCIPETCT(ck['config'],arm='m1',load_generic=False)
    state={k:v for k,v in ck['weights'].items() if not k.startswith(('decoder_sep.','semantic.'))}
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)
