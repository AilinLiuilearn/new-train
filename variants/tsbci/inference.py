"""Inference requires normalized CT/PET only; frozen text buffers in checkpoint."""
import torch
from .model import SHCIPETCT

def load_model(path,device='cuda'):
    b=torch.load(str(path),map_location='cpu',weights_only=False)
    if b['format']!='tsbci_conditioned_null_inference_v1':raise ValueError('Expected a selected SHCI inference bundle')
    t=b['weights'].get('shci.text_features')
    m=SHCIPETCT(b['config'],arm=b['config']['arm'],load_generic=False,text_features=t)
    m.load_state_dict(b['weights'],strict=True)
    return m.to(device).eval().requires_grad_(False)
