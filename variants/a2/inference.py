"""Inference uses CT/PET only; cached role buffers are restored from best.pt."""
import torch
from .model import Model

def load_model(path,device='cuda'):
    b=torch.load(str(path),map_location='cpu',weights_only=False)
    if b['format']!='prem1_candidates_inference_v1':raise ValueError('Wrong inference bundle')
    weights=b['weights']
    text=weights.get('front.text_features')
    if text is None:text=weights['front.role_text'].reshape(2,-1)
    m=Model(b['config'],load_generic=False,text_features=text)
    m.load_state_dict(weights,strict=True)
    return m.to(device).eval().requires_grad_(False)
