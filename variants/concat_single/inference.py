import torch
from .model import ConcatSingle


def load_model(path,device='cuda'):
    ck=torch.load(str(path),map_location='cpu',weights_only=False)
    if ck['format']!='concat_single_alignment_inference_v1':raise ValueError('Wrong checkpoint format')
    m=ConcatSingle(ck['config'],load_generic=False)
    m.load_state_dict(ck['weights'],strict=True)
    return m.to(device).eval().requires_grad_(False)
