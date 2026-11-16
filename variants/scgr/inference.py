"""Selected weights need normalized CT/PET only; no GT or text encoder."""
import torch
from .model import Model


def load_model(path,device='cuda'):
    bundle=torch.load(str(path),map_location='cpu',weights_only=False)
    if bundle['format']!='scgr_support_conditioned_v1':raise ValueError('Unexpected checkpoint format')
    model=Model(bundle['config'],load_generic=False,text_anchor=bundle['weights']['refiner.text_anchor'])
    model.load_state_dict(bundle['weights'],strict=True)
    return model.to(device).eval().requires_grad_(False)
