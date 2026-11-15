import torch
from .model import Model


def load_model(path,device='cuda'):
    ck=torch.load(str(path),map_location='cpu',weights_only=False)
    if ck['format']!='ccscmi_v1_inference':raise ValueError('Wrong CC-SCMI checkpoint')
    m=Model(ck['config'],load_generic=False,text_features=ck['weights']['front.text_features'])
    m.load_state_dict(ck['weights'],strict=True)
    return m.to(device).eval().requires_grad_(False)
