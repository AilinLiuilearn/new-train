import torch
from .model import Model


def load_model(path, device='cuda'):
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    if checkpoint['format'] != 'lpi_bi_v1_inference':
        raise ValueError('Wrong LPI-BI checkpoint')
    model = Model(checkpoint['config'], load_generic=False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    return model.to(device).eval().requires_grad_(False)
