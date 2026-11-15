"""Four-scale concatenation plus learnable channel compression."""
import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super().__init__()
        self.channels = tuple(channels)
        # Extra construction must not advance the old trainer's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleList([
                nn.Conv2d(2*c, c, 1, bias=False) for c in channels])
            with torch.no_grad():
                for c, layer in zip(channels, self.projections):
                    layer.weight.zero_()
                    index = torch.arange(c)
                    layer.weight[index, index, 0, 0] = 1
                    layer.weight[index, c+index, 0, 0] = 1

    def forward(self, ct, pet):
        if len(ct) != 4 or len(pet) != 4:
            raise ValueError('Exactly four modality feature scales required')
        result = []
        for c, p, width, projection in zip(ct, pet, self.channels, self.projections):
            if c.shape != p.shape or c.shape[1] != width:
                raise ValueError('Concat expects already aligned CT/PET features')
            if c.device != p.device or c.dtype != p.dtype:
                raise ValueError('Modality device/dtype mismatch')
            value = projection(torch.cat([c, p], dim=1))
            if not bool(value.isfinite().all()):
                raise FloatingPointError('Nonfinite concat projection')
            result.append(value)
        return result
