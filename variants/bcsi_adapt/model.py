import torch

from variants.tsbci.model import SHCIPETCT
from .front import BCSIAdapt


class Model(SHCIPETCT):
    def __init__(self, cfg, arm='full', load_generic=True):
        if arm != 'full':
            raise ValueError('This recipe runs the full 50% BCSI-Adapt candidate')
        fixed = {'topk_fraction': .5, 'channel_descriptor_grid': 8,
                 'channel_attention_width': 64, 'routing_temperature': 1.}
        for key, value in fixed.items():
            if cfg.get(key, value) != value:
                raise ValueError(f'First recipe requires {key}={value}')
        super().__init__(cfg, arm='m1', load_generic=load_generic)
        self.arm = 'full'
        with torch.random.fork_rng(devices=[]):
            self.front = BCSIAdapt(.5)

    def forward(self, ct, pet, capture=False):
        if ct.shape != pet.shape:
            raise ValueError('Paired CT/PET grids required')
        c = self.encode_ct(ct)
        p = self.pet_align(self.enc_pet(self._input(pet)))
        cc, pp, details = self.front(c, p, capture)
        fused = self.fusion(cc, pp)
        result = {'logits': self.decoder_full(fused, ct.shape[-2:])['logits']}
        if capture:
            result['capture'] = {'ct_aligned': c, 'pet_aligned': p, 'ct_interacted': cc,
                                 'pet_interacted': pp, 'fused': fused, 'interaction': details}
        return result
