"""Full-only LPI-BI -> unchanged M1 -> unchanged GN decoder."""
import torch

from variants.tsbci.model import SHCIPETCT
from .front import LPIBI


class Model(SHCIPETCT):
    def __init__(self, cfg, arm='full', load_generic=True):
        if arm != 'full':
            raise ValueError('This recipe runs only the complete LPI-BI model')
        fixed = {'localization_weight': .1, 'latent_width': 32, 'interaction_heads': 4,
                 'interaction_neighborhood': 3, 'spatial_beta_initial': .1,
                 'spatial_beta_max': 1., 'localization_epsilon': 1e-6}
        for key, value in fixed.items():
            if cfg.get(key, value) != value:
                raise ValueError(f'V1 requires {key}={value}')
        super().__init__(cfg, arm='m1', load_generic=load_generic)
        self.arm = 'full'
        # Do not change common parameters or the RNG seen by the original loader.
        with torch.random.fork_rng(devices=[]):
            self.front = LPIBI()

    def forward(self, ct, pet, capture=False):
        if ct.shape != pet.shape:
            raise ValueError('Paired CT/PET grids required')
        c = self.encode_ct(ct)
        p = self.pet_align(self.enc_pet(self._input(pet)))
        cc, pp, loc, details = self.front(c, p, capture)
        fused = self.fusion(cc, pp)
        out = {'logits': self.decoder_full(fused, ct.shape[-2:])['logits'],
               'localization_logits': loc}
        if capture:
            out['capture'] = {'ct_aligned': c, 'pet_aligned': p, 'ct_interacted': cc,
                              'pet_interacted': pp, 'fused': fused, 'interaction': details}
        return out
