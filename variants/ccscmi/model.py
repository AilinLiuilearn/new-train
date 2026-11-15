import json
import torch
from runtime import REPO
from variants.tsbci.model import SHCIPETCT
from .front import CCSCMI


class Model(SHCIPETCT):
    def __init__(self,cfg,arm='full',load_generic=True,text_features=None):
        super().__init__(cfg,arm='m1',load_generic=load_generic)
        if text_features is None:text_features=torch.tensor(json.loads((REPO/'resources/biomedclip_roles.json').read_text())['embeddings'])
        with torch.random.fork_rng(devices=[]):
            self.front=CCSCMI(text_features,cfg['text_enabled'],cfg.get('per_dimension',True))

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired PET/CT grids required')
        c=self.encode_ct(ct);p=self.pet_align(self.enc_pet(self._input(pet)))
        cc,pp,details=self.front(c,p,capture);f=self.fusion(cc,pp)
        o={'logits':self.decoder_full(f,ct.shape[-2:])['logits']}
        if capture:o['capture']={'ct_aligned':c,'pet_aligned':p,'ct_interacted':cc,'pet_interacted':pp,
                                'fused':f,'interaction':details}
        return o
