import json
import torch
from runtime import REPO
from variants.tsbci.model import SHCIPETCT
from .module import MultiScaleRefiner


class Model(SHCIPETCT):
    def __init__(self,cfg,arm='full',load_generic=True,text_anchor=None):
        if arm!='full' or cfg['recipe'] not in ('V','T','S'):raise ValueError('Unknown SCGR recipe')
        if cfg['text_enabled']!=(cfg['recipe']!='V') or cfg['score_enabled']!=(cfg['recipe']=='S'):
            raise ValueError('Recipe flags mismatch')
        super().__init__(cfg,arm='m1',load_generic=load_generic)
        self.arm='full';self.cfg=dict(cfg)
        if text_anchor is None:
            data=json.loads((REPO/'resources/scgr_text.json').read_text())
            if data['prompts']!={'fusion':'Pulmonary tumor lesion.'}:raise ValueError('Text prompt mismatch')
            text_anchor=data['embeddings'][0]
        with torch.random.fork_rng(devices=[]):
            # CPU construction must not reseed the CUDA stream used by the
            # original encoder's stochastic layers during common training.
            torch.random.default_generator.manual_seed(cfg['refiner_seed'])
            self.refiner=MultiScaleRefiner(text_anchor,cfg['text_enabled'],cfg['refiner_width'],cfg['context_pool'])

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired CT/PET grids required')
        c=self.encode_ct(ct);p=self.pet_align(self.enc_pet(self._input(pet)))
        fused=self.fusion(c,p)
        refined,scores,details=self.refiner(fused,capture)
        output={'logits':self.decoder_full(refined,ct.shape[-2:])['logits'],'support_logits':scores}
        if capture:output['capture']={'fused':fused,'refined':refined,'refiner':details}
        return output
