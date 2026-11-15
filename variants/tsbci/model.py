"""Fresh task training: generic encoder pretraining only; no task-weight import API."""
from pathlib import Path
import sys
import json
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'src'))
from models.staged_petct import StagedPETCT
from .shci import TSBCI


class SHCIPETCT(StagedPETCT):
    def __init__(self,cfg,arm='full',load_generic=True,text_features=None):
        if arm not in ('m1','full'):raise ValueError(arm)
        super().__init__(stage=2,ct_pretrained_path=cfg['ct_pretrained_path'] if load_generic else None,
            pet_pretrained_path=cfg['pet_pretrained_path'] if load_generic else None,
            decoder_norm='GN',fusion_type='local_region_residual',fusion_config=cfg['fusion_config'])
        del self.decoder_ct;self.arm=arm
        if arm=='full':
            if text_features is None:text_features=torch.tensor(json.loads((ROOT.parents[1]/'resources/biomedclip_roles.json').read_text())['embeddings'],dtype=torch.float32)
            with torch.random.fork_rng(devices=[]):self.shci=TSBCI(text_features)
        self.requires_grad_(True)

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired CT/PET grids required')
        c=self.ct_align(self.enc_ct(self._input(ct)));p=self.pet_align(self.enc_pet(self._input(pet)))
        cc,pp,diag=self.shci(c,p,capture) if self.arm=='full' else (c,p,{})
        f=self.fusion(cc,pp);z=self.decoder_full(f,ct.shape[-2:])['logits']
        out={'logits':z}
        if capture:out['capture']={'ct_aligned':c,'pet_aligned':p,'ct_interacted':cc,'pet_interacted':pp,'fused':f,'interaction':diag}
        return out
