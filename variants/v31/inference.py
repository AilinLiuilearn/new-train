from pathlib import Path
import torch
from runtime import sha
from .model import JointModel,CorrectedModel


def load_joint(path):
    ck=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    if ck['format']!='v31_joint_v1':raise ValueError('Expected complete joint checkpoint')
    m=JointModel(ck['config']);m.load_state_dict(ck['weights'],strict=True)
    return m,ck


def load_model(path,device='cuda',base_checkpoint=None):
    ck=torch.load(str(path),map_location='cpu',weights_only=False)
    if ck['format']!='v31_correction_v1':raise ValueError('Expected correction checkpoint')
    base=Path(base_checkpoint or ck['base_checkpoint'])
    if sha(base)!=ck['base_sha256']:raise ValueError('Joint base checksum mismatch')
    m,_=load_joint(base);m=CorrectedModel(m)
    m.corrector.load_state_dict(ck['weights'],strict=True)
    return m.to(device).eval().requires_grad_(False)
