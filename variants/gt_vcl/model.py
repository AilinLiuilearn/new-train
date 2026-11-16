import json
import torch
from torch.utils.checkpoint import checkpoint

from runtime import REPO
from variants.tsbci.model import SHCIPETCT
from models.components.group_unet_decoder import UNetStyleDecoder
from .semantic import SemanticRegularizer


class Model(SHCIPETCT):
    def __init__(self,cfg,arm='full',load_generic=True):
        if arm!='full' or cfg['recipe'] not in ['E1','E2','E4']:raise ValueError('Unknown GT-VCL recipe')
        if cfg['calibration_enabled']!=(cfg['recipe']!='E1') or cfg['semantic_enabled']!=(cfg['recipe']=='E4'):
            raise ValueError('E1/E2/E4 flags do not match recipe')
        super().__init__(cfg,arm='m1',load_generic=load_generic)
        self.arm='full';self.recipe=cfg['recipe'];self.cfg=dict(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['aux_decoder_seed'])
            self.decoder_sep=UNetStyleDecoder(norm='GN')
        if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in self.decoder_sep.modules()):
            raise ValueError('Checkpointed shared decoder must not update BN statistics')
        if cfg['semantic_enabled']:
            data=json.loads((REPO/'resources/gt_vcl_text.json').read_text())
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg['semantic_seed'])
                self.semantic=SemanticRegularizer(torch.tensor(data['embeddings']),cfg['semantic_margin'],cfg['semantic_chunk'])

    def decode_single(self,features,size):
        want_features=hasattr(self,'semantic')
        def decode(*xs):
            output=self.decoder_sep(list(xs),size,return_features=want_features)
            return (output['logits'],output['features']) if want_features else output['logits']
        if self.training and self.cfg['checkpoint_shared_decoder']:
            result=checkpoint(decode,*features,use_reentrant=False,preserve_rng_state=False)
        else:result=decode(*features)
        return result if want_features else (result,None)

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired CT/PET grids required')
        c=self.encode_ct(ct);p=self.pet_align(self.enc_pet(self._input(pet)))
        fused=self.fusion(c,p)
        z=self.decoder_full(fused,ct.shape[-2:])['logits']
        zc,hc=self.decode_single(c,ct.shape[-2:]);zp,hp=self.decode_single(p,ct.shape[-2:])
        output={'logits':z,'ct_logits':zc,'pet_logits':zp}
        if hasattr(self,'semantic'):output['semantic_features']=(hc,hp)
        if capture:output['capture']={'ct':c,'pet':p,'fused':fused}
        return output
