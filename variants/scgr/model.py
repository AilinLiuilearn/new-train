import json
import torch
from runtime import REPO
from models.components.backbones import create_feature_backbone, load_local_weights_safe
from models.components.channel_align import StageChannelAlign
from models.staged_petct import CHANNELS
from variants.tsbci.model import SHCIPETCT
from .module import MultiScaleRefiner


FIXED_TEXT_ANCHORS = {
    None: ('scgr_text.json', 'Pulmonary tumor lesion.'),
    'mismatched_text': ('scgr_mismatched_text.json', 'Benign hepatic cyst.'),
    'synonym_text': ('scgr_synonym_text.json', 'Lung tumor.'),
}
ABLATIONS = ('no_tccm', 'no_lcsg', 'learnable_vector', 'mismatched_text', 'synonym_text')


class Model(SHCIPETCT):
    def __init__(self,cfg,arm='full',load_generic=True,text_anchor=None):
        if arm!='full' or cfg['recipe'] not in ('V','T','S'):raise ValueError('Unknown SCGR recipe')
        if cfg['text_enabled']!=(cfg['recipe']!='V') or cfg['score_enabled']!=(cfg['recipe']=='S'):
            raise ValueError('Recipe flags mismatch')
        defaults = {'ct': 'convnextv2_nano', 'pet': 'mit_b1'}
        replacements = {branch: cfg[f'{branch}_backbone'] for branch, default in defaults.items()
                        if cfg.get(f'{branch}_backbone', default) != default}
        initial_cfg = dict(cfg)
        for branch in replacements:
            initial_cfg[f'{branch}_pretrained_path'] = None
        super().__init__(initial_cfg,arm='m1',load_generic=load_generic)
        for branch, backbone in replacements.items():
            # Isolate encoder construction so identical task modules retain the
            # historical initialization and the training RNG position.
            with torch.random.fork_rng(devices=[]):
                encoder = create_feature_backbone(backbone, in_channels=3)
                aligner = StageChannelAlign(encoder.feature_info.channels(), CHANNELS)
                if load_generic:
                    load_local_weights_safe(encoder, cfg[f'{branch}_pretrained_path'],
                                            name=f'{branch.upper()}_Encoder')
            setattr(self, f'enc_{branch}', encoder)
            setattr(self, f'{branch}_align', aligner)
        self.arm='full';self.cfg=dict(cfg)
        ablation = cfg.get('tcsr_ablation')
        if ablation is not None and (ablation not in ABLATIONS or cfg['recipe'] != 'T'):
            raise ValueError('TCSR ablations require text recipe T')
        if text_anchor is None:
            if ablation == 'learnable_vector':
                generator=torch.Generator().manual_seed(2803)
                text_anchor=torch.nn.functional.normalize(torch.randn(512,generator=generator),dim=0)
            else:
                cache,prompt=FIXED_TEXT_ANCHORS.get(ablation,FIXED_TEXT_ANCHORS[None])
                data=json.loads((REPO/'resources'/cache).read_text())
                if data['prompts']!={'fusion':prompt}:raise ValueError('Text prompt mismatch')
                text_anchor=data['embeddings'][0]
        with torch.random.fork_rng(devices=[]):
            # CPU construction must not reseed the CUDA stream used by the
            # original encoder's stochastic layers during common training.
            torch.random.default_generator.manual_seed(cfg['refiner_seed'])
            self.refiner=MultiScaleRefiner(text_anchor,cfg['text_enabled'],cfg['refiner_width'],cfg['context_pool'],
                                           trainable_anchor=ablation=='learnable_vector',
                                           channel_enabled=ablation!='no_tccm',
                                           spatial_enabled=ablation!='no_lcsg')

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired CT/PET grids required')
        c=self.encode_ct(ct);p=self.pet_align(self.enc_pet(self._input(pet)))
        fused=self.fusion(c,p)
        refined,scores,details=self.refiner(fused,capture)
        output={'logits':self.decoder_full(refined,ct.shape[-2:])['logits'],'support_logits':scores}
        if capture:output['capture']={'fused':fused,'refined':refined,'refiner':details}
        return output
