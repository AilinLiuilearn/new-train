"""Aligned Add baseline + independent pre-decoder CT-key/matrix-value banks."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.spatial_modulation_bank import SpatialModulationMemory


MATRIX_DEFAULTS = dict(enabled=True, clusters=8, decay=.99, temperature=.5, chunk_size=8192, mod_loss_enabled=True, mod_loss_weight=.1)


class SpatialModulationAddModel(DualSharedAddPETCTBaseline):
    def __init__(self, **kwargs):
        settings = {k: kwargs.pop('matrix_'+k, v) for k, v in MATRIX_DEFAULTS.items()}
        self.matrix_enabled = bool(settings.pop('enabled'))
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('matrix experiment uses final segmentation supervision only')
        super().__init__(**kwargs)
        self.matrix_settings = dict(enabled=self.matrix_enabled, **settings)
        self.matrix = SpatialModulationMemory((64,128,320,512), **{k:v for k,v in settings.items() if not k.startswith('mod_loss_')})
        self.matrix_read_mode = 'normal'

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto',
                return_knowledge=False, return_diagnostics=False):
        if return_knowledge and not self.training: raise RuntimeError('collection is training-only')
        if forward_mode in ('full', 'missing'):
            state = torch.full((len(ct),), int(forward_mode=='full'), device=ct.device, dtype=torch.long)
        elif forward_mode == 'auto':
            state = _validate_state(torch.ones(len(ct), dtype=torch.long) if pet_available is None else pet_available, len(ct)).to(ct.device)
        else: raise ValueError('unsupported forward mode')
        fi, mi = state.eq(1).nonzero(as_tuple=True)[0], state.eq(0).nonzero(as_tuple=True)[0]
        if len(fi) and pet is None: raise ValueError('Full requires PET')
        ct_features = self._encode_ct(ct)
        features = list(ct_features)
        knowledge = None
        if len(fi):
            cf = [x.index_select(0, fi) for x in ct_features]
            pf = self._encode_pet(pet.index_select(0, fi).contiguous())
            fused = self.fusion(cf, pf)
            features = [x.index_copy(0, fi, f.to(x.dtype)) for x,f in zip(features, fused)]
            if return_knowledge:
                knowledge = dict(full_index=fi, ct=[x.detach() for x in cf],
                                 full=[x.detach() for x in fused])
        stats = [{} for _ in range(4)] if return_diagnostics else None
        if self.matrix_enabled and len(mi):
            corrected = self.matrix([x.index_select(0, mi) for x in ct_features], stats, self.matrix_read_mode)
            features = [x.index_copy(0, mi, f.to(x.dtype)) for x,f in zip(features, corrected)]
        out = self._decode(features, ct.shape[-2:] if target_size is None else target_size)
        out.update(pet_available=state.detach().cpu(), num_full=len(fi), num_missing=len(mi))
        if return_knowledge: out['knowledge'] = knowledge
        if stats is not None: out['matrix_reads'] = stats
        return out



def build_matrix_model(cfg):
    if bool(cfg.use_deep_supervision or cfg.deep_supervision):
        raise ValueError('disable both deep supervision flags for the matrix experiment')
    model = SpatialModulationAddModel(ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels), pretrained=cfg.pretrained,
        **{'matrix_'+k: getattr(cfg, 'matrix_'+k, v) for k,v in MATRIX_DEFAULTS.items()})
    print(f'[spatial_modulation] independent_scales=4 unified_K={cfg.matrix_clusters} '
          f'all_positions=True first_epoch_no_read=True mod_loss={cfg.matrix_mod_loss_enabled} '
          f'weight={cfg.matrix_mod_loss_weight}',flush=True)
    return {'model':model}
