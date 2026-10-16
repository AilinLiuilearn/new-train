"""Aligned Add baseline + independent pre-decoder CT-key/matrix-value banks."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.ct_full_matrix import CTFullMatrixMemory


MATRIX_DEFAULTS = dict(enabled=True, clusters=4, capacity=256, min_candidates=24,
    min_cluster_samples=2, ridge=.01, decay=.99, iterations=15, outlier_filter=False,
    samples_per_class=8, temperature=.5, gate_init=.1, warmup_steps=100, refresh_steps=20,
    sampling_seed=2023, value_mode='analytic', value_rank=8, value_init='full')


class CTFullMatrixAddModel(DualSharedAddPETCTBaseline):
    def __init__(self, **kwargs):
        settings = {k: kwargs.pop('matrix_'+k, v) for k, v in MATRIX_DEFAULTS.items()}
        self.matrix_enabled = bool(settings.pop('enabled'))
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('matrix experiment uses final segmentation supervision only')
        super().__init__(**kwargs)
        self.matrix_settings = dict(enabled=self.matrix_enabled, **settings)
        self.matrix = CTFullMatrixMemory((64,128,320,512), **settings)
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

    @torch.no_grad()
    def prepare_candidates(self, knowledge, mask):
        if not self.training: raise RuntimeError('collection is training-only')
        return self.matrix.candidates(knowledge['ct'], knowledge['full'],
                                      mask.index_select(0, knowledge['full_index']))


def build_matrix_model(cfg):
    if bool(cfg.use_deep_supervision or cfg.deep_supervision):
        raise ValueError('disable both deep supervision flags for the matrix experiment')
    model = CTFullMatrixAddModel(ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels), pretrained=cfg.pretrained,
        **{'matrix_'+k: getattr(cfg, 'matrix_'+k, v) for k,v in MATRIX_DEFAULTS.items()})
    print('[ct_full_matrix] before_decoder=True independent_scales=True '
          f'K={cfg.matrix_clusters} spherical_CT_keys=True raw_transform_values=True '
          f'enabled={cfg.matrix_enabled} final_supervision_only=True', flush=True)
    print(f'[matrix_value] mode={getattr(cfg,"matrix_value_mode","analytic")} '
          f'rank={getattr(cfg,"matrix_value_rank",8)} init={getattr(cfg,"matrix_value_init","full")}', flush=True)
    return {'model': model}
