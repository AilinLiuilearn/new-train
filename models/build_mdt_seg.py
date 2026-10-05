# -*- coding: utf-8 -*-
"""Model builders for the two clean baselines (no registration system)."""
from models.ct_only_baseline import CTOnlySegmentationModel
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_dual_model(cfg):
    model = DualSharedAddPETCTBaseline(
        ct_backbone=_get(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=_get(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=_get(cfg, 'ct_pretrained_path', None),
        pet_pretrained_path=_get(cfg, 'pet_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=tuple(_get(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(_get(cfg, 'use_deep_supervision', False)
                                  or _get(cfg, 'deep_supervision', False)),
        pretrained=bool(_get(cfg, 'pretrained', True)),
    )
    print(f'[dual_baseline] ct={_get(cfg, "ct_backbone", "convnextv2_nano")} '
          f'pet={_get(cfg, "pet_backbone", "mit_b1")} '
          f'fusion=AddFusion shared_decoder=UNetStyleDecoder(GroupNorm) '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}


def build_ct_only_model(cfg):
    if str(_get(cfg, 'ct_backbone', 'convnextv2_nano')).lower().replace('-', '_') != 'convnextv2_nano':
        raise ValueError('CT-only baseline is fixed to convnextv2_nano, '
                         f'got {_get(cfg, "ct_backbone", None)!r}')
    model = CTOnlySegmentationModel(
        ct_backbone=_get(cfg, 'ct_backbone', 'convnextv2_nano'),
        ct_pretrained_path=_get(cfg, 'ct_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=tuple(_get(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(_get(cfg, 'use_deep_supervision', False)
                                  or _get(cfg, 'deep_supervision', False)),
        pretrained=bool(_get(cfg, 'pretrained', True)),
    )
    print(f'[ct_only_baseline] ct={_get(cfg, "ct_backbone", "convnextv2_nano")} '
          f'fusion=none pet_encoder=none shared_decoder=UNetStyleDecoder(GroupNorm) '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}


def build_mdt_seg_teacher(cfg):
    """Legacy entry kept for compatibility: forwards to the dual baseline."""
    print('[INFO] run_mdt_seg.py is a compatibility entry: '
          'actual training mode is the Full/Missing mixed baseline', flush=True)
    return build_dual_model(cfg)