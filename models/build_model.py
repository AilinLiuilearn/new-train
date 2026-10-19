"""Exactly two models: heterogeneous addition and its asymmetric extension."""
from models.asymmetric_petct import AsymmetricPETCT
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def build_model(cfg, *, load_encoder_pretrained=None):
    if cfg.ct_backbone!='convnextv2_nano' or cfg.pet_backbone!='mit_b1':
        raise ValueError('This branch fixes CT=ConvNeXtV2-Nano and PET=MiT-B1')
    pretrained=cfg.pretrained if load_encoder_pretrained is None else load_encoder_pretrained
    common=dict(ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path if pretrained else None,
        pet_pretrained_path=cfg.pet_pretrained_path if pretrained else None,
        decoder_channels=tuple(cfg.decoder_channels),use_deep_supervision=False,
        pretrained=bool(pretrained))
    if cfg.fusion=='add':
        return DualSharedAddPETCTBaseline(**common)
    if cfg.fusion=='asymmetric':
        return AsymmetricPETCT(**common,fusion_enabled=cfg.fusion_enabled,
            fusion_ranks=tuple(cfg.fusion_ranks),context_size=cfg.context_size,
            norm_mode=cfg.norm_mode)
    raise ValueError(f'Unknown fusion: {cfg.fusion}')
