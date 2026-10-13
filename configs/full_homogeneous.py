"""Independent Full-only homogeneous baseline configuration."""
from configs.full_structure_response import FullStructureConfig
from models.full_homogeneous import NATIVE_CHANNELS


DEFAULT_WEIGHTS = {
    'mit_b1': '/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1',
    'convnextv2_nano': '/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano',
}


class FullHomogeneousConfig(FullStructureConfig):
    @staticmethod
    def model_parser():
        p = FullStructureConfig.model_parser()
        for action in p._actions:
            if action.dest == 'model_arch':
                action.choices = ('full_homogeneous',)
        p.set_defaults(model_arch='full_homogeneous', ct_backbone=None, pet_backbone=None,
                       ct_pretrained_path=None, pet_pretrained_path=None)
        p.add_argument('--encoder_pair', choices=tuple(NATIVE_CHANNELS), default='mit_b1',
                       help='Same architecture in both modalities; independent weights, no pre-fusion alignment.')
        return p

    @staticmethod
    def task_specific_parser():
        p = FullStructureConfig.task_specific_parser()
        p.set_defaults(fusion_enabled=False)
        return p

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        for name in ('ct_backbone', 'pet_backbone'):
            requested = getattr(cfg, name)
            if requested is not None and requested != cfg.encoder_pair:
                raise ValueError(f'{name}={requested} conflicts with encoder_pair={cfg.encoder_pair}')
            setattr(cfg, name, cfg.encoder_pair)
        for name in ('ct_pretrained_path', 'pet_pretrained_path'):
            if getattr(cfg, name) is None and cfg.pretrained:
                setattr(cfg, name, DEFAULT_WEIGHTS[cfg.encoder_pair])
        if cfg.fusion_enabled and any((c//2) % cfg.fusion_heads for c in NATIVE_CHANNELS[cfg.encoder_pair]):
            raise ValueError('Fusion heads must divide the selected encoder half-widths')
        cfg.channel_alignment = 'identity'
        return cfg
