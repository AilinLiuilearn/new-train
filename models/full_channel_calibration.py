"""Baseline model with an isolated four-scale channel-calibration fusion seam."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.channel_calibration import ChannelCalibrationFusion, validate_settings

CONFIG_FIELDS = ('channel_calibration_enabled', 'channel_reduction', 'channel_strength')


class FullChannelCalibrationPETCT(DualSharedAddPETCTBaseline):
    """Full trains real CT+PET; inherited Missing/auto never read absent PET.

    Calibrated features enter only the decoder, never the next encoder stage.
    Both encoders and aligners retain the original pretrained-loading policy.
    """

    def __init__(self, channel_calibration_enabled=True, channel_reduction=8,
                 channel_strength=0.25, **kwargs):
        validate_settings(channel_reduction, channel_strength)
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('This experiment uses baseline main BCE+Dice only')
        super().__init__(**kwargs)
        # Add new parameters only after the entire baseline is initialized.
        # Restore CPU RNG so subsequent dropout/data RNG matches the baseline.
        with torch.random.fork_rng(devices=[]):
            self.fusion = ChannelCalibrationFusion(channel_calibration_enabled,
                                                    channel_reduction, channel_strength)


def build_full_channel_model(cfg):
    model = FullChannelCalibrationPETCT(
        **{name: getattr(cfg, name) for name in CONFIG_FIELDS},
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[fusion] MMTM-inspired adapted_channel_calibration enabled={cfg.channel_calibration_enabled} '
          'scales=S1,S2,S3,S4 modalities=CT,PET initialization=exact_Add '
          f'reduction={cfg.channel_reduction} strength={cfg.channel_strength} '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
