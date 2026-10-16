"""Stage-1 Full CT+PET training with MMTM-inspired calibrated addition."""
from configs.full_channel_calibration import FullChannelCalibrationConfig
from models.components.channel_calibration import ChannelCalibrationFusion
from models.full_channel_calibration import CONFIG_FIELDS, build_full_channel_model
from run_full_cmx_fusion import main as train_full


def main():
    train_full(config_cls=FullChannelCalibrationConfig, model_builder=build_full_channel_model,
               expected_fusion=ChannelCalibrationFusion, experiment_type='full_channel_calibration',
               result_config_fields=CONFIG_FIELDS)


if __name__ == '__main__':
    main()
