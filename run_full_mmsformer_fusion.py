"""Full real CT+PET with native MMSFormer FusionBlock, baseline BCE+Dice/EMA."""
from configs.full_mmsformer import FullMMSFormerConfig
from models.components.mmsformer_fusion import MMSFormerFusion
from models.full_mmsformer_fusion import CONFIG_FIELDS, build_full_mmsformer_model
from run_full_cmx_fusion import main as train_full


def main():
    train_full(config_cls=FullMMSFormerConfig, model_builder=build_full_mmsformer_model,
               expected_fusion=MMSFormerFusion, experiment_type='full_mmsformer_fusion',
               result_config_fields=CONFIG_FIELDS)


if __name__ == '__main__':
    main()
