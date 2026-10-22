"""Clean entrypoints preserve frozen settings and reject removed experiments."""
import argparse
import pytest
import run_staged as runner
from scripts.train_local_region import training_argv


def test_only_original_augmentation_is_exposed():
    for flag in ('--crop_guidance', '--copy_paste', '--guidance_map_dir', '--paste_index'):
        with pytest.raises(SystemExit): runner.parser().parse_args([flag, 'unused'])


def test_old_inert_checkpoint_config_remains_compatible():
    cfg = argparse.Namespace(crop_guidance='none', copy_paste='none',
                             guided_crop_prob=.5, guidance_map_dir=None,
                             paste_index=None, decoder_norm='GN')
    runner.clean_legacy_augmentation_config(cfg)
    assert vars(cfg) == {'decoder_norm': 'GN'}
    for name in ('crop_guidance', 'copy_paste'):
        with pytest.raises(ValueError, match='archived branch'):
            runner.clean_legacy_augmentation_config(argparse.Namespace(**{name: 'dual_state'}))


@pytest.mark.parametrize('fusion', ['add', 'local_region_residual'])
def test_portable_frozen_launcher(fusion):
    argv = training_argv('new-output', fusion_type=fusion, root='/new instance/data',
                         ct_pretrained_path='/weights/ct', pet_pretrained_path='/weights/pet', random_state=2024)
    cfg = runner.parser().parse_args(argv)
    runner.normalize_fusion_config(cfg); runner.validate(cfg)
    assert cfg.root == '/new instance/data' and cfg.ct_pretrained_path == '/weights/ct'
    assert cfg.pet_pretrained_path == '/weights/pet' and cfg.random_state == 2024
    assert cfg.fusion_type == fusion and cfg.decoder_norm == 'GN'
    assert cfg.image_size_2d == 512 and cfg.batch_size == 16 and cfg.learning_rate == 8e-5
    assert cfg.epochs == 60 and cfg.early_stopping_patience == 5 and cfg.mixed_precision and cfg.ema_enabled
    assert not cfg.save_last_checkpoint and cfg.aug_mode == 'cipa'
    assert not hasattr(cfg, 'copy_paste') and not hasattr(cfg, 'crop_guidance')
