import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss


def test_model_imports():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    assert model.decoder.use_deep_supervision is False


def test_forward_full_missing_shapes():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out_full = model(ct, pet, forward_mode='full')
    out_missing = model(ct, None, forward_mode='missing')
    assert out_full['logits'].shape == out_missing['logits'].shape


def test_bce_dice_loss_unpack():
    loss_fn = BCEDiceLoss()
    logits = torch.randn(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    loss, stats = loss_fn(logits, target)
    assert torch.is_tensor(loss)
    assert 'loss_bce' in stats and 'loss_dice' in stats


def test_task_train_step_unpacks_logits():
    cfg = type('C', (), {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
    })()
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, cfg)
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.is_tensor(loss)
    assert isinstance(outputs, dict)
    assert 'logits' in outputs


def test_build_teacher():
    cfg = type('C', (), {'ct_backbone': 'convnextv2_nano', 'pet_backbone': 'mit_b1', 'ct_pretrained_path': None, 'pet_pretrained_path': None, 'decoder_channels': (512, 256, 128, 64), 'use_deep_supervision': False, 'deep_supervision': False})()
    out = build_mdt_seg_teacher(cfg)
    assert 'model' in out
