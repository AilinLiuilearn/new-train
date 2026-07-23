import copy
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import PCLT20KSegDataset
from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.seg_losses import BCEDiceLoss


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
    }
    base.update(kwargs)
    return type('C', (), base)()


def test_model_imports():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    assert model.decoder.use_deep_supervision is False
    assert hasattr(model, 'apsf')
    assert not hasattr(model, 'fusion_legacy')


def test_forward_full_missing_shapes():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    model.eval()
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
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, _make_cfg())
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.is_tensor(loss)
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert torch.allclose(stats['loss_total'], stats['loss_seg'])
    assert 'loss_aux_apsf' not in stats


def test_build_teacher():
    cfg = _make_cfg(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, deep_supervision=False)
    out = build_mdt_seg_teacher(cfg)
    assert 'model' in out


def test_scheduler_state_dict_roundtrip():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_scheduler(opt, epochs=2, warmup_steps=1, min_lr=1e-6, steps_per_epoch=2, flat_ratio=0.3)
    state = copy.deepcopy(sched.state_dict())
    sched.step()
    sched.load_state_dict(state)
    assert isinstance(sched.state_dict(), dict)


def test_module_grad_norm_preserves_grad_and_value():
    from run_mdt_seg import module_grad_norm
    lin = torch.nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = lin(x).sum()
    y.backward()
    before = [p.grad.clone() for p in lin.parameters()]
    norm = module_grad_norm(lin)
    after = [p.grad for p in lin.parameters()]
    assert norm >= 0
    for b, a in zip(before, after):
        assert torch.allclose(b, a)


def test_missing_path_pet_encoder_not_called(monkeypatch):
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    calls = {'n': 0}
    orig = model.enc_pet.forward

    def wrapped(*args, **kwargs):
        calls['n'] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(model.enc_pet, 'forward', wrapped)
    ct = torch.randn(1, 1, 64, 64)
    out = model(ct, None, forward_mode='missing')
    assert 'logits' in out
    assert calls['n'] == 0


def test_checkpoint_save_and_eval_config_contract(tmp_path):
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, _make_cfg())
    path = tmp_path / 'ckpt.pth.tar'
    task.save_checkpoint(str(path), 1, best_joint=0.1, best_full=0.2, best_missing=0.3, best_joint_epoch=1, val_full={'dice': 0.2}, val_missing={'dice': 0.3}, joint_dice=0.25)
    ckpt = torch.load(path, map_location='cpu')
    saved_config = dict(ckpt['config'])
    saved_config.pop('checkpoint_dir', None)
    saved_config['root'] = '/tmp/root'
    saved_config['random_state'] = 123
    saved_config['ct_pretrained_path'] = None
    saved_config['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=saved_config)
    assert cfg.root == '/tmp/root'
