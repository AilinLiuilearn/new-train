import copy

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.seg_losses import BCEDiceLoss


def _test_text_embeddings(text_dim=512):
    embeddings = torch.zeros(2, text_dim)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    return embeddings


def _make_baseline(**kwargs):
    defaults = dict(
        use_deep_supervision=False,
        pet_text_embeddings=_test_text_embeddings(),
        edv_attention_backend='sdpa',
    )
    defaults.update(kwargs)
    return DualSharedAddPETCTBaseline(**defaults)


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
    model = _make_baseline()
    assert model.decoder.use_deep_supervision is False


def test_forward_full_missing_shapes():
    model = _make_baseline()
    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    out_full = model(ct, pet, forward_mode='full')
    out_missing = model(ct, pet, forward_mode='missing')
    assert out_full['logits'].shape == out_missing['logits'].shape


def test_bce_dice_loss_unpack():
    loss_fn = BCEDiceLoss()
    logits = torch.randn(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    loss, stats = loss_fn(logits, target)
    assert torch.is_tensor(loss)
    assert 'loss_bce' in stats and 'loss_dice' in stats


def test_task_train_step_unpacks_logits():
    task = MDTSegTeacher({'model': _make_baseline()}, _make_cfg())
    batch = {
        'ct': torch.randn(1, 1, 64, 64),
        'pet': torch.randn(1, 1, 64, 64),
        'mask': torch.zeros(1, 1, 64, 64),
    }
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.is_tensor(loss)
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert 'loss_total' in stats


def test_build_teacher(tmp_path):
    cfg = _make_cfg(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        checkpoint_dir=str(tmp_path),
        pet_text_embeddings=_test_text_embeddings(),
        edv_attention_backend='sdpa',
    )
    out = build_mdt_seg_teacher(cfg)
    assert 'model' in out
    from models.evidence_guided_sdnca_pet_ct import MultiScaleEvidenceGuidedSDNCA
    assert isinstance(out['model'].fusion, MultiScaleEvidenceGuidedSDNCA)


def test_scheduler_state_dict_roundtrip():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_scheduler(
        opt,
        epochs=2,
        warmup_steps=1,
        min_lr=1e-6,
        steps_per_epoch=2,
        flat_ratio=0.3,
    )
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


def test_missing_path_encodes_real_pet_but_fuses_calibrated_proxy(monkeypatch):
    """Missing still encodes real PET for CPPI collect; EDV gets calibrated proxy."""
    model = _make_baseline()
    pet_encoder_calls = {'n': 0}
    calib_inputs = []
    fusion_inputs = []
    orig_pet = model.enc_pet.forward
    orig_calib = model.pet_calibration.forward
    orig_fusion = model.fusion.forward

    def wrapped_pet(*args, **kwargs):
        pet_encoder_calls['n'] += 1
        return orig_pet(*args, **kwargs)

    def wrapped_calib(ct_feats, pet_feats, *args, **kwargs):
        calib_inputs.append([feat.detach().clone() for feat in pet_feats])
        return orig_calib(ct_feats, pet_feats, *args, **kwargs)

    def wrapped_fusion(ct_feats, pet_feats, *args, **kwargs):
        fusion_inputs.append([feat.detach().clone() for feat in pet_feats])
        return orig_fusion(ct_feats, pet_feats, *args, **kwargs)

    monkeypatch.setattr(model.enc_pet, 'forward', wrapped_pet)
    monkeypatch.setattr(model.pet_calibration, 'forward', wrapped_calib)
    monkeypatch.setattr(model.fusion, 'forward', wrapped_fusion)

    ct = torch.randn(1, 1, 64, 64)
    pet = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        real_pet_feats = [f.detach().clone() for f in model._encode_pet(pet)]
    pet_encoder_calls['n'] = 0
    out = model(ct, pet, forward_mode='missing')
    assert 'logits' in out
    assert pet_encoder_calls['n'] == 1
    assert len(calib_inputs) == 1
    assert len(fusion_inputs) == 1
    # Proxy path into calibration must not be identical to real PET features.
    for real, calib_in in zip(real_pet_feats, calib_inputs[0]):
        assert not torch.allclose(real, calib_in)
    # EDV must receive calibration outputs (identity when bank not ready).
    for calib_in, fusion_in in zip(calib_inputs[0], fusion_inputs[0]):
        assert torch.allclose(calib_in, fusion_in)


def test_checkpoint_save_and_eval_config_contract(tmp_path):
    task = MDTSegTeacher({'model': _make_baseline()}, _make_cfg())
    path = tmp_path / 'ckpt.pth.tar'
    task.save_checkpoint(
        str(path),
        1,
        best_joint=0.1,
        best_full=0.2,
        best_missing=0.3,
        best_joint_epoch=1,
        val_full={'dice': 0.2},
        val_missing={'dice': 0.3},
        joint_dice=0.25,
    )
    ckpt = torch.load(path, map_location='cpu')
    saved_config = dict(ckpt['config'])
    saved_config.pop('checkpoint_dir', None)
    saved_config['root'] = '/tmp/root'
    saved_config['random_state'] = 123
    saved_config['ct_pretrained_path'] = None
    saved_config['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=saved_config)
    assert cfg.root == '/tmp/root'
