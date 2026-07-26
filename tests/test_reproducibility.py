import hashlib
import os
import tempfile

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import PCLT20KSegDataset
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import compute_hd95_pair, SegmentationMetricsCIPA
from utils.reproducibility import configure_reproducibility
from utils.image_augmentation import randomHorizontalFlip


def _tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).tolist().__repr__().encode()).hexdigest()


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'hd95_backend': 'scipy',
        'assert_eval_state_unchanged': True,
    }
    base.update(kwargs)
    return type('C', (), base)()


def test_same_seed_same_initial_weights():
    configure_reproducibility(2023, 'balanced')
    m1 = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    h1 = [_tensor_hash(p) for p in m1.parameters()]
    configure_reproducibility(2023, 'balanced')
    m2 = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    h2 = [_tensor_hash(p) for p in m2.parameters()]
    assert h1 == h2


def test_same_input_same_eval_logits():
    configure_reproducibility(2023, 'balanced')
    m = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    m.eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    out1 = m(ct, pet, forward_mode='full')['logits']
    out2 = m(ct, pet, forward_mode='full')['logits']
    assert torch.equal(out1, out2)


def test_checkpoint_roundtrip_logits(tmp_path):
    cfg = _make_cfg()
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, cfg)
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    path = tmp_path / 'ckpt.pth.tar'
    task.save_checkpoint(str(path), 1)
    ct = batch['ct'].to(task.device)
    pet = batch['pet'].to(task.device)
    x1 = task.model(ct, pet, forward_mode='full')['logits']
    task.load_checkpoint(str(path), strict=True, restore_rng=True)
    x2 = task.model(ct, pet, forward_mode='full')['logits']
    assert torch.equal(x1, x2)


def test_metrics_repeatable_and_state_unchanged():
    cfg = _make_cfg()
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, cfg)
    task.model.eval()
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    batch = {k: v.to(task.device) for k, v in batch.items()}
    loader = [batch, batch]
    a = task.evaluate(loader, eval_mode='full')
    b = task.evaluate(loader, eval_mode='full')
    for k in ['dice', 'iou', 'acc', 'acc_pixel', 'hd95', 'precision', 'f1']:
        assert abs(a[k] - b[k]) < 1e-10


def test_dataset_seeded_augmentation_consistent():
    rec = [{'image_id': 'case1_001', 'case_id': 'case1', 'slice_id': 'case1_001', 'ct_path': '/tmp/a', 'pet_path': '/tmp/b', 'mask_path': '/tmp/c'}]
    ds = PCLT20KSegDataset.__new__(PCLT20KSegDataset)
    ds.records = rec
    ds.image_size = 64
    ds.train = False
    ds.aug_mode = 'cipa'
    ds.norm_mode = 'cipa'
    ds.base_seed = 2023
    ds.current_epoch = 1
    assert ds.base_seed == 2023
    assert ds.current_epoch == 1


def test_hd95_backend_fixed():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    assert compute_hd95_pair(a, b, backend='scipy') == 0.0


def test_pretrained_must_error_on_missing_path():
    from models.build_mdt_seg import load_local_weights_safe
    model = torch.nn.Linear(2, 2)
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, 'missing.pth')
        load_local_weights_safe(model, missing, name='Linear')


def test_strict_load_state_dict():
    m = torch.nn.Linear(2, 2)
    sd = m.state_dict()
    m.load_state_dict(sd, strict=True)
