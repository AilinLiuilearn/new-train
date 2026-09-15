# -*- coding: utf-8 -*-
"""Mixed Full/Missing training tests for the PSPI module1-clean branch.

Mechanism must be identical to the e1-api-masked-baseline-mix-full-missing
baseline: one forward, one backward, one optimizer update, one scheduler
update per batch; per-sample balanced Full/Missing states; subset losses
combined 50/50 (missing_loss_weight=1.0). The only addition on this branch
is the clean Module-1 (PSPI) prior imputation + prototype loss, which must
not change the training-loop mechanism.

Run:  python -m pytest -q tests/test_mixed_batch_training.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher
from run_mdt_seg import build_balanced_pet_available, _assert_mixed


class _FakeCfg:
    learning_rate = 1e-4
    weight_decay = 1e-4
    mixed_precision = False
    loss_smooth = 1.0
    bce_weight = 1.0
    dice_weight = 1.0
    random_state = 2023
    train_batch_mode = 'mixed'
    missing_loss_weight = 1.0
    grad_clip = 0.0
    pspi_proto_contrastive_weight = 0.01


def _make_task(pspi_enabled=True):
    model = DualSharedAddPETCTBaseline(
        ct_backbone='convnext_tiny',
        pet_backbone='mit_b0',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        use_deep_supervision=False,
        pspi_enabled=pspi_enabled,
        pspi_num_clusters=2,
        pspi_cluster_max_iter=2,
    )
    return MDTSegTeacher({'model': model}, _FakeCfg())


def _make_batch(bs=4, size=64, device='cpu'):
    return {
        'ct': torch.randn(bs, 1, size, size).to(device),
        'pet': torch.randn(bs, 1, size, size).to(device),
        'mask': (torch.rand(bs, 1, size, size) > 0.5).float().to(device),
    }


# ------------------------------------------------------------------
# 1. Balanced per-sample availability (same helper as the baseline)
# ------------------------------------------------------------------

def test_balanced_pet_available_is_half_and_half():
    state = build_balanced_pet_available(16, 0, 2023, torch.device('cpu'))
    assert state.dtype == torch.long
    assert state.numel() == 16
    assert int(state.sum()) == 8
    assert int((state == 0).sum()) == 8
    assert torch.all((state == 0) | (state == 1))


def test_balanced_pet_available_is_deterministic_and_step_dependent():
    a = build_balanced_pet_available(16, 0, 2023, torch.device('cpu'))
    b = build_balanced_pet_available(16, 0, 2023, torch.device('cpu'))
    c = build_balanced_pet_available(16, 1, 2023, torch.device('cpu'))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ------------------------------------------------------------------
# 2. Mixed training step: one forward, 50/50 loss, correct stats
# ------------------------------------------------------------------

def test_train_step_mixed_counts_single_forward():
    task = _make_task()
    task.model.train()
    calls = {'n': 0}
    orig_forward = task.model.forward

    def counting_forward(*args, **kwargs):
        calls['n'] += 1
        return orig_forward(*args, **kwargs)

    task.model.forward = counting_forward
    batch = _make_batch(4)
    state = build_balanced_pet_available(4, 0, 2023, task.device)
    loss, logits, outputs, stats = task.train_step_mixed(batch, state, missing_loss_weight=1.0)
    assert calls['n'] == 1, 'mixed step must run exactly one model forward'
    assert stats['num_full'] == 2 and stats['num_missing'] == 2
    assert stats['full_weight'] == 0.5 and stats['missing_weight'] == 0.5
    assert torch.isfinite(loss)
    assert logits.shape[0] == 4


def test_train_step_mixed_loss_equals_baseline_definition_plus_pspi_proto():
    task = _make_task()
    task.model.train()
    batch = _make_batch(4)
    state = build_balanced_pet_available(4, 3, 2023, task.device)
    loss, logits, outputs, stats = task.train_step_mixed(batch, state, missing_loss_weight=1.0)
    mask = batch['mask'].to(task.device).float()
    full_idx = state.eq(1)
    missing_idx = state.eq(0)
    l_full, _ = task.criterion(logits[full_idx], mask[full_idx])
    l_missing, _ = task.criterion(logits[missing_idx], mask[missing_idx])
    proto = outputs['prototype_contrastive_loss']
    expected = 0.5 * l_full + 0.5 * l_missing + 0.01 * proto
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-6)
    assert torch.allclose(stats['loss_full'].float(), l_full.detach().float(), atol=1e-6)
    assert torch.allclose(stats['loss_missing'].float(), l_missing.detach().float(), atol=1e-6)


def test_train_step_mixed_gradients_match_weighted_sum():
    # Within the SAME forward graph, the returned total loss gradient is
    # exactly the weighted combination of branch gradients; two separate
    # autograd.grad passes on GPU agree up to float32 summation order.
    torch.manual_seed(7)
    task = _make_task()
    task.model.train()
    batch = _make_batch(4)
    state = build_balanced_pet_available(4, 5, 2023, task.device)
    mask = batch['mask'].to(task.device).float()
    loss, logits, outputs, _ = task.train_step_mixed(batch, state, missing_loss_weight=1.0)
    full_idx = state.eq(1)
    missing_idx = state.eq(0)
    l_full, _ = task.criterion(logits[full_idx], mask[full_idx])
    l_missing, _ = task.criterion(logits[missing_idx], mask[missing_idx])
    proto = outputs['prototype_contrastive_loss']
    params = [p for p in task.model.parameters() if p.requires_grad]
    g_combined = torch.autograd.grad(0.5 * l_full + 0.5 * l_missing + 0.01 * proto, params, retain_graph=True, allow_unused=True)
    g_direct = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    for ga, gd in zip(g_combined, g_direct):
        if ga is None:
            continue
        assert torch.allclose(ga.float(), gd.float(), atol=5e-3, rtol=5e-2)


# ------------------------------------------------------------------
# 3. PSPI semantics inside the mixed batch
# ------------------------------------------------------------------

def test_missing_rows_independent_of_real_pet_full_rows_depend():
    task = _make_task()
    task.model.eval()
    dev = task.device
    batch_a = _make_batch(4, device=dev)
    batch_b = _make_batch(4, device=dev)
    batch_b['pet'] = torch.randn(4, 1, 64, 64).to(dev)
    state = build_balanced_pet_available(4, 9, 2023, dev)
    mask = batch_a['mask']
    with torch.no_grad():
        out_a = task.model(batch_a['ct'], pet=batch_a['pet'], pet_available=state, forward_mode='auto', mask=mask)
        out_b = task.model(batch_a['ct'], pet=batch_b['pet'], pet_available=state, forward_mode='auto', mask=mask)
    missing_idx = state.eq(0)
    full_idx = state.eq(1)
    assert torch.allclose(out_a['logits'][missing_idx], out_b['logits'][missing_idx], atol=1e-5), \
        'Missing rows must not leak real PET'
    assert not torch.allclose(out_a['logits'][full_idx], out_b['logits'][full_idx], atol=1e-5), \
        'Full rows must use real PET'


def test_mixed_step_collects_candidates_once_per_batch():
    task = _make_task()
    task.model.train()
    before = int(task.model.module1._collect_calls)
    batch = _make_batch(4)
    state = build_balanced_pet_available(4, 0, 2023, task.device)
    task.train_step_mixed(batch, state, missing_loss_weight=1.0)
    after = int(task.model.module1._collect_calls)
    assert after - before == 1, 'exactly one candidate collection per mixed batch (same as baseline)'


def test_fixed_missing_eval_matches_all_zero_auto():
    task = _make_task()
    task.model.eval()
    batch = _make_batch(4)
    ct = batch['ct'].to(task.device)
    pet = batch['pet'].to(task.device)
    mask = batch['mask'].to(task.device).float()
    with torch.no_grad():
        out_eval = task.model(ct, pet=None, forward_mode='missing', mask=mask)
        out_zero = task.model(ct, pet=pet, pet_available=torch.zeros(4, dtype=torch.long, device=task.device),
                              forward_mode='auto', mask=mask)
    assert torch.allclose(out_eval['logits'], out_zero['logits'], atol=1e-5)


# ------------------------------------------------------------------
# 4. Input validation (same errors as baseline mixed contract)
# ------------------------------------------------------------------

def _expect(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f'expected {exc.__name__}')


def test_invalid_availability_states_are_rejected():
    task = _make_task()
    task.model.train()
    batch = _make_batch(4)
    _expect(lambda: task.train_step_mixed(batch, torch.ones(3, dtype=torch.long), 1.0))
    _expect(lambda: task.train_step_mixed(batch, torch.tensor([0, 2, 0, 1]), 1.0))
    _expect(lambda: task.train_step_mixed(batch, torch.zeros(4, dtype=torch.long), 1.0))


def test_mixed_mode_requires_even_batch_drop_last_and_no_route_drop():
    cfg = _FakeCfg()
    cfg.batch_size = 16
    cfg.train_pet_drop_prob = 0.0
    loader = type('L', (), {'drop_last': True})()
    _assert_mixed(cfg, loader)  # no raise
    cfg2 = _FakeCfg()
    cfg2.batch_size = 15
    cfg2.train_pet_drop_prob = 0.0
    _expect(lambda: _assert_mixed(cfg2, loader))
    cfg3 = _FakeCfg()
    cfg3.batch_size = 16
    cfg3.train_pet_drop_prob = 0.0
    _expect(lambda: _assert_mixed(cfg3, type('L', (), {'drop_last': False})()))
    cfg4 = _FakeCfg()
    cfg4.batch_size = 16
    cfg4.train_pet_drop_prob = 0.25
    _expect(lambda: _assert_mixed(cfg4, loader))


# ------------------------------------------------------------------
# 5. Checkpoint + config contract
# ------------------------------------------------------------------

def test_checkpoint_records_train_batch_mode(tmp_path):
    task = _make_task()
    path = os.path.join(str(tmp_path), 'ckpt.pth.tar')
    task.save_checkpoint(path, 1, 0.5, 0.6, 0.4, 1, {'total_loss': 0.5}, {'total_loss': 0.6}, 0.5)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['train_batch_mode'] == 'mixed'
    assert 'global_batch_step' in ckpt and ckpt['epoch'] == 1
    fresh = _make_task()
    fresh.model.load_state_dict(ckpt['model'], strict=True)
    assert 'config' in ckpt


def test_legacy_alternating_mode_still_available():
    from configs.seg_mdt import SegMDTConfig
    import argparse
    found = None
    for parser in (SegMDTConfig.train_parser(),):
        for action in parser._actions:
            if '--train_batch_mode' in action.option_strings:
                found = action
    assert found is not None
    assert found.choices == ('alternating', 'mixed')
    assert found.default == 'mixed'


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as exc:
                print(f'FAIL {name}: {exc!r}')
                raise
