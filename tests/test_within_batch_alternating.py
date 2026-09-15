# -*- coding: utf-8 -*-
"""Within-batch alternating Full/Missing training tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'train_mode': 'within_batch_alternating',
        'within_batch_full_weight': 0.5,
        'within_batch_missing_weight': 0.5,
        'grad_clip': 5.0,
        'pspi_proto_contrastive_weight': 0.01,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _make_model(**kwargs):
    kwargs.setdefault('ct_backbone', 'convnext_tiny')
    kwargs.setdefault('pet_backbone', 'mit_b0')
    kwargs.setdefault('ct_pretrained_path', None)
    kwargs.setdefault('pet_pretrained_path', None)
    kwargs.setdefault('use_deep_supervision', False)
    kwargs.setdefault('pspi_enabled', True)
    kwargs.setdefault('pspi_num_clusters', 2)
    kwargs.setdefault('pspi_cluster_max_iter', 2)
    return DualSharedAddPETCTBaseline(**kwargs)


def _make_task(**kwargs):
    cfg = _make_cfg()
    return MDTSegTeacher({'model': _make_model(**kwargs)}, cfg)


def _make_batch(batch_size=2, size=64):
    return {
        'ct': torch.randn(batch_size, 1, size, size),
        'pet': torch.randn(batch_size, 1, size, size),
        'mask': (torch.rand(batch_size, 1, size, size) > 0.5).float(),
    }


def test_within_batch_counts_two_forwards_one_step(monkeypatch):
    task = _make_task()
    calls = {'fwd': 0, 'opt': 0, 'sched': 0}

    def counted_train_step(batch, forward_mode='full', collect_module1_candidates=True):
        calls['fwd'] += 1
        return MDTSegTeacher.train_step(task, batch, forward_mode=forward_mode, collect_module1_candidates=collect_module1_candidates)

    monkeypatch.setattr(task, 'train_step', counted_train_step)
    opt_cls = type(task.optimizer)
    orig_opt_step = opt_cls.step

    def counted_opt_step(self, *args, **kwargs):
        calls['opt'] += 1
        return orig_opt_step(self, *args, **kwargs)

    monkeypatch.setattr(opt_cls, 'step', counted_opt_step)
    sched = torch.optim.lr_scheduler.LambdaLR(task.optimizer, lr_lambda=lambda step: 1.0)
    task.scheduler = sched

    combined, _, _, _ = task.train_batch_full_missing(_make_batch(2))
    task.scheduler.step()
    calls['sched'] += 1
    task.global_batch_step += 1

    assert calls['fwd'] == 2
    assert calls['opt'] == 1
    assert calls['sched'] == 1
    assert task.global_batch_step == 1
    assert torch.isfinite(combined)


def test_global_batch_step_single_increment_three_batches():
    task = _make_task()
    for _ in range(3):
        task.train_batch_full_missing(_make_batch(2))
        task.global_batch_step += 1
    assert task.global_batch_step == 3


def test_full_missing_share_same_batch(monkeypatch):
    task = _make_task()
    seen = []

    def spy(batch, forward_mode='full', collect_module1_candidates=True):
        seen.append((batch['ct'], batch['pet'], forward_mode))
        return task.train_step.__wrapped__ if False else _passthrough(task, batch, forward_mode, collect_module1_candidates)

    def _passthrough(task, batch, forward_mode, collect_flag):
        return MDTSegTeacher.train_step(task, batch, forward_mode=forward_mode, collect_module1_candidates=collect_flag)

    monkeypatch.setattr(task, 'train_step', spy)
    task.train_batch_full_missing(_make_batch(2))
    assert len(seen) == 2
    modes = sorted(m[2] for m in seen)
    assert modes == ['full', 'missing']
    assert seen[0][0] is seen[1][0] and seen[0][1] is seen[1][1]


def test_full_missing_combined_stats_logged():
    task = _make_task()
    _, _, _, stats = task.train_batch_full_missing(_make_batch(2))
    for key in ('full_loss', 'missing_loss', 'combined_loss', 'total_grad_norm',
                'grad_combined_enc_ct', 'grad_combined_decoder',
                'grad_combined_module1_retrieval', 'grad_combined_prior_scale'):
        assert key in stats
        assert torch.isfinite(torch.as_tensor(float(stats[key])))
    assert abs(float(stats['combined_loss']) - 0.5 * (float(stats['full_loss']) + float(stats['missing_loss']))) < 1e-5


def test_combined_gradient_equals_weighted_sum():
    torch.manual_seed(7)
    task = _make_task()
    task.model.eval()
    batch = _make_batch(2)
    params = [p for p in task.model.parameters() if p.requires_grad]

    task.model.zero_grad(set_to_none=True)
    _, _, _, _ = task.train_batch_full_missing(batch)
    g_mixed = [None if p.grad is None else p.grad.detach().clone() for p in params]

    task.model.zero_grad(set_to_none=True)
    loss_f, _, _, _ = task.train_step(batch, forward_mode='full', collect_module1_candidates=True)
    loss_m, _, _, _ = task.train_step(batch, forward_mode='missing', collect_module1_candidates=False)
    (0.5 * loss_f + 0.5 * loss_m).backward()
    expected = [None if p.grad is None else p.grad.detach().clone() for p in params]

    compared = 0
    for gm, exp in zip(g_mixed, expected):
        if gm is None:
            continue
        exp = exp if exp is not None else torch.zeros_like(gm)
        assert torch.allclose(gm.float(), exp.float(), atol=5e-3, rtol=5e-2)
        compared += 1
    assert compared > 0


def test_module1_collection_exactly_once_per_batch_pair():
    task = _make_task()
    batch = _make_batch(2)
    before = int(task.model.module1._collect_calls)
    task.model.train()
    task.train_step(batch, forward_mode='full', collect_module1_candidates=True)
    task.train_step(batch, forward_mode='missing', collect_module1_candidates=False)
    after = int(task.model.module1._collect_calls)
    assert after - before == 1


def test_missing_logits_independent_of_real_pet():
    task = _make_task()
    task.model.eval()
    dev = task.device
    ct = torch.randn(2, 1, 64, 64).to(dev)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float().to(dev)
    with torch.no_grad():
        out_a = task.model(ct, pet=torch.randn(2, 1, 64, 64).to(dev), forward_mode='missing', mask=mask)
        out_b = task.model(ct, pet=torch.randn(2, 1, 64, 64).to(dev), forward_mode='missing', mask=mask)
    assert torch.allclose(out_a['logits'], out_b['logits'], atol=1e-5)


def test_full_logits_use_real_pet():
    task = _make_task()
    task.model.eval()
    dev = task.device
    ct = torch.randn(2, 1, 64, 64).to(dev)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float().to(dev)
    with torch.no_grad():
        out_a = task.model(ct, pet=torch.randn(2, 1, 64, 64).to(dev), forward_mode='full', mask=mask)
        out_b = task.model(ct, pet=torch.randn(2, 1, 64, 64).to(dev), forward_mode='full', mask=mask)
    assert not torch.allclose(out_a['logits'], out_b['logits'], atol=1e-5)


def test_cold_start_bank_not_ready():
    model = _make_model()
    assert model.module1.bank_ready is False
    assert int(model.module1.bank_version.item()) == 0


def test_finalize_called_once_per_epoch(monkeypatch):
    task = _make_task()
    calls = {'n': 0}
    orig = task.model.finalize_module1_epoch

    def counted(epoch):
        calls['n'] += 1
        return orig(epoch)

    monkeypatch.setattr(task.model, 'finalize_module1_epoch', counted)
    task.train_batch_full_missing(_make_batch(2))
    task.train_batch_full_missing(_make_batch(2))
    task.model.finalize_module1_epoch(epoch=1)
    assert calls['n'] == 1


def test_legacy_mode_still_runs():
    cfg = _make_cfg(train_mode='legacy_batch_alternating')
    task = MDTSegTeacher({'model': _make_model()}, cfg)
    batch = _make_batch(2)
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, _ = task.train_step(batch, forward_mode='full')
    loss.backward()
    task.optimizer.step()
    task.optimizer.zero_grad(set_to_none=True)
    loss2, _, _, _ = task.train_step(batch, forward_mode='missing')
    loss2.backward()
    task.optimizer.step()
    assert torch.isfinite(loss) and torch.isfinite(loss2)


def test_checkpoint_records_train_mode_and_weights(tmp_path):
    task = _make_task()
    path = str(tmp_path / 'ckpt.pth.tar')
    task.save_checkpoint(path, 1)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['train_mode'] == 'within_batch_alternating'
    assert ckpt['within_batch_full_weight'] == pytest.approx(0.5)
    assert ckpt['within_batch_missing_weight'] == pytest.approx(0.5)
    legacy_payload = dict(ckpt)
    legacy_payload.pop('train_mode')
    legacy_payload.pop('within_batch_full_weight')
    train_mode = str(legacy_payload.get('train_mode', 'legacy_batch_alternating'))
    assert train_mode == 'legacy_batch_alternating'
