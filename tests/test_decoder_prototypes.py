"""Synthetic contracts; fake encoders isolate decoder/prototype integration."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.decoder_dual_prototype import EMACentroidBank, DecoderPrototypeScale, gain_weights
from models.prototype_add_model import PrototypeAddModel
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.prototype_seg import PrototypeSegTask


class FakeEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in channels)
        self.feature_info = SimpleNamespace(channels=lambda: list(channels))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [layer(F.adaptive_avg_pool2d(x, (16 // 2**i, 16 // 2**i)))
                for i, layer in enumerate(self.layers)]


@pytest.fixture(autouse=True)
def fake_encoders(monkeypatch):
    import models.dual_shared_add_baseline as base
    monkeypatch.setattr(base, 'create_feature_backbone', lambda name, **kw:
                        FakeEncoder((8, 16, 24, 32) if 'convnext' in name
                                    else (64, 128, 320, 512)))
    torch.set_num_threads(1)


def model(**kw):
    return PrototypeAddModel(pretrained=False, decoder_channels=(32, 24, 16, 8),
                             prototype_dim=4, prototype_clusters=2,
                             prototype_capacity=16, prototype_min_candidates=2,
                             prototype_refresh_steps=1, prototype_warmup_steps=0, **kw)


def cfg(**kw):
    args = dict(learning_rate=1e-3, weight_decay=0., mixed_precision=False,
                ema_enabled=False, random_state=2023, prototype_ct_aux_weight=.1)
    args.update(kw)
    return SimpleNamespace(**args)


def batch():
    return dict(ct=torch.randn(4, 1, 32, 32), pet=torch.randn(4, 1, 32, 32),
                mask=(torch.rand(4, 1, 32, 32) > .6).float())


def populate(m):
    for scale in m.prototype.scales:
        for bank in (scale.spatial_bank, scale.channel_bank):
            bank.add(torch.randn(8, bank.dimension))
            bank.refresh()


def test_zero_init_matches_add_baseline_all_modes():
    m = model().eval()
    base = DualSharedAddPETCTBaseline(pretrained=False, decoder_channels=(32, 24, 16, 8)).eval()
    base.load_state_dict({k: v for k, v in m.state_dict().items()
                          if not k.startswith('prototype.')}, strict=True)
    populate(m)
    b = batch()
    for mode in ('full', 'missing', 'auto'):
        kw = dict(pet=b['pet'], forward_mode=mode, pet_available=torch.tensor([1, 0, 1, 0]))
        assert torch.equal(m(b['ct'], **kw)['logits'], base(b['ct'], **kw)['logits'])


def test_pet_isolation_with_active_nonzero_correction():
    m = model().eval()
    populate(m)
    for s in m.prototype.scales:
        nn.init.normal_(s.spatial_out.weight, std=.03)
        nn.init.normal_(s.channel_out.weight, std=.03)
    b = batch()
    state = torch.tensor([1, 0, 1, 0])
    a = m(b['ct'], pet=b['pet'], pet_available=state)['logits']
    pet = b['pet'].clone()
    pet[state == 0] = float('nan')
    z = m(b['ct'], pet=pet, pet_available=state)['logits']
    assert torch.equal(a, z)
    before = m.enc_pet.calls
    loss = m(b['ct'], pet=pet, forward_mode='missing')['logits'].square().mean()
    loss.backward()
    assert m.enc_pet.calls == before
    assert all(p.grad is None for p in m.enc_pet.parameters())


def test_bank_waits_for_success_and_eval_is_read_only():
    task = PrototypeSegTask({'model': model()}, cfg())
    b = batch()
    state = torch.tensor([1, 0, 1, 0])
    old = copy.deepcopy(task.model.prototype.state_dict())
    loss, _, _, stats = task.train_step_mixed(b, state)
    assert all(torch.equal(old[k], v) for k, v in task.model.prototype.state_dict().items())
    assert torch.allclose(loss.detach(), .5 * stats['loss_full'] + .5 * stats['loss_missing']
                          + .1 * stats['loss_ct_aux'])
    loss.backward()
    task.after_optimizer_step(False)
    assert all(torch.equal(old[k], v) for k, v in task.model.prototype.state_dict().items())
    task.train_step_mixed(b, state)
    task.after_optimizer_step(True)
    assert task.model.prototype.update_steps.item() == 1
    saved = copy.deepcopy(task.model.prototype.state_dict())
    task.model.eval()
    task.model(b['ct'], forward_mode='missing')
    assert all(torch.equal(saved[k], v) for k, v in task.model.prototype.state_dict().items())


def test_both_branches_learn_and_buffers_have_no_grads():
    m = model().train()
    populate(m)
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        m(torch.randn(2, 1, 32, 32), forward_mode='missing')['logits'].square().mean().backward()
        for scale in m.prototype.scales:
            assert scale.spatial_out.weight.grad.abs().sum() > 0
            assert scale.channel_out.weight.grad.abs().sum() > 0
            if step:
                assert scale.spatial_q.weight.grad.abs().sum() > 0
                assert scale.channel_q.weight.grad.abs().sum() > 0
        opt.step()
    assert all(not b.requires_grad for b in m.prototype.buffers())


def test_centroid_bank_euclidean_ema_filter_and_roundtrip():
    bank = EMACentroidBank(2, 2, capacity=12, min_candidates=4, decay=.5)
    x = torch.tensor([[-2., 0.], [-2.1, 0.], [2., 0.], [2.1, 0.]])
    bank.add(x)
    assert bank.refresh()
    old = bank.centers.clone()
    bank.pool.zero_(); bank.size.zero_(); bank.cursor.zero_()
    bank.add(x + torch.tensor([0., 2.]))
    bank.refresh()
    assert torch.allclose(bank.centers[:, 1], torch.ones(2))
    assert torch.allclose(bank.centers[:, 0].sort().values, old[:, 0].sort().values)
    fresh = EMACentroidBank(2, 2, capacity=12, min_candidates=4, decay=.5)
    fresh.load_state_dict(bank.state_dict(), strict=True)
    assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in bank.state_dict().items())
    with pytest.raises(ValueError):
        bank.add(torch.full((1, 2), float('nan')))


def test_outlier_filter_defaults_off_and_off_uses_all_cluster_members():
    values = torch.tensor([[0.], [.1], [.2], [.3], [5.], [100.], [100.1], [100.2], [100.3]])
    unfiltered = EMACentroidBank(2, 1, capacity=16, min_candidates=9)
    filtered = EMACentroidBank(2, 1, capacity=16, min_candidates=9, outlier_filter=True)
    for bank in (unfiltered, filtered):
        bank.add(values)
        assert bank.refresh()
    # The rare 5.0 descriptor stays in the left cluster's ordinary mean.
    assert not unfiltered.outlier_filter
    assert torch.allclose(unfiltered.centers.min(), values[:5].mean(), atol=1e-6)
    assert torch.allclose(filtered.centers.min(), values[:4].mean(), atol=1e-6)
    assert torch.equal(unfiltered.pool, filtered.pool)  # Trimming never mutates the FIFO.
    for enabled in (False, True):
        m = model(prototype_outlier_filter=enabled)
        assert m.prototype.diagnostics()['outlier_filter_enabled'] is enabled
        assert all(bank.outlier_filter is enabled for scale in m.prototype.scales
                   for bank in (scale.spatial_bank, scale.channel_bank))


def test_legacy_checkpoint_preserves_filter_policy_and_rejects_silent_change(tmp_path):
    task = PrototypeSegTask({'model': model(prototype_outlier_filter=True)}, cfg())
    path = str(tmp_path / 'legacy.pt')
    task.save_checkpoint(path, 0, best=0., best_epoch=0)
    legacy = task.load_state_dicts(path)
    legacy['config'].pop('prototype_outlier_filter')
    torch.save(legacy, path)
    same = PrototypeSegTask({'model': model(prototype_outlier_filter=True)}, cfg())
    same.resume(path)
    changed = PrototypeSegTask({'model': model()}, cfg())
    with pytest.raises(ValueError, match='prototype_outlier_filter'):
        changed.resume(path)


def test_gain_filter_and_signed_relations():
    scale = DecoderPrototypeScale(8, dim=4, clusters=2, capacity=16,
                                  min_candidates=2, refresh_steps=1)
    full = torch.randn(2, 8, 8, 8)
    ref = torch.randn_like(full)
    empty = torch.zeros(2, 1, 8, 8)
    s, c = scale.candidates(full, ref, empty)
    assert s.shape[0] == c.shape[0] == 0
    w = torch.ones_like(empty)
    _, plus = scale.candidates(full, ref, w)
    _, minus = scale.candidates(ref, full, w)
    assert torch.allclose(plus, -minus, atol=1e-6)
    assert torch.allclose(plus.view(-1, 4, 4), plus.view(-1, 4, 4).transpose(1, 2))


def test_ema_memory_is_copied_not_smoothed():
    task = PrototypeSegTask({'model': model()}, cfg(ema_enabled=True, ema_start_epoch=0,
                                                 ema_decay=.99, ema_decay_warmup=False))
    populate(task.model)
    task.update_ema()
    src = dict(task.model.prototype.named_buffers())
    dst = dict(task.ema.model.prototype.named_buffers())
    assert all(torch.equal(src[k], dst[k]) for k in src)


def test_invalid_configuration_and_eval_collection():
    with pytest.raises(ValueError):
        model(prototype_spatial=False, prototype_channel=False)
    m = model().eval()
    with pytest.raises(RuntimeError):
        m(torch.randn(2, 1, 32, 32), pet=torch.randn(2, 1, 32, 32),
          return_knowledge=True)


def test_resume_restores_optimizer_memory_and_rng(tmp_path):
    task = PrototypeSegTask({'model': model()}, cfg())
    task.loader_generators = {'train': torch.Generator().manual_seed(4)}
    populate(task.model)
    task.global_batch_step = 13
    task.current_epoch = 2
    task.no_improve = 1
    task.train_step_mixed(batch(), torch.tensor([1, 0, 1, 0]))[0].backward()
    task.optimizer.step()
    task.after_optimizer_step(True)
    path = str(tmp_path / 'last.pt')
    task.save_checkpoint(path, 2, best=.4, best_epoch=1)
    expected_rng = torch.rand(3)
    expected_loader_rng = torch.rand(3, generator=task.loader_generators['train'])
    fresh = PrototypeSegTask({'model': model()}, cfg())
    fresh.loader_generators = {'train': torch.Generator().manual_seed(1)}
    ckpt = fresh.resume(path)
    assert ckpt['epoch'] == 2 and fresh.global_batch_step == 13
    assert fresh.no_improve == 1
    assert torch.equal(expected_rng, torch.rand(3))
    assert torch.equal(expected_loader_rng, torch.rand(3, generator=fresh.loader_generators['train']))
    assert all(torch.equal(v, fresh.model.state_dict()[k])
               for k, v in task.model.state_dict().items())
    assert len(fresh.optimizer.state) == len(task.optimizer.state)


def test_gain_is_supervised_detached_and_rejects_harm():
    y = torch.tensor([[[[1., 0., 1., 0.]]]])
    p0 = torch.tensor([[[[.2, .8, .8, .2]]]])
    pf = torch.tensor([[[[.8, .2, .2, .8]]]])
    logits = torch.logit(pf).requires_grad_()
    w = gain_weights(logits, torch.logit(p0), y)
    assert not w.requires_grad
    assert torch.allclose(w, torch.tensor([[[[1.2, .6, 0., 0.]]]]), atol=1e-6)
    # Both predictions wrong: improvement alone is insufficient for admission.
    assert gain_weights(torch.logit(torch.full_like(y, .3)),
                        torch.logit(torch.full_like(y, .1)), torch.ones_like(y)).sum() == 0


def test_full_bypasses_nonzero_correction_and_mixed_order():
    m = model().eval()
    populate(m)
    b = batch()
    full_before = m(b['ct'], pet=b['pet'], forward_mode='full')['logits']
    for s in m.prototype.scales:
        nn.init.normal_(s.spatial_out.weight, std=.05)
        nn.init.normal_(s.channel_out.weight, std=.05)
    assert torch.equal(full_before, m(b['ct'], pet=b['pet'], forward_mode='full')['logits'])
    state = torch.tensor([0, 1, 0, 1])
    mixed = m(b['ct'], pet=b['pet'], pet_available=state)['logits']
    separate = m(b['ct'][state == 0], forward_mode='missing')['logits']
    assert torch.allclose(mixed[state == 0], separate, atol=1e-6)


def test_cpu_autocast_and_single_branch_ablations():
    for s, c in ((True, False), (False, True), (True, True)):
        m = model(prototype_spatial=s, prototype_channel=c).train()
        populate(m)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = m(torch.randn(2, 1, 32, 32), forward_mode='missing')['logits'].float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_memory_contents_affect_reads_and_only_full_rows_are_collected():
    m = model().eval()
    populate(m)
    for scale in m.prototype.scales:
        nn.init.normal_(scale.spatial_out.weight, std=.1)
        nn.init.normal_(scale.channel_out.weight, std=.1)
    b = batch()
    before = m(b['ct'], forward_mode='missing')['logits']
    for scale in m.prototype.scales:
        scale.spatial_bank.centers.normal_()
        scale.channel_bank.centers.normal_()
    after = m(b['ct'], forward_mode='missing')['logits']
    assert not torch.allclose(before, after)
    m.train()
    state = torch.tensor([0, 1, 0, 1])
    out = m(b['ct'], pet=b['pet'], pet_available=state, return_knowledge=True)
    knowledge = out['knowledge']
    assert torch.equal(knowledge['full_index'], torch.tensor([1, 3]))
    for level, channels in zip(knowledge['full_features'], (32, 24, 16, 8)):
        assert level.shape[:2] == (2, channels)
        assert not level.requires_grad
    assert all(not x.requires_grad for x in knowledge['reference_features'])


def test_protocol_defaults_and_imagenet_normalization(monkeypatch):
    import inspect
    import sys
    import numpy as np
    from configs.seg_mdt import SegMDTConfig
    from run_decoder_prototype_add import PrototypeConfig
    from datasets.pclt20k_seg import _normalize_ch, get_pclt20k_loaders_cipa_aligned
    monkeypatch.setattr(sys, 'argv', ['train'])
    for config in (SegMDTConfig.parse_arguments(), PrototypeConfig.parse_arguments()):
        assert config.norm_mode == 'imagenet'
        assert config.hd95_mode == 'cipa_official'
    assert PrototypeConfig.parse_arguments().prototype_outlier_filter is False
    assert PrototypeConfig.parse_arguments().log_interval == 100
    assert inspect.signature(get_pclt20k_loaders_cipa_aligned).parameters['norm_mode'].default == 'imagenet'
    gray = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    expected = (np.repeat(gray[None].astype(np.float32) / 255., 3, axis=0)
                - np.array([.485, .456, .406], dtype=np.float32)[:, None, None])
    expected /= np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    assert np.allclose(_normalize_ch(gray), expected)
    assert np.array_equal(_normalize_ch(gray), _normalize_ch(gray, 'imagenet'))
    with pytest.raises(ValueError):
        _normalize_ch(gray, 'unknown')


def test_both_aligners_precede_pure_add_and_pet_alignment_is_missing_isolated():
    from models.components.channel_align import StageChannelAlign
    m = model().train()
    assert isinstance(m.ct_align, StageChannelAlign)
    assert isinstance(m.pet_align, StageChannelAlign)
    for align in (m.ct_align, m.pet_align):
        assert [layer[0].out_channels for layer in align.proj] == [64, 128, 320, 512]
        assert all(isinstance(layer[1], nn.BatchNorm2d) and isinstance(layer[2], nn.ReLU)
                   for layer in align.proj)
    captured = {}
    def capture(_module, inputs, output):
        c, p = inputs
        captured['count'] = c[0].shape[0]
        assert all(torch.equal(a + b, f) for a, b, f in zip(c, p, output))
    handle = m.fusion.register_forward_hook(capture)
    b = batch()
    out = m(b['ct'], pet=b['pet'], pet_available=torch.tensor([1, 0, 1, 0]))
    out['logits'][[0, 2]].square().mean().backward()
    assert captured['count'] == 2
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pet_align.parameters())
    handle.remove()
    m.zero_grad(set_to_none=True)
    buffers = copy.deepcopy(m.pet_align.state_dict())
    m(b['ct'], forward_mode='missing')['logits'].square().mean().backward()
    assert all(p.grad is None for p in m.pet_align.parameters())
    assert all(torch.equal(v, m.pet_align.state_dict()[k]) for k, v in buffers.items())


def test_cipa_official_hd95_policy_does_not_modify_dice_masks():
    import numpy as np
    from medpy.metric.binary import hd95
    from utils.metrics_seg import hd95_cipa_official_pair, SegmentationMetricsCIPA
    a = np.zeros((512, 512), dtype=bool)
    b = a.copy()
    a[10, 10], b[13, 14] = True, True
    saved_a, saved_b = a.copy(), b.copy()
    assert hd95_cipa_official_pair(a, b) == hd95(a, b) == 5.
    assert np.array_equal(a, saved_a) and np.array_equal(b, saved_b)
    empty = np.zeros_like(a)
    assert hd95_cipa_official_pair(empty, empty) == 0.
    work = empty.copy(); work[256, 256] = True
    assert hd95_cipa_official_pair(empty, b) == hd95(work, b)
    assert not empty.any()
    with pytest.raises(ValueError, match='empty GT'):
        hd95_cipa_official_pair(a, empty)
    with pytest.raises(ValueError, match='512'):
        hd95_cipa_official_pair(a[:32, :32], b[:32, :32])
    metric = SegmentationMetricsCIPA(hd95_mode='cipa_official')
    metric.update(torch.full((1, 1, 512, 512), -100.), torch.tensor(b)[None, None])
    assert metric.compute()['dice'] == 0.
    assert metric.compute()['missed_positive_count'] == 1
    assert metric.tp == metric.fp == 0 and metric.fn == 1


def test_final_eval_keeps_task_official_mode():
    from utils.run_common import run_missing_rate_eval
    task = PrototypeSegTask({'model': model()}, cfg(hd95_mode='cipa_official'))
    assert task.metrics.hd95_mode == 'cipa_official'
    b = dict(ct=torch.randn(2, 1, 512, 512), pet=torch.randn(2, 1, 512, 512),
             mask=torch.ones(2, 1, 512, 512), case_id=['a', 'b'])
    result = run_missing_rate_eval(task, task.model, [b], {'a': 0, 'b': 1})
    assert result['hd95_mode'] == 'cipa_official'
    assert result['hd95_unit'] == 'pixel'
    assert result['slice_count'] == 2


def test_validation_progress_reports_completed_batches_without_changing_memory(capsys):
    task = PrototypeSegTask({'model': model()}, cfg(log_interval=1))
    populate(task.model)
    before = copy.deepcopy(task.model.prototype.state_dict())
    result = task.evaluate([batch(), batch()], eval_mode='missing', tag='validation_progress')
    output = capsys.readouterr().out
    assert '[validation_progress] start mode=missing batches=2' in output
    assert '[validation_progress] batch 1/2' in output
    assert '[validation_progress] batch 2/2' in output
    assert '[validation_progress] complete Dice=' in output
    assert 0 <= result['dice'] <= 1
    assert task.model.training  # Parent training-mode restoration is preserved.
    assert all(torch.equal(v, task.model.prototype.state_dict()[k]) for k, v in before.items())


def test_epoch_runner_final_selection_and_resume_match_uninterrupted(tmp_path, monkeypatch, capsys):
    import sys
    from torch.utils.data import DataLoader
    import run_decoder_prototype_add as runner

    def loaders(_cfg):
        b = batch()
        rows = [dict(ct=b['ct'][i], pet=b['pet'][i], mask=b['mask'][i], case_id=f'case{i}')
                for i in range(4)]
        return tuple(DataLoader(rows, batch_size=4, shuffle=(i == 0), drop_last=True,
                                generator=torch.Generator().manual_seed(10 + i)) for i in range(3))

    monkeypatch.setattr(runner, '_loaders', loaders)
    common = ['run', '--epochs', '2', '--batch_size', '4', '--num_workers', '0',
              '--hd95_mode', 'legacy',  # 32x32 synthetic data; formal default stays official.
              '--pretrained', 'false', '--mixed_precision', 'false', '--ema_enabled', 'true',
              '--ema_start_epoch', '1', '--decoder_channels', '32', '24', '16', '8',
              '--prototype_dim', '4', '--prototype_clusters', '2', '--prototype_capacity', '16',
              '--prototype_min_candidates', '2', '--prototype_refresh_steps', '1',
              '--prototype_warmup_steps', '0', '--cosine_warmup', '0',
              '--checkpoint_root', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'continuous', '--log_interval', '1'])
    runner.main()
    reference = PrototypeSegTask.load_state_dicts(str(tmp_path / 'MDT/continuous/ckpt.last.pth.tar'))

    class InterruptRun(Exception):
        pass

    original_save = PrototypeSegTask.save_checkpoint

    def interrupted_save(self, path, epoch, *args, **kwargs):
        original_save(self, path, epoch, *args, **kwargs)
        if path.endswith('ckpt.last.pth.tar') and epoch == 1:
            raise InterruptRun()

    monkeypatch.setattr(PrototypeSegTask, 'save_checkpoint', interrupted_save)
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'resumed'])
    with pytest.raises(InterruptRun):
        runner.main()
    monkeypatch.setattr(PrototypeSegTask, 'save_checkpoint', original_save)
    last = str(tmp_path / 'MDT/resumed/ckpt.last.pth.tar')
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'resumed', '--resume', last])
    runner.main()
    resumed = PrototypeSegTask.load_state_dicts(last)
    assert reference['global_batch_step'] == resumed['global_batch_step'] == 2
    for key, value in reference['model'].items():
        assert torch.equal(value, resumed['model'][key]), key
    for key, value in reference['model_ema'].items():
        assert torch.equal(value, resumed['model_ema'][key]), key
    assert (tmp_path / 'MDT/resumed/final_test_metrics.json').exists()
    output = capsys.readouterr().out
    assert output.count('[train epoch 1] batch 1/1') == 1
    assert 'first forward complete;' not in output
    for expected in ('[setup] train_batches=', 'waiting for first batch...',
                     '[prototype epoch 1 batch 1]',
                     '[checkpoint epoch 1] saving last...', '[final] evaluating best epoch'):
        assert expected in output

    import csv
    directory = tmp_path / 'MDT/continuous'
    for filename, per_epoch in (('prototype_banks.csv', 16), ('prototype_clusters.csv', 32),
                                ('prototype_gain.csv', 2)):
        with open(directory / filename, newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * per_epoch
        assert {row['event'] for row in rows} == {'interval', 'epoch_end'}
    assert reference['prototype_refresh_stats'] == resumed['prototype_refresh_stats']


def test_gain_diagnostics_counts_errors_and_selection_without_changing_weights():
    pf = torch.tensor([[[[.9, .4, .2, .6]]]])
    p0 = torch.tensor([[[[.5, .3, .4, .7]]]])
    y = torch.tensor([[[[1., 1., 0., 0.]]]])
    full, reference = torch.logit(pf), torch.logit(p0)
    stats = {}
    expected = gain_weights(full, reference, y)
    actual = gain_weights(full, reference, y, stats=stats)
    assert torch.equal(expected, actual)
    assert int(stats['full_images']) == int(stats['useful_full_images']) == 1
    assert int(stats['fg_selected']) == int(stats['bg_selected']) == 1
    assert int(stats['full_fn']) == int(stats['reference_fn']) == 1
    assert int(stats['full_fp']) == int(stats['reference_fp']) == 1
    assert float(stats['fg_gain_sum']) == pytest.approx(.24)
    assert float(stats['bg_gain_sum']) == pytest.approx(.12)
    for label in ('fg', 'bg'):
        assert stats[f'{label}_pixels'] == sum(stats[f'{label}_{k}'] for k in
                                              ('selected', 'full_wrong', 'below_gain'))
    assert all(not value.requires_grad for value in stats.values() if torch.is_tensor(value))


def test_bank_diagnostics_known_distribution_and_ema_movement_are_read_only():
    from utils.prototype_diagnostics import bank_snapshot
    bank = EMACentroidBank(2, 1, capacity=8, min_candidates=4, decay=.5)
    assert not bank.refresh()
    row, clusters = bank_snapshot(bank, True)
    assert row['status'] == 'collecting' and row['refresh_insufficient_pool'] == 1
    assert 'pool_count' not in clusters[0]  # Undefined before initialization.
    bank.add(torch.tensor([[0.], [0.], [10.], [10.]]))
    assert bank.refresh()
    before = copy.deepcopy(bank.state_dict())
    metadata = copy.deepcopy(bank.refresh_stats)
    rng = torch.get_rng_state().clone()
    row, clusters = bank_snapshot(bank, True)
    assert row['occupied_clusters'] == 2
    assert row['assignment_entropy_normalized'] == pytest.approx(1.)
    assert row['max_cluster_fraction'] == .5
    assert row['min_center_distance'] == 10.
    assert row['mean_nearest_distance'] == 0.
    assert [c['pool_count'] for c in clusters] == [2, 2]
    assert row['last_raw_center_drift'] is None  # Initialization is not an EMA update.
    assert torch.equal(rng, torch.get_rng_state())
    assert metadata == bank.refresh_stats
    assert all(torch.equal(v, bank.state_dict()[k]) for k, v in before.items())
    bank.pool[:4].add_(2.)
    assert bank.refresh()
    assert bank.refresh_stats['last_raw_drift'] == pytest.approx(2.)
    assert bank.refresh_stats['last_applied_drift'] == pytest.approx(1.)
    # FIFO ownership against EMA centers is distinct from last K-means ownership.
    bank.pool[:4].fill_(100.)
    row, clusters = bank_snapshot(bank, True)
    assert row['occupied_clusters'] == 1
    assert row['assignment_entropy_normalized'] == 0.
    assert row['max_cluster_fraction'] == 1.
    assert sum(c['last_success_count'] for c in clusters) == 4
    duplicate = EMACentroidBank(2, 1, capacity=4, min_candidates=4)
    duplicate.add(torch.ones(4, 1))
    assert not duplicate.refresh()
    assert duplicate.refresh_stats['last_status'] == 'insufficient_unique'


def test_candidate_composition_and_success_warmup_skip_accounting(tmp_path):
    import csv
    from utils.prototype_diagnostics import PrototypeCSVLogger
    task = PrototypeSegTask({'model': model()}, cfg())
    task.model.prototype.warmup_steps = 1
    task.begin_epoch(1)
    b, state = batch(), torch.tensor([1, 0, 1, 0])
    for succeeded in (True, False, True):
        task.train_step_mixed(b, state)
        # Supply a deterministic candidate count to test lifecycle accounting.
        task._pending_candidates = [(torch.ones(3, s.spatial_bank.dimension),
                                     torch.ones(2, s.channel_bank.dimension))
                                    for s in task.model.prototype.scales]
        task.after_optimizer_step(succeeded)
    stats = task.prototype_epoch_stats
    assert stats['attempted_batches'] == 3 and stats['successful_batches'] == 2
    assert stats['skipped_batches'] == 1 and stats['warmup_batches'] == 1
    assert stats['d4_spatial_written'] == 3 and stats['d4_channel_written'] == 2
    assert int(task.model.prototype.scales[0].spatial_bank.accepted) == 3
    logger = PrototypeCSVLogger(str(tmp_path))
    logger.snapshot(task, 1, 3, 'epoch_end')
    for filename in logger.paths:
        with open(logger.paths[filename], newline='') as stream:
            assert list(csv.DictReader(stream))
    # Preserve earlier completed epochs and remove uncheckpointed future rows.
    logger.snapshot(task, 2, 1, 'interval')
    PrototypeCSVLogger(str(tmp_path), start_epoch=2)
    with open(tmp_path / 'prototype_gain.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['epoch'] == '1'
    assert rows[0]['fg_selected_fraction'] != ''
    # Old checkpoints have no logging metadata and remain strictly loadable.
    path = str(tmp_path / 'legacy.pt')
    task.save_checkpoint(path, 1, best=0., best_epoch=1)
    legacy = task.load_state_dicts(path)
    legacy.pop('prototype_refresh_stats')
    torch.save(legacy, path)
    fresh = PrototypeSegTask({'model': model()}, cfg())
    fresh.resume(path)
    assert all(torch.equal(v, fresh.model.state_dict()[k]) for k, v in task.model.state_dict().items())
    # Foreground-patch counts describe patches with any positive FG gain mass.
    scale = task.model.prototype.scales[0]
    full = torch.randn(1, scale.channels, 10, 10)
    fg = torch.zeros(1, 1, 10, 10); fg[:, :, :5] = 1
    counts = {}
    spatial, channel = scale.candidates(full, full + .5, torch.ones_like(fg), fg, stats=counts)
    assert counts == dict(spatial_generated=4, channel_generated=1,
                          spatial_fg_patches=2, spatial_bg_patches=2)
    assert spatial.shape[0] == 4 and channel.shape[0] == 1


def test_default_validation_prints_only_every_100_batches(monkeypatch, capsys):
    from tasks.mdt_seg import MDTSegTeacher
    task = PrototypeSegTask({'model': model()}, cfg())

    def consume(self, loader, **kwargs):
        assert len(list(loader)) == 205
        return dict(dice=.5, hd95=1.)

    monkeypatch.setattr(MDTSegTeacher, 'evaluate', consume)
    task.evaluate([{}] * 205, tag='hundred')
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith('[hundred] batch ')]
    assert len(lines) == 2
    assert 'batch 100/205' in lines[0] and 'batch 200/205' in lines[1]


def test_sampled_attention_is_detached_and_does_not_change_outputs_or_gradients():
    scale = DecoderPrototypeScale(8, dim=4, clusters=2, capacity=16, min_candidates=2)
    for bank in (scale.spatial_bank, scale.channel_bank):
        bank.add(torch.randn(8, bank.dimension))
        bank.refresh()
    with torch.no_grad():
        scale.spatial_out.weight.normal_(std=.1)
        scale.channel_out.weight.normal_(std=.1)
    twin = copy.deepcopy(scale)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    stats = {}
    plain, sampled = scale(x), twin(xx, diagnostics=stats)
    assert torch.equal(plain, sampled)
    plain.sum().backward(); sampled.sum().backward()
    assert torch.equal(x.grad, xx.grad)
    for p, pp in zip(scale.parameters(), twin.parameters()):
        assert (p.grad is None and pp.grad is None) or torch.equal(p.grad, pp.grad)
    for label, queries in (('spatial', 128), ('channel', 2)):
        read = stats[label]
        assert read['available'] and read['query_count'] == queries
        assert 0 <= read['attention_entropy'] <= 1
        assert read['attention_mean'].sum() == pytest.approx(1.)
        assert read['top1_fraction'].sum() == pytest.approx(1.)
        assert read['correction_rms_ratio'] > 0
        assert all(not value.requires_grad for value in read.values() if torch.is_tensor(value))


def test_channel_correction_diagnostic_matches_broadcast_residual_rms():
    correction = torch.randn(2, 8, 5, 7)
    shift = torch.randn(2, 8, 1, 1)
    x = torch.randn_like(correction)
    attention = torch.randn(2, 4).softmax(-1)
    stats = DecoderPrototypeScale._read_diagnostics(attention, correction, x, shift)
    expected = (correction + shift).square().mean().sqrt() / x.square().mean().sqrt()
    assert stats['correction_rms_ratio'] == pytest.approx(float(expected), rel=1e-6)
