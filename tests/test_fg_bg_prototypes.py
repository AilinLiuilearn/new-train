"""Contracts for scale-supervised foreground/background decoder memories."""
import copy
import csv
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.decoder_fg_bg_prototype import ForegroundBackgroundScale, ForegroundBackgroundMemory
from models.fg_bg_prototype_add_model import ForegroundBackgroundAddModel
from tasks.fg_bg_prototype_seg import ForegroundBackgroundSegTask


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
def encoders(monkeypatch):
    import models.dual_shared_add_baseline as base
    monkeypatch.setattr(base, 'create_feature_backbone', lambda name, **kw:
                        FakeEncoder((8, 16, 24, 32) if 'convnext' in name
                                    else (64, 128, 320, 512)))
    torch.set_num_threads(1)


def model(**kw):
    return ForegroundBackgroundAddModel(pretrained=False, decoder_channels=(32, 24, 16, 8),
        prototype_dim=4, prototype_clusters=2, prototype_capacity=16,
        prototype_min_candidates=2, prototype_refresh_steps=1, prototype_warmup_steps=0, **kw)


def config(**kw):
    args = dict(learning_rate=1e-3, weight_decay=0., mixed_precision=False,
                ema_enabled=False, random_state=2023, prototype_ct_aux_weight=.1,
                prototype_deep_supervision_weight=.2)
    args.update(kw)
    return SimpleNamespace(**args)


def batch():
    return dict(ct=torch.randn(4, 1, 32, 32), pet=torch.randn(4, 1, 32, 32),
                mask=(torch.rand(4, 1, 32, 32) > .7).float())


def populate(m):
    for scale in m.prototype.scales:
        for bank in scale.banks:
            bank.add(torch.randn(8, bank.dimension)); assert bank.refresh()


def test_zero_head_exact_baseline_and_auxiliary_heads_shared():
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    m = model().eval(); populate(m)
    base = DualSharedAddPETCTBaseline(pretrained=False, decoder_channels=(32, 24, 16, 8),
                                     use_deep_supervision=True).eval()
    base.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith('prototype.')}, strict=True)
    b = batch()
    for mode in ('full', 'missing', 'auto'):
        kw = dict(pet=b['pet'], forward_mode=mode, pet_available=torch.tensor([1, 0, 1, 0]))
        assert torch.equal(m(b['ct'], **kw)['logits'], base(b['ct'], **kw)['logits'])
    assert not any('reference_head' in k or 'channel_bank' in k for k in m.state_dict())


def test_missing_pet_isolation_full_bypass_and_eval_does_not_call_aux_heads():
    m = model().eval(); populate(m)
    b = batch(); state = torch.tensor([1, 0, 1, 0])
    full = m(b['ct'], pet=b['pet'], forward_mode='full')['logits']
    for scale in m.prototype.scales:
        nn.init.normal_(scale.out.weight, std=.05)
    assert torch.equal(full, m(b['ct'], pet=b['pet'], forward_mode='full')['logits'])
    before = copy.deepcopy(m.prototype.state_dict())
    handles = [getattr(m.decoder, f'aux_head_d{i}').register_forward_hook(
        lambda *args: pytest.fail('inference computed auxiliary predictions')) for i in (4, 3, 2)]
    a = m(b['ct'], pet=b['pet'], pet_available=state)['logits']
    pet = b['pet'].clone(); pet[state == 0] = float('nan')
    assert torch.equal(a, m(b['ct'], pet=pet, pet_available=state)['logits'])
    calls = m.enc_pet.calls
    m(b['ct'], pet=None, forward_mode='missing')['logits'].square().mean().backward()
    assert m.enc_pet.calls == calls and all(p.grad is None for p in m.enc_pet.parameters())
    assert all(torch.equal(v, m.prototype.state_dict()[k]) for k, v in before.items())
    for handle in handles: handle.remove()
    with pytest.raises(RuntimeError):
        m(b['ct'], return_knowledge=True)


def test_auxiliary_loss_formula_and_gradients_into_all_heads_and_ct_reference():
    task = ForegroundBackgroundSegTask({'model': model()}, config())
    b = batch(); state = torch.tensor([1, 0, 1, 0])
    loss, _, _, stats = task.train_step_mixed(b, state)
    expected = (.5 * stats['loss_full'] + .5 * stats['loss_missing'] + .1 * stats['loss_ct_aux']
                + .2 * (.5 * stats['loss_full_deep'] + .5 * stats['loss_missing_deep']
                          + .1 * stats['loss_ct_deep']))
    assert torch.allclose(loss.detach(), expected)
    loss.backward()
    for i in (4, 3, 2):
        head = getattr(task.model.decoder, f'aux_head_d{i}')
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert task.model.enc_ct.layers[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in task.model.parameters())


def test_per_scale_gain_rejects_harm_even_when_final_prediction_is_helpful():
    m = model().train()
    y = torch.ones(2, 1, 32, 32)
    logits = lambda p: torch.logit(torch.full_like(y, p))
    features = [torch.randn(2, s.channels, 10, 10) for s in m.prototype.scales]
    knowledge = dict(full_index=torch.tensor([0, 1]), full_features=features,
        reference_features=[f + .2 for f in features],
        full_logits_by_scale=[logits(.3), logits(.8), logits(.8), logits(.8)],
        reference_logits_by_scale=[logits(.6), logits(.4), logits(.4), logits(.4)])
    values, stats = m.prepare_candidates(knowledge, y, return_diagnostics=True)
    assert values[0][0].shape[0] == 0 and values[0][1].shape[0] == 0
    assert all(v[0].shape[0] > 0 for v in values[1:])
    assert stats['d4_fg_selected'] == 0 and stats['d1_fg_selected'] == y.numel()
    m.gain_mode = 'final'
    values, _ = m.prepare_candidates(knowledge, y, return_diagnostics=True)
    assert values[0][0].shape[0] > 0
    assert all(not x.requires_grad for value in values for x in value)


def test_class_gain_multiply_before_downsampling_and_separate_descriptors():
    s = ForegroundBackgroundScale(4, dim=2, clusters=2, capacity=8, min_candidates=2,
                                  patch_size=1, max_patches=4)
    full, ref = torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)
    y = torch.zeros(1, 1, 4, 4); y[:, :, 0, 0] = 1
    w = torch.zeros_like(y); w[:, :, 1, 1] = 1
    fg, bg, counts = s.candidates(full, ref, w, y)
    assert fg.shape[0] == 0 and bg.shape[0] == 1
    assert counts.tolist() == [0, 1]
    w[:, :, 0, 0] = 1
    fg, bg, counts = s.candidates(full, ref, w, y)
    assert fg.shape[0] == bg.shape[0] == 1 and counts.tolist() == [1, 1]


def test_adaptive_budget_favors_rare_class_without_duplicates():
    s = ForegroundBackgroundScale(4, dim=2, clusters=2, capacity=16, min_candidates=2,
                                  patch_size=1, max_patches=8)
    full, ref = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
    y = torch.zeros(1, 1, 4, 4); y[:, :, :2] = 1
    s.sampling_mass.copy_(torch.tensor([1., 100.])); s.sampling_updates.fill_(1)
    fg, bg, _ = s.candidates(full, ref, torch.ones_like(y), y)
    assert fg.shape[0] == 7 and bg.shape[0] == 1
    s.adaptive_sampling = False
    fg, bg, _ = s.candidates(full, ref, torch.ones_like(y), y)
    assert fg.shape[0] == bg.shape[0] == 4
    y.zero_(); y[:, :, 0, 0] = 1
    s.adaptive_sampling = True
    fg, bg, _ = s.candidates(full, ref, torch.ones_like(y), y)
    assert fg.shape[0] == 1 and bg.shape[0] == 7


def test_commit_warmup_skips_and_sampling_statistics_are_checkpointed():
    m = ForegroundBackgroundMemory([8] * 4, dim=4, clusters=2, capacity=16,
        min_candidates=2, warmup_steps=1, refresh_steps=1).train()
    values = [(torch.randn(4, 12), torch.randn(4, 12), torch.tensor([2., 100.])) for _ in range(4)]
    m.commit(values)
    assert int(m.update_steps) == 1 and int(m.scales[0].foreground_bank.size) == 0
    assert int(m.scales[0].sampling_updates) == 1
    m.commit(values)
    assert all(bool(bank.valid) for scale in m.scales for bank in scale.banks)
    fresh = copy.deepcopy(m); fresh.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(fresh.scales[0].sampling_mass, m.scales[0].sampling_mass)
    m.eval()
    with pytest.raises(RuntimeError): m.commit(values)


@pytest.mark.parametrize('ready', ['foreground', 'background', 'both'])
def test_partial_banks_joint_attention_bound_zero_grad_and_diagnostics(ready):
    s = ForegroundBackgroundScale(8, dim=4, clusters=2, capacity=16, min_candidates=2)
    for label, bank in zip(('foreground', 'background'), s.banks):
        if ready in (label, 'both'):
            bank.add(torch.randn(8, bank.dimension)); assert bank.refresh()
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    assert torch.equal(s(x), x)
    s(x).square().mean().backward()
    assert s.out.weight.grad.abs().sum() > 0
    with torch.no_grad(): s.out.weight.normal_(std=10.)
    twin = copy.deepcopy(s); xx = x.detach().clone().requires_grad_()
    stats = {}
    a, b = s(x), twin(xx, stats)
    assert torch.equal(a, b)
    a.square().sum().backward(); b.square().sum().backward()
    assert torch.isfinite(xx.grad).all()
    ratios = (b - xx).flatten(1).norm(dim=1) / xx.flatten(1).norm(dim=1)
    assert (ratios <= .1 + 1e-6).all()
    mass = sum(float(stats[label]['class_mass']) for label in ('foreground', 'background')
               if stats[label]['available'])
    assert mass == pytest.approx(1.)
    assert float(stats['joint']['logit_min']) >= -2.000001
    assert float(stats['joint']['logit_max']) <= 2.000001
    assert torch.equal(s(torch.zeros_like(x)), torch.zeros_like(x))


def test_cpu_autocast_and_disabled_memory_baseline():
    m = model().train(); populate(m)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = m(torch.randn(2, 1, 32, 32), forward_mode='missing')['logits'].float().square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.prototype_enabled = False
    b = batch()
    assert not m(b['ct'], pet=b['pet'], return_knowledge=True)['knowledge']['full_features'][0].requires_grad


def test_resume_model_optimizer_sampling_ema_rng_and_policy_rejection(tmp_path):
    task = ForegroundBackgroundSegTask({'model': model()}, config(ema_enabled=True, ema_start_epoch=0))
    populate(task.model)
    b = batch(); state = torch.tensor([1, 0, 1, 0])
    task.train_step_mixed(b, state)[0].backward(); task.optimizer.step(); task.after_optimizer_step(True)
    task.update_ema()
    path = str(tmp_path / 'last.pt'); task.save_checkpoint(path, 2, best=.4, best_epoch=1)
    expected = torch.rand(3)
    fresh = ForegroundBackgroundSegTask({'model': model()}, config(ema_enabled=True, ema_start_epoch=0))
    fresh.resume(path)
    assert torch.equal(expected, torch.rand(3))
    assert all(torch.equal(v, fresh.model.state_dict()[k]) for k, v in task.model.state_dict().items())
    assert all(torch.equal(v, fresh.ema.model.state_dict()[k]) for k, v in task.ema.model.state_dict().items())
    assert all(torch.equal(v, dict(fresh.ema.model.prototype.named_buffers())[k])
               for k, v in task.model.prototype.named_buffers())
    changed = ForegroundBackgroundSegTask({'model': model(prototype_gain_mode='final')}, config())
    with pytest.raises(ValueError, match='configuration mismatch'): changed.resume(path)


def test_invalid_scale_supervision_and_csv_per_scale(tmp_path):
    from utils.fg_bg_prototype_diagnostics import ForegroundBackgroundCSVLogger
    with pytest.raises(ValueError): model(use_deep_supervision=False)
    with pytest.raises(ValueError):
        ForegroundBackgroundSegTask({'model': model()}, config(prototype_deep_supervision_weight=0.))
    task = ForegroundBackgroundSegTask({'model': model()}, config())
    populate(task.model); task.capture_prototype_reads = True
    task.train_step_mixed(batch(), torch.tensor([1, 0, 1, 0])); task.after_optimizer_step(False)
    logger = ForegroundBackgroundCSVLogger(str(tmp_path)); logger.snapshot(task, 1, 1, 'epoch_end')
    for filename, count in [('prototype_gain.csv', 4), ('prototype_banks.csv', 8),
                            ('prototype_clusters.csv', 16), ('scale_train.csv', 4)]:
        with open(tmp_path / filename, newline='') as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == count
    with open(tmp_path / 'prototype_gain.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {r['scale'] for r in rows} == {'d4', 'd3', 'd2', 'd1'}
    assert all(r['fg_selected_weight_fraction'] != '' for r in rows)
    ForegroundBackgroundCSVLogger(str(tmp_path), start_epoch=2)
    with open(tmp_path / 'prototype_gain.csv', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_reference_auxiliary_gradient_does_not_update_pet_and_heads_are_shared():
    m = model().train(); b = batch()
    calls = {i: 0 for i in (4, 3, 2)}
    handles = []
    for i in calls:
        def count(_module, _inputs, _outputs, level=i): calls[level] += 1
        handles.append(getattr(m.decoder, f'aux_head_d{i}').register_forward_hook(count))
    out = m(b['ct'], pet=b['pet'], pet_available=torch.tensor([1, 0, 1, 0]),
            return_knowledge=True, return_scale_logits=True)
    assert all(n == 2 for n in calls.values())  # Main mixed decode and CT reference.
    out['knowledge']['reference_logits_by_scale'][0].square().mean().backward()
    assert m.enc_ct.layers[-1].weight.grad.abs().sum() > 0
    assert m.decoder.aux_head_d4.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.enc_pet.parameters())
    for handle in handles: handle.remove()


@pytest.mark.parametrize('gain_mode,deep,weight', [('per_scale', True, .2), ('final', True, .2), ('final', False, 0.)])
def test_ablation_modes_and_disabled_memory_remain_trainable(gain_mode, deep, weight):
    task = ForegroundBackgroundSegTask({'model': model(prototype_gain_mode=gain_mode,
        use_deep_supervision=deep, prototype_enabled=False)}, config(prototype_deep_supervision_weight=weight))
    loss, _, _, stats = task.train_step_mixed(batch(), torch.tensor([1, 0, 1, 0]))
    loss.backward(); task.optimizer.step(); task.after_optimizer_step(True)
    assert torch.isfinite(loss) and task._pending_candidates is None
    assert int(task.model.prototype.update_steps) == 0
    assert not any(p.grad is not None for p in task.model.prototype.parameters())
    assert (stats['loss_deep_supervision'] > 0) == deep


def test_sampling_and_memory_are_unchanged_after_failed_optimizer_step():
    task = ForegroundBackgroundSegTask({'model': model()}, config())
    before = copy.deepcopy(task.model.prototype.state_dict())
    task.train_step_mixed(batch(), torch.tensor([1, 0, 1, 0]))
    assert all(torch.equal(v, task.model.prototype.state_dict()[k]) for k, v in before.items())
    task.after_optimizer_step(False)
    assert all(torch.equal(v, task.model.prototype.state_dict()[k]) for k, v in before.items())
    assert task._pending_candidates is None


def test_sampled_diagnostics_preserve_gradients_and_rng():
    s = ForegroundBackgroundScale(8, dim=4, clusters=2, capacity=16, min_candidates=2)
    for bank in s.banks:
        bank.add(torch.randn(8, bank.dimension)); bank.refresh()
    with torch.no_grad(): s.out.weight.normal_(std=.1)
    twin = copy.deepcopy(s)
    x = torch.randn(2, 8, 5, 7, requires_grad=True); xx = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state().clone(); stats = {}
    a, b = s(x), twin(xx, stats)
    assert torch.equal(a, b) and torch.equal(rng, torch.get_rng_state())
    a.square().mean().backward(); b.square().mean().backward()
    assert torch.equal(x.grad, xx.grad)
    for p, pp in zip(s.parameters(), twin.parameters()):
        assert (p.grad is None and pp.grad is None) or torch.equal(p.grad, pp.grad)
    assert all(not v.requires_grad for row in stats.values() for v in row.values() if torch.is_tensor(v))


def test_runner_resume_matches_continuous_and_evaluation_loads_new_checkpoint(tmp_path, monkeypatch):
    import sys
    from torch.utils.data import DataLoader
    import run_decoder_fg_bg_prototype_add as runner
    import eval_decoder_fg_bg_prototype_add as evaluator

    def loaders(_cfg):
        b = batch()
        rows = [dict(ct=b['ct'][i], pet=b['pet'][i], mask=b['mask'][i], case_id=f'case{i}') for i in range(4)]
        return tuple(DataLoader(rows, batch_size=4, shuffle=i == 0, drop_last=True,
            generator=torch.Generator().manual_seed(10 + i)) for i in range(3))

    monkeypatch.setattr(runner, '_loaders', loaders)
    common = ['run', '--epochs', '2', '--batch_size', '4', '--num_workers', '0',
        '--hd95_mode', 'legacy', '--pretrained', 'false', '--mixed_precision', 'false',
        '--ema_enabled', 'true', '--ema_start_epoch', '1', '--decoder_channels', '32', '24', '16', '8',
        '--prototype_dim', '4', '--prototype_clusters', '2', '--prototype_capacity', '16',
        '--prototype_min_candidates', '2', '--prototype_refresh_steps', '1',
        '--prototype_warmup_steps', '0', '--cosine_warmup', '0', '--checkpoint_root', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'continuous', '--log_interval', '1'])
    runner.main()
    path = str(tmp_path / 'MDT/continuous/ckpt.last.pth.tar')
    reference = ForegroundBackgroundSegTask.load_state_dicts(path)

    class InterruptRun(Exception): pass
    original = ForegroundBackgroundSegTask.save_checkpoint
    def interrupt(self, path, epoch, *args, **kw):
        original(self, path, epoch, *args, **kw)
        if path.endswith('ckpt.last.pth.tar') and epoch == 1: raise InterruptRun()
    monkeypatch.setattr(ForegroundBackgroundSegTask, 'save_checkpoint', interrupt)
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'resumed'])
    with pytest.raises(InterruptRun): runner.main()
    monkeypatch.setattr(ForegroundBackgroundSegTask, 'save_checkpoint', original)
    path = str(tmp_path / 'MDT/resumed/ckpt.last.pth.tar')
    monkeypatch.setattr(sys, 'argv', common + ['--hash', 'resumed', '--resume', path])
    runner.main()
    restored = ForegroundBackgroundSegTask.load_state_dicts(path)
    for tag in ('model', 'model_ema'):
        assert all(torch.equal(v, restored[tag][k]) for k, v in reference[tag].items())
    assert reference['prototype_refresh_stats'] == restored['prototype_refresh_stats']
    for filename, count in [('prototype_gain.csv', 16), ('prototype_banks.csv', 32), ('scale_train.csv', 16)]:
        with open(tmp_path / 'MDT/continuous' / filename, newline='') as stream:
            assert len(list(csv.DictReader(stream))) == count
    with open(tmp_path / 'MDT/continuous/train_log.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert all(r['val_full_hd95'] and r['val_missing_hd95'] and r['train_deep_supervision_loss'] for r in rows)
    monkeypatch.setattr(evaluator, '_loaders', loaders)
    monkeypatch.setattr(sys, 'argv', ['eval', '--checkpoint', path, '--output_dir', str(tmp_path / 'evaluation')])
    evaluator.main()
    assert (tmp_path / 'evaluation/final_test_metrics.json').exists()


def test_all_main_and_auxiliary_losses_use_float32_under_autocast():
    task = ForegroundBackgroundSegTask({'model': model()}, config())
    original = task.criterion
    dtypes = []
    def criterion(logits, target):
        dtypes.append(logits.dtype)
        return original(logits, target)
    task.criterion = criterion
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = task.train_step_mixed(batch(), torch.tensor([1, 0, 1, 0]))[0]
    assert len(dtypes) == 12 and all(dtype == torch.float32 for dtype in dtypes)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in task.model.parameters())
