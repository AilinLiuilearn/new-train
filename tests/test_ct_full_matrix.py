"""Contracts for independent pre-decoder CT-key / transform-value memories."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.ct_full_matrix import MatrixBank, MatrixScale, CTFullMatrixMemory


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def bank(**kw):
    return MatrixBank(4, clusters=2, capacity=32, min_candidates=8,
                      min_cluster_samples=2, ridge=.01, decay=.5, **kw)


def pairs(n=24, c=4):
    g = torch.Generator().manual_seed(9)
    x = torch.randn(n, c, generator=g)
    x[:n//2, 0] += 3; x[n//2:, 0] -= 3
    t = torch.eye(c); t[0, 0] = 1.4; t[1, 2] = .2
    return x, x @ t.T, torch.ones(n)


def test_ridge_solution_matches_primal_and_identity():
    x, y, w = pairs()
    t, lam, error = MatrixBank.fit_transform(x, y, w, .01)
    a = (x.T * w) @ x / w.sum()
    b = (y.T * w) @ x / w.sum()
    expected = torch.linalg.solve(a + lam * torch.eye(4),
                                 (b + lam * torch.eye(4)).T).T
    assert torch.allclose(t, expected, atol=1e-5)
    assert error < ((x-y).square().mean())
    identity = MatrixBank.fit_transform(x, x, w, .01)[0]
    assert torch.allclose(identity, torch.eye(4), atol=2e-5)


def test_fifo_preserves_raw_pairs_and_refresh_matches_ema_slots():
    b = bank(); x, y, w = pairs()
    b.add(x, y, w); assert b.refresh()
    assert bool(b.valid) and torch.allclose(b.keys.norm(dim=1), torch.ones(2))
    assert torch.equal(b.pool_full[:24], y) and not b.transforms.requires_grad
    before = b.transforms.clone()
    twin = copy.deepcopy(b); twin.decay = 0.
    more = y * 1.2
    b.pool_full[:24].copy_(more); twin.pool_full[:24].copy_(more)
    assert b.refresh() and twin.refresh()
    assert torch.allclose(b.transforms, .5 * before + .5 * twin.transforms, atol=1e-5)
    b.pool_full[:24].copy_(y)
    b.add(x, y, w); assert int(b.size) == 32
    assert torch.allclose(b.pool_full[:, 0], b.pool_ct[:, 0] * 1.4)


def test_reject_degenerate_clusters_without_destroying_old_state():
    b = bank(); x, y, w = pairs(); b.add(x, y, w); b.refresh()
    old = b.transforms.clone()
    b.pool_ct[:24].fill_(1)
    assert not b.refresh()
    assert torch.equal(old, b.transforms)


def test_gt_area_candidate_pairs_no_empty_foreground_or_duplicates():
    s = MatrixScale(4, clusters=2, capacity=32, min_candidates=8,
                    min_cluster_samples=2, samples_per_class=8)
    x = torch.randn(2, 4, 2, 2); y = x * 2
    gt = torch.zeros(2, 1, 8, 8); gt[0, 0, 0, 0] = 1
    fg, bg = s.candidates(x, y, gt)
    assert len(fg[0]) == 1 and float(fg[2][0]) == pytest.approx(1/16)
    assert len(bg[0]) == 7
    assert torch.equal(fg[1], fg[0] * 2) and torch.equal(bg[1], bg[0] * 2)
    assert not fg[0].requires_grad


def test_memory_transaction_warmup_independent_scales_and_eval_readonly():
    m = CTFullMatrixMemory([4]*4, warmup_steps=1, refresh_steps=1,
        clusters=2, capacity=32, min_candidates=8, min_cluster_samples=2).train()
    x, y, w = pairs()
    pending = [((x, y*(i+1), w), (x, y, w)) for i in range(4)]
    m.commit(pending)
    assert all(int(s.foreground_bank.size) == 24 for s in m.scales)
    assert not any(bool(s.foreground_bank.valid) for s in m.scales)
    m.commit(pending)
    assert all(bool(s.foreground_bank.valid) for s in m.scales)
    assert not torch.equal(m.scales[0].foreground_bank.transforms,
                           m.scales[1].foreground_bank.transforms)
    m.eval(); before = copy.deepcopy(m.state_dict())
    m([torch.randn(2, 4, 4, 4) for _ in range(4)])
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    with pytest.raises(RuntimeError): m.commit(pending)


def test_transform_application_and_gate_backward_and_ablation():
    s = MatrixScale(4, clusters=2, capacity=32, min_candidates=8, min_cluster_samples=2)
    for b in s.banks:
        b.valid.fill_(True); b.keys.copy_(F.normalize(torch.randn(2, 4), dim=1))
        b.transforms.copy_(torch.eye(4)[None].repeat(2, 1, 1) * 2)
    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    out = s(x)
    assert torch.allclose(out, 1.1*x, atol=1e-6)
    out.square().mean().backward()
    assert s.gate.weight.grad.abs().sum() > 0 and x.grad.abs().sum() > 0
    assert not any(b.transforms.grad is not None for b in s.banks)
    assert torch.equal(s(x, read_mode='identity'), x)
    assert torch.equal(s(x, read_mode='off'), x)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = s(x)
    assert torch.isfinite(result).all()


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in (8, 16, 24, 32))
        self.feature_info = SimpleNamespace(channels=lambda: [8, 16, 24, 32])
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [layer(F.adaptive_avg_pool2d(x, (16//2**i,)*2)) for i, layer in enumerate(self.layers)]


@pytest.fixture
def fake_backbones(monkeypatch):
    import models.dual_shared_add_baseline as base
    monkeypatch.setattr(base, 'create_feature_backbone', lambda *a, **kw: FakeEncoder())
    # Preserve real aligned channel dimensions: the matrices must be 64/128/320/512.


def make_model(**kw):
    from models.ct_full_matrix_add_model import CTFullMatrixAddModel
    return CTFullMatrixAddModel(pretrained=False, decoder_channels=(32, 24, 16, 8),
        matrix_clusters=2, matrix_capacity=16, matrix_min_candidates=4,
        matrix_min_cluster_samples=2, matrix_warmup_steps=0, matrix_refresh_steps=1, **kw)


def test_model_baseline_full_bypass_and_missing_pet_isolation(fake_backbones):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    m = make_model().eval()
    base = DualSharedAddPETCTBaseline(pretrained=False, decoder_channels=(32,24,16,8)).eval()
    base.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('matrix.')})
    ct, pet = torch.randn(4,1,32,32), torch.randn(4,1,32,32)
    state = torch.tensor([1,0,1,0])
    assert torch.equal(m(ct, pet, state)['logits'], base(ct, pet, state)['logits'])
    full = m(ct, pet, forward_mode='full')['logits']
    for s in m.matrix.scales:
        for b in s.banks:
            b.valid.fill_(True); b.keys.copy_(F.normalize(torch.randn_like(b.keys), dim=1))
            b.transforms.mul_(1.2)
    assert torch.equal(full, m(ct, pet, forward_mode='full')['logits'])
    pred = m(ct, pet, state)['logits']
    pet[state==0] = float('nan')
    assert torch.equal(pred, m(ct, pet, state)['logits'])
    calls = m.enc_pet.calls
    m(ct, None, forward_mode='missing')['logits'].square().mean().backward()
    assert m.enc_pet.calls == calls
    assert all(p.grad is None for p in m.enc_pet.parameters())
    assert not any('aux_head' in k for k in m.state_dict())


def test_task_only_final_loss_failed_step_and_checkpoint(fake_backbones, tmp_path):
    from tasks.ct_full_matrix_seg import CTFullMatrixSegTask
    cfg = SimpleNamespace(learning_rate=1e-3, weight_decay=0., mixed_precision=False,
                          ema_enabled=False, random_state=2023)
    task = CTFullMatrixSegTask({'model':make_model()}, cfg)
    b = dict(ct=torch.randn(4,1,32,32), pet=torch.randn(4,1,32,32),
             mask=(torch.rand(4,1,32,32)>.7).float())
    before = copy.deepcopy(task.model.matrix.state_dict())
    loss, _, _, stats = task.train_step_mixed(b, torch.tensor([1,0,1,0]))
    assert torch.allclose(loss.detach(), .5*(stats['loss_full']+stats['loss_missing']))
    loss.backward(); task.after_optimizer_step(False)
    assert all(torch.equal(v, task.model.matrix.state_dict()[k]) for k,v in before.items())
    task.optimizer.zero_grad()
    loss = task.train_step_mixed(b, torch.tensor([1,0,1,0]))[0]
    loss.backward(); task.optimizer.step(); task.after_optimizer_step(True)
    p = str(tmp_path/'last.pt'); task.save_checkpoint(p, 1, .4, 1)
    expected = torch.rand(3)
    other = CTFullMatrixSegTask({'model':make_model()}, copy.deepcopy(cfg))
    other.resume(p)
    assert torch.equal(expected, torch.rand(3))
    assert all(torch.equal(v, other.model.state_dict()[k]) for k,v in task.model.state_dict().items())
    changed = CTFullMatrixSegTask({'model':make_model(matrix_ridge=.2)}, copy.deepcopy(cfg))
    with pytest.raises(ValueError, match='configuration mismatch'): changed.resume(p)


def test_collection_does_not_change_global_rng_and_noop_snapshot(fake_backbones, tmp_path):
    from tasks.ct_full_matrix_seg import CTFullMatrixSegTask
    from utils.ct_full_matrix_diagnostics import MatrixCSVLogger
    m = make_model()
    cfg = SimpleNamespace(learning_rate=1e-3, weight_decay=0., mixed_precision=False, ema_enabled=False)
    task = CTFullMatrixSegTask({'model':m},cfg)
    x = torch.randn(2,4,16,16); y = x*1.2; gt = (torch.rand(2,1,32,32)>.7).float()
    s = MatrixScale(4, samples_per_class=2)
    rng = torch.get_rng_state().clone()
    first = s.candidates(x,y,gt,seed=9)
    second = s.candidates(x,y,gt,seed=9)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for aa,bb in zip(first,second) for a,b in zip(aa,bb))
    logger = MatrixCSVLogger(str(tmp_path))
    before = copy.deepcopy(m.state_dict())
    logger.snapshot(task, 1, 1, 'epoch_end')
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_shuffle_changes_value_pairing_without_changing_keys():
    s = MatrixScale(4, clusters=2, capacity=16, min_candidates=4, min_cluster_samples=2)
    b = s.foreground_bank
    b.valid.fill_(True); b.keys[0,0]=1; b.keys[1,0]=-1
    b.transforms[0].mul_(2); b.transforms[1].mul_(.5)
    x = torch.ones(1,4,2,2)
    keys = b.keys.clone()
    assert not torch.allclose(s(x), s(x,read_mode='shuffle'))
    assert torch.equal(keys,b.keys)


@pytest.mark.parametrize('value_mode', ['analytic', 'lowrank'])
def test_runner_resume_and_eval_controls(fake_backbones, tmp_path, monkeypatch, value_mode):
    import csv
    import sys
    from torch.utils.data import DataLoader
    import run_ct_full_matrix_add as runner
    import eval_ct_full_matrix_add as evaluator
    from tasks.ct_full_matrix_seg import CTFullMatrixSegTask

    def loaders(_cfg):
        g = torch.Generator().manual_seed(10)
        rows = [dict(ct=torch.randn(1,32,32,generator=g),
                     pet=torch.randn(1,32,32,generator=g),
                     mask=(torch.rand(1,32,32,generator=g)>.7).float(),case_id=f'case{i}') for i in range(4)]
        return tuple(DataLoader(rows,batch_size=4,shuffle=i==0,drop_last=True,
            generator=torch.Generator().manual_seed(10+i)) for i in range(3))

    monkeypatch.setattr(runner,'_loaders',loaders)
    common = ['run','--matrix_value_mode',value_mode,'--matrix_value_rank','2','--epochs','2','--batch_size','4','--num_workers','0',
        '--hd95_mode','legacy','--pretrained','false','--mixed_precision','false',
        '--ema_enabled','true','--ema_start_epoch','0','--cosine_warmup','0',
        '--decoder_channels','32','24','16','8','--matrix_clusters','2',
        '--matrix_capacity','16','--matrix_min_candidates','4','--matrix_min_cluster_samples','2',
        '--matrix_warmup_steps','0','--matrix_refresh_steps','1','--checkpoint_root',str(tmp_path)]
    monkeypatch.setattr(sys,'argv', common+['--hash','continuous','--log_interval','1'])
    runner.main()
    reference = CTFullMatrixSegTask.load_state_dicts(str(tmp_path/'MDT/continuous/ckpt.last.pth.tar'))
    original = CTFullMatrixSegTask.save_checkpoint
    class InterruptRun(Exception): pass
    def interrupt(self,path,epoch,*args,**kwargs):
        original(self,path,epoch,*args,**kwargs)
        if epoch==1 and path.endswith('ckpt.last.pth.tar'): raise InterruptRun()
    monkeypatch.setattr(CTFullMatrixSegTask,'save_checkpoint',interrupt)
    monkeypatch.setattr(sys,'argv',common+['--hash','resumed'])
    with pytest.raises(InterruptRun): runner.main()
    monkeypatch.setattr(CTFullMatrixSegTask,'save_checkpoint',original)
    last = str(tmp_path/'MDT/resumed/ckpt.last.pth.tar')
    monkeypatch.setattr(sys,'argv',common+['--hash','resumed','--resume',last])
    runner.main()
    restored = CTFullMatrixSegTask.load_state_dicts(last)
    for tag in ('model','model_ema'):
        assert all(torch.equal(v,restored[tag][k]) for k,v in reference[tag].items())
    assert reference['matrix_refresh_stats']==restored['matrix_refresh_stats']
    with open(tmp_path/'MDT/resumed/matrix_banks.csv',newline='') as f:
        assert len(list(csv.DictReader(f)))==16
    monkeypatch.setattr(evaluator,'_loaders',loaders)
    for mode in ('normal','off','identity','shuffle'):
        out = str(tmp_path/('eval_'+mode))
        monkeypatch.setattr(sys,'argv',['eval','--checkpoint',last,'--output_dir',out,'--read_mode',mode])
        evaluator.main()
        assert (tmp_path/('eval_'+mode)/'final_test_metrics.json').exists()
