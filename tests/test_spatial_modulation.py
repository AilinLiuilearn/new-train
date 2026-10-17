import copy
from types import SimpleNamespace
import pytest
import torch
from models.spatial_modulation_bank import SpatialModulationScale, UnifiedModulationBank


def test_every_position_and_paired_means():
    bank = UnifiedModulationBank(2, clusters=2, chunk_size=3)
    x = torch.cat(
        [
            torch.tensor([[1.0, 0.0]]).repeat(7, 1),
            torch.tensor([[0.0, 1.0]]).repeat(5, 1),
        ]
    )
    g = 2 * x
    b = 3 * x
    fg = torch.arange(12).float() / 12
    bank.add_all(x, g, b, fg, True)
    assert int(bank.accepted) == 12 and bank.count.sum() == 12
    assert torch.allclose(bank.gamma, 2 * bank.keys) and torch.allclose(
        bank.beta, 3 * bank.keys
    )
    before = bank.gamma.clone()
    bank.add_all(x, 4 * x, 6 * x, fg, False)
    assert torch.allclose(bank.gamma, 0.99 * before + 0.01 * 4 * bank.keys)
    assert bank.count.sum() == 24


def test_warmup_loss_detach_and_switch():
    scale = SpatialModulationScale(4, clusters=2)
    ct = torch.randn(2, 4, 4, 4, requires_grad=True)
    full = (ct.detach() + 1).requires_grad_()
    mask = torch.ones(2, 1, 16, 16)
    loss, p = scale.supervise(ct, full, mask, True)
    loss.backward()
    assert ct.grad is None and full.grad is None
    assert scale.generator[-1].bias.grad.abs().sum() > 0
    expected = (
        (ct.detach() - full.detach()).abs().mean((1, 2, 3))
        / (full.detach().square().mean((1, 2, 3)).sqrt() + 1e-6)
    ).mean()
    assert torch.allclose(loss, expected)
    scale.bank.add_all(*p, True)
    scale.bank.gamma.fill_(0.2)
    scale.bank.beta.fill_(0.1)
    assert torch.equal(scale(ct), ct)
    scale.epoch.fill_(2)
    assert torch.allclose(scale(ct), 1.2 * ct + 0.1, atol=1e-6)
    scale.zero_grad()
    off, _ = scale.supervise(ct, full, mask, False)
    assert not off.requires_grad and off == 0


def test_model_task_and_resume(monkeypatch, tmp_path):
    from tests.helpers import FakeEncoder
    import models.dual_shared_add_baseline as base

    monkeypatch.setattr(base, "create_feature_backbone", lambda *a, **kw: FakeEncoder())
    from models.spatial_modulation_add_model import SpatialModulationAddModel
    from tasks.spatial_modulation_seg import SpatialModulationSegTask

    def model():
        return SpatialModulationAddModel(
            pretrained=False, decoder_channels=(32, 24, 16, 8), matrix_clusters=2
        )

    cfg = SimpleNamespace(
        learning_rate=0.001,
        weight_decay=0.0,
        mixed_precision=False,
        ema_enabled=False,
        random_state=2023,
    )
    task = SpatialModulationSegTask({"model": model()}, cfg)
    task.begin_epoch(1)
    batch = dict(
        ct=torch.randn(4, 1, 32, 32),
        pet=torch.randn(4, 1, 32, 32),
        mask=(torch.rand(4, 1, 32, 32) > 0.5).float(),
    )
    state = torch.tensor([1, 0, 1, 0])
    loss = task.train_step_mixed(batch, state)[0]
    loss.backward()
    assert all(
        s.generator[-1].weight.grad is not None for s in task.model.matrix.scales
    )
    before = copy.deepcopy(task.model.matrix.state_dict())
    task.after_optimizer_step(False)
    assert all(
        torch.equal(v, task.model.matrix.state_dict()[k]) for k, v in before.items()
    )
    task.optimizer.zero_grad()
    loss = task.train_step_mixed(batch, state)[0]
    loss.backward()
    task.optimizer.step()
    task.after_optimizer_step(True)
    assert all(int(s.bank.accepted) > 0 for s in task.model.matrix.scales)
    path = str(tmp_path / "checkpoint.pt")
    task.save_checkpoint(path, 1, 0.2, 1)
    other = SpatialModulationSegTask({"model": model()}, copy.deepcopy(cfg))
    other.resume(path)
    assert all(
        torch.equal(v, other.model.state_dict()[k])
        for k, v in task.model.state_dict().items()
    )
    task.begin_epoch(2)
    task.model.eval()
    ct = batch["ct"]
    pet = batch["pet"]
    full = task.model(ct, pet, forward_mode="full")["logits"]
    task.model.matrix_read_mode = "off"
    assert torch.equal(full, task.model(ct, pet, forward_mode="full")["logits"])
    task.model.matrix_read_mode = "normal"
    before = copy.deepcopy(task.model.state_dict())
    pred = task.model(ct, pet, state)["logits"]
    pet = pet.clone()
    pet[state == 0] = float("nan")
    assert torch.equal(pred, task.model(ct, pet, state)["logits"])
    assert all(torch.equal(v, task.model.state_dict()[k]) for k, v in before.items())


def test_runner_two_epochs_and_eval(monkeypatch, tmp_path):
    import sys, json
    from torch.utils.data import DataLoader
    from tests.helpers import FakeEncoder
    import models.dual_shared_add_baseline as base
    import run_spatial_modulation_add as runner
    import eval_spatial_modulation_add as evaluator
    from tasks.spatial_modulation_seg import SpatialModulationSegTask

    monkeypatch.setattr(base, "create_feature_backbone", lambda *a, **kw: FakeEncoder())

    def loaders(cfg):
        g = torch.Generator().manual_seed(7)
        rows = [
            dict(
                ct=torch.randn(1, 32, 32, generator=g),
                pet=torch.randn(1, 32, 32, generator=g),
                mask=(torch.rand(1, 32, 32, generator=g) > 0.5).float(),
                case_id=str(i),
            )
            for i in range(4)
        ]
        return tuple(
            DataLoader(
                rows,
                batch_size=4,
                drop_last=True,
                generator=torch.Generator().manual_seed(i),
            )
            for i in range(3)
        )

    monkeypatch.setattr(runner, "_loaders", loaders)
    args = [
        "run",
        "--hash",
        "modtest",
        "--epochs",
        "2",
        "--batch_size",
        "4",
        "--num_workers",
        "0",
        "--hd95_mode",
        "legacy",
        "--pretrained",
        "false",
        "--mixed_precision",
        "false",
        "--ema_enabled",
        "false",
        "--cosine_warmup",
        "0",
        "--decoder_channels",
        "32",
        "24",
        "16",
        "8",
        "--matrix_clusters",
        "2",
        "--checkpoint_root",
        str(tmp_path),
        "--log_interval",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    last = str(tmp_path / "MDT/modtest/ckpt.last.pth.tar")
    ckpt = SpatialModulationSegTask.load_state_dicts(last)
    assert ckpt["epoch"] == 2
    assert ckpt["model"]["matrix.scales.0.epoch"] == 2
    assert ckpt["model"]["matrix.scales.0.bank.accepted"] == 2 * 2 * 16 * 16
    monkeypatch.setattr(evaluator, "_loaders", loaders)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--checkpoint",
            last,
            "--compare_on_off",
            "true",
            "--output_dir",
            str(tmp_path / "comparison"),
        ],
    )
    evaluator.main()
    result = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert result["on_minus_off"][0]["dice"] == 0
