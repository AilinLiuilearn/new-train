import copy
from types import SimpleNamespace
import pytest
import torch
from models.spatial_modulation_bank import SpatialModulationScale, UnifiedModulationBank


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
        s.bank.gamma.grad is None and s.bank.beta.grad is None
        for s in task.model.matrix.scales
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
    task.begin_epoch(2)
    task.optimizer.zero_grad(set_to_none=True)
    task.config.matrix_mod_loss_enabled = False
    task.train_step_mixed(batch, state)[0].backward()
    assert all(s.bank.beta.grad.abs().sum() > 0 for s in task.model.matrix.scales)
    task.optimizer.step()
    task.after_optimizer_step(True)
    assert all(torch.count_nonzero(s.bank.beta) > 0 for s in task.model.matrix.scales)
    task.config.matrix_mod_loss_enabled = True
    path = str(tmp_path / "checkpoint.pt")
    task.save_checkpoint(path, 2, 0.2, 2)
    other = SpatialModulationSegTask({"model": model()}, copy.deepcopy(cfg))
    other.resume(path)
    assert all(
        torch.equal(v, other.model.state_dict()[k])
        for k, v in task.model.state_dict().items()
    )
    assert len(other.optimizer.state) == len(task.optimizer.state)
    value_ids = {
        id(p) for group in task.optimizer.param_groups for p in group["params"]
    }
    assert all(id(s.bank.beta) in value_ids for s in task.model.matrix.scales)
    old = task.load_state_dicts(path)
    old["experiment_type"] = "ct_full_spatial_modulation"
    torch.save(old, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="learnable"):
        other.resume(str(tmp_path / "old.pt"))
    task.begin_epoch(2)
    task.model.eval()
    ct = batch["ct"]
    pet = batch["pet"]
    full = task.model(ct, pet, forward_mode="full")["logits"]
    task.model.matrix_read_mode = "off"
    assert torch.equal(full, task.model(ct, pet, forward_mode="full")["logits"])
    task.model.matrix_read_mode = "normal"
    baseline = base.DualSharedAddPETCTBaseline(
        pretrained=False, decoder_channels=(32, 24, 16, 8)
    )
    baseline.load_state_dict(
        {
            k: v
            for k, v in task.model.state_dict().items()
            if not k.startswith("matrix.")
        },
        strict=True,
    )
    baseline.eval()
    assert torch.equal(full, baseline(ct, pet, forward_mode="full")["logits"])
    # Zero Value gives exact baseline Missing outputs even after Key collection.
    zero_model = copy.deepcopy(task.model)
    with torch.no_grad():
        for s in zero_model.matrix.scales:
            s.bank.gamma.zero_()
            s.bank.beta.zero_()
    assert torch.equal(
        zero_model(ct, forward_mode="missing")["logits"],
        baseline(ct, forward_mode="missing")["logits"],
    )
    before = copy.deepcopy(task.model.state_dict())
    pred = task.model(ct, pet, state)["logits"]
    pet = pet.clone()
    pet[state == 0] = float("nan")
    assert torch.equal(pred, task.model(ct, pet, state)["logits"])

    def forbid_pet(*args):
        pytest.fail("pure Missing called PET encoder")

    handle = task.model.enc_pet.register_forward_pre_hook(forbid_pet)
    try:
        task.model(ct, forward_mode="missing")
    finally:
        handle.remove()
    assert all(torch.equal(v, task.model.state_dict()[k]) for k, v in before.items())
    with pytest.raises(RuntimeError, match="evaluation"):
        task.model.matrix.commit(
            [
                (torch.zeros(1, s.channels), torch.zeros(1))
                for s in task.model.matrix.scales
            ]
        )


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
    assert ckpt["experiment_type"] == "ct_key_learnable_affine_value"
    assert torch.count_nonzero(ckpt["model"]["matrix.scales.0.bank.beta"]) > 0
    import csv

    with open(tmp_path / "MDT/modtest/modulation_slots.csv") as f:
        rows = list(csv.DictReader(f))
    assert all(float(r["value_grad_norm"]) == 0 for r in rows if r["epoch"] == "1")
    assert any(float(r["value_grad_norm"]) > 0 for r in rows if r["epoch"] == "2")
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


def test_disabled_module_has_no_value_gradient_or_bank_writes(monkeypatch):
    from tests.helpers import FakeEncoder
    import models.dual_shared_add_baseline as base
    from models.spatial_modulation_add_model import SpatialModulationAddModel
    from tasks.spatial_modulation_seg import SpatialModulationSegTask

    monkeypatch.setattr(base, "create_feature_backbone", lambda *a, **kw: FakeEncoder())
    model = SpatialModulationAddModel(
        pretrained=False,
        decoder_channels=(32, 24, 16, 8),
        matrix_enabled=False,
        matrix_clusters=2,
    )
    cfg = SimpleNamespace(
        learning_rate=0.001, weight_decay=0.0, mixed_precision=False, ema_enabled=False
    )
    task = SpatialModulationSegTask({"model": model}, cfg)
    task.begin_epoch(2)
    batch = dict(
        ct=torch.randn(4, 1, 32, 32),
        pet=torch.randn(4, 1, 32, 32),
        mask=(torch.rand(4, 1, 32, 32) > 0.5).float(),
    )
    loss = task.train_step_mixed(batch, torch.tensor([1, 0, 1, 0]))[0]
    loss.backward()
    task.optimizer.step()
    task.after_optimizer_step(True)
    assert all(
        int(s.bank.accepted) == 0 and s.bank.beta.grad is None
        for s in model.matrix.scales
    )
    assert task.mod_loss_sum == 0
