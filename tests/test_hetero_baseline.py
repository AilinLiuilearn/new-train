"""Heterogeneous Add baseline, exact-half routing and selected weight contracts."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

import run_full_missing_baseline as runner
from models.build_mdt_seg import build_dual_model
from models.components.add_fusion import AddFusion
from models.components.backbones import TimmFeatureBackbone, SegformerFeatureBackbone
from tasks.mdt_seg import MDTSegTeacher
from utils.run_common import build_balanced_pet_available


def parse(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--hash",
            "smoke",
            "--checkpoint_root",
            str(tmp_path),
            "--pretrained",
            "false",
            "--mixed_precision",
            "false",
            "--batch_size",
            "2",
            "--image_size_2d",
            "64",
            "--epochs",
            "1",
            "--cosine_warmup",
            "0",
            "--num_workers",
            "0",
            "--hd95_mode",
            "surface",
            *extra,
        ],
    )
    return runner.MixedHeteroConfig.parse_arguments()


def test_launcher_actual_arguments(monkeypatch, tmp_path):
    shim = tmp_path / "python"
    capture = tmp_path / "args.json"
    shim.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(dict(argv=sys.argv[1:],'
        'omp=os.environ["OMP_NUM_THREADS"],mkl=os.environ["MKL_NUM_THREADS"])))\n'
    )
    shim.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "bash",
            str(root / "scripts/train_full_missing_hetero_lr4e5.sh"),
            "--hash",
            "override",
        ],
        env=dict(
            os.environ,
            PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
            CAPTURE_ARGS=str(capture),
        ),
        check=True,
    )
    data = json.loads(capture.read_text())
    assert data["omp"] == data["mkl"] == "1"
    assert data["argv"][0] == "-u" and data["argv"][1] == "run_full_missing_baseline.py"
    monkeypatch.setattr(sys, "argv", ["run", *data["argv"][2:]])
    cfg = runner.MixedHeteroConfig.parse_arguments()
    assert cfg.hash == "override" and cfg.ct_backbone == "convnextv2_nano" and cfg.pet_backbone == "mit_b1"
    assert cfg.ct_pretrained_path != cfg.pet_pretrained_path
    assert cfg.ct_pretrained_path.endswith("/pretrained/convnextv2_nano") and cfg.pretrained
    assert cfg.pet_pretrained_path.endswith("/pretrained/mit-b1")
    assert cfg.learning_rate == 4e-5 and cfg.train_batch_mode == "mixed"
    assert cfg.batch_size == 16 and cfg.epochs == cfg.early_stop_patience == 60
    assert cfg.ema_enabled and cfg.ema_start_epoch == 3 and cfg.ema_decay == 0.999
    assert cfg.norm_mode == "imagenet" and cfg.allow_val_equals_test
    assert not cfg.use_deep_supervision and not cfg.deep_supervision
    assert not hasattr(cfg, "local_enabled") and not hasattr(cfg, "matrix_enabled")


def test_actual_heterogeneous_alignment_routing_and_gradients(monkeypatch, tmp_path):
    cfg = parse(monkeypatch, tmp_path, "--ema_enabled", "false")
    task = runner.MixedHeteroTask(build_dual_model(cfg), cfg)
    model = task.model
    runner.assert_protocol(cfg, model)
    assert isinstance(model.fusion, AddFusion)
    assert isinstance(model.enc_ct, TimmFeatureBackbone)
    assert isinstance(model.enc_pet, SegformerFeatureBackbone)
    assert (
        type(model.enc_ct) is not type(model.enc_pet) and model.enc_ct is not model.enc_pet
    )
    assert {p.data_ptr() for p in model.enc_ct.parameters()}.isdisjoint(
        {p.data_ptr() for p in model.enc_pet.parameters()}
    )
    seen = []

    def record(module, inputs, output):
        seen.append((inputs[0][0].shape[0], [tuple(x.shape) for x in output]))

    hooks = [
        model.ct_align.register_forward_hook(record),
        model.pet_align.register_forward_hook(record),
    ]
    batch = dict(
        ct=torch.randn(2, 1, 64, 64),
        pet=torch.randn(2, 1, 64, 64),
        mask=torch.ones(2, 1, 64, 64),
    )
    state = build_balanced_pet_available(2, 0, 2023, task.device)
    loss, _, _, stats = task.train_step_mixed(batch, state)
    assert stats["num_full"] == stats["num_missing"] == 1
    assert torch.allclose(
        loss.detach(), 0.5 * stats["loss_full"] + 0.5 * stats["loss_missing"]
    )
    loss.backward()
    assert [row[0] for row in seen] == [2, 1]
    assert [x[1] for x in seen[0][1]] == [64, 128, 320, 512]
    for name in ("enc_ct", "enc_pet", "ct_align", "pet_align", "decoder"):
        grads = [
            p.grad for p in getattr(model, name).parameters() if p.grad is not None
        ]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    for hook in hooks:
        hook.remove()
    model.eval()
    ct, pet = batch["ct"].to(task.device), batch["pet"].to(task.device)
    pet[state.cpu() == 0] = float("nan")
    with torch.no_grad():
        output = model(ct, pet, state)["logits"]
        clean = model(ct, torch.nan_to_num(pet), state)["logits"]
    assert torch.equal(output, clean)
    model.zero_grad(set_to_none=True)

    def forbid(*args):
        pytest.fail("Missing called PET encoder")

    hook = model.enc_pet.register_forward_pre_hook(forbid)
    try:
        model(ct, forward_mode="missing")["logits"].square().mean().backward()
    finally:
        hook.remove()
    assert all(p.grad is None for p in model.enc_pet.parameters())
    with pytest.raises(ValueError, match="exact-half"):
        task.train_step_mixed(
            {k: v.repeat(2, 1, 1, 1) for k, v in batch.items()},
            torch.tensor([1, 1, 1, 0]),
        )


@pytest.mark.parametrize("start_epoch,expected", [(3, "raw"), (0, "ema")])
def test_training_checkpoint_and_selected_final_weights(
    monkeypatch, tmp_path, start_epoch, expected
):
    cfg = parse(
        monkeypatch,
        tmp_path,
        "--ema_start_epoch",
        str(start_epoch),
        "--log_interval",
        "1",
    )
    monkeypatch.setattr(
        runner.MixedHeteroConfig, "parse_arguments", classmethod(lambda cls: cfg)
    )
    g = torch.Generator().manual_seed(2)
    rows = [
        dict(
            ct=torch.randn(1, 64, 64, generator=g),
            pet=torch.randn(1, 64, 64, generator=g),
            mask=(torch.rand(1, 64, 64, generator=g) > 0.8).float(),
            case_id=str(i),
        )
        for i in range(2)
    ]
    monkeypatch.setattr(
        runner,
        "_loaders",
        lambda conf: tuple(
            DataLoader(rows, batch_size=2, drop_last=True) for _ in range(3)
        ),
    )
    runner.main()
    folder = Path(cfg.checkpoint_dir)
    ckpt = MDTSegTeacher.load_state_dicts(folder / "ckpt.best_joint.pth.tar")
    assert ckpt["eval_weights"] == expected and ckpt["global_batch_step"] == 1
    assert ckpt["config"]["pet_missing_rate"] == 0.5
    cfg.pretrained = False
    fresh = build_dual_model(cfg)["model"]
    fresh.load_state_dict(ckpt["model"], strict=True)
    fresh.load_state_dict(ckpt["model_ema"], strict=True)
    with open(folder / "train_log.csv") as f:
        record = list(csv.DictReader(f))[0]
    assert (
        float(record["train_full_samples"])
        == float(record["train_missing_samples"])
        == 1
    )
    with open(folder / "final_test_metrics.csv") as f:
        metrics = list(csv.DictReader(f))
    assert [float(r["missing_rate"]) for r in metrics] == [0, 0.25, 0.5, 0.75, 1]
    assert all(r["weights"] == expected for r in metrics)
    # Standalone reevaluation must use the same selected weights and metric protocol.
    import eval_full_missing_baseline as evaluation
    monkeypatch.setattr(evaluation, "_loaders", runner._loaders)
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", str(folder / "ckpt.best_joint.pth.tar"),
    ])
    evaluation.main()
    final = json.loads((folder / "final_test_metrics.json").read_text())
    reeval = json.loads((folder / "reeval_test_metrics.json").read_text())
    assert reeval == final
    # A fresh invocation must not overwrite the completed run.
    with pytest.raises(FileExistsError):
        runner.main()


def test_evaluator_rejects_missing_selected_ema():
    from eval_full_missing_baseline import select_checkpoint_weights
    with pytest.raises(ValueError, match="does not contain ema"):
        select_checkpoint_weights({"eval_weights": "ema", "model": {}, "model_ema": None})


def test_protocol_rejects_homogeneous_backbones(monkeypatch, tmp_path):
    cfg = parse(monkeypatch, tmp_path, "--ct_backbone", "mit_b1")
    with pytest.raises(ValueError, match="CT=convnextv2_nano"):
        runner.assert_protocol(cfg)
