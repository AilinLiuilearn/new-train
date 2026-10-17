"""Real backbones, synthetic data, no training-performance claim."""

import argparse
import torch
from models.spatial_modulation_add_model import build_matrix_model
from run_spatial_modulation_add import SpatialModulationConfig
from tasks.spatial_modulation_seg import SpatialModulationSegTask

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=64)
p.add_argument("--backward", action="store_true")
args = p.parse_args()
torch.set_num_threads(2)
cfg = SpatialModulationConfig(
    args=dict(
        use_deep_supervision=False,
        deep_supervision=False,
        ct_backbone="convnextv2_nano",
        pet_backbone="mit_b1",
        pretrained=False,
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        mixed_precision=False,
        ema_enabled=False,
        learning_rate=8e-5,
        weight_decay=1e-4,
        matrix_clusters=2 if args.size < 128 else 8,
        matrix_enabled=True,
        matrix_mod_loss_enabled=True,
        matrix_mod_loss_weight=0.1,
    )
)
task = SpatialModulationSegTask(build_matrix_model(cfg), cfg)
b = dict(
    ct=torch.randn(2, 1, args.size, args.size),
    pet=torch.randn(2, 1, args.size, args.size),
    mask=(torch.rand(2, 1, args.size, args.size) > 0.8).float(),
)
task.begin_epoch(1)
if args.backward:
    loss = task.train_step_mixed(b, torch.tensor([1, 0]))[0]
    loss.backward()
    assert all(
        s.bank.gamma.grad is None and s.bank.beta.grad is None
        for s in task.model.matrix.scales
    )
    task.optimizer.step()
    task.after_optimizer_step(True)
else:
    task.model.train()
    with torch.no_grad():
        out = task.model(b["ct"], b["pet"], torch.tensor([1, 0]), return_knowledge=True)
        knowledge = out["knowledge"]
        pending = []
        for s, c, f in zip(
            task.model.matrix.scales, knowledge["ct"], knowledge["full"]
        ):
            pending.append(s.supervise(c, f, b["mask"][:1], True)[1])
        task.model.matrix.commit(pending)
task.begin_epoch(2)
if args.backward:
    task.optimizer.zero_grad(set_to_none=True)
    loss = task.train_step_mixed(b, torch.tensor([1, 0]))[0]
    loss.backward()
    assert all(
        s.bank.beta.grad is not None and s.bank.beta.grad.abs().sum() > 0
        for s in task.model.matrix.scales
    )
    task.optimizer.step()
    task.after_optimizer_step(True)
task.model.eval()
with torch.no_grad():
    before = {k: v.clone() for k, v in task.model.state_dict().items()}
    out = task.model(b["ct"], b["pet"], torch.tensor([1, 0]))["logits"]
    assert out.shape == b["ct"].shape and torch.isfinite(out).all()
    assert all(torch.equal(v, task.model.state_dict()[k]) for k, v in before.items())
print("real-backbone modulation smoke passed", args.size, args.backward)
