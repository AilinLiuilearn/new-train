"""Two-step heterogeneous mixed baseline preflight, without real data/checkpoints."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from configs.base import str2bool
from models.build_mdt_seg import build_dual_model
from models.components.add_fusion import AddFusion
from run_full_missing_baseline import MixedHeteroTask
from utils.run_common import build_balanced_pet_available, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--pretrained", type=str2bool, default=True)
    p.add_argument(
        "--pet_pretrained_path",
        default="/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1",
    )
    p.add_argument("--ct_pretrained_path", default="/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano")
    args = p.parse_args()
    if (
        args.batch_size < 2
        or args.batch_size % 2
        or args.image_size < 32
        or args.image_size % 32
    ):
        raise ValueError("even batch_size>=2 and image_size divisible by 32 required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(2023)
    device = torch.device(args.device)
    cfg = SimpleNamespace(
        ct_backbone="convnextv2_nano",
        pet_backbone="mit_b1",
        ct_pretrained_path=args.ct_pretrained_path,
        pet_pretrained_path=args.pet_pretrained_path,
        pretrained=args.pretrained,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        learning_rate=4e-5,
        weight_decay=1e-4,
        mixed_precision=device.type == "cuda",
        ema_enabled=False,
        random_state=2023,
        hd95_mode="surface",
    )
    task = MixedHeteroTask(build_dual_model(cfg), cfg)
    # The regular task auto-selects CUDA when present; this preflight honors --device.
    task.device = device
    task.model.to(device)
    model = task.model
    assert isinstance(model.fusion, AddFusion)
    assert model.enc_ct is not model.enc_pet and type(model.enc_ct) is not type(
        model.enc_pet
    )
    assert {p.data_ptr() for p in model.enc_ct.parameters()}.isdisjoint(
        {p.data_ptr() for p in model.enc_pet.parameters()}
    )
    seen = []

    def pet_hook(module, inputs):
        seen.append(len(inputs[0]))

    def aligned(module, inputs, output):
        for x, c, stride in zip(output, (64, 128, 320, 512), (4, 8, 16, 32)):
            assert x.shape[1:] == (
                c,
                args.image_size // stride,
                args.image_size // stride,
            )

    handles = [
        model.enc_pet.register_forward_pre_hook(pet_hook),
        model.ct_align.register_forward_hook(aligned),
        model.pet_align.register_forward_hook(aligned),
    ]
    batch = dict(
        ct=torch.randn(args.batch_size, 1, args.image_size, args.image_size),
        pet=torch.randn(args.batch_size, 1, args.image_size, args.image_size),
        mask=torch.zeros(args.batch_size, 1, args.image_size, args.image_size),
    )
    batch["mask"][
        :,
        :,
        args.image_size // 3 : args.image_size // 2,
        args.image_size // 3 : args.image_size // 2,
    ] = 1
    task.scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda", init_scale=128.0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(2):
        state = build_balanced_pet_available(args.batch_size, step, 2023, device)
        task.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, logits, _, stats = task.train_step_mixed(batch, state)
        assert stats["num_full"] == stats["num_missing"] == args.batch_size // 2
        assert logits.shape == batch["mask"].shape and torch.isfinite(loss)
        task.scaler.scale(loss).backward()
        task.scaler.unscale_(task.optimizer)
        for name in ("enc_ct", "enc_pet", "ct_align", "pet_align", "decoder"):
            grads = [
                p.grad for p in getattr(model, name).parameters() if p.grad is not None
            ]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        task.scaler.step(task.optimizer)
        task.scaler.update()
        losses.append(float(loss.detach()))
    for handle in handles:
        handle.remove()
    assert seen == [args.batch_size // 2] * 2
    model.eval()

    def forbid_pet(*args):
        raise AssertionError("Missing invoked PET encoder")

    handle = model.enc_pet.register_forward_pre_hook(forbid_pet)
    try:
        with torch.no_grad():
            missing = model(batch["ct"].to(device), forward_mode="missing")["logits"]
            assert torch.isfinite(missing).all()
    finally:
        handle.remove()
    print(
        json.dumps(
            dict(
                status="passed",
                device=str(device),
                image_size=args.image_size,
                batch_size=args.batch_size,
                full_per_batch=args.batch_size // 2,
                missing_per_batch=args.batch_size // 2,
                pet_encoded_rows=seen,
                learning_rate=4e-5,
                ct_backbone="convnextv2_nano",
                pet_backbone="mit_b1",
                alignment=True,
                compensation=False,
                encoder_pretrained=args.pretrained,
                losses=losses,
                params_total=sum(p.numel() for p in model.parameters()),
                peak_cuda_gib=(
                    torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda"
                    else None
                ),
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
