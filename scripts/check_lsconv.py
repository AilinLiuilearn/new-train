"""Data-free preflight: source SKA parity and real-backbone training checks.

Run from repository root, e.g. python scripts/check_lsconv.py --device cuda
--backend triton --image_size 512 --batch_size 2. Does not load pretrained data.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.components.lsconv_ska import SKA, ska_torch
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_lsconv_fusion import FullLSConvPETCT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--backend', choices=('torch', 'triton', 'auto'), default='triton')
    p.add_argument('--image_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=2)
    args = p.parse_args()
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is not a CUDA validation')
    if args.batch_size < 2 or args.image_size < 64 or args.image_size % 32:
        raise ValueError('Use batch_size >= 2 and image_size >=64 divisible by 32')
    print(f'[INFO] python={sys.executable} torch={torch.__version__} '
          f'cuda={torch.version.cuda} backend={args.backend}', flush=True)
    # Check actual requested backend first, before allocating full models.
    for dtype in ([torch.float32, torch.float16] if device.type == 'cuda' else [torch.float32]):
        x = torch.randn(2, 64, 13, 17, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(2, 8, 9, 13, 17, device=device, dtype=dtype, requires_grad=True)
        a, b = SKA(args.backend)(x, w), ska_torch(x, w)
        tol = 2e-2 if dtype == torch.float16 else 2e-5
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        probe = torch.randn_like(x)
        ga = torch.autograd.grad((a * probe).sum(), (x, w))
        gb = torch.autograd.grad((b * probe).sum(), (x, w))
        for left, right in zip(ga, gb):
            assert torch.isfinite(left).all()
            torch.testing.assert_close(left, right, atol=tol, rtol=tol)
        print(f'[PASS] SKA {args.backend} forward/backward dtype={dtype}', flush=True)
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).to(device).eval()
    torch.manual_seed(2023)
    model = FullLSConvPETCT(lsconv_backend=args.backend, pretrained=False).to(device).eval()
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    with torch.no_grad():
        baseline_logits = base(ct, pet=pet, forward_mode='full')['logits']
        logits = model(ct, pet=pet)['logits']
        torch.testing.assert_close(logits, baseline_logits, atol=0, rtol=0)
    del base, baseline_logits, logits
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    target = torch.rand(args.batch_size, 1, args.image_size, args.image_size, device=device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
            logits = model(ct, pet=pet)['logits']
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        if step == 1:
            for block in model.fusion.blocks:
                assert block.lkp.cv1.c.weight.grad.abs().sum() > 0
        scaler.step(optimizer)
        scaler.update()
        print(f'[PASS] real-backbone optimizer step={step + 1} loss={float(loss.detach()):.6f}', flush=True)
    print(json.dumps(dict(status='passed', device=str(device), backend=args.backend,
        image_size=args.image_size, batch_size=args.batch_size,
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
