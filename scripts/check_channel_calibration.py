"""Data-free real-backbone Add equivalence and two-step training preflight."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_channel_calibration import FullChannelCalibrationPETCT
from utils.seg_losses import BCEDiceLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is not a CUDA validation')
    if args.image_size < 64 or args.image_size % 32 or args.batch_size < 2:
        raise ValueError('Use image_size>=64 divisible by 32, batch_size>=2')
    device = torch.device(args.device)
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).to(device).eval()
    torch.manual_seed(2023)
    model = FullChannelCalibrationPETCT(pretrained=False).to(device).eval()
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    with torch.no_grad():
        expected = base(ct, pet, forward_mode='full')['logits']
        actual = model(ct, pet, forward_mode='full')['logits']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del base, expected, actual
    print('[PASS] all shared weights equal; initial enabled fusion is exactly Add', flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    criterion = BCEDiceLoss()
    mask = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    mask[:, :, args.image_size//3:args.image_size//2, args.image_size//3:args.image_size//2] = 1
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs = model(ct, pet, forward_mode='full')
            loss, _ = criterion(outputs['logits'], mask)
        assert torch.isfinite(loss) and not outputs['aux']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        for block in model.fusion.blocks:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
            for head in (block.fc_ct, block.fc_pet):
                assert float(head.weight.grad.abs().sum()) > 0
            if step == 1:
                assert float(block.fc_squeeze.weight.grad.abs().sum()) > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        print(f'[PASS] step={step+1} main_BCE_Dice={float(loss.detach()):.6f}; gradients finite', flush=True)
    print(json.dumps(dict(status='passed', device=str(device), image_size=args.image_size,
        batch_size=args.batch_size, amp=device.type == 'cuda',
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
