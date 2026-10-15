"""Data-free real-backbone preflight: exact Add initialization and two AMP steps."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from configs.base import str2bool
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_lesion_guided_fusion import FullLesionGuidedPETCT
from models.components.lesion_guided_fusion import auxiliary_lesion_loss
from utils.seg_losses import BCEDiceLoss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lesion_aux_loss_enabled', type=str2bool, default=True)
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; not a CUDA validation')
    if args.image_size < 64 or args.image_size % 32 or args.batch_size < 2:
        raise ValueError('Use image_size>=64 divisible by 32, batch_size>=2')
    device = torch.device(args.device)
    print(f'[INFO] python={sys.executable} torch={torch.__version__} cuda={torch.version.cuda}', flush=True)
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False).to(device).eval()
    torch.manual_seed(2023)
    model = FullLesionGuidedPETCT(pretrained=False).to(device).eval()
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    with torch.no_grad():
        expected = baseline(ct, pet=pet, forward_mode='full')['logits']
        actual = model(ct, pet=pet)['logits']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del baseline, expected, actual
    print('[PASS] complete-model step-0 logits exactly equal Add', flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    target = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    target[:, :, args.image_size // 3:args.image_size // 2, args.image_size // 3:args.image_size // 2] = 1
    criterion = BCEDiceLoss()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(ct, pet=pet, return_aux=args.lesion_aux_loss_enabled)
            main_loss, _ = criterion(out['logits'], target)
            extra, stats = auxiliary_lesion_loss(out['aux'].get('lesion_logits', []), target,
                                                 enabled=args.lesion_aux_loss_enabled)
            loss = main_loss + extra
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        if step == 1:
            for block in model.fusion.blocks:
                assert block.gate[-1].weight.grad.abs().sum() > 0
                assert block.ct_context.project[0].weight.grad.abs().sum() > 0
                assert block.pet_context.project[0].weight.grad.abs().sum() > 0
        scaler.step(optimizer)
        scaler.update()
        print(f'[PASS] step={step + 1} total={float(loss.detach()):.6f} '
              f'main={float(main_loss.detach()):.6f} aux_raw={float(stats["aux_raw"]):.6f} '
              f'aux_weighted={float(stats["aux_weighted"]):.6f}', flush=True)
    print(json.dumps(dict(status='passed', device=str(device), image_size=args.image_size,
        batch_size=args.batch_size, aux_loss_enabled=args.lesion_aux_loss_enabled,
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
