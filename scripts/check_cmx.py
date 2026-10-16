"""Data-free real-backbone preflight for unchanged official CMX modules."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from configs.base import str2bool
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_cmx_fusion import FullCMXPETCT
from utils.seg_losses import BCEDiceLoss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--cmx_ffm_enabled', type=str2bool, default=True)
    p.add_argument('--cmx_frm_enabled', type=str2bool, default=False)
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; not a CUDA validation')
    if args.image_size < 64 or args.image_size % 32 or args.batch_size < 2:
        raise ValueError('Use image_size>=64 divisible by 32, batch_size>=2')
    device = torch.device(args.device)
    print(f'[INFO] python={sys.executable} torch={torch.__version__} '
          f'cuda={torch.version.cuda} ffm={args.cmx_ffm_enabled} frm={args.cmx_frm_enabled}', flush=True)
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).to(device).eval()
    torch.manual_seed(2023)
    model = FullCMXPETCT(cmx_ffm_enabled=args.cmx_ffm_enabled,
                         cmx_frm_enabled=args.cmx_frm_enabled, pretrained=False).to(device).eval()
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    with torch.no_grad():
        expected = base(ct, pet=pet, forward_mode='full')['logits']
        result = model(ct, pet=pet)['logits']
        if args.cmx_ffm_enabled or args.cmx_frm_enabled:
            assert not torch.equal(result, expected), 'Official initialization must not be replaced by Add'
        model.fusion.ffm_enabled = model.fusion.frm_enabled = False
        control = model(ct, pet=pet)['logits']
        torch.testing.assert_close(control, expected, atol=0, rtol=0)
        model.fusion.ffm_enabled, model.fusion.frm_enabled = args.cmx_ffm_enabled, args.cmx_frm_enabled
    del base, expected, result, control
    print('[PASS] shared baseline weights equal; disabled fusion exactly Add; enabled fusion uses official init', flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    mask = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    mask[:, :, args.image_size // 3:args.image_size // 2, args.image_size // 3:args.image_size // 2] = 1
    criterion = BCEDiceLoss()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs = model(ct, pet=pet)
            loss, _ = criterion(outputs['logits'], mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        for blocks in (model.fusion.ffms, model.fusion.frms):
            for block in blocks:
                grads = [p.grad for p in block.parameters()]
                assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
                assert sum(float(g.abs().sum()) for g in grads) > 0
        scaler.step(optimizer)
        scaler.update()
        print(f'[PASS] optimizer step={step + 1} main_BCE_Dice={float(loss.detach()):.6f}', flush=True)
    print(json.dumps(dict(status='passed', device=str(device), image_size=args.image_size,
        batch_size=args.batch_size, cmx_ffm_enabled=args.cmx_ffm_enabled, cmx_frm_enabled=args.cmx_frm_enabled,
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
