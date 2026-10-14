"""Real encoders + switchable CACR/GCM + decoder backward, without dataset/weights."""
import argparse
import json
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.base import str2bool
from models.full_cainet_fusion import FullCAINetPETCT
from utils.seg_losses import BCEDiceLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--cacr_enabled', type=str2bool, default=True)
    parser.add_argument('--gcm_enabled', type=str2bool, default=True)
    parser.add_argument('--amp', action='store_true', help='CUDA fp16 with GradScaler')
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp smoke is CUDA-only')
    torch.manual_seed(2023)
    model = FullCAINetPETCT(cacr_enabled=args.cacr_enabled,
                           gcm_enabled=args.gcm_enabled, pretrained=False).to(device).train()
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    mask = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    lo, hi = args.image_size // 3, 2 * args.image_size // 3
    mask[:, :, lo:hi, lo:hi] = 1
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
        output = model(ct, pet=pet, forward_mode='full')['logits']
        loss, _ = BCEDiceLoss()(output, mask)
    scaler.scale(loss).backward()
    checked = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Missing/nonfinite gradient: {name}')
            checked += 1
    if not torch.isfinite(loss) or not torch.isfinite(output).all():
        raise RuntimeError('Nonfinite logits/loss')
    print(json.dumps(dict(
        device=str(device), amp=args.amp,
        cacr_enabled=args.cacr_enabled, gcm_enabled=args.gcm_enabled,
        output_shape=list(output.shape), loss=float(loss.detach()),
        fusion_params=sum(p.numel() for p in model.fusion.parameters()),
        total_params=sum(p.numel() for p in model.parameters()),
        finite_gradient_parameter_tensors=checked,
        gcm_initial_projection_zero=(bool(torch.count_nonzero(model.fusion.gcm_to_s4.weight) == 0)
                                     if args.gcm_enabled else None),
        peak_cuda_mib=(torch.cuda.max_memory_allocated(device) / 1024**2
                       if device.type == 'cuda' else None),
        peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    ), indent=2))


if __name__ == '__main__':
    main()
