"""Real-backbone preflight for native four-scale MMSFormer FusionBlock."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.components.mmsformer_fusion import MMSFormerFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_mmsformer_fusion import FullMMSFormerPETCT
from utils.seg_losses import BCEDiceLoss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=2)
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is not a CUDA validation')
    if args.image_size < 64 or args.image_size % 32 or args.batch_size < 2:
        raise ValueError('Use image_size>=64 divisible by 32 and batch_size>=2')
    device = torch.device(args.device)
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False).to(device).eval()
    torch.manual_seed(2023)
    model = FullMMSFormerPETCT(pretrained=False).to(device).eval()
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    with torch.no_grad():
        expected = baseline(ct, pet, forward_mode='full')['logits']
        native = model(ct, pet, forward_mode='full')['logits']
        assert not torch.equal(native, expected), 'Native FusionBlock must not be replaced by Add'
        fusion = model.fusion
        model.fusion = MMSFormerFusion(enabled=False).to(device)
        control = model(ct, pet, forward_mode='full')['logits']
        torch.testing.assert_close(control, expected, atol=0, rtol=0)
        model.fusion = fusion
    del baseline, expected, native, control
    print('[PASS] shared baseline weights equal; disabled fusion exactly Add; enabled uses native initialization', flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    criterion = BCEDiceLoss()
    mask = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    mask[:, :, args.image_size//3:args.image_size//2, args.image_size//3:args.image_size//2] = 1
    se_active = [None] * 4
    hooks = [block.fc[1].register_forward_hook(
        lambda m, args, out, i=i: se_active.__setitem__(i, float((out.detach() > 0).float().mean())))
        for i, block in enumerate(model.fusion.block.channel_attns)]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(ct, pet, forward_mode='full')
            loss, _ = criterion(out['logits'], mask)
        assert torch.isfinite(loss) and not out['aux']
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        stage_gradients = []
        for i in range(4):
            modules = (model.fusion.block.liner_fusion_layers[i], model.fusion.block.mix_ffn[i],
                       model.fusion.block.channel_attns[i])
            magnitudes = [sum(float(p.grad.abs().sum()) for p in module.parameters()) for module in modules]
            assert magnitudes[0] > 0 and magnitudes[1] > 0, f'S{i+1}: linear/multiscale path has no gradient'
            # Native SE has a bias-free small ReLU bottleneck. All units can
            # be inactive for a particular batch. Check and report that state,
            # rather than changing source initialization to force gradients.
            if magnitudes[2] == 0:
                assert se_active[i] == 0, f'S{i+1}: unexpected disconnected SE path'
            stage_gradients.append(dict(stage=i+1, linear=magnitudes[0], multiscale=magnitudes[1],
                                        se=magnitudes[2], se_relu_active_fraction=se_active[i]))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        print(f'[PASS] step={step+1} main_BCE_Dice={float(loss.detach()):.6f}; four-scale gradients finite', flush=True)
        print(json.dumps(dict(step=step+1, stage_gradients=stage_gradients)), flush=True)
    for hook in hooks:
        hook.remove()
    print(json.dumps(dict(status='passed', device=str(device), image_size=args.image_size,
        batch_size=args.batch_size, amp=device.type == 'cuda',
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
