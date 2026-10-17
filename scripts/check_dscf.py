"""Real-backbone Add+DSCF preflight: exact Add init and two gradient updates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.components.dscf_residual import DSCFResidualFusion, stage_geometry
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_dscf_residual import FullDSCFResidualPETCT
from utils.seg_losses import BCEDiceLoss


def grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=2)
    args = p.parse_args()
    stage_geometry(args.image_size)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is not a CUDA validation')
    if args.batch_size < 2: raise ValueError('Use batch_size>=2 for the BN training preflight')
    device = torch.device(args.device)
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False).eval()
    torch.manual_seed(2023)
    model = FullDSCFResidualPETCT(pretrained=False, image_size=args.image_size).eval()
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    ct = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    pet = torch.randn_like(ct)
    baseline.to(device)
    with torch.no_grad():
        expected = baseline(ct, pet, forward_mode='full')['logits'].cpu()
    del baseline
    if device.type == 'cuda': torch.cuda.empty_cache()
    model.to(device)
    with torch.no_grad():
        initial = model(ct, pet, forward_mode='full')['logits'].cpu()
        torch.testing.assert_close(initial, expected, atol=0, rtol=0)
        fusion = model.fusion
        model.fusion = DSCFResidualFusion(enabled=False, image_size=args.image_size).to(device)
        control = model(ct, pet, forward_mode='full')['logits'].cpu()
        torch.testing.assert_close(control, expected, atol=0, rtol=0)
        model.fusion = fusion
    del expected, initial, control
    print('[PASS] shared weights match Add; enabled zero-init and disabled logits exactly Add', flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    # Conservative preflight loss scale isolates AMP support from initial
    # GradScaler calibration skips. The training runner keeps its baseline scale.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda', init_scale=128.)
    criterion = BCEDiceLoss()
    mask = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
    mask[:, :, args.image_size//3:args.image_size//2, args.image_size//3:args.image_size//2] = 1
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
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
            assert grad_sum(getattr(model, name)) > 0, name
        stages = []
        for i, block in enumerate(model.fusion.blocks):
            values = dict(stage=i+1, output_head=grad_sum(block.U_fc1),
                          ct_projection=grad_sum(block.D_fc1), pet_projection=grad_sum(block.D_fc2),
                          offset_ct=grad_sum(block.deform_atten.conv_offset_x),
                          offset_pet=grad_sum(block.deform_atten.conv_offset_y),
                          sample_weight=grad_sum(block.deform_atten.get_sample_weight))
            assert values['output_head'] > 0
            for name in ('ct_projection', 'pet_projection', 'offset_ct', 'offset_pet', 'sample_weight'):
                assert (values[name] == 0) if step == 0 else (values[name] > 0), (step, values)
            stages.append(values)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        print(json.dumps(dict(step=step+1, main_BCE_Dice=float(loss.detach()),
                              gradients=stages, diagnostics=model.fusion.pop_diagnostics())), flush=True)
    print(json.dumps(dict(status='passed', device=str(device), image_size=args.image_size,
        batch_size=args.batch_size, pretrained=False, amp=device.type == 'cuda',
        correction_precision='float32', preflight_initial_loss_scale=128.,
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None), indent=2))


if __name__ == '__main__':
    main()
