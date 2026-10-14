"""Real ConvNeXtV2-Nano/MiT-B1 smoke, random weights, no patient data.

Run from the repository root:
PYTHONPATH=. python tests/smoke_fg_bg_prototype.py --size 64 --backward
PYTHONPATH=. python tests/smoke_fg_bg_prototype.py --size 512
"""
import argparse
import json

import torch

from run_decoder_fg_bg_prototype_add import ForegroundBackgroundConfig
from models.fg_bg_prototype_add_model import build_fg_bg_model
from tasks.fg_bg_prototype_seg import ForegroundBackgroundSegTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2023)
    cfg = ForegroundBackgroundConfig(args=dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None,
        pet_pretrained_path=None, decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=True, deep_supervision=False, pretrained=False,
        prototype_enabled=True, prototype_dim=32, prototype_clusters=4,
        prototype_capacity=256, prototype_min_candidates=24, prototype_decay=.999,
        prototype_refresh_steps=20, prototype_warmup_steps=0,
        prototype_gain_mode='per_scale', prototype_adaptive_sampling=True,
        prototype_sampling_decay=.99, prototype_max_patches=16,
        prototype_deep_supervision_weight=.2, prototype_logit_scale=2., prototype_correction_alpha=.1,
        prototype_min_gain=.01, prototype_ct_aux_weight=.1, mixed_precision=False,
        learning_rate=1e-4, weight_decay=1e-4, ema_enabled=False))
    task = ForegroundBackgroundSegTask(build_fg_bg_model(cfg), cfg)
    for scale in task.model.prototype.scales:
        for bank in scale.banks:
            bank.add(torch.randn(24, bank.dimension, device=task.device))
            bank.refresh()
    # Exercise populated libraries with nonzero corrections, not only bypass.
    for scale in task.model.prototype.scales:
        torch.nn.init.normal_(scale.out.weight, std=.001)
    ct = torch.randn(2, 1, args.size, args.size, device=task.device)
    pet = torch.randn_like(ct)
    state = torch.tensor([1, 0], device=task.device)
    if args.backward:
        task.model.train()
        batch = dict(ct=ct, pet=pet, mask=(torch.rand_like(ct) > .8).float())
        loss, logits, _, _ = task.train_step_mixed(batch, state)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in task.model.parameters())
        task.optimizer.step()
        task.after_optimizer_step(True)
    else:
        task.model.eval()
        with torch.no_grad():
            logits = task.model(ct, pet=pet, pet_available=state)['logits']
            other = pet.clone()
            other[1] = float('nan')
            repeated = task.model(ct, pet=other, pet_available=state)['logits']
            assert torch.equal(logits, repeated)
            missing = task.model(ct[1:], forward_mode='missing')['logits']
            assert torch.allclose(logits[1:], missing, atol=1e-3)
    assert logits.shape == ct.shape and torch.isfinite(logits).all()
    print(json.dumps(dict(size=args.size, backward=args.backward, device=str(task.device),
                          parameters=sum(p.numel() for p in task.model.parameters()),
                          extra_parameters=sum(p.numel() for p in task.model.prototype.parameters()),
                          shape=list(logits.shape), diagnostics=task.model.prototype.diagnostics())))


if __name__ == '__main__':
    main()
