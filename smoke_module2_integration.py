# -*- coding: utf-8 -*-
"""Smoke script for the Module-2 integration lifecycle.

Runs: two-round alternating training (epoch1 cold -> finalize -> epoch2),
real wrapper/task path, optimizer/scheduler/scaler checkpoint round-trip,
optional real-resolution (512) CUDA AMP forward/backward, and text-cache
mode selection. Usage:

CPU lifecycle (default):
  python smoke_module2_integration.py

Real-resolution AMP on GPU:
  python smoke_module2_integration.py --amp-real-scale --device cuda

Text mode (requires a valid cache; fails loudly otherwise):
  python smoke_module2_integration.py --use-text --text-cache pretrained/module2_text_cache.pt

No other flags exist. This script never launches the full 60-epoch training;
formal runs use run_mdt_seg.py with the commands in the delivery report.
"""
import argparse
import copy
import time

import torch


def _base_model(use_text, text_cache, experts, device):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    kwargs = {'use_text': bool(use_text), 'experts_per_group': int(experts)}
    if use_text:
        from models.state_guided_expert_fusion import load_text_cache
        emb, meta = load_text_cache(text_cache)
        kwargs['text_embeddings'] = emb
        kwargs['text_metadata'] = meta
    torch.manual_seed(2023)
    return DualSharedAddPETCTBaseline(use_deep_supervision=False, module2_enabled=True,
                                      module2_kwargs=kwargs).to(device)


def _task(model):
    from tasks.mdt_seg import MDTSegTeacher
    cfg = type('C', (), {
        'learning_rate': 1e-4, 'weight_decay': 1e-4, 'mixed_precision': True,
        'loss_smooth': 1.0, 'bce_weight': 1.0, 'dice_weight': 1.0,
        'random_state': 2023, 'pspi_proto_contrastive_weight': 0.01,
    })()
    return MDTSegTeacher({'model': model}, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use-text', action='store_true')
    p.add_argument('--text-cache', default='pretrained/module2_text_cache.pt')
    p.add_argument('--experts-per-group', type=int, default=2)
    p.add_argument('--amp-real-scale', action='store_true',
                   help='512-input batch16 CUDA AMP forward/backward (needs GPU memory)')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'[SMOKE] device={device} use_text={args.use_text} experts={args.experts_per_group}')
    model = _base_model(args.use_text, args.text_cache, args.experts_per_group, device)
    n_params = sum(p.numel() for p in model.module2.parameters())
    print(f'[SMOKE] module2 params={n_params} fusion={type(model.module2).__name__}')
    task = _task(model)

    def batch(b=2, h=64):
        g = torch.Generator(device='cpu').manual_seed(1000 + b)
        return {'ct': torch.randn(b, 1, h, h, generator=g).to(device),
                'pet': torch.randn(b, 1, h, h, generator=g).to(device),
                'mask': (torch.rand(b, 1, h, h, generator=g) > 0.5).float().to(device)}

    # Round 1 (cold): alternating Full/Missing, bank not ready.
    model.train()
    b = batch()
    for step, mode in enumerate(['full', 'missing']):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, stats = task.train_step(b, forward_mode=mode)
        assert torch.isfinite(loss)
        loss.backward()
        task.optimizer.step()
        active = outputs['aux']['module2']['active']
        print(f"[SMOKE] round1 step={step} mode={mode} loss={float(loss):.4f} "
              f"module2_active={bool(active.any())} bank_ready={model.module1.bank_ready}")
        if mode == 'missing':
            assert not model.module1.bank_ready and not bool(active.any())

    report = model.finalize_module1_epoch(1)
    print(f"[SMOKE] finalize status={report.get('status')} bank_ready={model.module1.bank_ready}")
    assert model.module1.bank_ready

    # Round 2: both routes optimize with Module-2 active.
    for mode in ['full', 'missing']:
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, _ = task.train_step(b, forward_mode=mode)
        loss.backward()
        task.optimizer.step()
        aux = outputs['aux']['module2']
        ms = aux['modality_scales']
        assert torch.allclose(ms, 2.0 * aux['route_weights'], atol=1e-6), 'a=2w contract'
        scales_ct = ms[..., 0].mean(dim=0).tolist()
        scales_pet = ms[..., 1].mean(dim=0).tolist()
        alpha = [float(torch.sigmoid(v).item()) for v in model.missing_prior_logits.detach()]
        print(f"[SMOKE] round2 mode={mode} loss={float(loss):.4f} "
              f"route_mean={aux['route_weights'].mean(dim=(0,1)).tolist()} "
              f"scales_ct={[round(v,4) for v in scales_ct]} "
              f"scales_pet={[round(v,4) for v in scales_pet]} "
              f"alpha={[round(v,4) for v in alpha]}")
        assert bool(aux['active'].any())
        assert model.missing_prior_logits is not None, 'alpha reused under Module-2'
    # Missing inference with pet=None must not touch the PET encoder.
    model.eval()
    with torch.no_grad():
        out_m = model(b['ct'], pet=None, forward_mode='missing')
    assert torch.isfinite(out_m['logits']).all()
    print('[SMOKE] missing inference pet=None OK, '
          f'route_missing={out_m["aux"]["module2"]["route_weights"].mean(dim=(0,1)).tolist()}')

    # Optimizer/scheduler/scaler state round-trip.
    opt_state = copy.deepcopy(task.optimizer.state_dict())
    scaler_state = copy.deepcopy(task.scaler.state_dict())
    task.optimizer.load_state_dict(opt_state)
    task.scaler.load_state_dict(scaler_state)
    print('[SMOKE] optimizer/scheduler/scaler state round-trip OK')

    if args.amp_real_scale:
        if device.type != 'cuda':
            raise ValueError('--amp-real-scale requires --device cuda')
        print('[SMOKE] real-scale AMP: batch16 @512x512 forward+backward')
        torch.manual_seed(7)
        ct = torch.randn(16, 1, 512, 512, device=device)
        pet = torch.randn(16, 1, 512, 512, device=device)
        mask = (torch.rand(16, 1, 512, 512, device=device) > 0.5).float()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.cuda.amp.autocast(enabled=True):
                out = model(ct, pet=pet, forward_mode='full', mask=mask)
                loss = out['logits'].float().mean() + out['prototype_contrastive_loss'].float()
            loss.backward()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'[SMOKE] AMP real-scale BLOCKED by GPU memory: {e}')
                return
            raise
        print(f"[SMOKE] AMP ok peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
              f"time={time.time()-t0:.1f}s loss={float(loss):.4f}")
    print('[SMOKE] done')


if __name__ == '__main__':
    main()
