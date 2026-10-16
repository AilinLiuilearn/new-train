"""Real encoders, random weights, synthetic data; no patient-data performance claim.

PYTHONPATH=. python tests/smoke_ct_full_matrix.py --size 64 --backward
PYTHONPATH=. python tests/smoke_ct_full_matrix.py --size 512
"""
import argparse
import json
import torch
from models.ct_full_matrix_add_model import build_matrix_model
from run_ct_full_matrix_add import CTFullMatrixConfig
from tasks.ct_full_matrix_seg import CTFullMatrixSegTask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--backward', action='store_true')
    p.add_argument('--value_mode', choices=('analytic','lowrank'), default='analytic')
    args = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(2023)
    cfg = CTFullMatrixConfig(args=dict(ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        pretrained=False, ct_pretrained_path=None, pet_pretrained_path=None,
        use_deep_supervision=False, deep_supervision=False, decoder_channels=(512,256,128,64),
        learning_rate=8e-5, weight_decay=1e-4, mixed_precision=False, ema_enabled=False,
        matrix_enabled=True, matrix_clusters=4, matrix_value_mode=args.value_mode))
    task = CTFullMatrixSegTask(build_matrix_model(cfg), cfg)
    # Actually fit all banks, then exercise live gates and nonidentity transforms.
    for s in task.model.matrix.scales:
        for b in s.banks:
            x = torch.randn(64, s.channels, device=task.device)
            b.add(x, x*1.2, torch.ones(64, device=task.device))
            assert b.refresh(), b.refresh_stats
    ct = torch.randn(2,1,args.size,args.size, device=task.device)
    pet = torch.randn_like(ct); state = torch.tensor([1,0], device=task.device)
    if args.backward:
        task.model.train()
        loss, logits, _, _ = task.train_step_mixed(
            dict(ct=ct, pet=pet, mask=(torch.rand_like(ct)>.8).float()), state)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in task.model.parameters())
        assert all(s.gate.weight.grad is not None for s in task.model.matrix.scales)
        if args.value_mode=='lowrank':
            assert all(b.value_a.grad is not None and b.value_b.grad is not None for s in task.model.matrix.scales for b in s.banks)
        task.optimizer.step(); task.after_optimizer_step(True)
    else:
        task.model.eval()
        with torch.no_grad():
            logits = task.model(ct, pet, state)['logits']
            other = pet.clone(); other[1] = float('nan')
            assert torch.equal(logits, task.model(ct, other, state)['logits'])
            before = task.model(ct, pet, forward_mode='full')['logits']
            task.model.matrix_read_mode = 'off'
            assert torch.equal(before, task.model(ct, pet, forward_mode='full')['logits'])
    assert logits.shape == ct.shape and torch.isfinite(logits).all()
    print(json.dumps(dict(size=args.size, backward=args.backward, device=str(task.device),
        gate_parameters=sum(p.numel() for p in task.model.matrix.parameters()),
        matrix_buffers=sum(p.numel() for p in task.model.matrix.buffers()),
        shape=list(logits.shape), banks=task.model.matrix.diagnostics())))


if __name__ == '__main__': main()
