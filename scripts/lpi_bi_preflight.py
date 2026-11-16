"""Real 512/batch16 AMP check; all temporary updates and weights discarded."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from runtime import REPO, require_myenv, resolve_config, sha, gpu_lock_path
from variants.tsbci.model import SHCIPETCT
from variants.lpi_bi.model import Model
from variants.lpi_bi.train import objective, diagnostics
from variants.tsbci import train as engine
from datasets.staged import loaders
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from utils.amp import make_grad_scaler, autocast_context
from utils.ema import ModelEMA


def hashes(model):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()}


def main(args):
    require_myenv()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_num_threads(4)
    args.run = 'lpi_bi_preflight'; args.arm = 'full'
    args.config = str(REPO / 'configs/lpi_bi.json')
    cfg, _, _ = resolve_config(args, 'lpi_bi')
    cfg.num_workers = 0
    with open(gpu_lock_path(), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory() as scratch:
            _, val, _ = loaders(cfg, Path(scratch))
            batch = next(iter(val))
        ct, pet, target = [batch[name].cuda() for name in ['ct', 'pet', 'mask']]
        assert len(target) == 16 and ct.shape[-2:] == (512, 512)
        seed_everything(2023)
        base = SHCIPETCT(vars(cfg), arm='m1', load_generic=True).cuda().eval()
        reference_hash = hashes(base)
        reference_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            reference = base(ct[:1], pet[:1])['logits']
        del base
        gc.collect(); torch.cuda.empty_cache()
        seed_everything(2023)
        model = Model(vars(cfg), load_generic=True)
        candidate_hash = hashes(model)
        assert all(candidate_hash[k] == v for k, v in reference_hash.items())
        assert torch.equal(torch.get_rng_state(), reference_rng)
        assert sum(p.numel() for p in model.front.parameters()) == 654608
        model.cuda().eval()
        with torch.no_grad():
            initial = model(ct[:1], pet[:1])['logits']
        torch.testing.assert_close(initial, reference, rtol=0, atol=0)
        del initial, reference
        model.train()
        ema = ModelEMA(model, .999, warmup=True, device='cuda')
        criterion = BCEDiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
        scaler = make_grad_scaler(torch.device('cuda'), True)
        torch.cuda.reset_peak_memory_stats()
        attempts, successful, groups_seen = 0, 0, {}
        loss_snapshot = None
        while successful < 3 and attempts < 10:
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            scale_before = scaler.get_scale()
            with autocast_context(torch.device('cuda'), True):
                output = model(ct, pet, capture=True)
            loss, parts = objective(output, target, criterion)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            finite = bool(torch.isfinite(norm))
            scaler.step(optimizer); scaler.update()
            if finite:
                successful += 1
                ema.update(model)
                with tempfile.TemporaryDirectory() as scratch:
                    engine.OUT = Path(scratch)
                    diagnostics(model, output, 0, successful,
                                {'success': True, 'backward_scale': scale_before})
                    snapshot = json.loads((engine.OUT / 'diagnostics.jsonl').read_text())
                    for scale, row in snapshot['scales'].items():
                        groups_seen[scale] = row['unscaled_clipped_gradient_norms']
                loss_snapshot = parts
            print('PREFLIGHT', attempts, 'successful_steps', successful, 'gradient', float(norm), flush=True)
            del output, loss
            gc.collect()
        assert successful == 3, 'Could not complete three finite AMP updates'
        assert all(all(v > 0 for v in groups.values()) for groups in groups_seen.values()), groups_seen
        # Export/import the exact selected weight schema with no optimizer/RNG state.
        from variants.lpi_bi.inference import load_model
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / 'best.pt'
            torch.save({'format': 'lpi_bi_v1_inference', 'weights': ema.model.state_dict(),
                        'config': vars(cfg)}, checkpoint)
            restored = load_model(checkpoint, device='cpu')
            assert hashes(restored) == hashes(ema.model)
            del restored
        result = {
            'passed': True, 'parameters': sum(p.numel() for p in model.parameters()),
            'new_parameters': sum(p.numel() for p in model.front.parameters()),
            'shared_initialization_identical': True, 'loader_rng_identical': True,
            'initial_FP32_output_exact': True, 'AMP_batch16_three_updates_finite': True,
            'temporary_AMP_attempts': attempts, 'last_temporary_losses': loss_snapshot,
            'later_gradient_groups': groups_seen, 'best_only_roundtrip': True,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024**3,
            'EMA_and_optimizer_included_in_memory_check': True,
            'scope': 'Temporary preflight only; no performance result or reusable trained weights',
        }
        files = list((REPO / 'variants/lpi_bi').glob('*.py'))
        files += [REPO / name for name in ['variants/tsbci/train.py', 'configs/lpi_bi.json',
                                          'scripts/lpi_bi_preflight.py']]
        result['source_sha256'] = {str(p.relative_to(REPO)): sha(p) for p in files}
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', default=str(REPO / 'paths.local.json'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', default=str(REPO / 'docs/LPI_BI_GPU_PREFLIGHT.json'))
    main(parser.parse_args())
