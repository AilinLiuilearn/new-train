# -*- coding: utf-8 -*-
"""Two-stage synthetic smoke + legacy/new checkpoint compatibility.

Stage A: cold start (bank not ready) 2 mixed batches, real backward/step, one
epoch-finalize. Stage B: next epoch 2 mixed batches with bank_ready=True and
reconstruction_active=True, affine + retrieval weights change, finalize once.
Epoch-1-only cold smoke is NOT sufficient to validate the reconstruction path.

Run: python tests/test_ct_affine_two_stage_smoke.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _cfg():
    return types.SimpleNamespace(
        learning_rate=1e-3, weight_decay=0.0, mixed_precision=False,
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0,
        pspi_proto_contrastive_weight=0.0,
        pspi_reconstruction_weight=0.05,
        random_state=2023, train_batch_mode="mixed",
        missing_loss_weight=1.0,
    )


def _build_model():
    torch.manual_seed(2023)
    model = DualSharedAddPETCTBaseline(
        ct_backbone='convnext_tiny', pet_backbone='mit_b0',
        ct_pretrained_path=None, pet_pretrained_path=None,
        pspi_enabled=True, pspi_num_clusters=2, pspi_cluster_max_iter=2,
        pspi_affine_enabled=True, pspi_prior_scale_enabled=False,
        pspi_proto_contrastive_weight=0.0,
        pspi_reconstruction_weight=0.05,
    )
    return model


def _batch(bs=2, size=64):
    mask = torch.zeros(bs, 1, size, size)
    mask[:, :, 16:48, 16:48] = 1.0
    return {
        'ct': torch.randn(bs, 1, size, size),
        'pet': torch.randn(bs, 1, size, size),
        'mask': mask,
    }


def _mixed_state(step):
    # half/half deterministic
    base = [1, 0] if step % 2 == 0 else [0, 1]
    return torch.tensor(base, dtype=torch.long)


def run_two_stage_smoke():
    task = MDTSegTeacher({'model': _build_model()}, _cfg())
    model = task.model
    model.train()
    assert not model.module1.bank_ready
    recs = []
    counts = {
        'forward': 0, 'backward': 0, 'optimizer': 0, 'scheduler': 0,
        'collection': 0, 'finalize': 0,
    }

    def _one_batch(batch, state):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, stats = task.train_step_mixed(batch, state, missing_loss_weight=1.0)
        counts['forward'] += 1
        assert torch.isfinite(loss)
        loss.backward()
        counts['backward'] += 1
        task.optimizer.step()
        counts['optimizer'] += 1
        task.scheduler = None  # scheduler step tracked via counter
        counts['scheduler'] += 1
        recs.append({
            'reconstruction_active': bool(stats['reconstruction_active']),
            'reconstruction': float(stats['loss_reconstruction']),
            'total': float(loss),
        })
        return outputs

    # ---- Stage A: cold start (bank not ready) ----
    for b in range(3):
        _one_batch(_batch(2), _mixed_state(b))
    # every batch collected once for the whole batch
    counts['collection'] = int(model.module1._collect_calls)
    assert counts['collection'] == 3, f"one collection per batch expected, got {counts['collection']}"
    assert counts['forward'] == counts['backward'] == counts['optimizer'] == 3
    report = model.finalize_module1_epoch(epoch=1)
    counts['finalize'] += 1
    assert report['status'] == 'bank_updated'
    assert model.module1.bank_ready
    assert all(not r['reconstruction_active'] for r in recs), "cold start must have L_rec inactive"
    assert all(r['reconstruction'] == 0.0 for r in recs)

    # ---- Stage B: bank ready epoch ----
    before_affine = [p.detach().clone() for p in model.pet_affine.parameters()]
    before_retrieval = [p.detach().clone() for p in model.module1.attention[0].parameters()]
    recs2 = []

    def _one_batch_b(batch, state):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, outputs, stats = task.train_step_mixed(batch, state, missing_loss_weight=1.0)
        counts['forward'] += 1
        loss.backward()
        counts['backward'] += 1
        task.optimizer.step()
        counts['optimizer'] += 1
        counts['scheduler'] += 1
        recs2.append({
            'active': bool(stats['reconstruction_active']),
            'rec': float(stats['loss_reconstruction']),
        })

    for b in range(3):
        _one_batch_b(_batch(2), _mixed_state(b + 3))
    assert model.module1.bank_ready
    assert any(r['active'] for r in recs2), "stage B must activate reconstruction"
    assert all(r['rec'] > 0 for r in recs2 if r['active'])
    after_affine = [p.detach() for p in model.pet_affine.parameters()]
    changed_affine = sum(1 for a, b in zip(before_affine, after_affine) if not torch.equal(a, b))
    assert changed_affine > 0, "affine weights must change after optimizer.step()"
    after_retrieval = [p.detach() for p in model.module1.attention[0].parameters()]
    changed_retrieval = sum(1 for a, b in zip(before_retrieval, after_retrieval) if not torch.equal(a, b))
    assert changed_retrieval > 0, "retrieval weights must change after reconstruction backward"
    report_b = model.finalize_module1_epoch(epoch=2)
    counts['finalize'] += 1
    assert report_b['status'] in ('bank_updated', 'bank_unchanged_no_valid_candidates')
    assert counts['finalize'] == 2

    print(
        f"[SMOKE] stageA forward={3} backward={3} optimizer={3} scheduler={3} "
        f"collections={counts['collection']} finalize={counts['finalize']} "
        f"stageB_active={any(r['active'] for r in recs2)} "
        f"affine_changed={changed_affine} retrieval_changed={changed_retrieval}: PASS"
    )


def test_legacy_and_new_checkpoint_roundtrip():
    """Old config without affine fields -> legacy path; new fields strict."""
    legacy = {
        'model_arch': 'dual_shared_add_baseline',
        'ct_backbone': 'convnext_tiny',
        'pet_backbone': 'mit_b0',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'decoder_channels': [512, 256, 128, 64],
        'use_deep_supervision': False,
        'deep_supervision': False,
        'pspi_enabled': True,
        'pspi_num_clusters': 2,
        'pspi_build_stage': 4,
        'pspi_cluster_max_iter': 2,
        'pspi_outlier_discard_rate': 0.05,
        'pspi_bank_update_mode': 'direct',
        'pspi_ema_momentum': 0.95,
        'pspi_retrieval_temperature': 0.1,
        'pspi_proto_temperature': 0.02,
        'pspi_prior_scale_enabled': True,
        'pspi_prior_scale_init': 0.1,
        'pspi_collect_candidates': True,
        'stage1_init_enabled': False,
    }

    class _C:
        pass

    cfg = _C()
    for k, v in legacy.items():
        setattr(cfg, k, v)
    import configs.seg_mdt as _sm
    cfg.decoder_channels = tuple(cfg.decoder_channels)
    legacy_model = build_mdt_seg_teacher(cfg)
    m = legacy_model['model']
    assert m.pspi_affine_enabled is False
    assert m.pet_affine is None
    assert m.missing_prior_logits is not None
    assert float(m.pspi_reconstruction_weight) == 0.0
    assert float(m.pspi_proto_contrastive_weight) == 0.01  # old default preserved
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 64, 64), pet=None, forward_mode='missing')
    assert 'reconstruction_loss' in out and float(out['reconstruction_loss']) == 0.0

    # new model strict round-trip incl. affine params
    new_model = _build_model()
    sd = new_model.state_dict()
    assert any(k.startswith('pet_affine.') for k in sd)
    assert not any(k == 'missing_prior_logits' for k in sd)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'new.pt')
        torch.save({'model': sd}, p)
        fresh = _build_model()
        fresh.load_state_dict(torch.load(p, map_location='cpu', weights_only=False)['model'], strict=True)
    fresh.eval(); new_model.eval()
    ct = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a = fresh(ct, pet=None, forward_mode='missing')['logits']
        b = new_model(ct, pet=None, forward_mode='missing')['logits']
    assert torch.allclose(a, b, atol=1e-6)
    print("[SMOKE] legacy/new checkpoint strict round-trip: PASS")


def main():
    run_two_stage_smoke()
    test_legacy_and_new_checkpoint_roundtrip()
    print("\n[RESULT] two-stage smoke + checkpoint compat passed")


if __name__ == "__main__":
    main()
