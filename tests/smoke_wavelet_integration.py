"""Module-2 (wavelet fusion) integration smoke test.

Real path only: build_mdt_seg_teacher -> DualSharedAddPETCTBaseline ->
MDTSegTeacher with real ConvNeXtV2-Nano / MiT-B1 backbones (pretrained=None),
real Module-1, real shared decoder, synthetic CT/PET + FG+BG mask.

Usage:
    python tests/smoke_wavelet_integration.py --device cuda --image-size 512 --batch-size 2 --amp
    python tests/smoke_wavelet_integration.py --device cpu --image-size 64 --batch-size 2
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def make_config(**kw):
    d = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False,
        pspi_enabled=True, pspi_num_clusters=2, pspi_num_clusters_bg=0,
        pspi_num_clusters_fg=0, pspi_build_stage=4, pspi_cluster_max_iter=5,
        pspi_outlier_discard_rate=0.0, pspi_bank_update_mode='direct',
        pspi_ema_momentum=0.95, pspi_retrieval_temperature=0.1,
        pspi_retrieval_topk=0, pspi_retrieval_per_class_topk=0,
        pspi_retrieval_gate_temperature=1.0, pspi_proto_temperature=0.02,
        pspi_collect_candidates=True, pspi_prior_scale_enabled=False,
        pspi_prior_scale_init=0.1, pspi_affine_enabled=True,
        pspi_reconstruction_weight=0.0, pspi_proto_contrastive_weight=0.0,
        pspi_ct_proto_contrastive_weight=0.0, stage1_init_enabled=False,
        eval_full_pet=False, eval_fixed_missing_pet=True,
        m2_enabled=False, m2_checkpoint=False,
        learning_rate=1e-4, weight_decay=0.0, mixed_precision=False,
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, grad_clip=5.0,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def synth_battery(device, image_size, batch_size):
    g = torch.Generator(device='cpu').manual_seed(2023)
    ct = torch.randn(batch_size, 1, image_size, image_size, generator=g).to(device)
    pet = torch.randn(batch_size, 1, image_size, image_size, generator=g).to(device)
    mask = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    h0, h1 = image_size // 4, image_size // 2
    mask[:, :, h0:h1, h0:h1] = 1.0  # FG block + BG surroundings
    return ct, pet, mask


def assert_backbones_real(model):
    for name in ('enc_ct', 'enc_pet'):
        cls = type(getattr(model, name)).__name__
        assert 'Fallback' not in cls, f'{name} fell back to {cls}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(2023)

    from models.build_mdt_seg import build_mdt_seg_teacher
    from tasks.mdt_seg import MDTSegTeacher

    ct, pet, mask = synth_battery(device, args.image_size, args.batch_size)
    state = torch.tensor([1, 0] * ((args.batch_size + 1) // 2), device=device)[:args.batch_size]
    batch = {'ct': ct, 'pet': pet, 'mask': mask}

    # 1. M2 off: wrapper passthrough == AddFusion; old state_dict strict load.
    cfg0 = make_config()
    m0 = build_mdt_seg_teacher(cfg0)['model']
    assert_backbones_real(m0)
    assert type(m0.fusion).__name__ == 'AddFusion'
    m0.to(device).eval()
    with torch.no_grad():
        ct_f = m0._encode_ct(ct)
        pet_f = m0._encode_pet(pet)
        a = m0._fuse_features(ct_f, pet_f, True)
        b = m0.fusion(ct_f, pet_f, None)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
    buf = io.BytesIO()
    torch.save(m0.state_dict(), buf)
    buf.seek(0)
    m0b = build_mdt_seg_teacher(cfg0)['model']
    m0b.load_state_dict(torch.load(buf, map_location='cpu', weights_only=True), strict=True)
    print('[1] M2-off wrapper==AddFusion 1e-6 OK; strict reload OK')

    # 2. M2 on: cold-start mixed forward/backward/step; Missing rows == C.
    cfg1 = make_config(m2_enabled=True)
    m1 = build_mdt_seg_teacher(cfg1)['model']
    assert_backbones_real(m1)
    assert type(m1.fusion).__name__ == 'PETCTWaveletFusion'
    assert not bool(m1.module1.bank_ready)
    m1.to(device)
    use_amp = bool(args.amp) and device.type == 'cuda'
    teacher = MDTSegTeacher({'model': m1}, make_config(m2_enabled=True, mixed_precision=use_amp))
    teacher.device = device
    teacher.model.to(device)
    from contextlib import nullcontext
    amp_ctx = (lambda: torch.autocast(device.type, dtype=torch.float16)) if use_amp else nullcontext

    def _opt_step(loss):
        # Mirrors run_mdt_seg.py: scaled backward, unscale, clip, step.
        teacher.optimizer.zero_grad(set_to_none=True)
        if teacher.scaler.is_enabled():
            teacher.scaler.scale(loss).backward()
            teacher.scaler.unscale_(teacher.optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.model.parameters(), 5.0)
            teacher.scaler.step(teacher.optimizer)
            teacher.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.model.parameters(), 5.0)
            teacher.optimizer.step()
    opt_ids = {id(p) for p in teacher.optimizer.param_groups[0]['params']}
    m2_ids = {id(p) for p in m1.fusion.parameters()}
    assert m2_ids <= opt_ids and len(m2_ids) > 0, 'M2 params missing from optimizer'
    m1.eval()
    with torch.no_grad():
        ct_f = m1._encode_ct(ct)
        pet_f = m1._encode_pet(pet)
        miss_only = m1._fuse_features(ct_f, pet_f, False)
        for x, y in zip(miss_only, ct_f):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
    print('[2a] cold-start pure-Missing == C OK')
    m1.train()
    loss, _, outputs, _ = teacher.train_step_mixed(batch, state, missing_loss_weight=1.0)
    assert outputs['logits'].shape == (args.batch_size, 1, args.image_size, args.image_size)
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(outputs['logits']).all())
    loss.backward()  # caller performs backward/step (train_step_mixed is forward+loss only)
    teacher.optimizer.step()
    teacher.optimizer.zero_grad(set_to_none=True)
    print(f'[2b] cold-start mixed step OK, loss={float(loss):.4f} (amp={use_amp})')

    # 3. Real collect/finalize -> bank ready; 2 mixed steps; affine grad flows.
    m1.train()
    with torch.no_grad():
        for _ in range(8):
            ct_f = m1._encode_ct(ct)
            pet_f = m1._encode_pet(pet)
            m1.module1.collect_candidates(ct_f, pet_f, mask)
    rep = m1.module1.finalize_epoch(epoch=1)
    assert bool(m1.module1.bank_ready), rep
    for _ in range(2):
        with amp_ctx():
            loss, _, outputs, _ = teacher.train_step_mixed(batch, state, missing_loss_weight=1.0)
        _opt_step(loss)
    # Extra forward+backward to inspect gradients: the loop ends with
    # zero_grad, so grads must be read before clearing.
    with amp_ctx():
        loss, _, outputs, _ = teacher.train_step_mixed(batch, state, missing_loss_weight=1.0)
    if teacher.scaler.is_enabled():
        teacher.scaler.scale(loss).backward()
    else:
        loss.backward()
    aff_grads = [p.grad for p in m1.pet_affine.parameters() if p.grad is not None]
    assert aff_grads and all(bool(torch.isfinite(g).all()) for g in aff_grads), 'no affine grad'
    teacher.optimizer.step()
    teacher.optimizer.zero_grad(set_to_none=True)
    opened = sum(float(s.low_missing.out.weight.abs().sum()) for s in m1.fusion.scales)
    assert opened > 0, 'M2 Missing zero-heads never opened'
    print(f'[3] bank ready, 2 post-bank steps OK, loss={float(loss):.4f}')

    # 4. Eval Full/Missing isolation + fusion input contract.
    m1.eval()
    calls = {'enc_pet': 0, 'low_full': 0, 'high_full': 0, 'pet_full_project': 0,
             'retrieve': 0, 'affine': 0}
    orig_enc_pet = m1.enc_pet
    orig_retrieve = m1.module1.retrieve_pet_prior
    orig_affine = m1.pet_affine

    class EncPetGuard(torch.nn.Module):
        def forward(self, *a, **k):
            calls['enc_pet'] += 1
            raise AssertionError('eval Missing called PET encoder')

    def retrieve_counter(*a, **k):
        calls['retrieve'] += 1
        return orig_retrieve(*a, **k)

    class AffineGuard(torch.nn.Module):
        def forward(self, *a, **k):
            calls['affine'] += 1
            raise AssertionError('eval Full called pet_affine')

    m1.enc_pet = EncPetGuard()
    m1.module1.retrieve_pet_prior = retrieve_counter
    for s in m1.fusion.scales:
        s.low_full.register_forward_pre_hook(lambda *a: calls.__setitem__('low_full', calls['low_full'] + 1))
        if s.high_full is not None:
            s.high_full.register_forward_pre_hook(lambda *a: calls.__setitem__('high_full', calls['high_full'] + 1))
        s.pet_full_project.register_forward_pre_hook(lambda *a: calls.__setitem__('pet_full_project', calls['pet_full_project'] + 1))
    n_collect = int(m1.module1._collect_calls)
    version = int(m1.module1.bank_version.item())
    keys_before = [getattr(m1.module1, f'ct_keys_s{s + 1}').detach().cpu().clone() for s in range(4)]
    with torch.no_grad():
        out_m = m1(ct, pet=None, forward_mode='missing', mask=mask)
    assert out_m['logits'].shape == (args.batch_size, 1, args.image_size, args.image_size)
    assert calls['enc_pet'] == 0 and calls['low_full'] == 0 and calls['high_full'] == 0
    assert calls['pet_full_project'] == 0 and calls['retrieve'] >= 1
    assert int(m1.module1._collect_calls) == n_collect and int(m1.module1.bank_version.item()) == version
    for s in range(4):
        torch.testing.assert_close(
            getattr(m1.module1, f'ct_keys_s{s + 1}').detach().cpu(), keys_before[s])
    print('[4a] eval Missing: no PET encoder, no Full branch, bank untouched OK')
    m1.enc_pet = orig_enc_pet
    m1.module1.retrieve_pet_prior = orig_retrieve
    m1.pet_affine = AffineGuard()

    def retrieve_forbidden(*a, **k):
        raise AssertionError('eval Full called retrieve_pet_prior')

    m1.module1.retrieve_pet_prior = retrieve_forbidden
    with torch.no_grad():
        out_f = m1(ct, pet=pet, forward_mode='full', mask=mask)
    assert out_f['logits'].shape == (args.batch_size, 1, args.image_size, args.image_size)
    assert calls['affine'] == 0
    print('[4b] eval Full OK: no retrieve/affine calls')
    m1.pet_affine = orig_affine
    m1.module1.retrieve_pet_prior = orig_retrieve

    # 5. Checkpoint roundtrip with M2 config; strict load; output match.
    ck = {'config': vars(make_config(m2_enabled=True)), 'model': m1.state_dict()}
    buf2 = io.BytesIO()
    torch.save(ck, buf2)
    buf2.seek(0)
    ck2 = torch.load(buf2, map_location='cpu', weights_only=False)
    assert ck2['config']['m2_enabled'] is True
    m2 = build_mdt_seg_teacher(SimpleNamespace(**ck2['config']))['model']
    m2.load_state_dict(ck2['model'], strict=True)
    m2.to(device).eval()
    m1.eval()
    with torch.no_grad():
        r1 = m1(ct, pet=pet, pet_available=state, forward_mode='auto', mask=mask)['logits']
        r2 = m2.to(device)(ct, pet=pet, pet_available=state, forward_mode='auto', mask=mask)['logits']
        torch.testing.assert_close(r1, r2, atol=1e-6, rtol=1e-6)
    n_total = sum(p.numel() for p in m1.parameters())
    n_m2 = sum(p.numel() for p in m1.fusion.parameters())
    print(f'[5] checkpoint strict reload OK; params total={n_total} m2={n_m2}')
    print('SMOKE PASSED')


if __name__ == '__main__':
    main()
