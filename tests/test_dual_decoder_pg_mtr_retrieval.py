import csv
import os

import torch

from models.dual_decoder_pg_mtr_retrieval import DualDecoderPGMTRRetrieval
from run_mdt_seg import _resolve_checkpoint_selection
from utils.train_logger import append_epoch_log, init_train_log


class DummyFeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return self._channels


class DummyBackbone(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature_info = DummyFeatureInfo(channels)

    def forward(self, x):
        b, _, h, w = x.shape
        feats = []
        for i, c in enumerate(self.feature_info.channels(), start=1):
            scale = 2 ** i
            feats.append(torch.randn(b, c, max(1, h // scale), max(1, w // scale), device=x.device, dtype=x.dtype))
        return feats


def make_model(stage_mode='all'):
    model = DualDecoderPGMTRRetrieval.__new__(DualDecoderPGMTRRetrieval)
    torch.nn.Module.__init__(model)
    model.use_deep_supervision = False
    model.pg_mtr_detach_bank_missing = True
    model.enc_ct = DummyBackbone([64, 128, 320, 512])
    model.enc_pet = DummyBackbone([64, 128, 320, 512])
    model.ct_align = torch.nn.Identity()
    model.fusion = torch.nn.Identity()
    model.full_decoder = torch.nn.Identity()
    model.missing_decoder = torch.nn.Identity()
    from models.pg_mtr import PETGroundedMetabolicTokenRetrieval
    model.pg_mtr = PETGroundedMetabolicTokenRetrieval([64, 128, 320, 512], num_tokens=8, temperature=0.07, stage_mode=stage_mode)
    model.retrieval_projs = torch.nn.ModuleDict({str(k): torch.nn.Conv2d(model.pg_mtr.stage_modules[str(k)].latent_dim, [64, 128, 320, 512][k-1], 1, bias=False) for k in model.pg_mtr.active_stage_numbers})
    for p in model.retrieval_projs.parameters():
        torch.nn.init.zeros_(p)
    return model


def _module_has_any_grad(module):
    return any(p.grad is not None for p in module.parameters() if p.requires_grad)


def _module_has_no_grad(module):
    return all(p.grad is None for p in module.parameters() if p.requires_grad)


def _assert_existing_grads_finite(module):
    for p in module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_all_stage_activation():
    model = make_model('all')
    assert model.pg_mtr.active_stage_numbers == (1, 2, 3, 4)
    assert set(model.pg_mtr.stage_modules.keys()) == {'1', '2', '3', '4'}
    assert set(model.retrieval_projs.keys()) == {'1', '2', '3', '4'}


def test_retrieved_memory_shape():
    from models.pg_mtr import PETGroundedMetabolicTokenRetrieval
    pg = PETGroundedMetabolicTokenRetrieval([64, 128, 320, 512], stage_mode='all')
    feats = [torch.randn(2, c, 16 // (2 ** i), 16 // (2 ** i)) for i, c in enumerate([64, 128, 320, 512])]
    retrieved, _, diag = pg(feats, mode='missing')
    assert set(retrieved.keys()) == {1, 2, 3, 4}
    for i, mem in retrieved.items():
        assert mem.shape[0] == 2
        assert mem.shape[1] == pg.stage_modules[str(i)].latent_dim
        assert mem.shape[-2:] == feats[i - 1].shape[-2:]
        assert torch.isfinite(mem).all()
    assert torch.isfinite(torch.stack([v for v in diag.values() if torch.is_tensor(v)])).all()


def test_zero_init_retrieval_degenerates_to_ct():
    model = make_model('all')
    ct = [torch.randn(2, c, 16 // (2 ** i), 16 // (2 ** i)) for i, c in enumerate([64, 128, 320, 512])]
    retrieved, _, _ = model.pg_mtr(ct, mode='missing')
    missing_feats = [ct[i] + model.retrieval_projs[str(i + 1)](retrieved[i + 1]) for i in range(4)]
    for a, b in zip(ct, missing_feats):
        assert torch.allclose(a, b, atol=1e-6)


def test_full_auxiliary_loss_gradient_isolation():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all', pg_mtr_num_tokens=8, pg_mtr_temperature=0.07)
    model.zero_grad(set_to_none=True)
    ct = torch.randn(2, 3, 64, 64, requires_grad=True)
    pet = torch.randn(2, 3, 64, 64, requires_grad=True)
    full_out = model._forward_full(ct, pet, (64, 64))
    loss = full_out['aux_losses']['pg_mtr_route_loss'] + full_out['aux_losses']['pg_mtr_mem_loss']
    loss.backward()
    assert _module_has_any_grad(model.pg_mtr)
    assert _module_has_no_grad(model.full_decoder)
    assert _module_has_no_grad(model.missing_decoder)
    assert _module_has_no_grad(model.enc_ct)
    assert _module_has_no_grad(model.enc_pet)
    assert _module_has_no_grad(model.retrieval_projs)
    _assert_existing_grads_finite(model.pg_mtr)


def test_full_total_loss_gradient_routes():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all', pg_mtr_num_tokens=8, pg_mtr_temperature=0.07)
    model.zero_grad(set_to_none=True)
    ct = torch.randn(2, 3, 64, 64, requires_grad=True)
    pet = torch.randn(2, 3, 64, 64, requires_grad=True)
    full_out = model._forward_full(ct, pet, (64, 64))
    seg_loss = full_out['logits'].float().mean()
    route_loss = full_out['aux_losses']['pg_mtr_route_loss']
    mem_loss = full_out['aux_losses']['pg_mtr_mem_loss']
    total_loss = seg_loss + 0.1 * route_loss + 0.05 * mem_loss
    total_loss.backward()
    assert _module_has_any_grad(model.full_decoder)
    assert _module_has_any_grad(model.enc_ct)
    assert _module_has_any_grad(model.enc_pet)
    assert _module_has_any_grad(model.pg_mtr)
    assert _module_has_no_grad(model.missing_decoder)
    assert _module_has_no_grad(model.retrieval_projs)


def test_missing_route_gradient_isolation():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all')
    model.zero_grad(set_to_none=True)
    ct = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = model._forward_missing(ct, (64, 64))
    seg_loss = out['logits'].float().mean()
    seg_loss.backward()
    assert _module_has_any_grad(model.missing_decoder)
    assert _module_has_any_grad(model.enc_ct)
    assert _module_has_any_grad(model.retrieval_projs)
    assert _module_has_no_grad(model.full_decoder)
    assert _module_has_no_grad(model.enc_pet)
    assert _module_has_no_grad(model.pg_mtr)
    assert model.pg_mtr.memory_tokens.grad is None
    assert model.pg_mtr.token_key.weight.grad is None
    assert model.pg_mtr.token_value.weight.grad is None


def test_checkpoint_select_resolver():
    val_full = {'dice': 0.80}
    val_missing = {'dice': 0.70}
    score, name, ckpt = _resolve_checkpoint_selection('full_dice', val_full, val_missing)
    assert score == 0.80 and name == 'full_dice' and ckpt == 'ckpt.best_full_dice.pth.tar'
    score, name, ckpt = _resolve_checkpoint_selection('missing_dice', val_full, val_missing)
    assert score == 0.70 and name == 'missing_dice' and ckpt == 'ckpt.best_missing_dice.pth.tar'
    score, name, ckpt = _resolve_checkpoint_selection('joint_dice', val_full, val_missing)
    assert score == 0.75 and name == 'joint_dice' and ckpt == 'ckpt.best_joint_dice.pth.tar'
    try:
        _resolve_checkpoint_selection('bad', val_full, val_missing)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected ValueError')


def test_pg_mtr_diagnostics_accumulation():
    from tasks.mdt_seg import MDTSegTeacher
    trainer = MDTSegTeacher.__new__(MDTSegTeacher)
    pg_diag_sum, pg_diag_count = {}, {}
    trainer._accumulate_pg_diagnostics(pg_diag_sum, pg_diag_count, {'pg_mtr_s1_ct_route_entropy': 0.8, 'pg_mtr_s2_ct_route_entropy': 0.6})
    trainer._accumulate_pg_diagnostics(pg_diag_sum, pg_diag_count, {'pg_mtr_s1_ct_route_entropy': 0.6, 'pg_mtr_s2_ct_route_entropy': 0.4})
    assert pg_diag_sum['pg_mtr_s1_ct_route_entropy'] / pg_diag_count['pg_mtr_s1_ct_route_entropy'] == 0.7
    assert pg_diag_sum['pg_mtr_s2_ct_route_entropy'] / pg_diag_count['pg_mtr_s2_ct_route_entropy'] == 0.5


def test_csv_field_order_and_missing_fields(tmp_path):
    log_path = tmp_path / 'train_log.csv'
    init_train_log(str(log_path), extra_headers=['metric_a', 'metric_b', 'metric_c'])
    append_epoch_log(str(log_path), 1, 1.0, {'total_loss': 1.0, 'dice': 0.5, 'iou': 0.4, 'acc': 0.3, 'acc_pixel': 0.2, 'hd95': 1.0}, extra_metrics={'metric_c': 3, 'metric_a': 1, 'metric_b': 2})
    with open(log_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert rows[0].index('metric_a') < rows[0].index('metric_b') < rows[0].index('metric_c')
    assert rows[1][rows[0].index('metric_a')] == '1.000000'
    assert rows[1][rows[0].index('metric_b')] == '2.000000'
    assert rows[1][rows[0].index('metric_c')] == '3.000000'
    log_path2 = tmp_path / 'train_log2.csv'
    init_train_log(str(log_path2), extra_headers=['metric_a', 'metric_b'])
    append_epoch_log(str(log_path2), 1, 1.0, {'total_loss': 1.0, 'dice': 0.5, 'iou': 0.4, 'acc': 0.3, 'acc_pixel': 0.2, 'hd95': 1.0}, extra_metrics={'metric_b': 2})
    with open(log_path2, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert rows[1][rows[0].index('metric_a')] == ''
    assert rows[1][rows[0].index('metric_b')] == '2.000000'


def test_missing_injection_diagnostics_and_logits_consistency():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all')
    ct = torch.randn(2, 3, 64, 64)
    out = model._forward_missing(ct, (64, 64))
    diag = out['diagnostics']
    for stage in model.pg_mtr.active_stage_numbers:
        assert f'pg_mtr_s{stage}_ct_rms' in diag
        assert f'pg_mtr_s{stage}_injection_rms' in diag
        assert f'pg_mtr_s{stage}_injection_ct_ratio' in diag
        assert float(diag[f'pg_mtr_s{stage}_injection_rms']) >= 0.0
        assert float(diag[f'pg_mtr_s{stage}_injection_ct_ratio']) >= 0.0
    aligned_ct_feats = model._encode_ct(ct)
    retrieved_memories, _, pg_diag = model.pg_mtr(aligned_ct_feats=aligned_ct_feats, pet_feats=None, mode='missing', detach_bank=True)
    manual_feats, _ = model._apply_retrieved_memory(aligned_ct_feats, retrieved_memories, pg_diag)
    manual_out = model._decode_with(model.missing_decoder, manual_feats, (64, 64))
    assert torch.allclose(out['logits'], manual_out['logits'], atol=1e-6, rtol=1e-5)
    for stage in model.pg_mtr.active_stage_numbers:
        assert float(diag[f'pg_mtr_s{stage}_injection_rms']) >= 0.0


def test_mixed_route_order():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all')
    ct = torch.randn(4, 3, 64, 64)
    pet = torch.randn(4, 3, 64, 64)
    pet_available = torch.tensor([1, 0, 1, 0])
    out = model(ct, pet, pet_available=pet_available, forward_mode='auto')
    assert out['logits'].shape[0] == 4
    assert torch.isfinite(out['logits']).all()
