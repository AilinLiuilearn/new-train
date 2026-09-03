import torch
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'pet_recon_weight': 0.1,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _make_inputs(batch_size=2, size=64):
    ct = torch.randn(batch_size, 1, size, size)
    pet = torch.randn(batch_size, 1, size, size)
    mask = torch.zeros(batch_size, 1, size, size)
    return ct, pet, mask


def _mark_bank_ready(model):
    memory = model.prototype_memory
    memory.prototype_ready.fill_(True)
    for scale_idx in range(memory.num_scales):
        keys = getattr(memory, f'ct_keys_s{scale_idx + 1}')
        values = getattr(memory, f'pet_values_s{scale_idx + 1}')
        keys.data.normal_()
        values.data.normal_()


def _forward_missing_with_recon(model, ct, pet, mask):
    model.train()
    ct_feats = model._encode_ct(ct)
    pet_feats_real = model._encode_pet(pet)
    model._collect_cppi(ct_feats, pet_feats_real, mask)
    pet_feats_proxy, _, _ = model._retrieve_cppi(
        ct_feats,
        compute_report=False,
        save_diagnostics=False,
        print_info=False,
        return_ct_reference=True,
    )
    pet_recon_losses = [
        F.mse_loss(proxy.float(), real.detach().float(), reduction='mean')
        for proxy, real in zip(pet_feats_proxy, pet_feats_real)
    ]
    pet_recon_loss = torch.stack(pet_recon_losses).mean()
    return pet_recon_loss, pet_recon_losses, pet_feats_proxy, pet_feats_real


def test_missing_bank_ready_pet_recon_loss():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    _mark_bank_ready(model)
    ct, pet, mask = _make_inputs()
    out = model(ct, pet, forward_mode='missing', mask=mask)
    aux = out['aux']
    pet_recon_loss = aux['pet_recon_loss']
    assert pet_recon_loss.ndim == 0
    assert torch.isfinite(pet_recon_loss)
    assert float(pet_recon_loss) >= 0.0
    for scale_idx in range(1, 5):
        scale_loss = aux[f'pet_recon_s{scale_idx}']
        assert scale_loss.ndim == 0
        assert torch.isfinite(scale_loss)


def test_full_route_excludes_pet_recon_loss():
    task = MDTSegTeacher(
        {'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)},
        _make_cfg(),
    )
    ct, pet, mask = _make_inputs(batch_size=1, size=64)
    batch = {'ct': ct, 'pet': pet, 'mask': mask}
    loss, _, _, stats = task.train_step(batch, forward_mode='full')
    assert torch.allclose(loss, stats['loss_seg'])
    assert float(stats['loss_pet_recon']) == 0.0


def test_bank_not_ready_pet_recon_loss_is_zero():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    assert not model.prototype_memory.bank_ready
    ct, pet, mask = _make_inputs()
    out = model(ct, pet, forward_mode='missing', mask=mask)
    aux = out['aux']
    assert float(aux['pet_recon_loss']) == 0.0
    for scale_idx in range(1, 5):
        assert float(aux[f'pet_recon_s{scale_idx}']) == 0.0


def test_real_pet_target_is_stop_gradient():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    _mark_bank_ready(model)
    ct, pet, mask = _make_inputs(batch_size=1, size=64)
    model.zero_grad(set_to_none=True)
    pet_recon_loss, _, _, _ = _forward_missing_with_recon(model, ct, pet, mask)
    pet_recon_loss.backward()
    pet_encoder_grads = [
        p.grad for p in model.enc_pet.parameters() if p.requires_grad
    ]
    assert all(g is None for g in pet_encoder_grads)


def test_proxy_path_receives_gradients():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    _mark_bank_ready(model)
    ct, pet, mask = _make_inputs(batch_size=1, size=64)
    model.zero_grad(set_to_none=True)
    pet_recon_loss, _, _, _ = _forward_missing_with_recon(model, ct, pet, mask)
    pet_recon_loss.backward()
    attn = model.prototype_memory.attention[0]
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        module = getattr(attn, name)
        weight = module.weight
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0.0


def test_output_shapes_match_baseline_contract():
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False)
    _mark_bank_ready(model)
    ct = torch.randn(16, 1, 512, 512)
    pet = torch.randn(16, 1, 512, 512)
    mask = torch.zeros(16, 1, 512, 512)
    out_full = model(ct, pet, forward_mode='full', mask=mask)
    out_missing = model(ct, pet, forward_mode='missing', mask=mask)
    assert out_full['logits'].shape == (16, 1, 512, 512)
    assert out_missing['logits'].shape == (16, 1, 512, 512)
    assert out_full['logits'].shape == out_missing['logits'].shape
    assert 'aux' in out_missing
    assert 'pet_recon_loss' in out_missing['aux']
