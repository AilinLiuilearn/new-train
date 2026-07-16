import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.dual_decoder_add_baseline import DualDecoderAddPETCTBaseline
from models.dual_decoder_ptgc import DualDecoderPTGC
from utils.seg_losses import BCEDiceLoss


class MiniConfig:
    bce_weight = 1.0
    dice_weight = 1.0
    loss_smooth = 1.0
    pos_weight = None
    use_deep_supervision = False
    deep_supervision = False
    ptgc_alpha = 0.25
    gpnd_rank_weight = 1.0
    gpnd_support_weight = 0.05
    ptgc_delta_active_threshold = 1e-4
    use_gpnd = False
    model_arch = 'dual_decoder_ptgc'
    gpus = [0]
    mixed_precision = False
    learning_rate = 1e-4
    weight_decay = 0.0
    optimizer = 'adamw'
    decoder_lr = 1e-4
    freeze_non_adc = False


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _max_abs(x):
    return float(x.detach().abs().max().cpu())


def main():
    torch.manual_seed(2023)
    base = DualDecoderAddPETCTBaseline(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, use_deep_supervision=False)
    base_ckpt = tempfile.NamedTemporaryFile(suffix='.pth.tar', delete=False)
    base_ckpt.close()
    torch.save({'model': base.state_dict()}, base_ckpt.name)

    model = DualDecoderPTGC(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        use_deep_supervision=False,
        ptgc_base_checkpoint=base_ckpt.name,
        use_gpnd=False,
    )

    B, H, W = 2, 128, 128
    ct = torch.randn(B, 1, H, W)
    pet = torch.randn(B, 1, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.5).float()

    model.train()
    out_full = model(ct, pet, forward_mode='full')
    out_missing = model(ct, None, forward_mode='missing')

    assert out_full['logits'].shape == mask.shape
    assert out_missing['logits'].shape == mask.shape
    assert out_missing['ptgc_delta_d4'].shape[-2:] == out_missing['ptgc_gain_pred'].shape[-2:]
    assert out_missing['ptgc_gain_pred'].shape == out_missing['ptgc_benefit_pred'].shape

    with torch.no_grad():
        ct_only = base(ct, pet, forward_mode='missing')['logits']
    delta_abs = _max_abs(out_missing['ptgc_delta_d4'])
    comp_err = float((out_missing['logits'] - ct_only).abs().mean().item())

    assert all(p.requires_grad for p in model.ptgc.parameters())
    frozen_ok = all(not p.requires_grad for m in (model.enc_ct, model.enc_pet, model.ct_align, model.full_decoder, model.missing_decoder) for p in m.parameters())
    assert frozen_ok
    assert not model.enc_ct.training and not model.enc_pet.training and not model.ct_align.training and not model.full_decoder.training and not model.missing_decoder.training

    loss_seg, _ = BCEDiceLoss()(out_missing['logits'], mask)

    class TaskShim:
        def __init__(self, model):
            self.networks = {'model': model}
            self.config = MiniConfig()
            self.loss_seg = BCEDiceLoss()

        def _compute_segmentation_loss(self, outputs, mask):
            pred = outputs['logits']
            loss_seg, stats = self.loss_seg(pred, mask)
            stats = dict(stats)
            stats['loss_seg'] = loss_seg.detach()
            return loss_seg, pred, stats

        def _compute_gpnd_loss(self, outputs, mask):
            from tasks.mdt_seg import MDTSegTeacher as _Task
            return _Task._compute_gpnd_loss(self, outputs, mask)

        def _compute_total_loss(self, outputs, mask):
            from tasks.mdt_seg import MDTSegTeacher as _Task
            return _Task._compute_total_loss(self, outputs, mask)

    task = TaskShim(model)
    loss_total, _, loss_dict = task._compute_total_loss(out_missing, mask)
    assert torch.allclose(loss_total, loss_seg, atol=1e-6, rtol=1e-6)
    assert torch.allclose(loss_dict['loss_total'], loss_seg.detach(), atol=1e-6, rtol=1e-6)

    task.config.use_gpnd = True
    loss_total_gpnd, _, loss_dict_gpnd = task._compute_total_loss(out_full, mask)
    assert 'weighted_loss_gpnd' in loss_dict_gpnd
    assert torch.isfinite(loss_total_gpnd)

    loss = loss_total_gpnd
    model.zero_grad(set_to_none=True)
    loss.backward()
    ptgc_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.ptgc.parameters())
    assert ptgc_grad_ok
    assert all(p.grad is None for p in model.enc_ct.parameters())
    assert all(p.grad is None for p in model.enc_pet.parameters())
    assert all(p.grad is None for p in model.ct_align.parameters())
    assert all(p.grad is None for p in model.full_decoder.parameters())
    assert all(p.grad is None for p in model.missing_decoder.parameters())

    full_eval = model(ct, pet, forward_mode='full')
    assert 'logits' in full_eval and full_eval['logits'].shape == mask.shape
    missing_eval = model(ct, None, forward_mode='missing')
    assert 'logits' in missing_eval and missing_eval['logits'].shape == mask.shape
    assert torch.isfinite(full_eval['logits']).all() and torch.isfinite(missing_eval['logits']).all()

    print('check_ptgc_module passed')
    print(f'initial_delta_max_abs={delta_abs:.6g}')
    print(f'initial_comp_ct_mae={comp_err:.6g}')
    print(f'ptgc_params={_count_params(model.ptgc)}')
    print(f'frozen_base_params={_count_params(model.enc_ct)+_count_params(model.enc_pet)+_count_params(model.ct_align)+_count_params(model.full_decoder)+_count_params(model.missing_decoder)}')
    os.unlink(base_ckpt.name)


if __name__ == '__main__':
    main()
