import tempfile

import torch

from models.dual_decoder_ptgc import DualDecoderPTGC
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss
from run_mdt_seg import _resolve_train_route


class MiniConfig:
    bce_weight = 1.0
    dice_weight = 1.0
    loss_smooth = 1.0
    pos_weight = None
    use_deep_supervision = False
    deep_supervision = False
    ptgc_ablation_mode = 'ptgc'
    ptgc_alpha = 0.25
    ptgc_loss_weight = 0.2
    gpnd_rank_weight = 1.0
    gpnd_support_weight = 0.05
    ptgc_delta_active_threshold = 1e-4
    gpus = [0]
    mixed_precision = False
    learning_rate = 1e-4
    decoder_lr = 1e-4
    weight_decay = 0.0
    optimizer = 'adamw'
    freeze_non_adc = False
    model_arch = 'dual_decoder_ptgc'


class TaskShim(MDTSegTeacher):
    def __init__(self, model, config):
        self.networks = {'model': model}
        self.config = config
        self.device = torch.device('cpu')
        self.scaler = None
        self.loss_seg = BCEDiceLoss()
        self.optimizer = None
        self.scheduler = None


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(2023)
    model = DualDecoderPTGC(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        use_deep_supervision=False,
        ptgc_ablation_mode='ptgc',
    )
    B, H, W = 2, 128, 128
    ct = torch.randn(B, 1, H, W)
    pet = torch.randn(B, 1, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.5).float()

    assert _resolve_train_route('dual_decoder_ptgc', 0) == 'full'
    assert _resolve_train_route('dual_decoder_ptgc', 1) == 'full'
    assert _resolve_train_route('dual_decoder_ptgc', 101) == 'full'

    model.train()
    out = model(ct, pet, forward_mode='auto')
    assert out['ptgc_ablation_mode'] == 'ptgc'
    assert out['ptgc_delta_d4'].abs().max().item() < 1e-8
    assert (out['logits'] - out['ptgc_ct_logits']).abs().max().item() < 1e-6

    baseline = DualDecoderPTGC(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, use_deep_supervision=False, ptgc_ablation_mode='baseline')
    base_out = baseline(ct, pet, forward_mode='auto')
    assert 'ptgc_comp_logits' not in base_out
    assert all(not p.requires_grad for p in baseline.ptgc.parameters())
    assert all(p.requires_grad for p in baseline.enc_ct.parameters())
    assert all(p.requires_grad for p in baseline.enc_pet.parameters())
    assert all(p.requires_grad for p in baseline.ct_align.parameters())
    assert all(p.requires_grad for p in baseline.full_decoder.parameters())
    assert all(p.requires_grad for p in baseline.missing_decoder.parameters())

    task = TaskShim(model, MiniConfig())
    loss_total, _, loss_dict = task._compute_total_loss(out, mask)
    assert torch.isfinite(loss_total)
    assert torch.allclose(loss_total, (loss_dict['loss_ptgc_ct'] + loss_dict['loss_ptgc_full'] + loss_dict['loss_ptgc_comp']) / 3.0, atol=1e-6, rtol=1e-6)

    task.config.ptgc_ablation_mode = 'baseline'
    base_loss_total, _, base_loss_dict = task._compute_total_loss(base_out, mask)
    assert torch.allclose(base_loss_total, 0.5 * (base_loss_dict['loss_ptgc_ct'] + base_loss_dict['loss_ptgc_full']), atol=1e-6, rtol=1e-6)
    assert base_loss_dict['loss_ptgc_comp'].abs().item() == 0.0

    task.config.ptgc_ablation_mode = 'ptgc_gpnd'
    gpnd_out = model(ct, pet, forward_mode='auto')
    gpnd_loss_total, _, gpnd_loss_dict = task._compute_total_loss(gpnd_out, mask)
    assert 'weighted_loss_gpnd' in gpnd_loss_dict
    assert torch.isfinite(gpnd_loss_total)

    model.zero_grad(set_to_none=True)
    loss_total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.enc_ct.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.enc_pet.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.ct_align.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.full_decoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.missing_decoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.ptgc.parameters())

    model.eval()
    with torch.no_grad():
        full_eval = model(ct, pet, forward_mode='full')
        missing_eval = model(ct, None, forward_mode='missing')
    assert torch.isfinite(full_eval['logits']).all()
    assert torch.isfinite(missing_eval['logits']).all()

    print('check_ptgc_module passed')
    print(f'ptgc_params={_count_params(model.ptgc)}')


if __name__ == '__main__':
    main()
