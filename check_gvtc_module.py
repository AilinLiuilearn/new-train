import torch

from models.dual_decoder_ptgc import DualDecoderPTGC
from models.gvtc_module import GainVerifiedVirtualTaskCompensation, sparsemax
from run_mdt_seg import _resolve_train_route
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss


class MiniConfig:
    bce_weight = 1.0
    dice_weight = 1.0
    loss_smooth = 1.0
    pos_weight = None
    use_deep_supervision = False
    deep_supervision = False
    ptgc_ablation_mode = 'gvtc_pgmr'
    pgmr_weight = 0.1
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


def main():
    torch.manual_seed(2023)
    z = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, -2.0]], requires_grad=True)
    y = sparsemax(z, dim=-1)
    assert (y >= 0).all()
    assert torch.allclose(y.sum(dim=-1), torch.ones(y.size(0)), atol=1e-6)
    assert (y == 0).any()
    y.sum().backward()
    assert torch.isfinite(z.grad).all()

    model = DualDecoderPTGC(ptgc_ablation_mode='gvtc_pgmr', ct_pretrained_path=None, pet_pretrained_path=None, use_deep_supervision=False)
    B, H, W = 2, 128, 128
    ct = torch.randn(B, 1, H, W)
    pet = torch.randn(B, 1, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.5).float()

    assert _resolve_train_route('dual_decoder_ptgc', 0) == 'full'
    model.train()
    out = model(ct, pet, forward_mode='auto')
    assert out['gvtc_delta_d4'].abs().max().item() < 1e-8
    assert (out['gvtc_comp_logits'] - out['gvtc_ct_logits']).abs().max().item() < 1e-6
    assert out['gvtc_routing_weights'].shape[-1] == 9
    assert (out['gvtc_routing_weights'] >= 0).all()
    assert torch.allclose(out['gvtc_routing_weights'].sum(dim=-1), torch.ones_like(out['gvtc_routing_weights'][..., 0]), atol=1e-5)

    gvtc = GainVerifiedVirtualTaskCompensation(task_channels=512)
    assert not any(n.startswith('null') for n, _ in gvtc.named_parameters())
    null_routing = torch.zeros_like(out['gvtc_routing_weights']); null_routing[..., 0] = 1
    assert torch.allclose(null_routing[..., 0], torch.ones_like(null_routing[..., 0]))

    task = TaskShim(model, MiniConfig())
    task.config.ptgc_ablation_mode = 'gvtc'
    loss_total, pred, loss_dict = task._compute_total_loss(out, mask)
    assert pred is out['logits']
    assert torch.allclose(loss_total, 0.5 * loss_dict['loss_gvtc_full'] + 0.5 * loss_dict['loss_gvtc_comp'], atol=1e-6, rtol=1e-6)

    task.config.ptgc_ablation_mode = 'gvtc_pgmr'
    loss_pgmr, pgmr_stats = task._compute_pgmr_loss(out, mask)
    assert torch.isfinite(loss_pgmr)
    assert 'pgmr_violation_ratio' in pgmr_stats

    model.zero_grad(set_to_none=True)
    loss_total.backward()
    assert model.gvtc.output_projection.weight.grad is not None
    assert torch.isfinite(model.gvtc.output_projection.weight.grad).all()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    model.zero_grad(set_to_none=True)
    out2 = model(ct, pet, forward_mode='auto')
    loss2, _, _ = task._compute_total_loss(out2, mask)
    loss2.backward()
    assert model.gvtc.local_query_projection.weight.grad is not None
    assert torch.isfinite(model.gvtc.local_query_projection.weight.grad).all()
    assert model.gvtc.region_query_projection.weight.grad is not None
    assert torch.isfinite(model.gvtc.region_query_projection.weight.grad).all()
    assert model.gvtc.operator_down.grad is not None
    assert torch.isfinite(model.gvtc.operator_down.grad).all()
    assert model.gvtc.operator_up.grad is not None
    assert torch.isfinite(model.gvtc.operator_up.grad).all()
    assert model.gvtc.output_projection.weight.grad is not None
    assert torch.isfinite(model.gvtc.output_projection.weight.grad).all()

    model.eval()
    with torch.no_grad():
        missing = model(ct, None, forward_mode='missing')
    assert torch.isfinite(missing['logits']).all()
    print('check_gvtc_module passed')


if __name__ == '__main__':
    main()
