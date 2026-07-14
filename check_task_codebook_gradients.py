import argparse

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from utils.seg_losses import BCEDiceLoss


def _grad_stats(param):
    if param.grad is None:
        return False, False, 0.0
    grad = param.grad.detach()
    return True, torch.isfinite(grad).all().item(), grad.norm().item()


def _print_param_stats(name, tensor):
    has_grad, grad_finite, grad_norm = _grad_stats(tensor)
    print(f'{name}: has_grad={has_grad} grad_finite={grad_finite} grad_norm={grad_norm:.12e}')


def _make_config(args):
    config = SegMDTConfig()
    config.root = args.root
    config.use_aligned_loader = args.use_aligned_loader
    config.ct_backbone = 'convnextv2_nano'
    config.pet_backbone = 'mit_b1'
    config.ct_pretrained_path = args.ct_pretrained_path
    config.pet_pretrained_path = args.pet_pretrained_path
    config.model_arch = 'dual_decoder_task_codebook_retrieval'
    config.task_codebook_stages = args.task_codebook_stages
    config.task_codebook_num_tokens = args.task_codebook_num_tokens
    config.task_codebook_temperature = args.task_codebook_temperature
    config.use_deep_supervision = False
    config.deep_supervision = False
    config.pg_mtr_route_weight = 0.0
    config.pg_mtr_mem_weight = 0.0
    config.missing_loss_weight = 1.0
    config.batch_size = 2
    config.num_workers = 0
    config.train_mode = 'alternating_full_missing'
    config.train_pet_drop_prob = 0.0
    config.loss_smooth = 1.0
    config.bce_weight = 1.0
    config.dice_weight = 1.0
    config.pos_weight = None
    config.decoder_channels = (512, 256, 128, 64)
    config.gpus = [0] if torch.cuda.is_available() else []
    config.mixed_precision = False
    config.optimizer = 'adamw'
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    return config


def _get_batch(loader):
    batch = next(iter(loader))
    if isinstance(batch, dict):
        return batch['ct'], batch['pet'], batch['mask']
    return batch[0], batch[1], batch[2]


def _module_has_grad(module):
    return any(p.grad is not None for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--use_aligned_loader', action='store_true')
    parser.add_argument('--task_codebook_stages', type=str, default='all')
    parser.add_argument('--task_codebook_num_tokens', type=int, default=8)
    parser.add_argument('--task_codebook_temperature', type=float, default=0.07)
    parser.add_argument('--ct_pretrained_path', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    parser.add_argument('--pet_pretrained_path', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    args = parser.parse_args()

    config = _make_config(args)
    task = build_mdt_seg_teacher(config)['model']
    if hasattr(task, 'train'):
        task.train()
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0, pos_weight=None)
    device = next(task.parameters()).device
    ct = torch.randn(2, 3, 256, 256, device=device)
    pet = torch.randn(2, 3, 256, 256, device=device)
    mask = (torch.rand(2, 1, 256, 256, device=device) > 0.5).float()
    print('[warn] using synthetic batch because dataset loader API is unavailable in this checkout')

    print('=== Missing backward ===')
    task.train()
    task.zero_grad(set_to_none=True)
    out = task(ct=ct, pet=None, forward_mode='missing', target_size=mask.shape[-2:])
    loss, _ = criterion(out['logits'], mask)
    print(f'missing_loss={loss.item():.6f}')
    loss.backward()
    _print_param_stats('task_codebook.shared_codebook_tokens', task.task_codebook.shared_codebook_tokens)
    _print_param_stats('task_codebook.codebook_key.weight', task.task_codebook.codebook_key.weight)
    _print_param_stats('task_codebook.codebook_value.weight', task.task_codebook.codebook_value.weight)
    for s in ('1', '2', '3', '4'):
        if s in task.task_codebook.stage_queries:
            _print_param_stats(f'task_codebook.stage_queries[{s}].proj.weight', task.task_codebook.stage_queries[s].proj.weight)
            _print_param_stats(f'retrieval_adapters[{s}][0].weight', task.retrieval_adapters[s][0].weight)
            _print_param_stats(f'gamma_s{s}', getattr(task, f'gamma_s{s}'))
    _print_param_stats('enc_ct', next(task.enc_ct.parameters()))
    _print_param_stats('ct_align', next(task.ct_align.parameters()))
    _print_param_stats('missing_decoder', next(task.missing_decoder.parameters()))
    print(f'enc_pet has_grad={_module_has_grad(task.enc_pet)}')
    print(f'full_decoder has_grad={_module_has_grad(task.full_decoder)}')

    print('=== Full backward ===')
    task.zero_grad(set_to_none=True)
    out = task(ct=ct, pet=pet, forward_mode='full', target_size=mask.shape[-2:])
    loss, _ = criterion(out['logits'], mask)
    print(f'full_loss={loss.item():.6f}')
    loss.backward()
    print(f'enc_ct has_grad={_module_has_grad(task.enc_ct)}')
    print(f'ct_align has_grad={_module_has_grad(task.ct_align)}')
    print(f'enc_pet has_grad={_module_has_grad(task.enc_pet)}')
    print(f'full_decoder has_grad={_module_has_grad(task.full_decoder)}')
    print(f'task_codebook has_grad={_module_has_grad(task.task_codebook)}')
    print(f'retrieval_adapters has_grad={_module_has_grad(task.retrieval_adapters)}')
    print(f'gamma has_grad={any(getattr(task, f"gamma_s{s}").grad is not None for s in task.task_codebook.active_stage_numbers)}')
    print(f'missing_decoder has_grad={_module_has_grad(task.missing_decoder)}')

    print('=== Parameter update check ===')
    opt = torch.optim.AdamW(task.parameters(), lr=1e-4)
    task.zero_grad(set_to_none=True)
    tokens_before = task.task_codebook.shared_codebook_tokens.detach().clone()
    key_before = task.task_codebook.codebook_key.weight.detach().clone()
    value_before = task.task_codebook.codebook_value.weight.detach().clone()
    out = task(ct=ct, pet=None, forward_mode='missing', target_size=mask.shape[-2:])
    loss, _ = criterion(out['logits'], mask)
    loss.backward()
    opt.step()
    tokens_after = task.task_codebook.shared_codebook_tokens.detach().clone()
    key_after = task.task_codebook.codebook_key.weight.detach().clone()
    value_after = task.task_codebook.codebook_value.weight.detach().clone()
    print(f'token_update={tokens_after.sub(tokens_before).abs().mean().item():.12e}')
    print(f'key_update={key_after.sub(key_before).abs().mean().item():.12e}')
    print(f'value_update={value_after.sub(value_before).abs().mean().item():.12e}')


if __name__ == '__main__':
    main()
