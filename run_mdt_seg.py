# import os
# import sys
# import json
# import torch
# import numpy as np

# if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.getcwd())

# from configs.seg_mdt import SegMDTConfig
# from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
# from models.build_mdt_seg import build_mdt_seg_teacher
# from tasks.mdt_seg import MDTSegTeacher
# from utils.optimization import get_cosine_scheduler
# from utils.train_logger import init_train_log, append_epoch_log
# from utils.model_profile import print_baseline_profile


# def main():
#     config = SegMDTConfig.parse_arguments()
#     config.task = 'MDT_Teacher'

#     g0 = int(config.gpus[0]) if config.gpus else 0
#     config.gpus = [g0]
#     if torch.cuda.is_available():
#         torch.cuda.set_device(g0)

#     np.random.seed(config.random_state)
#     torch.manual_seed(config.random_state)
#     torch.cuda.manual_seed_all(config.random_state)

#     print('GPU={} backbone={} single_modality={}'.format(g0, config.backbone, getattr(config, 'single_modality', False)))
#     print('lr={} wd={} bs={}'.format(config.learning_rate, config.weight_decay, config.batch_size))

#     os.makedirs(config.checkpoint_dir, exist_ok=True)
#     with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
#         json.dump(vars(config), f, indent=4)

#     if getattr(config, 'cipa_aligned', False):
#         train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
#             config.root, config.image_size_2d, config.batch_size,
#             config.num_workers, config.random_state, getattr(config, 'aug_strong', False))
#     else:
#         train_loader, val_loader, test_loader = get_pclt20k_loaders(
#             config.root, config.image_size_2d, config.batch_size,
#             config.num_workers, val_ratio=config.val_ratio,
#             random_state=config.random_state,
#             use_case_split=getattr(config, 'use_case_split', True),
#             aug_strong=getattr(config, 'aug_strong', False))

#     networks = build_mdt_seg_teacher(config)
#     print('\n' + '=' * 30 + ' MODEL PROFILE ' + '=' * 30)
#     print_baseline_profile(networks, config)
#     print('=' * 75 + '\n')

#     task = MDTSegTeacher(networks, config)
#     spe = len(train_loader)
#     task.scheduler = get_cosine_scheduler(
#         task.optimizer, config.epochs,
#         warmup_steps=config.cosine_warmup * spe,
#         min_lr=config.cosine_min_lr, steps_per_epoch=spe)

#     log_path = os.path.join(config.checkpoint_dir, 'train_log.csv')
#     init_train_log(log_path)

#     grad_clip = getattr(config, 'grad_clip', 5.0)
#     clip_params = [p for net in task.networks.values() for p in net.parameters()]

#     best_dice, best_epoch, no_improve = -1.0, 0, 0
#     patience = getattr(config, 'early_stop_patience', 15)

#     for epoch in range(1, config.epochs + 1):
#         tloss, tn = 0.0, 0
#         for i, batch in enumerate(train_loader):
#             task.optimizer.zero_grad()
#             with torch.amp.autocast('cuda', enabled=config.mixed_precision):
#                 loss, _, _, ld = task.train_step(batch)

#             if task.scaler:
#                 task.scaler.scale(loss).backward()
#                 if grad_clip > 0:
#                     task.scaler.unscale_(task.optimizer)
#                     torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
#                 task.scaler.step(task.optimizer)
#                 task.scaler.update()
#             else:
#                 loss.backward()
#                 if grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
#                 task.optimizer.step()

#             if task.scheduler:
#                 task.scheduler.step()

#             tloss += loss.item()
#             tn += 1

#             if (i + 1) % 50 == 0:
#                 print('  Ep{}[{}/{}] loss={:.4f} seg={:.4f}'.format(
#                     epoch, i + 1, spe, loss.item(), ld['loss_seg'].item()))

#         val_m = task.evaluate(val_loader)
#         append_epoch_log(log_path, epoch, tloss / max(tn, 1), val_m)
#         print('Epoch {} loss={:.4f} Dice={:.4f} IoU={:.4f} HD95={:.2f}'.format(
#             epoch, tloss / max(tn, 1), val_m['dice'], val_m['iou'], val_m['hd95']))

#         if val_m['dice'] > best_dice:
#             best_dice, best_epoch, no_improve = val_m['dice'], epoch, 0
#             task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
#         else:
#             no_improve += 1

#         if patience > 0 and no_improve >= patience:
#             print('Early stop at epoch', epoch)
#             break

#     task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)

#     ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
#     for k, v in task.networks.items():
#         if k in ckpt:
#             v.load_state_dict(ckpt[k], strict=False)

#     test_m = task.evaluate(test_loader)
#     print('\n=== TEST Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
#         test_m['dice'], test_m['iou'], test_m['acc'], test_m['hd95']))

#     with open(os.path.join(config.checkpoint_dir, 'test_results.json'), 'w') as f:
#         json.dump({k: float(v) for k, v in test_m.items()}, f, indent=2)

#     with open(os.path.join(config.checkpoint_dir, 'summary.json'), 'w') as f:
#         summary = {
#             'best_epoch': int(best_epoch),
#             'best_val_dice': float(best_dice),
#             'test_dice': float(test_m['dice']),
#             'test_hd95': float(test_m['hd95']),
#             'final_epoch': int(epoch),
#         }
#         json.dump(summary, f, indent=4)

#     print('Best epoch:', best_epoch, ' Val Dice:', round(best_dice, 4))


# if __name__ == '__main__':
#     main()


# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import numpy as np

# 强制开启显存碎片优化，防止 512x512 时的 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.train_logger import init_train_log, append_epoch_log
from utils.model_profile import print_baseline_profile

def main():
    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT_Teacher'

    # GPU 环境初始化
    g0 = int(config.gpus[0]) if config.gpus else 0
    config.gpus = [g0]
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)

    # 随机种子设定
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed_all(config.random_state)

    print('GPU={} backbone={} single_modality={}'.format(g0, config.backbone, getattr(config, 'single_modality', False)))
    print('lr={} wd={} bs={}'.format(config.learning_rate, config.weight_decay, config.batch_size))

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    # 数据集加载
    if getattr(config, 'cipa_aligned', False):
        train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
            config.root, config.image_size_2d, config.batch_size,
            config.num_workers, config.random_state, getattr(config, 'aug_strong', False))
    else:
        train_loader, val_loader, test_loader = get_pclt20k_loaders(
            config.root, config.image_size_2d, config.batch_size,
            config.num_workers, val_ratio=config.val_ratio,
            random_state=config.random_state,
            use_case_split=getattr(config, 'use_case_split', True),
            aug_strong=getattr(config, 'aug_strong', False))

    # 网络构建与参数概览
    networks = build_mdt_seg_teacher(config)
    print('\n' + '=' * 30 + ' MODEL PROFILE ' + '=' * 30)
    print_baseline_profile(networks, config)
    print('=' * 75 + '\n')

    # 训练任务初始化
    task = MDTSegTeacher(networks, config)
    
    # 学习率调度器配置
    spe = len(train_loader)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, config.epochs,
        warmup_steps=config.cosine_warmup * spe,
        min_lr=config.cosine_min_lr, steps_per_epoch=spe)

    # 日志初始化
    log_path = os.path.join(config.checkpoint_dir, 'train_log.csv')
    init_train_log(log_path)

    # 梯度剪切配置
    grad_clip = getattr(config, 'grad_clip', 5.0)
    clip_params = [p for net in task.networks.values() for p in net.parameters()]

    # 训练策略：梯度累加步数（针对 bs=4 拟合 bs=16）
    accum_iter = getattr(config, 'accumulation_steps', 1) 
    best_dice, best_epoch, no_improve = -1.0, 0, 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tn = 0.0, 0
        task.optimizer.zero_grad() # 在 epoch 开始时清零

        for i, batch in enumerate(train_loader):
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                loss, _, _, ld = task.train_step(batch)
                # 如果使用梯度累加，则对 loss 进行缩放
                loss = loss / accum_iter 

            # 反向传播
            if task.scaler:
                task.scaler.scale(loss).backward()
                # 梯度累加逻辑
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        task.scaler.unscale_(task.optimizer)
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    task.optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.optimizer.step()
                    task.optimizer.zero_grad()

            # --- 核心修正：Scheduler 必须在 Optimizer 之后 step ---
            if task.scheduler:
                task.scheduler.step()

            tloss += loss.item() * accum_iter
            tn += 1

            if (i + 1) % 50 == 0:
                curr_lr = task.optimizer.param_groups[0]['lr']
                print(' Ep{}[{}/{}] loss={:.4f} seg={:.4f} lr={:.6f}'.format(
                    epoch, i + 1, spe, loss.item() * accum_iter, ld['loss_seg'].item(), curr_lr))

        # 验证与早停逻辑
        val_m = task.evaluate(val_loader)
        append_epoch_log(log_path, epoch, tloss / max(tn, 1), val_m)
        print('Epoch {} loss={:.4f} Dice={:.4f} IoU={:.4f} HD95={:.2f}'.format(
            epoch, tloss / max(tn, 1), val_m['dice'], val_m['iou'], val_m['hd95']))

        if val_m['dice'] > best_dice:
            best_dice, best_epoch, no_improve = val_m['dice'], epoch, 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print('Early stop at epoch', epoch)
            break

    # 最终测试
    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)
    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            # 兼容性处理，支持 strict=False
            v.load_state_dict(ckpt[k], strict=False)

    test_m = task.evaluate(test_loader)
    print('\n=== TEST Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m['dice'], test_m['iou'], test_m['acc'], test_m['hd95']))

    # 结果持久化
    with open(os.path.join(config.checkpoint_dir, 'test_results.json'), 'w') as f:
        json.dump({k: float(v) for k, v in test_m.items()}, f, indent=2)

    with open(os.path.join(config.checkpoint_dir, 'summary.json'), 'w') as f:
        summary = {
            'best_epoch': int(best_epoch),
            'best_val_dice': float(best_dice),
            'test_dice': float(test_m['dice']),
            'test_hd95': float(test_m['hd95']),
            'final_epoch': int(epoch),
        }
        json.dump(summary, f, indent=4)

    print('Best epoch:', best_epoch, ' Val Dice:', round(best_dice, 4))

if __name__ == '__main__':
    main()

    