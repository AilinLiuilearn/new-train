# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from models.build_mdt_seg import build_mdt_seg_teacher
# from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
# from configs.seg_mdt import SegMDTConfig

# def visualize_results():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root', type=str, required=True)
#     parser.add_argument('--ckpt_path', type=str, required=True)
#     parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
#     parser.add_argument('--save_dir', type=str, default='vis_results')
#     parser.add_argument('--num_samples', type=int, default=15)
#     parser.add_argument('--cipa_aligned', type=str, default='False') # 接收字符串
#     args, _ = parser.parse_known_args()

#     # 1. 参数预处理
#     cipa_aligned = args.cipa_aligned.lower() == 'true'
    
#     # 2. 初始化配置
#     config = SegMDTConfig()
#     config.root = args.root
#     config.backbone = args.backbone
    
#     # 【核心修正】：强制将列表 [512, 512] 转为 整数 512
#     # 这样在 dataset 内部 (512, 512) 才是合法的
#     raw_size = [512, 512] 
#     target_size = raw_size[0] if isinstance(raw_size, list) else raw_size
#     config.image_size_2d = [target_size, target_size]

#     # 3. 加载对应的数据集
#     print(f"正在加载数据集 (cipa_aligned={cipa_aligned})...")
#     if cipa_aligned:
#         _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
#             config.root, target_size, batch_size=1, num_workers=0, random_state=42)
#     else:
#         _, _, test_loader = get_pclt20k_loaders(
#             config.root, target_size, batch_size=1, num_workers=0, random_state=42)

#     # 4. 构建模型
#     networks = build_mdt_seg_teacher(config)
#     model = networks['model'].cuda()
    
#     # 5. 加载权重
#     print(f"加载权重自: {args.ckpt_path}")
#     ckpt = torch.load(args.ckpt_path, map_location='cuda')
#     state_dict = ckpt['model'] if 'model' in ckpt else ckpt
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()

#     # 6. 生成可视化 (四列对比布局)
#     os.makedirs(args.save_dir, exist_ok=True)
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             if i >= args.num_samples: break
            
#             ct = batch['ct'].cuda()
#             pet = batch['pet'].cuda()
#             gt = batch['mask'].cpu().numpy()[0, 0]
            
#             # 推理
#             preds = model(ct, pet)
#             # 处理多阶段输出，取最高分辨率的那一个
#             pred_raw = preds[0] if isinstance(preds, (list, tuple)) else preds
#             pred_prob = torch.sigmoid(pred_raw).cpu().numpy()[0, 0]
#             pred_binary = (pred_prob > 0.5).astype(np.float32)

#             # 绘图
#             fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            
#             # CT 灰度图
#             axes[0].imshow(ct.cpu().numpy()[0, 0], cmap='gray')
#             axes[0].set_title("CT (Anatomy)")
            
#             # PET 热力图
#             axes[1].imshow(pet.cpu().numpy()[0, 0], cmap='hot')
#             axes[1].set_title("PET (Metabolism)")
            
#             # GT
#             axes[2].imshow(gt, cmap='gray')
#             axes[2].set_title("Ground Truth")
            
#             # Prediction
#             axes[3].imshow(pred_binary, cmap='jet')
#             axes[3].set_title("Prediction")

#             for ax in axes: ax.axis('off')
#             plt.savefig(os.path.join(args.save_dir, f'sample_{i}.png'), bbox_inches='tight')
#             plt.close()
#             print(f"Saved: sample_{i}.png")

# if __name__ == '__main__':
#     visualize_results()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models.build_mdt_seg import build_mdt_seg_teacher
from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
from configs.seg_mdt import SegMDTConfig

def visualize_diagnostic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--save_dir', type=str, default='vis_diagnostic')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--cipa_aligned', type=str, default='False')
    args, _ = parser.parse_known_args()

    cipa_aligned = args.cipa_aligned.lower() == 'true'
    config = SegMDTConfig()
    config.root = args.root
    config.backbone = args.backbone
    target_size = 512 
    config.image_size_2d = [target_size, target_size]

    # 1. 加载数据集
    if cipa_aligned:
        _, _, test_loader = get_pclt20k_loaders_cipa_aligned(config.root, target_size, batch_size=1, num_workers=0)
    else:
        _, _, test_loader = get_pclt20k_loaders(config.root, target_size, batch_size=1, num_workers=0)

    # 2. 加载模型
    networks = build_mdt_seg_teacher(config)
    model = networks['model'].cuda()
    ckpt = torch.load(args.ckpt_path, map_location='cuda')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if count >= args.num_samples: break
            
            gt = batch['mask'].cpu().numpy()[0, 0]
            # 【筛选逻辑】：只看有病灶的切片，否则根因分析没意义
            if np.sum(gt) == 0: continue 

            ct = batch['ct'].cuda()
            pet = batch['pet'].cuda()
            
            # 推理
            preds = model(ct, pet)
            pred_raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            prob = torch.sigmoid(pred_raw).cpu().numpy()[0, 0]
            pred_binary = (prob > 0.5).astype(np.float32)

            # --- 核心：计算诊断误差图 ---
            # 绿色 (TP): 预测正确
            # 红色 (FN): 漏检（GT有，Pred没有）
            # 蓝色 (FP): 误检（GT没有，Pred有）
            error_map = np.zeros((target_size, target_size, 3))
            error_map[..., 1] = (gt * pred_binary) # TP -> Green
            error_map[..., 0] = (gt * (1 - pred_binary)) # FN -> Red (漏检)
            error_map[..., 2] = ((1 - gt) * pred_binary) # FP -> Blue (误检)

            # 3. 绘图 (五列布局)
            fig, axes = plt.subplots(1, 5, figsize=(30, 6))
            
            # 1: CT
            axes[0].imshow(ct.cpu().numpy()[0, 0], cmap='gray')
            axes[0].set_title("CT (Anatomy)", fontsize=15)
            
            # 2: PET
            axes[1].imshow(pet.cpu().numpy()[0, 0], cmap='hot')
            axes[1].set_title("PET (Position)", fontsize=15)
            
            # 3: 置信度热力图 (看模型是不是“虚弱”)
            im3 = axes[2].imshow(prob, cmap='jet')
            axes[2].set_title("Confidence Heatmap", fontsize=15)
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 4: 真值轮廓
            axes[3].imshow(ct.cpu().numpy()[0, 0], cmap='gray')
            axes[3].contour(gt, colors='yellow', linewidths=1) # 黄色线是GT
            axes[3].set_title("Yellow: Ground Truth", fontsize=15)
            
            # 5: 误差诊断图 (红色代表漏检，最关键！)
            axes[4].imshow(error_map)
            axes[4].set_title("Red: FN (Missed) | Blue: FP", fontsize=15)

            for ax in axes: ax.axis('off')
            plt.savefig(os.path.join(args.save_dir, f'case_{i}_diag.png'), bbox_inches='tight', dpi=150)
            plt.close()
            count += 1
            print(f"已生成第 {count} 个诊断样本: case_{i}")

if __name__ == '__main__':
    visualize_diagnostic()