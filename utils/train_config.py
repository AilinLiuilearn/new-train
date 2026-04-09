# -*- coding: utf-8 -*-
"""
训练配置增强：将模型结构、计算成本、数据划分、测试结果等写入 configs.json，便于一目了然。
"""

import os
import copy
import json
from utils.model_profile import count_params_m, get_flops_params_gm


def _get_flops_params(networks, config, is_teacher=True):
    """获取 FLOPs (G) 和 Params (M)"""
    import torch
    params_m = count_params_m(networks)
    flops_g = None
    image_size = getattr(config, 'image_size_2d', 512)
    device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
    use_specific = getattr(config, 'use_specific', True)
    use_cmx = getattr(config, 'use_cmx', True)
    try:
        from utils.model_profile import TeacherSegWrapper, StudentSegWrapper
        nets_copy = {k: copy.deepcopy(v) for k, v in networks.items()}
        if is_teacher:
            wrapper = TeacherSegWrapper(nets_copy, use_cmx).to(device).eval()
            ct = torch.randn(1, 1, image_size, image_size, device=device)
            pet = torch.randn(1, 1, image_size, image_size, device=device)
            with torch.no_grad():
                flops_g, _ = get_flops_params_gm(wrapper, (ct, pet))
        else:
            wrapper = StudentSegWrapper(nets_copy, use_specific).to(device).eval()
            ct = torch.randn(1, 1, image_size, image_size, device=device)
            with torch.no_grad():
                flops_g, _ = get_flops_params_gm(wrapper, (ct,))
    except Exception:
        pass
    return params_m, flops_g


def build_model_summary(task_name, config, is_teacher=True, teacher_backbone=None):
    """构建模型结构描述，便于 configs.json 中一目了然"""
    backbone = getattr(config, 'backbone', 'unknown')
    use_cmx = getattr(config, 'use_cmx', True)
    use_specific = getattr(config, 'use_specific', True)
    use_projector = getattr(config, 'use_projector', True)
    hidden = getattr(config, 'hidden', 256)
    fpn_out = getattr(config, 'fpn_out_channels', 256)

    lines = []
    if task_name == 'MDT':
        td = getattr(config, 'teacher_decoder', 'mlp')
        if td == 'unet':
            dec_name = 'UNet 解码'
        elif td == 'fpn':
            dec_name = 'FPN 解码（Semantic FPN）'
        else:
            dec_name = 'SegFormerMLP 解码（四尺度嵌入+融合，配 MiT）'
        lines.append(f"【教师】双流 CT+PET → 浅层 CT-anchor 非对称融合 → 深层 UMSD(+小波) → {dec_name}")
        lines.append(f"  - Backbone×2: {backbone}（CT / PET 独立权重）")
        lines.append(f"  - Projector 1×1: {hidden}ch" if use_projector else "  - 无 Projector")
        fpn_do = getattr(config, 'fpn_dropout', 0.0)
        shallow = getattr(config, "shallow_fusion", "ct_anchor")
        if shallow == "hl_cim":
            lines.append("  - 浅层: HLMamba 启发 CIM 桥 + 选择性空间混合 ×3（hl_cim；医学适配，非官方复现）")
        else:
            lines.append("  - 浅层: ShallowFusionCTAnchorPETtoCT ×3（非对称 CRM + 轻量 PET→CT FFM）")
        lines.append("  - 深层: UMSD（仅最后一层）+ 可选并行 Haar 小波")
        lines.append(f"  - 分割头 ({td}): out={fpn_out}, dropout={fpn_do}")
    elif task_name == 'MDT_Student':
        lines.append("【学生】单流 CT → 双头残差解耦 → FPN 分割（对齐教师 CT 表征）")
        lines.append(f"  - Backbone: {backbone}")
        if teacher_backbone and teacher_backbone != backbone:
            lines.append(f"  - 教师 Backbone: {teacher_backbone}")
        lines.append(f"  - Projector: {hidden} 维" if use_projector else "  - 无 Projector")
        lines.append(f"  - Encoder: general + mri 解耦，half={hidden//2}")
        lines.append(f"  - FPN: fusion_s=concat(z_general,z_mri), use_specific={use_specific}")
    elif task_name == 'Student_Baseline_Aligned':
        lines.append("【学生 Baseline】单流 CT → 解耦 → FPN 分割（无蒸馏）")
        lines.append(f"  - Backbone: {backbone}")
        lines.append(f"  - Projector: {hidden} 维" if use_projector else "  - 无 Projector")
        lines.append(f"  - Encoder: general + mri 解耦，half={hidden//2}")
        lines.append(f"  - FPN: fusion_s=concat(z_general,z_mri), use_specific={use_specific}")
    elif task_name == 'MDT_Plus':
        lines.append("【MDT+】增强教师：学生 h_mri_s 蒸馏到教师 h_mri")
        lines.append(f"  - 教师 Backbone: {backbone} (双流独立)")
        lines.append(f"  - 学生 Backbone: {backbone} (单流，冻结)")
        if use_cmx:
            lines.append("  - 教师浅层: FSF ×3")
        lines.append("  - 损失: seg + FDMF 辅助项 + loss_kd_repr")
    return "\n".join(lines)


def build_data_summary(train_loader, val_loader, test_loader, task_name='MDT',
                       train_paired_loader=None, train_mri_loader=None, missing_rate=0.3):
    """构建数据划分描述"""
    info = {}
    if train_loader is not None:
        info['train_size'] = len(train_loader.dataset)
    if val_loader is not None:
        info['val_size'] = len(val_loader.dataset)
    if test_loader is not None:
        info['test_size'] = len(test_loader.dataset)
    info['missing_rate'] = missing_rate
    if task_name in ('MDT_Student', 'Student_Baseline_Aligned') and train_paired_loader is not None and train_mri_loader is not None:
        info['train_paired_size'] = len(train_paired_loader.dataset)
        info['train_mri_size'] = len(train_mri_loader.dataset)
        info['split_note'] = "配对(蒸馏+seg) + 非配对(仅seg)"
    elif task_name == 'MDT' or task_name == 'MDT_Plus':
        info['split_note'] = "仅 complete 配对数据"
    return info


def save_full_config(config, extras=None):
    """
    保存完整 configs.json，合并 extras 中的额外信息。
    extras 可包含: model_summary, data_split, params_m, flops_g, test_results, best_epoch, best_val_dice 等
    """
    import copy
    attrs = copy.deepcopy(vars(config))
    attrs['task'] = config.task
    attrs['checkpoint_dir'] = config.checkpoint_dir
    if 'checkpoint_dir' in attrs and attrs['checkpoint_dir'] is None:
        attrs['checkpoint_dir'] = config.checkpoint_dir
    if extras:
        for k, v in extras.items():
            if v is not None:
                attrs[k] = v
    save_path = os.path.join(config.checkpoint_dir, 'configs.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 确保 JSON 可序列化
    def _serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        return str(obj)
    attrs = _serialize(attrs)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(attrs, f, indent=2, ensure_ascii=False)
    return save_path


def update_config_with_test_results(config, test_metrics, best_epoch, best_val_dice):
    """训练结束后，将测试集结果写入 configs.json"""
    metrics_plain = {k: float(v) if hasattr(v, 'item') else v for k, v in test_metrics.items()}
    extras = {
        'test_results': metrics_plain,
        'best_epoch': int(best_epoch),
        'best_val_dice': float(best_val_dice) if hasattr(best_val_dice, 'item') else float(best_val_dice),
    }
    # 读取现有 configs，合并 test 结果后保存
    config_path = os.path.join(config.checkpoint_dir, 'configs.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing.update(extras)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    else:
        save_full_config(config, extras)


def print_test_results(test_metrics, task_name='MDT', checkpoint_dir=None):
    """训练完成后醒目打印测试集结果"""
    sep = "=" * 60
    print("\n" + sep)
    print("【测试集最终结果】")
    print(sep)
    print(f"  Dice:        {test_metrics.get('dice', 0):.4f}")
    print(f"  IoU:         {test_metrics.get('iou', 0):.4f}")
    print(f"  HD95:        {test_metrics.get('hd95', 0):.4f}  (与 CIPA pred.py 一致，越小越好)")
    print(f"  Acc (CIPA):  {test_metrics.get('acc', 0):.4f}  (mean(Sens,Spec)，对齐 github.com/mj129/CIPA)")
    print(f"  Acc_pixel:   {test_metrics.get('acc_pixel', 0):.4f}  (像素准确率，背景多时易虚高)")
    print(f"  Sensitivity: {test_metrics.get('sensitivity', 0):.4f}")
    print(f"  Specificity: {test_metrics.get('specificity', 0):.4f}")
    print(f"  Precision:   {test_metrics.get('precision', 0):.4f}")
    print(f"  F1:          {test_metrics.get('f1', 0):.4f}")
    if checkpoint_dir:
        print(sep)
        print(f"  权重目录: {checkpoint_dir}")
        print(f"  configs.json 已包含完整配置与测试结果，可直接查看")
    print(sep + "\n")
