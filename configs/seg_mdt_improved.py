# -*- coding: utf-8 -*-
"""
Improved MDT Teacher Configuration
优化后的教师模型配置 - 用于提升性能和拉大与学生模型的差距
"""

import argparse
from configs.base import ConfigBase, str2bool


class SegMDTImprovedConfig(ConfigBase):
    """
    改进版教师模型配置
    主要优化点:
    1. 更保守的学习率策略
    2. 更强的正则化
    3. 启用CUDM对比学习
    4. 更长的训练周期
    """

    @staticmethod
    def data_parser():
        p = argparse.ArgumentParser("Data", add_help=False)
        p.add_argument("--root", type=str, default="/root/autodl-tmp/data/PCLT20K")
        p.add_argument("--random_state", type=int, default=2023)
        p.add_argument("--val_ratio", type=float, default=0.1)
        p.add_argument("--use_case_split", type=str2bool, default=True)
        p.add_argument("--cipa_aligned", type=str2bool, default=True)
        p.add_argument("--image_size_2d", type=int, default=512)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--pin_memory", type=str2bool, default=True)
        return p

    @staticmethod
    def model_parser():
        p = argparse.ArgumentParser("Model", add_help=False)
        p.add_argument("--backbone", type=str, default="convnext_nano",
                       help="Backbone: convnext_nano(当前) / convnext_small(推荐升级)")
        p.add_argument("--pretrained_path", type=str, default="./pretrained/convnext_nano")
        p.add_argument("--use_tcpm", type=str2bool, default=True,
                       help="启用TCPM跨模态融合模块")
        return p

    @staticmethod
    def train_parser():
        p = argparse.ArgumentParser("Train", add_help=False)
        p.add_argument("--epochs", type=int, default=100,
                       help="增加到100 epochs (原80)")
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--accumulation_steps", type=int, default=1)
        p.add_argument("--optimizer", type=str, default="adamw", choices=("sgd", "adamw"))
        
        # 优化后的学习率策略
        p.add_argument("--learning_rate", type=float, default=8e-5,
                       help="降低学习率: 1.2e-4 → 8e-5 提升稳定性")
        p.add_argument("--decoder_lr", type=float, default=1.0e-4,
                       help="解码器学习率稍高: 1.2e-4 → 1.0e-4")
        
        # 增强正则化
        p.add_argument("--weight_decay", type=float, default=1e-3,
                       help="增强正则化: 5e-4 → 1e-3")
        
        # 学习率调度优化
        p.add_argument("--cosine_warmup", type=int, default=5,
                       help="更长预热: 3 → 5 epochs")
        p.add_argument("--cosine_min_lr", type=float, default=1e-6)
        p.add_argument("--lr_flat_ratio", type=float, default=0.15,
                       help="更长平台期: 0.1 → 0.15")
        
        # 梯度裁剪优化
        p.add_argument("--grad_clip", type=float, default=0.5,
                       help="更严格梯度裁剪: 1.0 → 0.5")
        
        p.add_argument("--mixed_precision", type=str2bool, default=True)
        p.add_argument("--vis_every_epoch", type=str2bool, default=True)
        p.add_argument("--vis_epoch_samples", type=int, default=2)
        p.add_argument("--early_stop_patience", type=int, default=25,
                       help="增加耐心: 20 → 25")
        p.add_argument("--eval_threshold", type=float, default=0.5)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser("Task", add_help=False)
        p.add_argument("--loss_smooth", type=float, default=1.0)
        
        # 损失权重优化
        p.add_argument("--bce_weight", type=float, default=0.8,
                       help="BCE权重: 1.0 → 0.8")
        p.add_argument("--dice_weight", type=float, default=1.2,
                       help="Dice权重: 1.0 → 1.2 (更关注Dice)")
        p.add_argument("--pos_weight", type=float, default=None)
        
        # 启用CUDM对比学习损失
        p.add_argument("--cudm_tumor_weight", type=float, default=1.0,
                       help="肿瘤区域对比学习: 0.0 → 1.0")
        p.add_argument("--cudm_bg_weight", type=float, default=0.5,
                       help="背景区域对比: 0.0 → 0.5")
        p.add_argument("--cudm_orth_weight", type=float, default=0.3,
                       help="正交约束: 0.0 → 0.3")
        p.add_argument("--cudm_loss_start_stage", type=int, default=3)
        
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [
            cls.ddp_parser(),
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
            cls.task_specific_parser(),
        ]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        return config
