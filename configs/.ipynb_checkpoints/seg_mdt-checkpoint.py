# -*- coding: utf-8 -*-
"""MDT 分割任务配置：2D CT+PET → 二值 mask，三阶段教师-学生-互蒸馏"""

import argparse
import os
from configs.base import ConfigBase, str2bool


class SegMDTConfig(ConfigBase):
    """2D CT/PET 分割 + MDT 解耦 + CIPA 数据对齐"""

    @staticmethod
    def data_parser():
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--root', type=str, default='../data/PCLT20K', help='PCLT20K 根目录')
        parser.add_argument('--random_state', type=int, default=2023)
        parser.add_argument('--missing_rate', type=float, default=0.3, help='训练集非配对比例，教师用配对、学生用配对+非配对；教师与学生需一致')
        parser.add_argument('--val_ratio', type=float, default=0.1, help='从 train 分出验证集比例，对齐 MKD validation_size')
        parser.add_argument('--use_case_split', type=str2bool, default=True,
                            help='True: 按病例 70/15/15 划分 train/val/test（推荐）；False: 用 train.txt/test.txt，有 val.txt 则用 val')
        parser.add_argument('--image_size_2d', type=int, default=512, help='CT/PET/mask 尺寸')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--use_full_paired', type=str2bool, default=False,
                            help='True: steps=len(paired), 用满配对数据，mri 循环；False: steps=min(paired,mri)，原逻辑')
        return parser

    @staticmethod
    def model_parser():
        parser = argparse.ArgumentParser("Model", add_help=False)
        parser.add_argument('--backbone', type=str, default='tf_efficientnetv2_s',
                            help='timm 骨干：efficientnet_b2/b3/b4/b5 或 tf_efficientnetv2_s（更强预训练）')
        parser.add_argument('--pretrained_backbone', type=str2bool, default=True, help='是否加载预训练')
        parser.add_argument('--pretrained_path', type=str, default=None,
                            help='骨干预训练权重本地路径，无法访问 HuggingFace 时在本地下载后指定')
        parser.add_argument('--hidden', type=int, default=256, help='projector/encoder 通道')
        parser.add_argument('--fpn_out_channels', type=int, default=256, help='FPN 解码器通道')
        parser.add_argument('--use_projector', type=str2bool, default=True)
        parser.add_argument('--use_specific', type=str2bool, default=True,
                            help='True: FPN 第4层用 concat(通用,专属)；False: 仅用通用特征(z_mri_g+z_pet_g 或 z_general)，便于消融')
        parser.add_argument('--add_type', type=str, default='add', choices=('concat', 'add'))
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--cipa_pretrained', type=str, default=None,
                            help='CIPA VMamba 预训练权重路径，默认 CIPA-main/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth')
        parser.add_argument('--cipa_root', type=str, default=None, help='CIPA-main 项目根目录')
        return parser

    @staticmethod
    def train_parser():
        parser = argparse.ArgumentParser("Train", add_help=False)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'))
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--cosine_warmup', type=int, default=2)
        parser.add_argument('--cosine_cycles', type=int, default=1)
        parser.add_argument('--cosine_min_lr', type=float, default=1e-6)
        parser.add_argument('--mixed_precision', type=str2bool, default=True)
        parser.add_argument('--early_stop_patience', type=int, default=15, help='0 表示不早停')
        return parser

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser("Task", add_help=False)
        parser.add_argument('--alpha_seg', type=float, default=1.0)
        parser.add_argument('--alpha_sim', type=float, default=0.3)
        parser.add_argument('--alpha_diff', type=float, default=0.3)
        parser.add_argument('--alpha_recon', type=float, default=0.3)
        parser.add_argument('--loss_sim', type=str, default='cosine', choices=('cosine', 'cmd', 'l2', 'mse'))
        parser.add_argument('--loss_diff', type=str, default='cosine', choices=('cosine', 'fro', 'mse'))
        parser.add_argument('--loss_seg', type=str, default='dice_bce', choices=('dice_bce', 'bce', 'dice'))
        parser.add_argument('--dice_smooth', type=float, default=1.0)
        parser.add_argument('--n_moments', type=int, default=5)
        return parser

    @staticmethod
    def student_parser():
        parser = argparse.ArgumentParser("Student", add_help=False)
        parser.add_argument('--teacher_ckpt', type=str, default=None, help='教师 best 权重路径')
        parser.add_argument('--alpha_feat', type=float, default=0.5, help='表示蒸馏权重')
        parser.add_argument('--alpha_logit', type=float, default=0.5, help='logit 蒸馏权重')
        parser.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪，0 表示不裁剪')
        return parser

    @staticmethod
    def mdt_plus_parser():
        parser = argparse.ArgumentParser("MDT+", add_help=False)
        parser.add_argument('--student_ckpt', type=str, default=None, help='学生 best 权重路径')
        parser.add_argument('--alpha_kd_repr', type=float, default=1.0, help='表示级蒸馏权重')
        parser.add_argument('--temperature', type=float, default=5.0, help='KD 温度')
        return parser

    @classmethod
    def parse_arguments(cls):
        from configs.base import ConfigBase
        parents = [
            cls.ddp_parser(),
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
            cls.task_specific_parser(),
            cls.student_parser(),
            cls.mdt_plus_parser(),
        ]
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        config = cls()
        parser.parse_args(namespace=config)
        return config
