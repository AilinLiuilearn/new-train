# -*- coding: utf-8 -*-
"""Student model configuration for single-modality knowledge distillation.

Defaults are aligned with the teacher config (2026-05-09_12-18-03) for fair comparison.
"""

import argparse
from configs.base import ConfigBase, str2bool


class SegStudentConfig(ConfigBase):

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
        p.add_argument("--student_backbone", type=str, default="convnext_pico",
                        choices=["convnext_atto", "convnext_femto",
                                 "convnext_pico", "convnext_nano"])
        p.add_argument("--student_pretrained_path", type=str, default=None,
                        help="Path to local pretrained encoder weights. "
                             "Auto-resolved from ./pretrained/<backbone>/ if not set.")
        p.add_argument("--no_pretrained", type=str2bool, default=False,
                        help="If True, skip pretrained encoder weights (train from scratch).")
        p.add_argument("--student_decoder_type", type=str, default="attention",
                        choices=["attention", "light"])
        return p

    @staticmethod
    def train_parser():
        p = argparse.ArgumentParser("Train", add_help=False)
        p.add_argument("--epochs", type=int, default=80)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--accumulation_steps", type=int, default=1)
        p.add_argument("--optimizer", type=str, default="adamw",
                        choices=("sgd", "adamw"))
        p.add_argument("--learning_rate", type=float, default=1.2e-4)
        p.add_argument("--decoder_lr", type=float, default=1.2e-4)
        p.add_argument("--weight_decay", type=float, default=5e-4)
        p.add_argument("--cosine_warmup", type=int, default=3)
        p.add_argument("--cosine_min_lr", type=float, default=1e-6)
        p.add_argument("--mixed_precision", type=str2bool, default=True)
        p.add_argument("--vis_every_epoch", type=str2bool, default=True)
        p.add_argument("--vis_epoch_samples", type=int, default=2)
        p.add_argument("--early_stop_patience", type=int, default=20)
        p.add_argument("--eval_threshold", type=float, default=0.5)
        p.add_argument("--grad_clip", type=float, default=1.0)
        p.add_argument("--ema_decay", type=float, default=0.9995)
        p.add_argument("--ema_warmup_epochs", type=int, default=5)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser("Task", add_help=False)
        p.add_argument("--loss_smooth", type=float, default=1.0)
        p.add_argument("--bce_weight", type=float, default=1.0)
        p.add_argument("--dice_weight", type=float, default=1.0)
        p.add_argument("--pos_weight", type=float, default=None)
        p.add_argument("--lr_flat_ratio", type=float, default=0.1)
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
