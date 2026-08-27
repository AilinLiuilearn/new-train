# -*- coding: utf-8 -*-
import argparse

from configs.base import ConfigBase, str2bool


class SegMDTConfig(ConfigBase):
    @staticmethod
    def data_parser():
        p = argparse.ArgumentParser('Data', add_help=False)
        p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
        p.add_argument('--train_split_file', type=str, default='train_original.txt')
        p.add_argument('--val_split_file', type=str, default='test.txt')
        p.add_argument('--test_split_file', type=str, default='test.txt')
        p.add_argument('--image_size_2d', type=int, default=512)
        p.add_argument('--num_workers', type=int, default=4)
        p.add_argument('--pin_memory', type=str2bool, default=True)
        p.add_argument('--aug_mode', type=str, default='cipa', choices=('cipa', 'light', 'none'))
        p.add_argument('--norm_mode', type=str, default='cipa', choices=('imagenet', 'cipa'))
        return p

    @staticmethod
    def model_parser():
        p = argparse.ArgumentParser('Model', add_help=False)
        p.add_argument(
            '--model_arch',
            type=str,
            default='dual_shared_add_baseline',
            choices=('dual_shared_add_baseline', 'prompt_role_expert_stage2'),
            help='prompt_role_expert_stage2 = joint end-to-end Recovery+SPRE+Decoder',
        )
        p.add_argument('--ct_backbone', type=str, default='convnextv2_nano')
        p.add_argument('--pet_backbone', type=str, default='mit_b1')
        p.add_argument('--ct_pretrained_path', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
        p.add_argument('--pet_pretrained_path', type=str, default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
        p.add_argument('--no_encoder_pretrained', type=str2bool, default=False)
        p.add_argument('--decoder_channels', type=int, nargs=4, default=[512, 256, 128, 64])
        p.add_argument('--use_deep_supervision', type=str2bool, default=False)
        p.add_argument('--deep_supervision', type=str2bool, default=False)
        p.add_argument('--joint_full_weight', type=float, default=0.5)
        p.add_argument('--joint_missing_weight', type=float, default=0.5)
        p.add_argument('--cppi_num_clusters', type=int, default=6)
        p.add_argument('--cppi_build_stage', type=int, default=4)
        p.add_argument('--enable_gradient_diagnostics', type=str2bool, default=False)
        p.add_argument('--gradient_diagnostics_interval', type=int, default=5)
        p.add_argument('--gradient_diagnostics_num_samples', type=int, default=1)
        # Resume a full joint-model training run (not a frozen Stage-1 warm start).
        p.add_argument('--resume_checkpoint', type=str, default=None)
        # SPRE hyperparameters (Text OFF: external_prompt_dim=None).
        p.add_argument('--stage2_expert_dim', type=int, default=128)
        p.add_argument('--stage2_atom_num', type=int, default=32)
        p.add_argument('--stage2_atom_dim', type=int, default=256)
        p.add_argument('--stage2_prompt_hidden_channels', type=int, default=64)
        p.add_argument('--stage2_mlp_ratio', type=float, default=2.0)
        p.add_argument('--stage2_dropout', type=float, default=0.0)
        return p

    @staticmethod
    def train_parser():
        p = argparse.ArgumentParser('Train', add_help=False)
        p.add_argument('--epochs', type=int, default=60)
        p.add_argument('--batch_size', type=int, default=16)
        p.add_argument('--accumulation_steps', type=int, default=1)
        p.add_argument('--optimizer', type=str, default='adamw', choices=('adamw', 'sgd'))
        p.add_argument('--learning_rate', type=float, default=8e-5)
        p.add_argument('--decoder_lr', type=float, default=8e-5)
        p.add_argument('--weight_decay', type=float, default=1e-4)
        p.add_argument('--cosine_warmup', type=int, default=3)
        p.add_argument('--cosine_min_lr', type=float, default=1e-6)
        p.add_argument('--lr_flat_ratio', type=float, default=0.3)
        p.add_argument('--mixed_precision', type=str2bool, default=True)
        p.add_argument('--grad_clip', type=float, default=5.0)
        p.add_argument('--early_stop_patience', type=int, default=10)
        p.add_argument('--validation_frequency', type=int, default=1)
        p.add_argument('--random_state', type=int, default=2023)
        p.add_argument('--final_test_missing_rates', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
        p.add_argument('--train_pet_drop_prob', type=float, default=0.0)
        p.add_argument('--missing_loss_weight', type=float, default=1.0)
        p.add_argument('--vis_every_epoch', type=str2bool, default=False)
        p.add_argument('--eval_full_pet', type=str2bool, default=True)
        p.add_argument('--eval_fixed_missing_pet', type=str2bool, default=True)
        p.add_argument('--eval_random_missing_pet', type=str2bool, default=False)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser('Task', add_help=False)
        p.add_argument('--loss_smooth', type=float, default=1.0)
        p.add_argument('--bce_weight', type=float, default=1.0)
        p.add_argument('--dice_weight', type=float, default=1.0)
        p.add_argument('--boundary_loss_weight', type=float, default=0.0)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(), cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        return config
