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
        p.add_argument('--model_arch', type=str, default='dual_shared_add_baseline', choices=('dual_shared_add_baseline',))
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
        p.add_argument('--cppi_build_stage', type=int, default=3)
        p.add_argument('--enable_gradient_diagnostics', type=str2bool, default=False)
        p.add_argument('--gradient_diagnostics_interval', type=int, default=5)
        p.add_argument('--gradient_diagnostics_num_samples', type=int, default=1)
        p.add_argument('--resume_checkpoint', type=str, default=None)
        p.add_argument(
            '--stage1_checkpoint',
            type=str,
            default=None,
            help='Stage-1 best checkpoint for frozen TaskMoE Stage-2 training',
        )
        p.add_argument(
            '--taskmoe_mode',
            type=str,
            default='independent',
            choices=('independent', 'cross_scale_shared', 'state_scale_factorized'),
            help=(
                'independent: per-scale TaskMoE (controlled by --taskmoe_scales). '
                'cross_scale_shared: one CrossScaleSharedTaskMoE on S1-S4 with '
                'scale-specific Prompt/Router and a shared Expert Bank '
                '(requires --taskmoe_scales all). '
                'state_scale_factorized: SSF-SP TaskMoE with 1 shared + 4 scale-private '
                '+ 2 state-private experts (requires --taskmoe_scales all).'
            ),
        )
        p.add_argument(
            '--taskmoe_scales',
            type=str,
            default='s4',
            help=(
                'TaskMoE insertion scales for Stage-2 ablation. '
                'Examples: s4 | s3s4 | s2s3s4 | s1s2s3s4 | all | s1,s2,s3,s4. '
                'For --taskmoe_mode cross_scale_shared, must be all/s1s2s3s4.'
            ),
        )
        p.add_argument(
            '--taskmoe_residual_mode',
            type=str,
            default='zero_start',
            choices=('zero_start', 'paper'),
            help=(
                'Shared TaskMoE residual combination. '
                'zero_start: F_out = F_base + beta * DeltaF, learnable beta initialized to zero. '
                'paper: F_out = F_base + DeltaF, no learnable residual scaling. '
                'Applies to --taskmoe_mode cross_scale_shared; default preserves prior behavior.'
            ),
        )
        p.add_argument(
            '--taskmoe_num_experts',
            type=int,
            default=6,
            help=(
                'Number of experts in TaskMoE. '
                'Used for expert-number ablation. '
                'Recommended controlled settings: 4, 6, 8 with TopK fixed at 2.'
            ),
        )
        p.add_argument(
            '--stage2_train_decoder',
            type=str2bool,
            default=False,
            help=(
                'Whether to fine-tune the pretrained shared decoder together with TaskMoE '
                'during Stage2. False preserves the original MoE-only Stage2 behavior.'
            ),
        )
        p.add_argument(
            '--taskmoe_use_text_prior',
            type=str2bool,
            default=False,
            help=(
                'Use fixed biomedical text semantics as an expert-routing prior '
                'for cross-scale shared TaskMoE.'
            ),
        )
        p.add_argument(
            '--taskmoe_text_model_path',
            type=str,
            default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model',
            help='Local BioMedCLIP model root (existence check; local_files_only).',
        )
        p.add_argument(
            '--taskmoe_text_tower_path',
            type=str,
            default='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower',
            help='Local biomedical text tower used to encode fixed Full/Missing texts.',
        )
        p.add_argument(
            '--taskmoe_private_rank',
            type=int,
            default=16,
            choices=(8, 16, 32),
            help='Low-rank width for scale/state private experts in state_scale_factorized.',
        )
        p.add_argument(
            '--taskmoe_beta_max',
            type=float,
            default=1.0,
            help='Bounded residual: beta_s = beta_max * tanh(raw_beta_s).',
        )
        p.add_argument(
            '--taskmoe_role_loss_weight',
            type=float,
            default=0.02,
            help=(
                'lambda_role for Factorized Expert Role Supervision (FERS). '
                'L_total = L_seg + lambda_role * L_FERS. Use 0 to disable.'
            ),
        )
        p.add_argument(
            '--taskmoe_fers_mode',
            type=str,
            default='both',
            choices=('both', 'scale', 'state', 'none'),
            help=(
                'FERS components: both=0.5*(scale+state), scale only, state only, or none. '
                'Replaces the removed shared Full-Missing consistency loss.'
            ),
        )
        p.add_argument(
            '--stage2_decoder_adapter',
            type=str2bool,
            default=False,
            help='Enable zero-start Stage2 decoder adapter after frozen d1 (not old-decoder fine-tune).',
        )
        p.add_argument(
            '--stage2_decoder_adapter_level',
            type=str,
            default='d1',
            choices=('d1',),
            help='Decoder adapter insertion level. v1 only supports d1.',
        )
        p.add_argument(
            '--stage2_strategy',
            type=str,
            default='legacy_taskmoe',
            choices=('legacy_taskmoe', 'logit_residual_decoder'),
            help=(
                'Stage-2 strategy. legacy_taskmoe keeps existing TaskMoE/FERS paths; '
                'logit_residual_decoder freezes Stage1 and trains only a lightweight '
                'logit residual decoder (E1/E2).'
            ),
        )
        p.add_argument(
            '--stage2_residual_channels',
            type=int,
            default=64,
            help='Hidden channels for Stage2 logit residual decoder (E1/E2).',
        )
        p.add_argument(
            '--stage2_residual_state_conditioned',
            type=str2bool,
            default=False,
            help=(
                'E1: False (no state modulation). '
                'E2: True enables deterministic Full/Missing FiLM on residual decoder.'
            ),
        )
        p.add_argument(
            '--stage2_delta_logit_max',
            type=float,
            default=2.0,
            help='Bound M for delta = M * tanh(raw / M) in logit residual decoder.',
        )
        p.add_argument(
            '--stage2_residual_dropout',
            type=float,
            default=0.0,
            help='Dropout before residual delta_head. Must be in [0, 1).',
        )
        p.add_argument(
            '--stage2_residual_lr',
            type=float,
            default=5e-5,
            help='Learning rate for stage2_residual_decoder only (logit_residual_decoder).',
        )
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
