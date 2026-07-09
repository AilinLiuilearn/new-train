# -*- coding: utf-8 -*-
import argparse
from configs.base import ConfigBase, str2bool


class SegMDTConfig(ConfigBase):

    @staticmethod
    def data_parser():
        p = argparse.ArgumentParser("Data", add_help=False)
        p.add_argument("--root", type=str, default="/root/autodl-tmp/data/PCLT20K")
        p.add_argument("--random_state", type=int, default=2023)
        p.add_argument("--val_ratio", type=float, default=0.1)
        p.add_argument("--use_case_split", type=str2bool, default=True)
        p.add_argument("--train_split_file", type=str, default="train.txt")
        p.add_argument("--val_split_file", type=str, default="val.txt")
        p.add_argument("--test_split_file", type=str, default="test.txt")
        p.add_argument("--cipa_aligned", type=str2bool, default=True)
        p.add_argument("--use_aligned_loader", type=str2bool, default=True)
        p.add_argument("--train_list", type=str, default="train_original.txt")
        p.add_argument("--val_list", type=str, default="test.txt")
        p.add_argument("--test_list", type=str, default="test.txt")
        p.add_argument("--image_size_2d", type=int, default=512)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--pin_memory", type=str2bool, default=True)
        p.add_argument("--aug_mode", type=str, default="cipa", choices=("cipa", "light", "none"))
        p.add_argument("--norm_mode", type=str, default="cipa", choices=("imagenet", "cipa"))
        return p

    @staticmethod
    def model_parser():
        p = argparse.ArgumentParser("Model", add_help=False)
        p.add_argument("--backbone", type=str, default="mit_b1")
        p.add_argument("--pretrained_path", type=str, default=None)
        p.add_argument("--model_arch", type=str, default="dual_shared_add_baseline", choices=("dual_shared_add_baseline",), help="dual_shared_add_baseline: dual encoder (ConvNeXt-Nano + MiT-B1), stage-wise sum fusion, shared UNet decoder.")
        p.add_argument("--missing_mode", type=str, default="ac_hmtr", choices=("ct", "ac_hmtr"))
        p.add_argument("--ac_hmtr_num_tokens", type=int, default=8)
        p.add_argument("--ac_hmtr_residual_scale_init", type=float, default=0.1)
        p.add_argument("--use_pet_mrp_gsa", type=str2bool, default=True)
        p.add_argument("--pet_mrp_stages", type=str, default="all", choices=("all", "c34", "c4"))
        p.add_argument("--pet_mrp_prior_mode", type=str, default="minmax", choices=("minmax", "full", "local"))
        p.add_argument("--pet_prior_type", type=str, default="lap_hgl", choices=("none", "intensity", "lap_hgl"))
        p.add_argument("--pet_prior_size", type=str, default="lite", choices=("lite", "full", "minimal"))
        p.add_argument("--pet_prior_c4_channels", type=int, default=64)
        p.add_argument("--pet_prior_mid_channels", type=int, default=32)
        p.add_argument("--pet_prior_channels", type=int, nargs=4, default=[24, 32, 48, 64])
        p.add_argument("--pet_fuse_mid_channels", type=int, default=32)
        p.add_argument("--pet_gn_groups", type=int, default=8)
        p.add_argument("--dinov3_model_name", type=str, default="vit_small_patch16_dinov3")
        p.add_argument("--dinov3_pretrained_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/dinov3_small")
        p.add_argument("--pet_prompt_base_channels", type=int, default=256)
        p.add_argument("--ct_backbone", type=str, default="convnextv2_nano")
        p.add_argument("--pet_backbone", type=str, default="mit_b1")
        p.add_argument("--encoder_name", type=str, default="mit_b1")
        p.add_argument("--pretrained", type=str2bool, default=True)
        p.add_argument("--ct_pretrained_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano")
        p.add_argument("--pet_pretrained_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1")
        p.add_argument("--freq_method", type=str, default="fft", choices=("fft", "fft_gaussian", "avgpool", "blur"))
        p.add_argument("--use_pet_proxy", type=str2bool, default=True)
        p.add_argument("--proxy_loss_weight", type=float, default=0.05)
        p.add_argument("--consistency_loss_weight", type=float, default=0.0)
        p.add_argument("--fusion_type", type=str, default="dmome", choices=("concat_conv", "add", "dmome", "dmome_channel_prior_gate", "hybrid_concat_dmome"))
        p.add_argument("--dmome_expert_reduction", type=int, default=4)
        p.add_argument("--dmome_use_status_token", type=str2bool, default=True)
        p.add_argument("--dmome_temperature", type=float, default=1.0)
        p.add_argument("--dmome_init_ct_bias", type=float, default=0.0)
        p.add_argument("--dmome_output_proj", type=str2bool, default=False)
        p.add_argument("--dmome_norm_groups", type=int, default=8)
        p.add_argument("--use_channel_prior_gate", type=str2bool, default=False)
        p.add_argument("--prior_gate_stages", type=str, default="1,2,3,4")
        p.add_argument("--hybrid_concat_stages", type=str, default="1,2,3")
        p.add_argument("--hybrid_dmome_stages", type=str, default="4")
        p.add_argument("--biomedclip_model_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model")
        p.add_argument("--log_dmome_weights", type=str2bool, default=True)
        p.add_argument("--decoder_type", type=str, default="unet", choices=("unet",))
        p.add_argument("--use_deep_supervision", type=str2bool, default=False)
        p.add_argument("--print_trainable_only", type=str2bool, default=True)
        return p

    @staticmethod
    def train_parser():
        p = argparse.ArgumentParser("Train", add_help=False)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--accumulation_steps", type=int, default=1)
        p.add_argument("--optimizer", type=str, default="adamw", choices=("sgd", "adamw"))
        p.add_argument("--learning_rate", type=float, default=8e-5)
        p.add_argument("--decoder_lr", type=float, default=8e-5)
        p.add_argument("--weight_decay", type=float, default=1e-4)
        p.add_argument("--cosine_warmup", type=int, default=3)
        p.add_argument("--cosine_min_lr", type=float, default=1e-6)
        p.add_argument("--mixed_precision", type=str2bool, default=True)
        p.add_argument("--nan_safe_mode", type=str2bool, default=True)
        p.add_argument("--vis_every_epoch", type=str2bool, default=True)
        p.add_argument("--vis_epoch_samples", type=int, default=2)
        p.add_argument("--early_stop_patience", type=int, default=10)
        p.add_argument("--eval_threshold", type=float, default=0.5)
        p.add_argument("--grad_clip", type=float, default=5.0)
        p.add_argument("--train_pet_drop_prob", type=float, default=0.0)
        p.add_argument("--eval_full_pet", type=str2bool, default=True)
        p.add_argument("--eval_fixed_missing_pet", type=str2bool, default=False)
        p.add_argument("--eval_random_missing_pet", type=str2bool, default=False)
        p.add_argument("--eval_random_pet_drop_prob", type=float, default=0.4)
        p.add_argument("--train_mode", type=str, default="alternating_full_missing", choices=("alternating_full_missing",))
        p.add_argument("--missing_loss_weight", type=float, default=1.0)
        p.add_argument("--eval_random_seed", type=int, default=2026)
        p.add_argument("--lr_find_start", type=float, default=1e-7)
        p.add_argument("--lr_find_end", type=float, default=1e-2)
        p.add_argument("--lr_find_num_iter", type=int, default=200)
        p.add_argument("--lr_find_stop_factor", type=float, default=4.0)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser("Task", add_help=False)
        p.add_argument("--loss_smooth", type=float, default=1.0)
        p.add_argument("--bce_weight", type=float, default=1.0)
        p.add_argument("--dice_weight", type=float, default=1.0)
        p.add_argument("--pos_weight", type=float, default=None)
        p.add_argument("--deep_supervision", type=str2bool, default=False)
        p.add_argument("--deep_supervision_weights", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125])
        p.add_argument("--boundary_loss_weight", type=float, default=0.0)
        p.add_argument("--lr_flat_ratio", type=float, default=0.3)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(), cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        return config
