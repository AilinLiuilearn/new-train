# -*- coding: utf-8 -*-
import argparse
from configs.base import ConfigBase, str2bool


class SegMDTConfig(ConfigBase):

    @staticmethod
    def data_parser():
        p = argparse.ArgumentParser("Data", add_help=False)
        p.add_argument("--root", type=str, default="../data/PCLT20K")
        p.add_argument("--random_state", type=int, default=2023)
        p.add_argument("--val_ratio", type=float, default=0.1)
        p.add_argument("--use_case_split", type=str2bool, default=True)
        p.add_argument("--cipa_aligned", type=str2bool, default=True)
        p.add_argument("--image_size_2d", type=int, default=512)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--pin_memory", type=str2bool, default=True)
        p.add_argument("--dataset", type=str, default="pclt20k", choices=("pclt20k", "sts"))
        p.add_argument("--split_json", type=str, default=None)
        p.add_argument("--test_ratio", type=float, default=0.2)
        p.add_argument("--sts_aug_preset", type=str, default="stable", choices=("none", "stable", "light", "default"))
        p.add_argument("--val_threshold_search", type=str2bool, default=False)
        return p

    @staticmethod
    def model_parser():
        p = argparse.ArgumentParser("Model", add_help=False)
        p.add_argument("--backbone", type=str, default="pvt_v2_b1")
        p.add_argument("--pretrained_path", type=str, default=None)
        p.add_argument("--use_tcpm", type=str2bool, default=False)
        p.add_argument("--disable_text_fusion", type=str2bool, default=False,
                       help="Disable text-guided query while keeping CUDM visual fusion.")
        p.add_argument("--pet_drop_rate", type=float, default=0.0,
                       help="Probability of dropping PET modality during training for robustness.")
        p.add_argument("--use_shaspec_disentangle", type=str2bool, default=False,
                       help="Enable stage-wise shared/specific disentanglement before MT-CUDM fusion.")
        p.add_argument("--light_teacher", type=str2bool, default=False,
                       help="Use student-scale ConvNeXt dual encoders as a lightweight multimodal teacher.")
        p.add_argument("--light_backbone", type=str, default="convnext_atto",
                       choices=("convnext_atto", "convnext_femto", "convnext_pico", "convnext_nano"))
        p.add_argument("--light_pretrained_path", type=str, default=None)
        p.add_argument("--light_no_pretrained", type=str2bool, default=False)
        p.add_argument("--light_decoder_type", type=str, default="light", choices=("light", "attention"))
        p.add_argument("--light_share_encoder", type=str2bool, default=False,
                       help="Share CT/PET encoder weights to minimize parameters.")
        return p

    @staticmethod
    def train_parser():
        p = argparse.ArgumentParser("Train", add_help=False)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--accumulation_steps", type=int, default=2, help="梯度累加步数")
        p.add_argument("--optimizer", type=str, default="adamw", choices=("sgd", "adamw"))
        p.add_argument("--learning_rate", type=float, default=8e-5)
        p.add_argument("--decoder_lr", type=float, default=8e-5)
        p.add_argument("--weight_decay", type=float, default=1e-4)
        p.add_argument("--cosine_warmup", type=int, default=2)
        p.add_argument("--cosine_min_lr", type=float, default=1e-6)
        p.add_argument("--mixed_precision", type=str2bool, default=True)
        p.add_argument("--vis_every_epoch", type=str2bool, default=True)
        p.add_argument("--vis_epoch_samples", type=int, default=2)
        p.add_argument("--early_stop_patience", type=int, default=15)
        p.add_argument("--eval_threshold", type=float, default=0.5)
        p.add_argument("--grad_clip", type=float, default=0.5)
        p.add_argument("--ema_decay", type=float, default=0.999)
        p.add_argument("--ema_warmup_epochs", type=int, default=3)
        return p

    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser("Task", add_help=False)
        p.add_argument("--loss_smooth", type=float, default=1.0)
        p.add_argument("--bce_weight", type=float, default=1.0)
        p.add_argument("--dice_weight", type=float, default=1.0)
        p.add_argument("--pos_weight", type=float, default=None)
        p.add_argument("--cudm_tumor_weight", type=float, default=0.0)
        p.add_argument("--cudm_bg_weight", type=float, default=0.0)
        p.add_argument("--cudm_orth_weight", type=float, default=0.0)
        p.add_argument("--cudm_loss_start_stage", type=int, default=3)
        p.add_argument("--dis_common_weight", type=float, default=0.0)
        p.add_argument("--dis_orth_weight", type=float, default=0.0)
        p.add_argument("--dis_specific_weight", type=float, default=0.0)
        p.add_argument("--dis_recon_weight", type=float, default=0.0)
        p.add_argument("--dis_loss_start_stage", type=int, default=1)
        p.add_argument("--lr_flat_ratio", type=float, default=0.3)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(), cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        return config
