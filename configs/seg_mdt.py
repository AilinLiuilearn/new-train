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
        p.add_argument("--aug_mode", type=str, default="cipa", choices=("cipa", "light", "none"))
        p.add_argument("--norm_mode", type=str, default="imagenet", choices=("imagenet", "cipa"))
        return p

    @staticmethod
    def model_parser():
        p = argparse.ArgumentParser("Model", add_help=False)
        p.add_argument("--backbone", type=str, default="pvt_v2_b1")
        p.add_argument("--pretrained_path", type=str, default=None)
        p.add_argument("--model_arch", type=str, default="dual", choices=("dual", "hetero_convnext_mit"), help="dual: homogeneous sum baseline; hetero_convnext_mit: CT ConvNeXt + PET MiT with MPA-BioCLIP fusion.")
        p.add_argument("--ct_backbone", type=str, default="convnext_tiny")
        p.add_argument("--pet_backbone", type=str, default="mit_b0")
        p.add_argument("--ct_pretrained_path", type=str, default=None)
        p.add_argument("--pet_pretrained_path", type=str, default=None)
        p.add_argument("--fusion_channels", type=int, nargs="+", default=None, help="Stage channels after projection alignment, e.g. 32 64 160 256 for MiT-B0 scale.")
        p.add_argument("--use_tcpm", type=str2bool, default=False)
        p.add_argument(
            "--fusion_type",
            type=str,
            default="mpa_bioclip_sum",
            choices=("auto", "sum", "project_sum", "mpa_bioclip_sum", "bcg_pa_sum"),
            help="Feature fusion type. Heterogeneous model supports project_sum and mpa_bioclip_sum/bcg_pa_sum; homogeneous baseline supports auto/sum.",
        )
        p.add_argument("--bioclip_model_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model", help="Local BiomedCLIP OpenCLIP config/weight path for MPA-BioCLIP text prompt encoding.")
        p.add_argument("--bioclip_text_tower_path", type=str, default="/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower", help="Local HuggingFace BiomedBERT text tower path used by BiomedCLIP.")
        p.add_argument("--bioclip_text", type=str, default="focal abnormal metabolic lung lesion on PET-CT scan", help="Text prompt encoded by BiomedCLIP/BiomedBERT for text-guided fusion.")
        p.add_argument("--decoder_type", type=str, default="light", choices=("light",))
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
        p.add_argument("--deep_supervision_weights", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.125])
        p.add_argument("--lr_flat_ratio", type=float, default=0.3)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(), cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        return config
