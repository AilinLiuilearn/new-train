"""Configuration for the isolated Full-only TACS fusion experiment."""
import argparse

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig


class TACSConfig(SegMDTConfig):
    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest == "model_arch":
                action.default = "tacs_channel_spatial"
                action.choices = ("tacs_channel_spatial",)
        p.add_argument("--tacs_channel_ratio", type=float, default=0.25)
        p.add_argument("--tacs_use_text", type=str2bool, default=False)
        p.add_argument("--tacs_text_embedding_path", type=str, default=None)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [
            cls.ddp_parser(), cls.data_parser(), cls.model_parser(),
            cls.train_parser(), cls.logging_parser(), cls.task_specific_parser(),
        ]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        if config.baseline != "full_add":
            raise ValueError("TACS experiment requires --baseline full_add")
        if config.tacs_use_text and not config.tacs_text_embedding_path:
            raise ValueError("--tacs_use_text true requires --tacs_text_embedding_path")
        if not (0.125 <= config.tacs_channel_ratio <= 1.0):
            raise ValueError("--tacs_channel_ratio must be in [0.125, 1.0]")
        if config.optimizer != "adamw" or config.accumulation_steps != 1:
            raise ValueError("TACS comparison requires AdamW and accumulation_steps=1")
        if config.use_deep_supervision or config.deep_supervision:
            raise ValueError("TACS comparison keeps deep supervision disabled")
        return config
