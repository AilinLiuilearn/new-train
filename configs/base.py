# -*- coding: utf-8 -*-
"""配置基类，与 mkd 风格一致"""

import os
import copy
import json
import argparse
import datetime


def str2bool(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class ConfigBase(object):
    def __init__(self, args=None, **kwargs):
        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = {}
        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)
        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._task = "MDT"

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
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        config = cls()
        parser.parse_args(namespace=config)
        return config

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            configs = json.load(f)
        # checkpoint_dir 为只读属性（由 checkpoint_root/task/hash 计算得到），从 json 中加载时跳过
        if 'checkpoint_dir' in configs:
            configs.pop('checkpoint_dir')
        return cls(args=configs)

    def save(self, path: str = None):
        if path is None:
            save_path = os.path.join(self.checkpoint_dir, 'configs.json')
        else:
            save_path = os.path.join(self.checkpoint_dir, f'configs_{path}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir
        with open(save_path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def checkpoint_dir(self) -> str:
        task = getattr(self, '_task', 'MDT')
        if task == 'MDT_Student':
            subdir = 'MDT-student'
        elif task == 'MDT_Plus':
            subdir = 'MDT-plus'
        elif task == 'Student_Baseline_Aligned':
            subdir = 'MDT-baseline'
        elif task == 'CIPA_Baseline_Teacher':
            subdir = 'CIPA-baseline-teacher'
        elif task == 'CIPA_VMamba_Baseline_Teacher':
            subdir = 'CIPA-vmamba-baseline-teacher'
        elif task == 'MDT_STS_Teacher':
            subdir = 'MDT-STS-teacher'
        elif task == 'MDT_Light_Teacher':
            subdir = 'MDT-light-teacher'
        else:
            subdir = 'MDT'
        ckpt = os.path.join(self.checkpoint_root, subdir, self.hash)
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @staticmethod
    def ddp_parser():
        parser = argparse.ArgumentParser("DDP", add_help=False)
        parser.add_argument('--gpus', type=str, nargs='+', default=['0'])
        parser.add_argument('--server', type=str, default='main')
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--node_rank', type=int, default=0)
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500')
        parser.add_argument('--dist_backend', type=str, default='nccl')
        return parser

    @staticmethod
    def data_parser():
        raise NotImplementedError

    @staticmethod
    def model_parser():
        raise NotImplementedError

    @staticmethod
    def train_parser():
        raise NotImplementedError

    @staticmethod
    def logging_parser():
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_new/')
        parser.add_argument('--save_every', type=int, default=5)
        parser.add_argument('--enable_wandb', type=str2bool, default=False)
        return parser
