"""Baseline supervision/EMA policy, preserving truly fixed Scharr buffers."""
import copy
import torch
from tasks.mdt_seg import MDTSegTeacher
from utils.ema import ModelEMA


class StructureResponseEMA(ModelEMA):
    @torch.no_grad()
    def update(self,model):
        decay=super().update(model)
        source=dict(model.named_buffers())
        for name,value in self.model.named_buffers():
            if name.endswith(('kernel_x','kernel_y')):
                # They are constants, not learnable floating statistics. Avoid
                # accumulating numerical drift from EMA arithmetic on buffers.
                value.copy_(source[name])
        return decay


class FullStructureTask(MDTSegTeacher):
    def __init__(self,networks,config):
        plain=copy.copy(config);plain.ema_enabled=False
        super().__init__(networks,plain);self.config=config
        if config.ema_enabled:
            self.ema=StructureResponseEMA(self.model,decay=config.ema_decay,
                warmup=config.ema_decay_warmup,device=self.device)
            self.ema_active=self.ema_start_epoch<=0
