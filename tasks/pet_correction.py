"""Joint supervised training with read-before-write paired correction memory.

PNDC reference: yansheng-qiu/PNDC, train_PNDC.py findPNmap/mask_*.
Adaptation: GT gain + correctness checksum, binary entropy confidence,
negative-region BCE reweighting. Not a copy of the four-MRI audit algorithm.
SAR-Net reference: synchronized paired entries, momentum source, top-k lookup.
"""
import copy
import hashlib
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from tasks.mdt_seg import MDTSegTeacher
from utils.ema import ModelEMA
from utils.metrics_seg import SegmentationMetricsCIPA
from models.pet_correction_memory import audit_regions


@contextmanager
def isolated_source_forward(device):
    """No-grad bank reads must not poison the surrounding training AMP cache.

In affected PyTorch versions a no-grad forward on the ONLINE model can cache
detached low-precision weights, reused by the later grad-enabled forward in
the SAME outer autocast context. Disable caching only for the source pass,
retain its current autocast precision, and isolate token-sampling RNG so
collection-only does not change the main model's DropPath/dropout trajectory.
"""
    try:
        enabled=torch.is_autocast_enabled(device.type)
    except TypeError:  # older PyTorch API
        enabled=torch.is_autocast_enabled() if device.type=='cuda' else torch.is_autocast_cpu_enabled()
    try:
        dtype=torch.get_autocast_dtype(device.type)
    except AttributeError:
        dtype=torch.get_autocast_gpu_dtype() if device.type=='cuda' else torch.get_autocast_cpu_dtype()
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad(), \
         torch.autocast(device_type=device.type,enabled=enabled,dtype=dtype,cache_enabled=False):
        yield


def assert_core_gradients(model):
    """Cheap first-batch guard against silently disconnected encoder/decoder kernels."""
    checked=[]
    for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder'):
        item=next(((n,p) for n,p in getattr(model,name).named_parameters()
                   if p.requires_grad and p.ndim>=2),None)
        if item is None or item[1].grad is None:
            parameter='no trainable kernel' if item is None else item[0]
            raise RuntimeError(f'Core gradient disconnected: {name}.{parameter}. '
                               'Optimizer and memory were not updated; check source/autocast isolation.')
        checked.append(name)
    return checked


def stable_ids(names,device='cpu'):
    """Stable signed-int64 identity, unchanged across Python processes/augmentations."""
    vals=[int.from_bytes(hashlib.blake2b(str(n).encode('utf-8'),digest_size=8).digest(),'little')
          & ((1<<63)-1) for n in names]
    return torch.tensor(vals,dtype=torch.long,device=device)


def batch_ids(batch,device):
    names=batch.get('image_id',batch.get('slice_id'))
    if names is None:
        raise ValueError('Dataset must return stable image_id/slice_id')
    samples=stable_ids(names,device)
    groups=stable_ids(batch['case_id'],device) if 'case_id' in batch else torch.full_like(samples,-1)
    return samples,groups


def build_ratio_state(batch_size,step,seed,full_ratio,device):
    if batch_size<2 or not 0<full_ratio<1:
        raise ValueError('Invalid mixed ratio/batch size')
    n=int(np.floor((step+1)*batch_size*full_ratio+1e-9)-np.floor(step*batch_size*full_ratio+1e-9))
    if not 0<n<batch_size:
        raise ValueError('Ratio requires a larger batch to keep both states')
    state=torch.cat([torch.ones(n,dtype=torch.long),torch.zeros(batch_size-n,dtype=torch.long)])
    generator=torch.Generator().manual_seed(seed+step*1000003)
    return state[torch.randperm(batch_size,generator=generator)].to(device)


def is_memory_buffer(key):
    return '.memory.' in key or key in ('memory_step','retrieval_active')


class CorrectionEMA(ModelEMA):
    """EMA parameters/BN buffers, but paired queues are copied atomically."""
    @torch.no_grad()
    def sync_memory(self,model):
        source=model.state_dict(); target=self.model.state_dict()
        for key in source:
            if is_memory_buffer(key):
                target[key].copy_(source[key])

    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        decay=min(self.decay,(1+self.updates)/(10+self.updates)) if self.warmup else self.decay
        target,source=self.model.state_dict(),model.state_dict()
        if target.keys()!=source.keys():
            raise RuntimeError('EMA key mismatch')
        for key,v in target.items():
            if is_memory_buffer(key) or not v.is_floating_point():
                v.copy_(source[key])
            else:
                v.mul_(decay).add_(source[key].detach(),alpha=1-decay)
        return decay


class PETCorrectionTask(MDTSegTeacher):
    def __init__(self,networks,config):
        plain=copy.copy(config); plain.ema_enabled=False
        super().__init__(networks,plain)
        self.config=config
        if config.ema_enabled:
            self.ema=CorrectionEMA(self.model,decay=config.ema_decay,
                                  warmup=config.ema_decay_warmup,device=self.device)
            self.ema_active=self.ema_start_epoch<=0
        # Bank momentum has its OWN schedule. Evaluation EMA remains inherited
        # from the baseline (normally activates after three warmup epochs).
        self.bank_ema=None
        self.bank_ema_start_epoch=int(getattr(config,'bank_ema_start_epoch',0))
        self.bank_ema_active=False
        if config.memory_enabled and getattr(config,'bank_ema_enabled',True):
            self.bank_ema=CorrectionEMA(self.model,decay=getattr(config,'bank_ema_decay',.999),
                warmup=getattr(config,'bank_ema_decay_warmup',True),device=self.device)
            self.bank_ema_active=self.bank_ema_start_epoch<=0
        self.epoch=0; self.pending=[None]*4

    def memory_source_model(self):
        return self.bank_ema.model if self.bank_ema_active else self.model

    def restore_bank_ema(self,checkpoint,required=False):
        state=checkpoint.get('model_bank_ema')
        if state is None:
            if required and self.bank_ema is not None:
                raise ValueError('Checkpoint has no independent bank EMA; start a new run')
            return
        if self.bank_ema is None:
            raise ValueError('Checkpoint bank EMA requires bank_ema_enabled=True and memory_enabled=True')
        self.bank_ema.model.load_state_dict(state,strict=True)
        self.bank_ema.updates=int(checkpoint['bank_ema_updates'])
        self.bank_ema_active=bool(checkpoint['bank_ema_active'])

    def begin_epoch(self,epoch):
        super().begin_epoch(epoch)
        self.epoch=int(epoch)
        if self.bank_ema is not None and not self.bank_ema_active and epoch>self.bank_ema_start_epoch:
            self.bank_ema.reset(self.model)
            self.bank_ema_active=True
        active=bool(self.config.memory_enabled and epoch>=self.config.retrieval_start_epoch)
        self.model.retrieval_active.fill_(active)
        if self.ema is not None:
            self.ema.sync_memory(self.model)
        if self.bank_ema is not None:
            self.bank_ema.sync_memory(self.model)

    def train_step_mixed(self,batch,pet_available):
        cfg=self.config
        ct=batch['ct'].to(self.device,non_blocking=True)
        pet=batch['pet'].to(self.device,non_blocking=True)
        y=batch['mask'].to(self.device,non_blocking=True).float()
        state=pet_available.to(self.device)
        fi=state.eq(1).nonzero(as_tuple=True)[0]; mi=state.eq(0).nonzero(as_tuple=True)[0]
        if fi.numel()==0 or mi.numel()==0:
            raise ValueError('Training requires both Full and Missing rows')
        sid,gid=batch_ids(batch,self.device)
        collecting=cfg.memory_enabled and self.epoch>cfg.memory_warmup_epochs
        querying=bool(self.model.retrieval_active)
        self.pending=[None]*4; queries=None
        if collecting or querying:
            source=self.memory_source_model()
            previous=source.training
            source.eval()
            try:
                with isolated_source_forward(ct.device):
                    # Queries and candidate CT keys share this source encoder.
                    queries=source._encode_ct(ct)
                    if collecting:
                        self.pending=source.make_candidates(queries,pet.index_select(0,fi),fi,y,sid,gid,
                            cfg.memory_gain_margin,cfg.memory_min_support,cfg.memory_max_negative_fraction,
                            cfg.memory_tokens_per_full)
            finally:
                source.train(previous)
        out=self.model(ct,pet,state,sample_ids=sid,group_ids=gid,query_features=queries,
                       training_details=True)
        lf,_=self.criterion(out['logits'][fi].float(),y[fi])
        lm,_=self.criterion(out['logits'][mi].float(),y[mi])
        lc,_=self.criterion(out['ct_aux_logits'].float(),y)
        scales=[]; neg_fraction=[]; pos_fraction=[]
        for s,(zr,zp) in enumerate(zip(out['reference_logits'],out['plus_logits'])):
            zr=F.interpolate(zr.float(),size=y.shape[-2:],mode='bilinear',align_corners=False)
            zp=F.interpolate(zp.float(),size=y.shape[-2:],mode='bilinear',align_corners=False)
            target=y[fi]
            lr,_=self.criterion(zr,target); lp,_=self.criterion(zp,target)
            audit=audit_regions(zr,zp,target,out['delta'][s].shape[-2:],
                                cfg.memory_gain_margin,cfg.memory_min_support,cfg.memory_max_negative_fraction)
            if cfg.region_calibration:
                weight=1+cfg.negative_region_weight*audit['negative'].float()
                pixel=F.binary_cross_entropy_with_logits(zp,target,reduction='none')
                # Replace the ordinary BCE component; Dice is unchanged.
                lp=lp+(pixel*weight).sum()/weight.sum().clamp_min(1)-pixel.mean()
            scales.append(.5*(lr+lp))
            neg_fraction.append(audit['negative'].float().mean())
            pos_fraction.append(audit['positive'].float().mean())
        ls=torch.stack(scales).mean()
        loss=cfg.full_loss_weight*lf+cfg.missing_loss_weight*lm+cfg.ct_aux_weight*lc+cfg.scale_loss_weight*ls
        if not torch.isfinite(loss):
            self.pending=[None]*4
            raise RuntimeError('Non-finite correction loss')
        stats={'loss_full':lf.detach(),'loss_missing':lm.detach(),'loss_ct_aux':lc.detach(),
               'loss_scale':ls.detach(),'num_full':fi.numel(),'num_missing':mi.numel(),
               'positive_fraction':torch.stack(pos_fraction).mean().detach(),
               'negative_fraction':torch.stack(neg_fraction).mean().detach(),
               'candidate_tokens':sum(0 if p is None else p[0].shape[0] for p in self.pending)}
        return loss,out['logits'],out,stats

    @torch.no_grad()
    def after_successful_step(self):
        step=int(self.model.memory_step)+1
        self.model.memory_step.fill_(step)
        # Update stable source only after a real optimizer update.
        self.update_ema()
        if self.bank_ema_active:
            self.bank_ema.update(self.model)
        self.model.commit_candidates(self.pending,step)
        self.pending=[None]*4
        if self.ema is not None:
            self.ema.sync_memory(self.model)
        if self.bank_ema is not None:
            self.bank_ema.sync_memory(self.model)

    def save_checkpoint(self,path,epoch,best=None,best_epoch=None,val=None):
        # Same baseline checkpoint contract plus the independent bank source;
        # one write, no save/reload/rewrite of large checkpoints.
        os.makedirs(os.path.dirname(path),exist_ok=True)
        payload={
            'epoch':epoch,'global_batch_step':self.global_batch_step,
            'model':self.model.state_dict(),
            'model_ema':self.ema.state_dict() if self.ema is not None else None,
            'ema_updates':self.ema.updates if self.ema is not None else 0,
            'model_bank_ema':self.bank_ema.state_dict() if self.bank_ema is not None else None,
            'bank_ema_updates':self.bank_ema.updates if self.bank_ema is not None else 0,
            'bank_ema_active':self.bank_ema_active,
            'optimizer':self.optimizer.state_dict(),
            'scheduler':None if self.scheduler is None else self.scheduler.state_dict(),
            'scaler':self.scaler.state_dict(),'best':best,'best_epoch':best_epoch,'val':val,
            'eval_weights':self.eval_weights_tag(),
            'random_state':getattr(self.config,'random_state',None),
            'seed':getattr(self.config,'random_state',None),'config':vars(self.config),
            'random_state_python':random.getstate(),'random_state_numpy':np.random.get_state(),
            'random_state_torch':torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload['random_state_cuda']=torch.cuda.get_rng_state_all()
        torch.save(payload,path)

    @torch.no_grad()
    def evaluate(self,loader,eval_mode='full',tag='val',model=None):
        active=self.eval_model() if model is None else model
        previous=active.training; active.eval()
        source=self.memory_source_model()
        source_previous=source.training; source.eval()
        metric=SegmentationMetricsCIPA(threshold=self.metrics.threshold,hd95_mode=self.metrics.hd95_mode)
        total,count=0.,0
        try:
            for batch in loader:
                ct=batch['ct'].to(self.device); y=batch['mask'].to(self.device).float()
                sid,gid=batch_ids(batch,self.device)
                if eval_mode=='ct':
                    logits=active.ct_aux_decoder(active._encode_ct(ct),ct.shape[-2:])['logits']
                elif eval_mode in ('full','missing'):
                    pet=batch['pet'].to(self.device) if eval_mode=='full' else None
                    queries=None
                    if eval_mode=='missing' and bool(active.retrieval_active) and source is not active:
                        # The segmenter may use delayed evaluation EMA, while
                        # keys/queries consistently use the bank momentum encoder.
                        queries=source._encode_ct(ct)
                    logits=active(ct,pet,forward_mode=eval_mode,sample_ids=sid,group_ids=gid,
                                  query_features=queries)['logits']
                else:
                    raise ValueError('Unsupported correction evaluation mode')
                loss,_=self.criterion(logits.float(),y)
                metric.update(logits,y); total+=float(loss)*ct.shape[0]; count+=ct.shape[0]
        finally:
            source.train(source_previous)
            active.train(previous)
        return {**metric.compute(),'total_loss':total/max(1,count),'eval_mode':eval_mode,
                'weights':self.eval_weights_tag(),'tag':tag}
