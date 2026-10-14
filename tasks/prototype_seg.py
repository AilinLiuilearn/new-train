"""Prototype experiment task; baseline task remains unchanged."""
import os
import random
import time

import numpy as np
import torch

from tasks.mdt_seg import MDTSegTeacher
from models.dual_shared_add_baseline import _validate_state


class PrototypeSegTask(MDTSegTeacher):
    def __init__(self, networks, config):
        super().__init__(networks, config)
        self.ct_aux_weight = float(getattr(config, 'prototype_ct_aux_weight', .1))
        if self.ct_aux_weight < 0:
            raise ValueError('CT auxiliary weight must be nonnegative')
        self._pending_candidates = None
        self.current_epoch, self.no_improve = 0, 0
        self.loader_generators = {}
        self.prototype_epoch_stats = {}
        self.capture_prototype_reads = False
        self.prototype_read_stats = None
        # Persist the actual effective setting, including legacy configs.
        self.config.prototype_outlier_filter = self.model.prototype.scales[0].spatial_bank.outlier_filter

    def begin_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self.prototype_epoch_stats = {}
        self.prototype_read_stats = None
        super().begin_epoch(epoch)

    @torch.no_grad()
    def _record_prototype_stats(self, stats):
        for key, value in stats.items():
            value = torch.as_tensor(value, dtype=torch.float64, device=self.device).detach()
            if key not in self.prototype_epoch_stats:
                self.prototype_epoch_stats[key] = value.clone()
            else:
                self.prototype_epoch_stats[key].add_(value)

    def train_step_mixed(self, batch, pet_available):
        if not self.model.training:
            raise RuntimeError('mixed training requires model.train()')
        if self._pending_candidates is not None:
            raise RuntimeError('call after_optimizer_step(success) before another training forward')
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        state = _validate_state(pet_available, ct.shape[0]).to(self.device)
        full, missing = state.eq(1), state.eq(0)
        nf, nm = int(full.sum()), int(missing.sum())
        if nf == 0 or nf != nm:
            raise ValueError('prototype training requires exact-half Full/Missing batches')
        collect = self.model.prototype_enabled or self.ct_aux_weight > 0
        out = self.model(ct, pet=pet, pet_available=state, return_knowledge=collect,
                         return_diagnostics=self.capture_prototype_reads)
        if self.capture_prototype_reads:
            self.prototype_read_stats = dict(global_batch_step=self.global_batch_step + 1,
                                             scales=out.pop('prototype_reads'))
        logits = out['logits']
        full_loss, _ = self.criterion(logits[full], mask[full])
        missing_loss, _ = self.criterion(logits[missing], mask[missing])
        ct_loss = logits.new_zeros(())
        if collect:
            knowledge = out.pop('knowledge')
            if self.ct_aux_weight:
                ct_loss, _ = self.criterion(knowledge['reference_logits'], mask[full])
            if self.model.prototype_enabled:
                self._pending_candidates, diagnostics = self.model.prepare_candidates(
                    knowledge, mask, return_diagnostics=True)
                self._record_prototype_stats(diagnostics)
        total = .5 * full_loss + .5 * missing_loss + self.ct_aux_weight * ct_loss
        if not torch.isfinite(total):
            self._pending_candidates = None
            raise RuntimeError('nonfinite prototype training loss')
        stats = dict(loss_total=total.detach(), loss_full=full_loss.detach(),
                     loss_missing=missing_loss.detach(), loss_ct_aux=ct_loss.detach(),
                     num_full=nf, num_missing=nm)
        return total, logits, out, stats

    def after_optimizer_step(self, succeeded):
        pending, self._pending_candidates = self._pending_candidates, None
        if succeeded and pending is not None:
            self.model.prototype.commit(pending)
        if pending is not None:
            writing = succeeded and int(self.model.prototype.update_steps) > self.model.prototype.warmup_steps
            stats = dict(attempted_batches=1, successful_batches=int(succeeded),
                         skipped_batches=int(not succeeded),
                         warmup_batches=int(succeeded and not writing))
            for i, (spatial, channel) in enumerate(pending):
                for label, values in (('spatial', spatial), ('channel', channel)):
                    stats[f'd{4-i}_{label}_written'] = values.shape[0] if writing else 0
            self._record_prototype_stats(stats)

    def evaluate(self, loader, eval_mode='full', tag='val', model=None):
        """Keep parent metrics/dataflow; report completed evaluation batches."""
        interval = int(getattr(self.config, 'log_interval', 100))
        if interval < 1:
            raise ValueError('log_interval must be positive')
        total = len(loader) if hasattr(loader, '__len__') else None
        start = time.time()
        print(f'[{tag}] start mode={eval_mode} batches={total if total is not None else "?"}; '
              'waiting for first batch...', flush=True)

        def progress_batches():
            for index, batch in enumerate(loader, 1):
                yield batch
                # Parent has completed forward, loss and metrics by now.
                if index % interval == 0:
                    print(f'[{tag}] batch {index}/{total if total is not None else "?"} '
                          f'elapsed={time.time() - start:.0f}s', flush=True)

        result = super().evaluate(progress_batches(), eval_mode=eval_mode, tag=tag, model=model)
        print(f'[{tag}] complete Dice={result["dice"]:.4f} HD95={result["hd95"]:.4f} '
              f'elapsed={time.time() - start:.1f}s', flush=True)
        return result

    def update_ema(self):
        decay = super().update_ema()
        if self.ema is not None and self.ema_active:
            # Centroid EMA is already applied by the library. Hard-copy ALL
            # prototype buffers, including queues/validity/counters/basis.
            with torch.no_grad():
                source = dict(self.model.prototype.named_buffers())
                for name, target in self.ema.model.prototype.named_buffers():
                    target.copy_(source[name])
        return decay

    def save_checkpoint(self, path, epoch, best=None, best_epoch=None, val=None):
        if self._pending_candidates is not None:
            raise RuntimeError('checkpoint only after committing/discarding pending knowledge')
        # Same baseline schema, plus explicit lifecycle/loader state. Atomic
        # replacement avoids destroying a valid checkpoint on an interrupted write.
        ckpt = dict(
            epoch=epoch, global_batch_step=self.global_batch_step,
            model=self.model.state_dict(),
            model_ema=self.ema.state_dict() if self.ema is not None else None,
            ema_updates=self.ema.updates if self.ema is not None else 0,
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            scaler=self.scaler.state_dict(), best=best, best_epoch=best_epoch, val=val,
            eval_weights=self.eval_weights_tag(), random_state=getattr(self.config, 'random_state', None),
            seed=getattr(self.config, 'random_state', None), config=vars(self.config),
            random_state_python=random.getstate(), random_state_numpy=np.random.get_state(),
            random_state_torch=torch.get_rng_state(), ema_active=self.ema_active,
            no_improve=self.no_improve, experiment_type='decoder_dual_prototype_add',
            prototype_diagnostics=self.model.prototype.diagnostics(),
            prototype_refresh_stats=[
                [bank.refresh_stats for bank in (scale.spatial_bank, scale.channel_bank)]
                for scale in self.model.prototype.scales],
            loader_generator_states={name: gen.get_state() for name, gen in self.loader_generators.items()})
        if torch.cuda.is_available():
            ckpt['random_state_cuda'] = torch.cuda.get_rng_state_all()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + '.tmp'
        try:
            torch.save(ckpt, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def resume(self, path):
        ckpt = self.load_state_dicts(path)
        if ckpt.get('experiment_type') != 'decoder_dual_prototype_add':
            raise ValueError('resume requires a prototype checkpoint; baseline weights are a separate warm start')
        # Non-tensor settings affect the meaning of restored banks. Never
        # silently resume using a different projection/collection policy.
        old, current = dict(ckpt['config']), vars(self.config)
        old.setdefault('prototype_outlier_filter', True)  # Legacy checkpoints used MAD trimming.
        keys = {k for k in old.keys() | current.keys() if k.startswith('prototype_')}
        keys.update({'decoder_channels', 'ct_backbone', 'pet_backbone', 'batch_size',
                     'ema_enabled', 'ema_decay', 'ema_decay_warmup', 'ema_start_epoch',
                     'learning_rate', 'weight_decay', 'epochs', 'cosine_warmup',
                     'cosine_min_lr', 'lr_flat_ratio', 'random_state', 'steps_per_epoch',
                     'train_split_file', 'val_split_file', 'test_split_file',
                     'image_size_2d', 'aug_mode', 'norm_mode', 'hd95_mode',
                     'mixed_precision', 'num_workers'})
        for key in keys:
            a, b = old.get(key), current.get(key)
            if key == 'decoder_channels' and a is not None and b is not None:
                a, b = tuple(a), tuple(b)
            if a != b:
                raise ValueError(f'resume configuration mismatch for {key}: {a!r} != {b!r}')
        self.model.load_state_dict(ckpt['model'], strict=True)
        if 'prototype_refresh_stats' in ckpt:
            for scale, stats in zip(self.model.prototype.scales, ckpt['prototype_refresh_stats']):
                for bank, saved in zip((scale.spatial_bank, scale.channel_bank), stats):
                    bank.refresh_stats = dict(saved)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None and ckpt.get('scheduler') is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt.get('scaler', {}))
        if self.ema is not None:
            if ckpt.get('model_ema') is None:
                raise ValueError('EMA requested but checkpoint has no EMA model')
            self.ema.model.load_state_dict(ckpt['model_ema'], strict=True)
            self.ema.updates = ckpt.get('ema_updates', 0)
        self.ema_active = bool(ckpt.get('ema_active', False))
        self.global_batch_step = int(ckpt['global_batch_step'])
        self.current_epoch = int(ckpt['epoch'])
        self.no_improve = int(ckpt.get('no_improve', 0))
        states = ckpt.get('loader_generator_states', {})
        if set(states) != set(self.loader_generators):
            raise ValueError('resume DataLoader generator names do not match checkpoint')
        for name, generator in self.loader_generators.items():
            generator.set_state(states[name].cpu())
        random.setstate(ckpt['random_state_python'])
        np.random.set_state(ckpt['random_state_numpy'])
        torch.set_rng_state(ckpt['random_state_torch'].cpu())
        if torch.cuda.is_available() and 'random_state_cuda' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['random_state_cuda'])
        return ckpt
