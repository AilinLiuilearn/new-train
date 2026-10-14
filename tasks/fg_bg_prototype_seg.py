"""Scale-supervised FG/BG experiment; baseline task remains unchanged."""
import os
import random
import time

import numpy as np
import torch

from tasks.mdt_seg import MDTSegTeacher
from models.dual_shared_add_baseline import _validate_state


class ForegroundBackgroundSegTask(MDTSegTeacher):
    def __init__(self, networks, config):
        super().__init__(networks, config)
        self.ct_aux_weight = float(getattr(config, 'prototype_ct_aux_weight', .1))
        if not np.isfinite(self.ct_aux_weight) or self.ct_aux_weight < 0:
            raise ValueError('CT auxiliary weight must be finite and nonnegative')
        self._pending_candidates = None
        self.current_epoch, self.no_improve = 0, 0
        self.loader_generators = {}
        self.prototype_epoch_stats = {}
        self.capture_prototype_reads = False
        self.prototype_read_stats = None
        self.deep_weight = float(getattr(config, 'prototype_deep_supervision_weight', .2))
        if not np.isfinite(self.deep_weight) or self.deep_weight < 0:
            raise ValueError('deep supervision weight must be finite and nonnegative')
        if self.model.gain_mode == 'per_scale' and self.deep_weight <= 0:
            raise ValueError('per_scale gain requires positive deep supervision weight')
        scale = self.model.prototype.scales[0]
        effective = dict(gain_mode=self.model.gain_mode, enabled=self.model.prototype_enabled,
            adaptive_sampling=scale.adaptive_sampling, sampling_decay=scale.sampling_decay,
            logit_scale=scale.logit_scale, correction_alpha=scale.correction_alpha,
            min_gain=self.model.min_gain, deep_supervision_weight=self.deep_weight,
            warmup_steps=self.model.prototype.warmup_steps, dim=scale.dim,
            max_patches=scale.max_patches, clusters=scale.foreground_bank.clusters,
            capacity=scale.foreground_bank.capacity, min_candidates=scale.foreground_bank.min_candidates,
            decay=scale.foreground_bank.decay, refresh_steps=scale.refresh_steps,
            ct_aux_weight=self.ct_aux_weight)
        for name, value in effective.items(): setattr(self.config, f'prototype_{name}', value)
        self.config.use_deep_supervision = self.model.use_deep_supervision

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
            raise RuntimeError('commit/discard pending candidates before next forward')
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        state = _validate_state(pet_available, ct.shape[0]).to(ct.device)
        full, missing = state.eq(1), state.eq(0)
        nf, nm = int(full.sum()), int(missing.sum())
        if nf == 0 or nf != nm:
            raise ValueError('requires exact-half Full/Missing batches')
        deep = self.model.use_deep_supervision and self.deep_weight > 0
        collect = self.model.prototype_enabled or self.ct_aux_weight > 0
        out = self.model(ct, pet=pet, pet_available=state, return_knowledge=collect,
                         return_scale_logits=deep, return_diagnostics=self.capture_prototype_reads)
        if self.capture_prototype_reads:
            self.prototype_read_stats = dict(global_batch_step=self.global_batch_step + 1,
                                             scales=out.pop('prototype_reads'))
        logits = out['logits']
        predictions = out.pop('scale_logits', [logits] * 4)
        knowledge = out.pop('knowledge', None)
        losses, reductions = [], {}
        for i in range(4):
            if i < 3 and not deep:
                losses.append((logits.new_zeros(()),) * 3)
                continue
            # Large 512x512 sigmoid sums can overflow fp16 in early auxiliary
            # heads. Keep every segmentation loss/reduction in float32 while
            # the encoders/decoder/heads still benefit from ambient AMP.
            lf, _ = self.criterion(predictions[i][full].float(), mask[full])
            lm, _ = self.criterion(predictions[i][missing].float(), mask[missing])
            lc = logits.new_zeros(())
            if self.ct_aux_weight:
                lc, _ = self.criterion(knowledge['reference_logits_by_scale'][i].float(), mask[full])
            losses.append((lf, lm, lc))
            for label, value in zip(('full', 'missing', 'ct'), losses[-1]):
                reductions[f'd{4-i}_loss_{label}'] = value.detach()
            reductions[f'd{4-i}_supervised_batches'] = 1
        full_loss, missing_loss, ct_loss = losses[3]
        if deep:
            deep_losses = [torch.stack([row[i] for row in losses[:3]]).mean() for i in range(3)]
        else:
            deep_losses = [logits.new_zeros(())] * 3
        deep_total = .5 * deep_losses[0] + .5 * deep_losses[1] + self.ct_aux_weight * deep_losses[2]
        total = .5 * full_loss + .5 * missing_loss + self.ct_aux_weight * ct_loss + self.deep_weight * deep_total
        if not torch.isfinite(total): raise RuntimeError('nonfinite FG/BG loss')
        if self.model.prototype_enabled:
            self._pending_candidates, diagnostics = self.model.prepare_candidates(knowledge, mask, True)
            reductions.update(diagnostics)
        self._record_prototype_stats(reductions)
        stats = dict(loss_total=total.detach(), loss_full=full_loss.detach(),
            loss_missing=missing_loss.detach(), loss_ct_aux=ct_loss.detach(),
            loss_full_deep=deep_losses[0].detach(), loss_missing_deep=deep_losses[1].detach(),
            loss_ct_deep=deep_losses[2].detach(), loss_deep_supervision=deep_total.detach(),
            num_full=nf, num_missing=nm)
        return total, logits, out, stats

    def after_optimizer_step(self, succeeded):
        pending, self._pending_candidates = self._pending_candidates, None
        if succeeded and pending is not None: self.model.prototype.commit(pending)
        writing = pending is not None and succeeded and int(self.model.prototype.update_steps) > self.model.prototype.warmup_steps
        stats = dict(attempted_batches=1, successful_batches=int(succeeded), skipped_batches=int(not succeeded),
                     warmup_batches=int(pending is not None and succeeded and not writing))
        if pending is not None:
            for i, (fg, bg, _) in enumerate(pending):
                for label, values in (('foreground', fg), ('background', bg)):
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
            no_improve=self.no_improve, experiment_type='decoder_fg_bg_scale_prototype_add',
            prototype_diagnostics=self.model.prototype.diagnostics(),
            prototype_refresh_stats=[
                [bank.refresh_stats for bank in scale.banks]
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
        if ckpt.get('experiment_type') != 'decoder_fg_bg_scale_prototype_add':
            raise ValueError('resume requires a prototype checkpoint; baseline weights are a separate warm start')
        # Non-tensor settings affect the meaning of restored banks. Never
        # silently resume using a different projection/collection policy.
        old, current = dict(ckpt['config']), vars(self.config)
        keys = {k for k in old.keys() | current.keys() if k.startswith('prototype_')}
        keys.update({'decoder_channels', 'ct_backbone', 'pet_backbone', 'batch_size',
                     'ema_enabled', 'ema_decay', 'ema_decay_warmup', 'ema_start_epoch',
                     'learning_rate', 'weight_decay', 'epochs', 'cosine_warmup',
                     'cosine_min_lr', 'lr_flat_ratio', 'random_state', 'steps_per_epoch',
                     'train_split_file', 'val_split_file', 'test_split_file',
                     'image_size_2d', 'aug_mode', 'norm_mode', 'hd95_mode',
                     'mixed_precision', 'num_workers', 'use_deep_supervision', 'deep_supervision',
                     'accumulation_steps', 'train_batch_mode'})
        for key in keys:
            a, b = old.get(key), current.get(key)
            if key == 'decoder_channels' and a is not None and b is not None:
                a, b = tuple(a), tuple(b)
            if a != b:
                raise ValueError(f'resume configuration mismatch for {key}: {a!r} != {b!r}')
        self.model.load_state_dict(ckpt['model'], strict=True)
        if 'prototype_refresh_stats' in ckpt:
            for scale, stats in zip(self.model.prototype.scales, ckpt['prototype_refresh_stats']):
                for bank, saved in zip(scale.banks, stats):
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
