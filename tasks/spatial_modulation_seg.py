"""Mixed segmentation with optional normalized L1 supervision of affine modulation."""

import os
import random
import time
import numpy as np
import torch
from tasks.mdt_seg import MDTSegTeacher
from models.dual_shared_add_baseline import _validate_state


class SpatialModulationSegTask(MDTSegTeacher):
    def __init__(self, networks, config):
        super().__init__(networks, config)
        self._pending_candidates = None
        self.current_epoch, self.no_improve = 0, 0
        self.loader_generators = {}
        self.matrix_epoch_stats = {}
        self.capture_matrix_reads = False
        self.matrix_read_stats = None
        self.mod_loss_sum, self.mod_loss_batches = 0.0, 0
        for name, value in self.model.matrix_settings.items():
            setattr(config, "matrix_" + name, value)
        self.config.use_deep_supervision = False
        self.config.deep_supervision = False

    def begin_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self.matrix_epoch_stats = {}
        self.matrix_read_stats = None
        super().begin_epoch(epoch)
        self.mod_loss_sum, self.mod_loss_batches = 0.0, 0
        self.model.matrix.set_epoch(epoch)
        if self.ema is not None:
            self.ema.model.matrix.set_epoch(epoch)

    def train_step_mixed(self, batch, pet_available):
        if not self.model.training:
            raise RuntimeError("requires train mode")
        if self._pending_candidates is not None:
            raise RuntimeError("commit/discard pending candidates first")
        ct = batch["ct"].to(self.device, non_blocking=True)
        pet = batch["pet"].to(self.device, non_blocking=True)
        target = batch["mask"].to(self.device, non_blocking=True).float()
        state = _validate_state(pet_available, len(ct)).to(self.device)
        full, missing = state.eq(1), state.eq(0)
        nf, nm = int(full.sum()), int(missing.sum())
        if nf == 0 or nf != nm:
            raise ValueError("requires exact-half Full/Missing batch")
        out = self.model(
            ct,
            pet,
            state,
            return_knowledge=self.model.matrix_enabled,
            return_diagnostics=self.capture_matrix_reads,
        )
        self.matrix_read_stats = out.pop("matrix_reads", None)
        logits = out["logits"]
        lf, _ = self.criterion(logits[full].float(), target[full])
        lm, _ = self.criterion(logits[missing].float(), target[missing])
        loss = 0.5 * (lf + lm)

        if self.model.matrix_enabled:
            knowledge = out.pop("knowledge")
            reference = target.index_select(0, knowledge["full_index"])
            losses = []
            pending = []
            for scale, c, f in zip(
                self.model.matrix.scales, knowledge["ct"], knowledge["full"]
            ):
                value, payload = scale.supervise(
                    c, f, reference, self.config.matrix_mod_loss_enabled
                )
                losses.append(value)
                pending.append(payload)
            mod_loss = torch.stack(losses).mean()
            loss = loss + self.config.matrix_mod_loss_weight * mod_loss
            self._pending_candidates = pending
            self.mod_loss_sum = getattr(self, "mod_loss_sum", 0.0) + float(
                mod_loss.detach()
            )
            self.mod_loss_batches = getattr(self, "mod_loss_batches", 0) + 1
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite modulation loss")
        stats = dict(
            loss_total=loss.detach(),
            loss_full=lf.detach(),
            loss_missing=lm.detach(),
            num_full=nf,
            num_missing=nm,
        )
        return loss, logits, out, stats

    def after_optimizer_step(self, succeeded):
        pending, self._pending_candidates = self._pending_candidates, None
        if succeeded and pending is not None:
            self.model.matrix.commit(pending)
        for key, value in dict(
            attempted_batches=1,
            successful_batches=int(succeeded),
            skipped_batches=int(not succeeded),
        ).items():
            self.matrix_epoch_stats[key] = self.matrix_epoch_stats.get(key, 0) + value

    def evaluate(self, loader, eval_mode="full", tag="val", model=None):
        """Keep parent metrics/dataflow; report completed evaluation batches."""
        interval = int(getattr(self.config, "log_interval", 100))
        if interval < 1:
            raise ValueError("log_interval must be positive")
        total = len(loader) if hasattr(loader, "__len__") else None
        start = time.time()
        print(
            f'[{tag}] start mode={eval_mode} batches={total if total is not None else "?"}; '
            "waiting for first batch...",
            flush=True,
        )

        def progress_batches():
            for index, batch in enumerate(loader, 1):
                yield batch
                # Parent has completed forward, loss and metrics by now.
                if index % interval == 0:
                    print(
                        f'[{tag}] batch {index}/{total if total is not None else "?"} '
                        f"elapsed={time.time() - start:.0f}s",
                        flush=True,
                    )

        result = super().evaluate(
            progress_batches(), eval_mode=eval_mode, tag=tag, model=model
        )
        print(
            f'[{tag}] complete Dice={result["dice"]:.4f} HD95={result["hd95"]:.4f} '
            f"elapsed={time.time() - start:.1f}s",
            flush=True,
        )
        return result

    def update_ema(self):
        decay = super().update_ema()
        if self.ema is not None and self.ema_active:
            with torch.no_grad():
                source = dict(self.model.matrix.named_buffers())
                for name, target in self.ema.model.matrix.named_buffers():
                    target.copy_(source[name])
        return decay

    def save_checkpoint(self, path, epoch, best=None, best_epoch=None, val=None):
        if self._pending_candidates is not None:
            raise RuntimeError(
                "checkpoint only after committing/discarding pending knowledge"
            )
        # Same baseline schema, plus explicit lifecycle/loader state. Atomic
        # replacement avoids destroying a valid checkpoint on an interrupted write.
        ckpt = dict(
            epoch=epoch,
            global_batch_step=self.global_batch_step,
            model=self.model.state_dict(),
            model_ema=self.ema.state_dict() if self.ema is not None else None,
            ema_updates=self.ema.updates if self.ema is not None else 0,
            optimizer=self.optimizer.state_dict(),
            scheduler=(
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            scaler=self.scaler.state_dict(),
            best=best,
            best_epoch=best_epoch,
            val=val,
            eval_weights=self.eval_weights_tag(),
            random_state=getattr(self.config, "random_state", None),
            seed=getattr(self.config, "random_state", None),
            config=vars(self.config),
            random_state_python=random.getstate(),
            random_state_numpy=np.random.get_state(),
            random_state_torch=torch.get_rng_state(),
            ema_active=self.ema_active,
            no_improve=self.no_improve,
            experiment_type="ct_full_spatial_modulation",
            matrix_diagnostics=self.model.matrix.diagnostics(),
            matrix_refresh_stats=[
                [bank.refresh_stats for bank in (scale.bank,)]
                for scale in self.model.matrix.scales
            ],
            loader_generator_states={
                name: gen.get_state() for name, gen in self.loader_generators.items()
            },
        )
        if torch.cuda.is_available():
            ckpt["random_state_cuda"] = torch.cuda.get_rng_state_all()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        try:
            torch.save(ckpt, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def resume(self, path):
        ckpt = self.load_state_dicts(path)
        if ckpt.get("experiment_type") != "ct_full_spatial_modulation":
            raise ValueError(
                "resume requires a matrix checkpoint; baseline weights are a separate warm start"
            )
        # Non-tensor settings affect the meaning of restored banks. Never
        # silently resume using a different projection/collection policy.
        old, current = dict(ckpt["config"]), vars(self.config)
        keys = {k for k in old.keys() | current.keys() if k.startswith("matrix_")}
        keys.update(
            {
                "decoder_channels",
                "ct_backbone",
                "pet_backbone",
                "batch_size",
                "ema_enabled",
                "ema_decay",
                "ema_decay_warmup",
                "ema_start_epoch",
                "learning_rate",
                "weight_decay",
                "epochs",
                "cosine_warmup",
                "cosine_min_lr",
                "lr_flat_ratio",
                "random_state",
                "steps_per_epoch",
                "train_split_file",
                "val_split_file",
                "test_split_file",
                "image_size_2d",
                "aug_mode",
                "norm_mode",
                "hd95_mode",
                "mixed_precision",
                "num_workers",
                "use_deep_supervision",
                "deep_supervision",
                "accumulation_steps",
                "train_batch_mode",
            }
        )
        for key in keys:
            a, b = old.get(key), current.get(key)
            if key == "decoder_channels" and a is not None and b is not None:
                a, b = tuple(a), tuple(b)
            if a != b:
                raise ValueError(
                    f"resume configuration mismatch for {key}: {a!r} != {b!r}"
                )
        self.model.load_state_dict(ckpt["model"], strict=True)
        if "matrix_refresh_stats" in ckpt:
            for scale, stats in zip(
                self.model.matrix.scales, ckpt["matrix_refresh_stats"]
            ):
                for bank, saved in zip((scale.bank,), stats):
                    bank.refresh_stats = dict(saved)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt.get("scaler", {}))
        if self.ema is not None:
            if ckpt.get("model_ema") is None:
                raise ValueError("EMA requested but checkpoint has no EMA model")
            self.ema.model.load_state_dict(ckpt["model_ema"], strict=True)
            self.ema.updates = ckpt.get("ema_updates", 0)
        self.ema_active = bool(ckpt.get("ema_active", False))
        self.global_batch_step = int(ckpt["global_batch_step"])
        self.current_epoch = int(ckpt["epoch"])
        self.no_improve = int(ckpt.get("no_improve", 0))
        states = ckpt.get("loader_generator_states", {})
        if set(states) != set(self.loader_generators):
            raise ValueError(
                "resume DataLoader generator names do not match checkpoint"
            )
        for name, generator in self.loader_generators.items():
            generator.set_state(states[name].cpu())
        random.setstate(ckpt["random_state_python"])
        np.random.set_state(ckpt["random_state_numpy"])
        torch.set_rng_state(ckpt["random_state_torch"].cpu())
        if torch.cuda.is_available() and "random_state_cuda" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["random_state_cuda"])
        return ckpt
