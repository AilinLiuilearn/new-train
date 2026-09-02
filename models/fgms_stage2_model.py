"""FGMS Stage2 wrapper with Progressive Co-Adaptation phase control."""

from __future__ import annotations

import contextlib
import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from models.fbase_guided_modality_specialized_taskmoe import (
    FBaseGuidedCrossScaleModalitySpecializedTaskMoE,
)

STAGE1_CORE_PREFIXES = (
    "enc_ct.",
    "enc_pet.",
    "ct_align.",
    "prototype_memory.",
    "pet_calibration.",
    "fusion.",
    "decoder.",
)

FORBIDDEN_STAGE1_PREFIXES = (
    "stage1.enc_ct.",
    "stage1.enc_pet.",
    "stage1.ct_align.",
    "stage1.prototype_memory.",
    "stage1.decoder.",
)

BOUNDARY_STAGE1_PREFIXES = (
    "stage1.pet_calibration.",
    "stage1.fusion.",
)

EXPECTED_CHANNELS = (64, 128, 320, 512)

PHASE_MOE_WARMUP = "moe_warmup"
PHASE_STAGE2_ADAPT = "stage2_adapt"
PHASE_BOUNDARY_COADAPT = "boundary_coadapt"


def _disable_amp_context(device: torch.device):
    """FGMS expert bank mixes fp32 accumulators with fp16 expert outputs under AMP."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def _unwrap_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _is_core_key(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in STAGE1_CORE_PREFIXES)


def load_stage1_checkpoint(stage1: nn.Module, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(checkpoint)
    result = stage1.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    core_missing = [k for k in missing_keys if _is_core_key(k)]
    if core_missing:
        raise RuntimeError(
            f"Stage1 checkpoint missing core keys ({len(core_missing)}): {core_missing[:10]}"
        )

    proto = stage1.prototype_memory
    if not proto.bank_ready:
        raise RuntimeError(
            "Stage1 CPPI prototype bank is not ready. "
            "Provide a fully trained Stage1 checkpoint with populated prototype buffers."
        )

    print(f"[STAGE1] checkpoint path={checkpoint_path}")
    print(f"[STAGE1] checkpoint epoch={checkpoint.get('epoch', 'N/A')}")
    print(f"[STAGE1] best_joint={checkpoint.get('best_joint', 'N/A')}")
    print(f"[STAGE1] best_joint_epoch={checkpoint.get('best_joint_epoch', 'N/A')}")
    print(f"[STAGE1] missing_keys={missing_keys}")
    print(f"[STAGE1] unexpected_keys={unexpected_keys}")
    print(f"[STAGE1] CPPI bank_version={int(proto.bank_version.item())}")
    print(f"[STAGE1] CPPI ready_slots={int(proto.prototype_ready.sum().item())}")

    return checkpoint


def freeze_stage1(stage1: nn.Module) -> None:
    for p in stage1.parameters():
        p.requires_grad = False
    stage1.eval()


def copy_stage2_decoder(stage1_decoder: nn.Module) -> nn.Module:
    stage2_decoder = copy.deepcopy(stage1_decoder)
    for old_p, new_p in zip(stage1_decoder.parameters(), stage2_decoder.parameters()):
        if not torch.equal(old_p.data, new_p.data):
            raise RuntimeError(
                "Stage2 decoder copy mismatch: parameter values differ from Stage1 decoder."
            )
        if old_p.data_ptr() == new_p.data_ptr():
            raise RuntimeError("Stage2 decoder shares storage with Stage1 decoder; deepcopy failed.")
    for p in stage2_decoder.parameters():
        p.requires_grad = False
    return stage2_decoder


def get_cppi_fingerprint(stage1: nn.Module) -> Dict[str, torch.Tensor]:
    proto = stage1.prototype_memory
    fingerprint = {
        "bank_version": proto.bank_version.detach().clone(),
        "prototype_ready": proto.prototype_ready.detach().clone(),
    }
    for scale_idx in range(1, 5):
        fingerprint[f"ct_keys_s{scale_idx}"] = getattr(proto, f"ct_keys_s{scale_idx}").detach().clone()
        fingerprint[f"pet_values_s{scale_idx}"] = getattr(proto, f"pet_values_s{scale_idx}").detach().clone()
    return fingerprint


def assert_cppi_unchanged(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> None:
    for key, val_before in before.items():
        val_after = after[key]
        if not torch.equal(val_before, val_after):
            raise RuntimeError(f"CPPI readonly violation: {key} changed during Stage2 training.")


class FGMSStage2PETCTModel(nn.Module):
    """Frozen Stage1 front-end + progressive Stage2 / boundary co-adaptation."""

    is_fgms_stage2 = True

    def __init__(self, stage1: nn.Module, config) -> None:
        super().__init__()
        self.config = config
        self.stage1 = stage1

        self.progressive_enabled = bool(getattr(config, "fgms_progressive_coadapt", True))
        self.decoder_unfreeze_epoch = int(getattr(config, "fgms_decoder_unfreeze_epoch", 2))
        self.boundary_unfreeze_epoch = int(getattr(config, "fgms_boundary_unfreeze_epoch", 4))

        self.current_phase = PHASE_STAGE2_ADAPT
        self.decoder_unlocked = not self.progressive_enabled
        self.stage1_boundary_unlocked = False

        channels = tuple(stage1.enc_pet.feature_info.channels())
        if tuple(channels) != EXPECTED_CHANNELS:
            raise ValueError(f"Expected channels {EXPECTED_CHANNELS}, got {channels}")

        self.stage2_moe = FBaseGuidedCrossScaleModalitySpecializedTaskMoE(
            channels=channels,
            expert_dim=int(getattr(config, "fgms_expert_dim", 128)),
            num_experts=int(getattr(config, "fgms_num_experts", 6)),
            top_k=int(getattr(config, "fgms_top_k", 2)),
            atom_num=32,
            atom_dim=256,
            mlp_ratio=2.0,
            enable_balance_loss=bool(getattr(config, "fgms_enable_balance_loss", True)),
            balance_loss_weight=float(getattr(config, "fgms_balance_loss_weight", 0.1)),
            residual_mode=str(getattr(config, "fgms_residual_mode", "zero_start")),
        )
        self.stage2_decoder = copy_stage2_decoder(stage1.decoder)
        self._boundary_param_ref = self._snapshot_boundary_params()
        self.configure_trainable_phase(1)
        self._print_startup_summary()

    @property
    def prototype_memory(self):
        return self.stage1.prototype_memory

    def _snapshot_boundary_params(self) -> Dict[str, torch.Tensor]:
        refs = {}
        for name, param in self.stage1.pet_calibration.named_parameters():
            refs[f"pet_calibration.{name}"] = param.detach().cpu().clone()
        for name, param in self.stage1.fusion.named_parameters():
            refs[f"fusion.{name}"] = param.detach().cpu().clone()
        return refs

    def _phase_from_epoch(self, epoch: int) -> str:
        if not self.progressive_enabled:
            return PHASE_STAGE2_ADAPT
        if epoch < self.decoder_unfreeze_epoch:
            return PHASE_MOE_WARMUP
        if epoch < self.boundary_unfreeze_epoch:
            return PHASE_STAGE2_ADAPT
        return PHASE_BOUNDARY_COADAPT

    def _boundary_trainable(self) -> bool:
        return self.stage1_boundary_unlocked

    def _freeze_forbidden_stage1(self) -> None:
        for module in (
            self.stage1.enc_ct,
            self.stage1.enc_pet,
            self.stage1.ct_align,
            self.stage1.prototype_memory,
            self.stage1.decoder,
        ):
            for p in module.parameters():
                p.requires_grad = False

    def _apply_phase_requires_grad(self) -> None:
        self._freeze_forbidden_stage1()

        for p in self.stage2_moe.parameters():
            p.requires_grad = True

        for p in self.stage2_decoder.parameters():
            p.requires_grad = self.decoder_unlocked

        boundary_train = self.stage1_boundary_unlocked
        for p in self.stage1.pet_calibration.parameters():
            p.requires_grad = boundary_train
        for p in self.stage1.fusion.parameters():
            p.requires_grad = boundary_train

    def assert_trainable_parameters_phase(self) -> None:
        allowed_prefixes = ("stage2_moe.", "stage2_decoder.")
        if self.stage1_boundary_unlocked:
            allowed_prefixes = allowed_prefixes + BOUNDARY_STAGE1_PREFIXES

        trainable_names = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            trainable_names.append(name)
            if not any(name.startswith(prefix) for prefix in allowed_prefixes):
                raise RuntimeError(f"Unexpected trainable parameter in phase={self.current_phase}: {name}")

        if self.current_phase == PHASE_MOE_WARMUP:
            expected = [n for n in trainable_names if n.startswith("stage2_moe.")]
            if not expected:
                raise RuntimeError("Phase A requires trainable stage2_moe parameters.")
            bad = [n for n in trainable_names if not n.startswith("stage2_moe.")]
            if bad:
                raise RuntimeError(f"Phase A has unexpected trainable params: {bad[:5]}")
        elif self.current_phase == PHASE_STAGE2_ADAPT:
            if not any(n.startswith("stage2_moe.") for n in trainable_names):
                raise RuntimeError("Phase B requires trainable stage2_moe parameters.")
            if not any(n.startswith("stage2_decoder.") for n in trainable_names):
                raise RuntimeError("Phase B requires trainable stage2_decoder parameters.")
            bad = [n for n in trainable_names if not (
                n.startswith("stage2_moe.") or n.startswith("stage2_decoder.")
            )]
            if bad:
                raise RuntimeError(f"Phase B has unexpected trainable params: {bad[:5]}")
        elif self.current_phase == PHASE_BOUNDARY_COADAPT:
            for prefix in ("stage2_moe.", "stage2_decoder.", "stage1.pet_calibration.", "stage1.fusion."):
                if not any(n.startswith(prefix) for n in trainable_names):
                    raise RuntimeError(f"Phase C missing trainable parameters for {prefix}")

        forbidden_trainable = [
            n for n in trainable_names
            if any(n.startswith(prefix) for prefix in FORBIDDEN_STAGE1_PREFIXES)
        ]
        if forbidden_trainable:
            raise RuntimeError(
                f"Forbidden Stage1 modules are trainable: {forbidden_trainable[:5]}"
            )

    def configure_trainable_phase(self, epoch: int) -> None:
        prev_phase = getattr(self, "current_phase", None)
        self.current_phase = self._phase_from_epoch(epoch)
        self.decoder_unlocked = (
            not self.progressive_enabled
            or self.current_phase in (PHASE_STAGE2_ADAPT, PHASE_BOUNDARY_COADAPT)
        )
        self.stage1_boundary_unlocked = self.current_phase == PHASE_BOUNDARY_COADAPT
        self._apply_phase_requires_grad()
        self.assert_trainable_parameters_phase()
        if prev_phase != self.current_phase:
            self._print_progressive_status(epoch)

    def _print_progressive_status(self, epoch: int) -> None:
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"[PROGRESSIVE] epoch={epoch}", flush=True)
        print(f"[PROGRESSIVE] phase={self.current_phase}", flush=True)
        print(f"[PROGRESSIVE] decoder_unlocked={self.decoder_unlocked}", flush=True)
        print(f"[PROGRESSIVE] stage1_boundary_unlocked={self.stage1_boundary_unlocked}", flush=True)
        print(f"[PROGRESSIVE] trainable modules:", flush=True)
        groups = {
            "stage2_moe": [n for n in trainable if n.startswith("stage2_moe.")],
            "stage2_decoder": [n for n in trainable if n.startswith("stage2_decoder.")],
            "stage1.pet_calibration": [n for n in trainable if n.startswith("stage1.pet_calibration.")],
            "stage1.fusion": [n for n in trainable if n.startswith("stage1.fusion.")],
        }
        for group_name, names in groups.items():
            if names:
                print(f"  - {group_name} ({len(names)} tensors)", flush=True)
        frozen = [
            "stage2_decoder" if not self.decoder_unlocked else None,
            "stage1.pet_calibration" if not self.stage1_boundary_unlocked else None,
            "stage1.fusion" if not self.stage1_boundary_unlocked else None,
            "encoders/ct_align/CPPI/old_decoder (always)",
        ]
        frozen = [x for x in frozen if x is not None]
        print(f"[PROGRESSIVE] frozen: {', '.join(frozen)}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.enc_ct.eval()
        self.stage1.enc_pet.eval()
        self.stage1.ct_align.eval()
        self.stage1.decoder.eval()

        if self._boundary_trainable() and mode:
            self.stage1.pet_calibration.train(True)
            self.stage1.fusion.train(True)
        else:
            self.stage1.pet_calibration.eval()
            self.stage1.fusion.eval()

        if mode:
            self.stage2_moe.train(True)
            if self.decoder_unlocked:
                self.stage2_decoder.train(True)
            else:
                self.stage2_decoder.eval()
        else:
            self.stage2_moe.eval()
            self.stage2_decoder.eval()
        return self

    def _print_startup_summary(self) -> None:
        proto = self.stage1.prototype_memory
        layout = self.stage2_moe.expert_layout()
        total = sum(p.numel() for p in self.parameters())
        moe_trainable = sum(p.numel() for p in self.stage2_moe.parameters() if p.requires_grad)
        dec_trainable = sum(p.numel() for p in self.stage2_decoder.parameters() if p.requires_grad)
        boundary_trainable = sum(
            p.numel()
            for p in list(self.stage1.pet_calibration.parameters()) + list(self.stage1.fusion.parameters())
            if p.requires_grad
        )
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("[STAGE2] mode=FGMS+ProgressiveCoAdapt")
        print(f"[STAGE2] progressive_enabled={self.progressive_enabled}")
        print(f"[STAGE2] decoder_unfreeze_epoch={self.decoder_unfreeze_epoch}")
        print(f"[STAGE2] boundary_unfreeze_epoch={self.boundary_unfreeze_epoch}")
        print(f"[STAGE2] Stage1 checkpoint={getattr(self.config, 'stage1_checkpoint', None)}")
        print("[STAGE2] CPPI readonly=True")
        print(f"[STAGE2] CPPI bank_version={int(proto.bank_version.item())}")
        print(f"[STAGE2] CPPI ready_slots={int(proto.prototype_ready.sum().item())}")
        print(f"[STAGE2] experts={self.stage2_moe.num_experts}")
        print(f"[STAGE2] expert_layout={layout}")
        print(f"[STAGE2] top_k={self.stage2_moe.top_k}")
        print(f"[STAGE2] balance_loss={self.stage2_moe.enable_balance_loss}")
        print(f"[STAGE2] MoE lr={float(getattr(self.config, 'learning_rate', 8e-5))}")
        print(f"[STAGE2] decoder lr={float(getattr(self.config, 'decoder_lr', 2e-5))}")
        print(f"[STAGE2] boundary lr={float(getattr(self.config, 'stage1_boundary_lr', 5e-6))}")
        print(f"[STAGE2] current_phase={self.current_phase}")
        print(f"[STAGE2] Total params={total:,}")
        print(f"[STAGE2] Stage2 MoE trainable params={moe_trainable:,}")
        print(f"[STAGE2] Stage2 Decoder trainable params={dec_trainable:,}")
        print(f"[STAGE2] Stage1 boundary trainable params={boundary_trainable:,}")
        print(f"[STAGE2] Total trainable params={total_trainable:,}")

    def _run_boundary(self, ct_feats, pet_feats, ct_reference_feats, mode: str):
        if self._boundary_trainable():
            pet_feats_cal = self.stage1.pet_calibration(
                ct_feats,
                pet_feats,
                ct_reference_feats,
                reference_valid=True,
            )
            fbase_feats = self.stage1.fusion(ct_feats, pet_feats_cal, mode=mode)
            return pet_feats_cal, fbase_feats
        with torch.no_grad():
            pet_feats_cal = self.stage1.pet_calibration(
                ct_feats,
                pet_feats,
                ct_reference_feats,
                reference_valid=True,
            )
            fbase_feats = self.stage1.fusion(ct_feats, pet_feats_cal, mode=mode)
        return pet_feats_cal, fbase_feats

    def _extract_full_features(
        self, ct: torch.Tensor, pet: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        with torch.no_grad():
            ct_feats = self.stage1._encode_ct(ct)
            pet_feats_real = self.stage1._encode_pet(pet)
            _, ct_reference_feats, _ = self.stage1._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_feats = [x.detach() for x in ct_feats]
            pet_feats_real = [x.detach() for x in pet_feats_real]
            ct_reference_feats = [x.detach() for x in ct_reference_feats]

        pet_feats_cal, fbase_feats = self._run_boundary(
            ct_feats, pet_feats_real, ct_reference_feats, mode="full"
        )
        return ct_feats, pet_feats_cal, fbase_feats

    def _extract_missing_features(
        self, ct: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        with torch.no_grad():
            ct_feats = self.stage1._encode_ct(ct)
            pet_feats_proxy, ct_reference_feats, _ = self.stage1._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_feats = [x.detach() for x in ct_feats]
            pet_feats_proxy = [x.detach() for x in pet_feats_proxy]
            ct_reference_feats = [x.detach() for x in ct_reference_feats]

        pet_feats_cal, fbase_feats = self._run_boundary(
            ct_feats, pet_feats_proxy, ct_reference_feats, mode="missing"
        )
        return ct_feats, pet_feats_cal, fbase_feats

    def _decode_stage2(self, refined_feats: Sequence[torch.Tensor], target_size) -> dict:
        out = self.stage2_decoder(refined_feats, target_size)
        out["pred"] = out["logits"]
        out["aux"] = {}
        return out

    def _forward_route(self, ct, pet, route: str, target_size):
        if route == "full":
            if pet is None:
                raise ValueError("Full route requires PET input.")
            ct_feats, pet_feats_cal, fbase_feats = self._extract_full_features(ct, pet)
        elif route == "missing":
            ct_feats, pet_feats_cal, fbase_feats = self._extract_missing_features(ct)
        else:
            raise ValueError(f"Unsupported route={route!r}")

        device = ct_feats[0].device
        with _disable_amp_context(device):
            moe_out = self.stage2_moe(
                ct_features=[x.float() for x in ct_feats],
                pet_features=[x.float() for x in pet_feats_cal],
                fbase_features=[x.float() for x in fbase_feats],
                route=route,
            )
            decoder_out = self._decode_stage2(moe_out.features, target_size)
        decoder_out["balance_loss"] = moe_out.balance_loss
        decoder_out["moe_stats"] = moe_out.stats
        return decoder_out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        if pet_available is None:
            pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError("pet_available must contain one state per sample")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0 or 1")

        if torch.all(availability == 1):
            return self._forward_route(ct, pet, "full", target_size)
        if torch.all(availability == 0):
            return self._forward_route(ct, pet, "missing", target_size)

        full_idx = torch.nonzero(availability == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(availability == 0, as_tuple=False).flatten()
        batch_size = ct.shape[0]
        logits = ct.new_zeros((batch_size, 1, *target_size))
        balance_loss = ct.new_zeros((), dtype=torch.float32)
        moe_stats = None
        total_weight = 0.0

        if full_idx.numel() > 0:
            out_full = self._forward_route(
                ct.index_select(0, full_idx), pet.index_select(0, full_idx), "full", target_size
            )
            logits.index_copy_(0, full_idx, out_full["logits"])
            w = float(full_idx.numel())
            balance_loss = balance_loss + out_full["balance_loss"] * w
            total_weight += w
            moe_stats = out_full["moe_stats"]

        if missing_idx.numel() > 0:
            out_missing = self._forward_route(
                ct.index_select(0, missing_idx), None, "missing", target_size
            )
            logits.index_copy_(0, missing_idx, out_missing["logits"])
            w = float(missing_idx.numel())
            balance_loss = balance_loss + out_missing["balance_loss"] * w
            total_weight += w
            moe_stats = out_missing["moe_stats"]

        if total_weight > 0:
            balance_loss = balance_loss / total_weight

        return {
            "logits": logits,
            "pred": logits,
            "aux": {},
            "balance_loss": balance_loss,
            "moe_stats": moe_stats,
        }

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode: str = "auto",
        mask=None,
    ):
        del mask
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == "full":
            return self._forward_route(ct, pet, "full", target_size)
        if forward_mode == "missing":
            return self._forward_route(ct, pet, "missing", target_size)
        if forward_mode == "auto":
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f"Unsupported forward_mode={forward_mode!r}")

    def get_cppi_fingerprint(self) -> Dict[str, torch.Tensor]:
        return get_cppi_fingerprint(self.stage1)

    def _count_nonzero_grads(self, prefixes: Sequence[str]) -> int:
        count = 0
        for name, p in self.named_parameters():
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                count += 1
        return count

    def count_forbidden_stage1_nonzero_grads(self) -> int:
        return self._count_nonzero_grads(FORBIDDEN_STAGE1_PREFIXES)

    def count_boundary_nonzero_grads(self) -> int:
        return self._count_nonzero_grads(BOUNDARY_STAGE1_PREFIXES)

    def get_boundary_drift_metrics(self) -> Dict[str, float]:
        drift_vals = []
        for name, ref in self._boundary_param_ref.items():
            module_name, param_name = name.split(".", 1)
            module = self.stage1.pet_calibration if module_name == "pet_calibration" else self.stage1.fusion
            current = dict(module.named_parameters())[param_name]
            current_cpu = current.detach().cpu()
            rel = float((current_cpu - ref).norm() / (ref.norm() + 1e-8))
            drift_vals.append(rel)

        fusion = self.stage1.fusion
        alpha_full = fusion.alpha_full.detach().float()
        alpha_missing = fusion.alpha_missing.detach().float()
        metrics = {
            "calibration_param_relative_drift": float(sum(drift_vals) / max(1, len(drift_vals))),
            "fusion_alpha_full_current": float(alpha_full.mean().item()),
            "fusion_alpha_missing_current": float(alpha_missing.mean().item()),
        }
        for idx in range(fusion.num_scales):
            metrics[f"alpha_full_s{idx + 1}"] = float(alpha_full[idx].item())
            metrics[f"alpha_missing_s{idx + 1}"] = float(alpha_missing[idx].item())
        return metrics

    def get_beta_metrics(self) -> Dict[str, float]:
        if self.stage2_moe.beta is None:
            return {}
        beta = self.stage2_moe.beta.detach().float()
        return {f"s{idx + 1}_beta": float(beta[idx].item()) for idx in range(len(beta))}
