# PSPI: CT-conditioned direct affine + Missing SmoothL1 feature reconstruction

## Branch / base

- New branch: `e1-pspi-ct-affine-smoothl1-add`
- Base: `e1-api-masked-baseline-PSPI-module1-clean-within-batch-alternating`
  @ `a5ca2d5103e4fadfd7ad9af931051a1d833f062c`
- Baseline branch NOT modified; work done in a separate worktree.

## Model contract

- Full: `decoder(AddFusion(C, P_real))`. No retrieval/affine; reconstruction 0.
- Missing: `decoder(AddFusion(C, gamma(C.detach()) * P_prior + beta(C.detach())))`.
  `P_prior` = existing Module-1 retrieval. Direct add; no AdaIN, no gates,
  no extra prior skip, no `(1+gamma)*P` reformulation.
- `gamma`/`beta` are per-sample per-voxel `[B_m,C_l,H_l,W_l]` from CT-only
  generators (`Conv1x1 -> GELU -> DWConv3x3 -> GELU -> 1x1 heads`).
- Identity init makes `pet_comp == P_prior` exactly.
- Cold start (bank not ready): affine skipped entirely; compensation and
  reconstruction are exact zeros even if beta bias is nonzero.
- `pspi_enabled=False`: zero-PET baseline, no affine/reconstruction.
- `affine=True` + `prior_scale=True` is a hard error (no silent alpha).

## Loss

`L_total = 0.5 * L_seg_full + 0.5 * L_seg_missing + 0.05 * L_rec_missing`.
`L_rec_missing` is the only aux objective in the main experiment: balanced
FG/BG per-sample, 4-scale equal-weight SmoothL1 on post-affine `pet_comp`
against detached same-sample `pet_real`, with per-scale RMS loss-unit
normalization. Legacy proto loss default is `0.0`; weight `>0` skips the
graph. Gradient contract verified: only Module-1 q/k/v/out + affine get
`L_rec` gradients; CT/PET encoders, align, decoder, buffers get none.

## Files

- NEW `models/ct_conditioned_pet_affine.py` — `CTConditionedPETAffine`.
- NEW `utils/pet_feature_reconstruction.py` — balanced SmoothL1 helper.
- `models/dual_shared_add_baseline.py` — `_compensate_missing_rows`,
  `_expand_missing_to_full_batch`, `_assemble_mixed_pet_fusion`, affine
  registration, legacy prior-scale compat, reconstruction/affine outputs.
- `models/build_mdt_seg.py` — new param pass-through, config validation,
  legacy fallback (`affine=False, recon=0, prior_scale=True`), new log line
  (`ct_only_direct_affine, reconstruction_target_detached, direct_add,
  proto_loss_disabled, S4, K_per_class=6, mixed_50_50`).
- `configs/seg_mdt.py` — new defaults (`affine=True, prior_scale=False,
  proto=0.0, recon=0.05`).
- `tasks/mdt_seg.py` — `train_step[_mixed]` adds weighted `L_rec` once.
- `run_mdt_seg.py` — reconstruction/affine logging; overflow-aware step
  counting (`mixed_optimizer_steps/mixed_scheduler_steps/skipped_updates`).
- `tests/test_ct_affine_reconstruction.py` — 16 contract tests.
- `tests/test_ct_affine_two_stage_smoke.py` — synthetic two-stage smoke.
- `tests/test_pspi_missing_only_design.py` — test 58 updated for the always
  exposed (zero-valued on legacy) `reconstruction_loss` key.

## Verification (synthetic, toy backbones, CPU)

- `python -m pytest -q tests` → **99 passed**.
- Subset `test_ct_affine_reconstruction + test_mixed_batch_training +
  test_baseline_contract` → 37 passed (includes new file's 16).
- Two-stage smoke: Stage A (cold, 3 mixed batches, 3 collections, finalize
  once, `L_rec` inactive zero); Stage B (bank ready, `reconstruction_active`,
  affine + retrieval weights change, finalize once). Epoch-1-only smoke was
  explicitly NOT accepted as reconstruction validation.
- Checkpoint compat: legacy config (no affine fields) builds the legacy path
  strict-loadable; new `pet_affine` checkpoint strict round-trips with
  identical Missing logits.
- Added params: `+293,632` (affine generators), `-4` (removed trainable
  alpha), net per toy-backbone totals `33,889,297` vs `33,595,669`.

## Main command template (placeholders — set to real paths)

```bash
python run_mdt_seg.py \
  --root "$PSPI_DATA_ROOT" \
  --ct_pretrained_path "$PSPI_CT_WEIGHTS" \
  --pet_pretrained_path "$PSPI_PET_WEIGHTS" \
  --train_split_file "$PSPI_TRAIN_SPLIT" \
  --val_split_file "$PSPI_VAL_SPLIT" \
  --test_split_file "$PSPI_TEST_SPLIT" \
  --train_batch_mode mixed --batch_size 16 \
  --missing_loss_weight 1.0 --train_pet_drop_prob 0.0 \
  --pspi_enabled true --pspi_collect_candidates true \
  --pspi_build_stage 4 --pspi_num_clusters 6 \
  --pspi_bank_update_mode direct \
  --pspi_affine_enabled true --pspi_prior_scale_enabled false \
  --pspi_proto_contrastive_weight 0.0 \
  --pspi_reconstruction_weight 0.05 \
  --hash pspi_ct_affine_rec_add_s2023 --random_state 2023
```

Real smoke: same command with `--batch_size 2 --num_workers 0 --epochs 2`
and an independent hash.

## NOT run

- Real-data/GPU training and real `eval_joint_baseline.py` (no dataset +
  weights + GPU asserted in this environment; not fabricated).
- K/scale/loss-type grid searches; only the `recon_weight=0` ablation is
  reserved.

## Risks recorded

- Identity affine starts compensation at `P_prior`, vs old `~0.1*P_prior`;
  Missing init is NOT equivalent to the old model.
- `beta` is a CT-conditioned additive term and may dominate the prototype;
  monitor `affine_beta_rms_*` vs `affine_gamma_mean_*`.
- Mixed-batch BN statistics still couple Full/Missing rows in CT align and
  the shared decoder; strict Missing-independence is guaranteed only for the
  fusion rows / eval-mode logits, as specified.
