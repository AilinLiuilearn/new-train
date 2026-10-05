# Clean segmentation baselines

Branch: `exp/clean-group-baseline`. Two independent, runnable baselines that
share config / loss / metrics / logging / EMA code (no duplicated utilities).

## A. CT-only — `run_ct_only_seg.py`

- All training slices, CT + mask only. The loader never requires PET files.
- `ConvNeXtV2-Nano` CT encoder → `StageChannelAlign` (Conv1x1+BN+ReLU, fixed
  output `(64, 128, 320, 512)`) → shared GroupNorm `UNetStyleDecoder`.
- No PET encoder, no fusion. Best by val CT Dice → `ckpt.best_ct.pth.tar`.

## B. Full/Missing mixed — `run_full_missing_baseline.py`

- CT `convnextv2_nano` + PET `mit_b1`, same align spec as CT-only.
- Every batch is exact-half Full/Missing (`build_balanced_pet_available`,
  reproducible per step). Each sample takes exactly one state per batch.
- Full: `aligned_CT + PET` → shared decoder. Missing: `aligned_CT + zeros` →
  same decoder; Missing rows' PET is never encoded (`pet=None` allowed).
- `loss = 0.5 * full_subset + 0.5 * missing_subset`, one backward/step.
- Val runs whole-set Full and Missing; `Joint = 0.5*(Full+Missing)` Dice.
- Best by val Joint → `ckpt.best_joint.pth.tar`. Final Full/Missing/Joint
  all come from this one checkpoint.

`run_mdt_seg.py` is a compatibility shim: it prints the forwarding rule and
delegates `--train_batch_mode mixed` to the mixed runner; other modes raise.

## Run

```bash
# CT-only (formal: real splits + pretrained; needs a disjoint val split)
python3 run_ct_only_seg.py --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --checkpoint_root <out> --epochs 60 --batch_size 16

# Full/Missing mixed
python3 run_full_missing_baseline.py --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --checkpoint_root <out> --epochs 60 --batch_size 16

# Evaluate a checkpoint at missing rates 0/0.25/0.5/0.75/1 (same seeded
# patient assignments for both experiments)
python3 eval_joint_baseline.py --checkpoint <ckpt> --mode dual --use_ema True
python3 eval_joint_baseline.py --checkpoint <ckpt> --mode ct_only
```

Notes:

- `--val_split_file` must differ from `--test_split_file`; train/val/test are
  enforced patient-disjoint (the shipped `val.txt` overlaps train cases, so
  provide an explicit disjoint val split — the loader reports the condition
  instead of silently re-splitting).
- Both runners auto-run the missing-rate final eval from the best checkpoint
  and record whether `raw` or `ema` weights were used.
- Formal runs require local pretrained weights (`--ct_pretrained_path`,
  `--pet_pretrained_path`); missing/unparseable/zero-match paths raise.
  `--pretrained False` opts out explicitly for smoke tests only.
- Results land only in the run's own `checkpoint_dir`; existing experiment
  outputs are never touched.

## Smoke (no data/weights needed)

```bash
PYTHONPATH=. python3 /tmp/smoke_clean.py   # real encoders, 512x512, CUDA
python3 -m pytest tests/test_baseline_contract.py tests/test_ema.py \
  tests/test_mixed_batch_training.py -q -p no:cacheprovider
```

## Future fusion seam

Only the Full path calls `AddFusion` (`models/add_fusion.py`). A new fusion
module replaces that call; Missing keeps feeding `aligned_CT` to the same
decoder. No fusion/prototype/retrieval/text code is wired in this round.

## Changed files

- `models/backbones.py` (new): timm/HF encoders, strict offline loading.
- `models/channel_align.py` (new): `StageChannelAlign` moved unchanged.
- `models/group_unet_decoder.py` (new): GroupNorm decoder moved + finite check.
- `models/add_fusion.py` (new): strict per-scale add (4 scales, channels,
  finite; no NaN masking).
- `models/ct_only_baseline.py` (new): `CTOnlySegmentationModel`.
- `models/dual_shared_add_baseline.py`: rewritten auto (Full-only PET encode,
  order restore, `pet=None` support); strict state validation.
- `models/build_mdt_seg.py`: two builders + compat forward note.
- `models/baseline_blocks.py` (deleted): split into the files above.
- `datasets/pclt20k_seg.py`: `ct_only` mode, disjointness incl. val/test,
  val≠test enforcement.
- `tasks/mdt_seg.py`: `train_step_ct` / fixed-0.5 `train_step_mixed`,
  `evaluate(full|missing|ct)`, unified single-LR AdamW, new ckpt schema.
- `utils/run_common.py` (new): seed/grad-norm/AMP-step/balanced-state/
  missing-rate eval shared by both runners.
- `utils/ema.py`: tensor-only EMA (text/extra-state machinery removed).
- `utils/region_tversky_loss.py` (deleted): unused.
- `configs/seg_mdt.py`: removed `decoder_lr`, loss weights, joint weights,
  `missing_loss_weight`, `train_pet_drop_prob`, `boundary_loss_weight`,
  `train_batch_mode` choices legacy, viz/eval toggles, `validation_frequency`;
  val default is now `val.txt`; added `--pretrained`.
- `run_ct_only_seg.py` / `run_full_missing_baseline.py` (new logic):
  independent entries on shared utils.
- `run_mdt_seg.py`: compat shim with explicit forwarding rule.
- `eval_joint_baseline.py`: `--checkpoint --mode dual|ct_only --use_ema`.
- Deleted unused: `models/petct_text_region_fusion.py`,
  `tests/test_dcim_wiring.py`, `tools/smoke_dcim_full.py`.
- `tests/`: updated to the new API + added the §八 contract checks.
