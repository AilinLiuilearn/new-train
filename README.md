# Clean segmentation baselines

Branch: `exp/full-mafi-add`. Independent baselines, weighted-add and MAFI candidates that
share config / loss / metrics / logging / EMA code (no duplicated utilities).


## Current experiment: Full-only unconstrained weighted addition

Entry: `run_full_weighted_add.py`. Model: `models/full_weighted_add_model.py`.
Module: `models/weighted_add_fusion.py`.

Four scales share the same form `F_s = alpha_s * CT_s + beta_s * PET_s`.
The eight coefficients initialize to 1 and learn directly from BCE+Dice;
there is no sigmoid, softmax, clipping or sign/range constraint. Negative
weights are allowed. They are global per-scale coefficients, not local
confidence scores or interpretable modality-contribution percentages.

This entry trains Full only, with real CT/PET, and uses neither MAFI nor
deep supervision. Shared encoders/alignment/decoder, loss, optimizer,
scheduler, EMA and HD95 are unchanged. Coefficients are registered before
optimizer and EMA construction. The log records all eight raw-model
coefficients each epoch; final JSON records the actual selected raw/EMA
model's coefficients. Final testing and standalone evaluation rebuild
`WeightedAddFusion` before strict state loading.

The MAFI and fixed-Add entry points remain independently available. New
weighted checkpoints cannot be loaded as fixed Add or CT-only checkpoints.
`--weighted_fusion_enabled False` disables the eight parameters and builds
the exact original Add architecture.

Run with the same settings as your fixed-Add reference:

```bash
python3 run_full_weighted_add.py \
  --hash full-weighted-add-unconstrained-imagenet-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file YOUR_VAL.txt \
  --test_split_file test.txt --image_size_2d 512 --batch_size 16 \
  --norm_mode imagenet --aug_mode cipa --random_state 2023 \
  --ema_enabled True --ema_decay 0.999 --hd95_mode cipa_official --epochs 60
```

Replace `YOUR_VAL.txt` with your existing patient-disjoint validation split.
Existing pretrained paths are reused. MedPy is required for official HD95.
No dataset split, shared parser default or historical output is changed.

Lightweight CPU contracts (fake encoders, no real data/MedPy computation):

```bash
python3 -m unittest discover -s tests -p 'test_weighted_add.py' -v
```

## A. CT-only — `run_ct_only_seg.py`

- All training slices, CT + mask only. The loader never requires PET files.
- `ConvNeXtV2-Nano` CT encoder → `StageChannelAlign` (Conv1x1+BN+ReLU, fixed
  output `(64, 128, 320, 512)`) → shared GroupNorm `UNetStyleDecoder`.
- No PET encoder, no fusion. Best by val CT Dice → `ckpt.best_ct.pth.tar`.

## B. Full/Missing mixed — `run_full_missing_baseline.py`

- CT `convnextv2_nano` + PET `mit_b1`, same align spec as CT-only.
- Every batch is exact-half Full/Missing (`build_balanced_pet_available`,
  reproducible per step). Each sample takes exactly one state per batch.
- Full: `aligned_CT + aligned_PET` → shared decoder. Missing: `aligned_CT + zeros` →
  same decoder; Missing rows' PET is never encoded (`pet=None` allowed).
- `loss = 0.5 * full_subset + 0.5 * missing_subset`, one backward/step.
- Val runs whole-set Full and Missing; `Joint = 0.5*(Full+Missing)` Dice.
- Best by val Joint → `ckpt.best_joint.pth.tar`. Final Full/Missing/Joint
  all come from this one checkpoint.

`run_mdt_seg.py` is a compatibility shim: it prints the forwarding rule and
delegates `--train_batch_mode mixed` to the mixed runner; other modes raise.

## C. Full-only add — `run_full_add_baseline.py`

- CT `convnextv2_nano` and PET `mit_b1`, with separate channel alignment.
- All samples use both modalities: `aligned_CT + aligned_PET` → shared decoder.
- Best by val Full Dice; final test uses this checkpoint and its recorded
  raw/EMA weight choice. No experimental fusion or background loss is enabled.

## Run

```bash
# CT-only (formal: real splits + pretrained; needs a disjoint val split)
python3 run_ct_only_seg.py --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --checkpoint_root <out> --epochs 60 --batch_size 16

# Full-only add (stage-one baseline)
python3 run_full_add_baseline.py --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --checkpoint_root <out> --epochs 60 --batch_size 16

# Full/Missing mixed
python3 run_full_missing_baseline.py --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --checkpoint_root <out> --epochs 60 --batch_size 16

# Evaluate a checkpoint at missing rates 0/0.25/0.5/0.75/1 (same seeded
# patient assignments for both experiments)
python3 eval_joint_baseline.py --checkpoint <ckpt> --mode dual --use_ema True \
  --hd95_mode cipa_official --output_dir <new_eval_dir>
python3 eval_joint_baseline.py --checkpoint <ckpt> --mode ct_only \
  --hd95_mode cipa_official --output_dir <new_eval_dir>
```

Notes:

- `--val_split_file` must differ from `--test_split_file`; train/val/test are
  enforced patient-disjoint (the shipped `val.txt` overlaps train cases, so
  provide an explicit disjoint val split — the loader reports the condition
  instead of silently re-splitting).
- The CT-only and Full/Missing runners auto-run the missing-rate final eval from the best checkpoint
  and record whether `raw` or `ema` weights were used.
- Formal runs require local pretrained weights (`--ct_pretrained_path`,
  `--pet_pretrained_path`); missing/unparseable/zero-match paths raise.
  `--pretrained False` opts out explicitly for smoke tests only.
- Results land only in the run's own `checkpoint_dir`; existing experiment
  outputs are never touched.

## Evaluation protocol

- Training defaults to `--hd95_mode cipa_official`; final evaluation uses
  the same selected mode and threshold. Final result files record mode and unit.
- Dice/IoU accumulate pixel counts over the dataset; HD95 is the mean of
  per-slice MedPy HD95 in pixels, without physical voxel spacing.
- `cipa_official` follows CIPA `pred.py` at commit
  `5c90d86ba1331e8043a92b8e732aa36234c548dc`: 512×512 masks, threshold 0.5,
  both-empty = 0, empty prediction with positive GT = insert a center pixel
  on a copy for HD95 only. Dice and missed-case counts keep the original mask.
- MedPy is required (`pip install MedPy`); no silent approximate fallback.
  Empty GT with nonempty prediction raises, since the official script does
  not define a working score for that case.
- Standalone evaluation honors a checkpoint's saved metric mode unless
  `--hd95_mode` overrides it; absent saved mode defaults to `cipa_official`.
  Re-evaluate old checkpoints with the explicit flag above for comparison;
  historical `legacy`/`surface` scores do not change automatically.

## Smoke (no data/weights needed)

```bash
PYTHONPATH=. python3 /tmp/smoke_clean.py   # real encoders, 512x512, CUDA
python3 -m pytest tests/test_baseline_contract.py tests/test_ema.py \
  tests/test_mixed_batch_training.py -q -p no:cacheprovider
```

## Future fusion seam

Only the Full path calls `AddFusion` (`models/components/add_fusion.py`). A new fusion
module replaces that call; Missing keeps feeding `aligned_CT` to the same
decoder. No fusion/prototype/retrieval/text code is wired in this round.

## Changed files

- `models/components/backbones.py` (new): timm/HF encoders, strict offline loading.
- `models/components/channel_align.py` (new): `StageChannelAlign` moved unchanged.
- `models/components/group_unet_decoder.py` (new): GroupNorm decoder moved + finite check.
- `models/components/add_fusion.py` (new): strict per-scale add (4 scales, channels,
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


## D. Full-only MAFI+Add candidate — `run_full_mafi_add.py`

Source: [EG-CMFNet official code](https://github.com/masurq/EG-CMFNet/blob/eed0ad865eea8b0dc0e18db1da11afc062a36b52/models/new_model/modules.py),
TGRS 2026, DOI `10.1109/TGRS.2026.3680719`.

- Four independent MAFI blocks after CT/PET alignment, then AddFusion and
  the unchanged decoder. No encoder-stage feedback, LFF or edge branch from
  the full source network is transplanted; this tests MAFI+Add in our framework.
- CCA/CSA retain mean, maximum and sample standard deviation, the exact
  CT-first sequential order, and source convolution initialization.
  Gamma is unconstrained and zero-initialized (16 scalars total); step-0
  output equals Add. Ratio=16 and kernel=7 at every scale: 97,448 new parameters.
- `--mafi_enabled False` builds the original Add architecture; original
  baseline files/entry points remain available. No auxiliary loss is added.
- The same AdamW, scheduler, data split, normalization, EMA policy and HD95
  selection apply. The source SHA is recorded in configs and final results.
  Fusion gradient norm is sampled once per 100 batches.
- New checkpoints rebuild MAFI for strict raw/EMA loading. The shared
  evaluator selects the builder from saved `fusion_type`; a MAFI checkpoint
  cannot load into the Add architecture or CT-only model.
- This runner trains Full only, without a memory bank. In inherited Missing
  routing PET and MAFI are bypassed; missing robustness is not trained here.

Use the same settings as the Add control, changing only entry/hash:

```bash
python3 run_full_mafi_add.py \
  --hash full-mafi-add-aligned-imagenet-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --image_size_2d 512 --batch_size 16 \
  --norm_mode imagenet --aug_mode cipa --random_state 2023 \
  --ema_enabled True --hd95_mode cipa_official --epochs 60

# Add control: own hash, same data/training settings.
python3 run_full_mafi_add.py \
  --hash full-mafi-disabled-aligned-imagenet-s2023 --mafi_enabled False \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file <your_val>.txt \
  --test_split_file test.txt --image_size_2d 512 --batch_size 16 \
  --norm_mode imagenet --aug_mode cipa --random_state 2023 \
  --ema_enabled True --hd95_mode cipa_official --epochs 60
```

Existing local pretrained paths are reused. Supply a disjoint validation
split. Install MedPy for official HD95. EMA/imagenet above are explicit run
arguments, not changes to shared baseline defaults.

Lightweight checks, with no dataset/download/real encoder execution:

```bash
python3 -m unittest discover -s tests -p 'test_mafi_add.py' -v
```

Tests use small fake encoders to exercise actual model routing, optimizer,
EMA and checkpoint plumbing. They do not measure GPU runtime or accuracy.
