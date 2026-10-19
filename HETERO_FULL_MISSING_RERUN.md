# Heterogeneous Full/Missing addition baseline rerun

Branch: `exp/hetero-full-missing-add-lr8e5`.
Base: `exp/aligned-add-baseline` at `663f9786122365d8b2774e4497df92bc06b715d5`.
This branch adds a reproducible launch recipe; it retains the existing clean
model, losses, mixed runner, metric implementation and scheduler.

## Run

```bash
bash scripts/run_hetero_full_missing_add_lr8e5.sh
```

Paths can be set using `DATA_ROOT`, `CT_WEIGHTS`, `PET_WEIGHTS`,
`CHECKPOINT_ROOT`, and `PYTHON_BIN`. `RUN_HASH` optionally names a run;
otherwise a timestamp is appended. Use a fresh run name for every launch.

## Fixed experiment

| Item | Setting |
| --- | --- |
| Encoders | CT ConvNeXtV2-Nano; PET MiT-B1; local pretrained weights required |
| Alignment | CT per-scale 1x1 Conv + BN + ReLU to (64,128,320,512), matching PET |
| Full | Aligned CT + PET, then shared GN decoder |
| Missing | Aligned CT + zero; PET encoder is not called for Missing rows |
| Fusion/retrieval | Pure addition; no prototype bank or retrieval |
| Training | Batch 16, exact half Full/Missing per batch, step-seeded reshuffle |
| Loss | Existing loss, 0.5 Full subset + 0.5 Missing subset |
| Data | train_original.txt; val=test.txt; input 512; CIPA augmentation; ImageNet normalization |
| Optimizer | AdamW; LR 8e-5; weight decay 1e-4; clip 5 |
| Schedule | Existing cosine schedule; warmup 3; min LR 1e-6; flat ratio 0.3 |
| Budget | 60 epochs maximum; patience 10; AMP; seed 2023 |
| EMA | Enabled; decay 0.999; start epoch 3; existing decay warmup |
| Selection | Joint Dice = (Full Dice + Missing Dice)/2 |
| Evaluation | Same best_joint checkpoint and weight type for Full/Missing/Joint; HD95 cipa_official in pixels |

The launch recipe explicitly restores the previously used EMA and ImageNet
settings (the generic parser defaults are EMA off and CIPA normalization).
The architecture is not modified. BN in the CT alignment remains BN; the
decoder remains GN.

Validation intentionally reuses test.txt for continuity with the existing
experiments, with the loader's explicit opt-in. Therefore these results involve
test-set model selection and must not be described as held-out test estimates.

## Results

Under `checkpoints_new/MDT/<run_hash>/`:

- `config_args.json`: actual parsed settings.
- `train_log.csv`: Full/Missing/Joint every epoch, subset counts and gradients.
- `ckpt.best_joint.pth.tar`: one checkpoint selected by Joint.
- `final_test_metrics.json` and `.csv`: missing rates 0/0.25/0.5/0.75/1;
  rate 0 is Full, rate 1 is Missing, JSON also includes Joint Dice.
- `final_test_case_assignments.json`: reproducible patient-level assignments.

This is the joint-trained baseline. Full-only training is a different experiment
and should not be substituted for its Full result.
