# Retained fusion design — completed experiment

Publication branch: `exp/local-region-residual-gn-v1`. This release preserves the
Python/shell source used by training commit `29e55f8729a12dfc49170c9993af0ef0accbec38`.
It is based on the original baseline commit with the actual frozen GN framework
changes included. Local diagnostic images, case-level exports, notebook backups,
model weights and datasets are excluded from this publication.

Training completed 10 epochs and stopped after 5 epochs without strict Joint
improvement. The selected checkpoint is epoch5 EMA. The design is retained as a
candidate; the GN add baseline remains the fixed comparison.

| Metric | GN add baseline | Fusion | Difference |
| --- | ---: | ---: | ---: |
| Full Dice | 0.77866676 | 0.78438079 | +0.5714 percentage points |
| Missing Dice | 0.76875582 | 0.76652556 | -0.2230 percentage points |
| Joint | 0.77371129 | 0.77545318 | +0.1742 percentage points |
| Full HD95 (lower better) | 17.804063 | 17.037423 | -0.766641 |
| Missing HD95 (lower better) | 22.554602 | 22.975225 | +0.420623 |

The records in `results/` contain the original selected-checkpoint metrics,
per-epoch training CSV, actual training configuration and checkpoint SHA256.
The weights remain at the server path in `results/archive_manifest.json`.
These single-seed results on reused val/test data do not establish statistical
superiority, independent generalization or a Gaussian-specific mechanism.

The following implementation report was recorded before formal training;
references below to pending training describe that earlier stage. Completed
metrics above supersede those status statements.

---

# Local / region residual fusion: implementation and launch record

Source workspace `/root/autodl-tmp/mkd-main/new-train` was left untouched. Its branch
was `exp/ct-pretrain-dual-decoder-add-lr8e5`, HEAD
`3c1c5d491a762552835ecd869524d9249329967c`. All actual tracked edits and untracked
source files were copied to an isolated worktree before implementation; exact dirty
file list is in `../fusion_source_snapshot.txt`. Snapshot commit:
`8056f3a4c97cfa462ff150089a4525ca58ad1878`.
Implementation branch: `exp/local-region-residual-gn-s2023`.
Worktree: `/root/autodl-tmp/mkd-main/fusion-local-region`.

## Source correspondence

CIPA was inspected from the existing local repository at commit
`5c90d86ba1331e8043a92b8e732aa36234c548dc`.
`models/encoders/local_vmamba/region_mamba.py` blob:
`0bd6e0a347bdeaf9433a34ee1023b05041ebb0d6` (exactly matches the requested reference).
`models/encoders/dual_vmamba.py` lines 99 and 107 explicitly retain
`out_rgb + out_x + DCIM_output`.

| CIPA element | This implementation |
| --- | --- |
| CT inner convolution stem | Same width, 3x3 convolution topology, biases, BN and ReLU flags |
| PET outer convolution stem | Same width, four 3x3 convolutions and strides 1/2/2/1, BN and ReLU flags |
| Local / regional roles and 4x4 regions | Retained; rectangular row/column reconstruction explicitly tested |
| Mamba blocks | Replaced by separate pre-norm 4-head SDPA and ratio-2 GELU FFN blocks |
| Region-to-local cross-modal token addition | Replaced by independent residual projections into original C and P; only final sum mixes modalities |
| External additive residual | Retained as `(C + R_ct) + (P + R_pet)` with zero output projections |

CIPA source: https://github.com/mj129/CIPA, paper https://arxiv.org/abs/2503.17261.
MIT attribution for the adapted stem is included in `CIPA_LICENSE.txt`.
HDNet provides the multiscale local-contrast motivation, not a reproduced MAC:
https://github.com/Yucl0521/HDNet-TGRS.
The three signed separable Gaussian differences are custom candidates; RAC-Net
full-text/RASFE/code were not audited here. No HU interpretation, localization
improvement, noise suppression, novelty, or metric gain is established.

## Preserved protocol

`configs/frozen_gn_training.json` is copied from the user's finalized GN framework.
The separate launcher serializes every frozen CLI setting explicitly and rejects
unknown fields. Original `scripts/train_joint.sh` and AddFusion are unchanged.
Both generic encoder pretrained files are loaded; no task-trained add checkpoint
or optimizer is used. Decoder/backbone common initialization equality, Full/Missing
initial logits, and the CT-once path are checked. The new module is constructed
last, after old modules and generic loading; add constructs no random fusion layers.

Fixed: align BN, two GN decoders, 512 input, batch16, lr8e-5, paired Full/Missing
loss 0.5/0.5, original CIPA random paired augmentation, AMP, EMA, 60-epoch schedule,
seed2023, Joint selection, patience5, best.pt only. PET regional results use exact
repeat_interleave broadcasting. Zero output projections are retained after
initialization. No gates, auxiliary losses, CRM, joint token attention or Mamba.

v1 deliberately rejects feature dimensions not divisible by4 before running the
stem. Gaussian reflect filtering also rejects dimensions <= maximum radius (4).
The integrated square image CLI requires >=256 and a multiple of128. Standard
512 scales satisfy both. No silent resize/crop/padding fallback is used; the
original AddFusion PET spatial alignment rule is retained.

Validation and test both use `test.txt`, as frozen. This is not an independent
blind test set. Only the full Gaussian experiment is launched; the other two
commands below are controlled comparisons to run separately, not a search queue.

## Verification

- `fusion_tests.log`: **9 passed**, including real train/save/resume/evaluate with
  fusion config restored from checkpoint and strict loading, and the frozen launcher.
- `regression.log`: **12 passed**, covering legacy add strict evaluation/resume,
  stage1 import, independent GN/BN decoders, and checkpoint policy.
- Rectangular 20x28 coordinate partition/reverse and exact 4x4 region broadcasts;
  constants/impulses/signed Gaussian differences; inputs unchanged; finite state;
  first-step projection and later internal gradients; optimizer and EMA coverage;
  Missing exception hooks and unchanged fusion buffers are covered.
- `preflight_512_b16.log`: actual generic-pretrained 512/batch16 AMP forward,
  backward, AdamW + EMA update; first run needed no skipped update.
- `performance.json` / `profile.log`: target batch inference and 3 successful
  training updates. The latter required 2 GradScaler overflow skips; skipped
  attempts are separately recorded and never counted as successful updates.
  After projection updates every scale's stem/local/regional/Gaussian projection
  has finite, nonzero gradients. Both raw and EMA states reconstruct strictly
  and produce exactly equal CPU eval logits after reconstruction.

Measured on RTX3090 24GiB, torch2.1.2+cu118. Model 46,977,746 parameters;
fusion adds 11,070,208. The target shapes and per-scale fusion parameters:

| Scale | C/P (batch16) | PET regional map | Region tokens | Params |
| --- | --- | --- | --- | --- |
| 1 | 16x64x128x128 | 16x64x32x32 | 1024 | 119,524 |
| 2 | 16x128x64x64 | 16x128x16x16 | 256 | 473,864 |
| 3 | 16x320x32x32 | 16x320x8x8 | 64 | 2,945,780 |
| 4 | 16x512x16x16 | 16x512x4x4 | 16 | 7,531,040 |

Same-feature initial fusion error: FP32 **0**, AMP **0**. CPU common-weight Full
and Missing logits also agree with add within atol1e-6/rtol1e-5.
Synthetic batch16 FP32 inference: 0.243s, peak allocated1.451GiB/reserved1.609GiB;
AMP inference: 0.206s, peak allocated1.295GiB/reserved1.963GiB. These inference
measurements exclude EMA/optimizer and are single device passes, not dataset I/O
benchmarks. AMP training with model+EMA+AdamW: 3.383s for 3 successful updates
plus 2 skipped attempts, peak allocated15.235GiB/reserved16.258GiB. No batch,
width, layer or resolution reduction and no activation checkpointing was needed.
Actual throughput and metrics must be read from the formal training logs.

## Reproducible commands

Run from the isolated worktree; each output must be new. Launchers explicitly
inherit the frozen settings rather than relying on parser defaults.

```bash
cd /root/autodl-tmp/mkd-main/fusion-local-region
# Full Gaussian version (the only formal experiment launched by this task)
bash scripts/train_local_region.sh /root/autodl-tmp/mkd-main/new-train/checkpoints_new/AUGMENTATION/local-region-residual-gn-s2023-v1
# Add control (new output, if wanted later)
bash scripts/train_local_region.sh /root/autodl-tmp/mkd-main/new-train/checkpoints_new/AUGMENTATION/add-gn-s2023-control-v1 --fusion_type add
# Same architecture, Gaussian bypassed (new output, if wanted later)
bash scripts/train_local_region.sh /root/autodl-tmp/mkd-main/new-train/checkpoints_new/AUGMENTATION/local-region-no-gaussian-gn-s2023-v1 --no-fusion_gaussian
# Exact full underlying CLI without executing
bash scripts/train_local_region.sh UNUSED --print_command
# Target-batch preflight; run only when GPU is free
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_staged.py --joint_from_pretrained --decoder_norm GN --fusion_type local_region_residual --image_size 512 --batch_size 16
# Target-batch profiling; run only when GPU is free
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/profile_local_region.py
# Tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_local_region_residual_fusion.py tests/test_staged.py tests/test_decoder_norm.py tests/test_checkpoint_policy.py -q
# Evaluate strictly using the saved fusion configuration
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_staged.py --evaluate /root/autodl-tmp/mkd-main/new-train/checkpoints_new/AUGMENTATION/local-region-residual-gn-s2023-v1/best.pt --device cuda
# Resume ONLY a run that opted into --save_last_checkpoint at creation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_staged.py --resume /PATH/TO/OPTED_IN_RUN/last.pt --device cuda
```

The launched run preserves best-only policy, so it deliberately has no last.pt.
The resume command's path is a placeholder for an opted-in recovery checkpoint,
not an assertion that such a file exists in the formal run. Actual fusion resume
was exercised successfully by the synthetic integration test. Resume/evaluate
reject structural CLI overrides and use the saved fusion_config before strict
load. Legacy checkpoints without new fields default to add.

No quantitative efficacy results are claimed yet. A successful implementation
and stable synthetic updates establish wiring/shape/gradient/checkpoint behavior,
not segmentation benefit. Add/no-Gaussian/full comparisons remain necessary for
mechanism attribution; long training and independent data validation are pending.
