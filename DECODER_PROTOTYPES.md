# Decoder spatial/channel prototypes on the 50% mixed Add baseline

Aligned baseline: `exp/aligned-add-baseline`, commit `663f978`.
Experiment branch:
`exp/decoder-dual-prototype-add`.

The aligned baseline's PET/CT alignment and official HD95 evaluation are
preserved. Input normalization defaults to ImageNet in config and loaders.
Two pre-existing tests use the repository's checkpoint loader, which
explicitly supports checkpoint RNG metadata under recent PyTorch.

## Exact experimental scope

- Every batch has exactly B/2 Full and B/2 Missing rows, with the baseline's
  seeded per-step assignment, even batch size, and `drop_last=True`.
- CT: ConvNeXtV2-Nano. PET: MiT-B1. Existing pretrained loading and data
  augmentation/normalization are reused without downloads or fallback encoders.
- Full fusion is **only AddFusion**: aligned CT + aligned real PET. Both
  modalities have independent Conv1x1 + BatchNorm + ReLU adapters at all scales.
  No MAFI, Mamba,
  text encoder, learned fusion block, PET reconstruction or distillation loss.
- CT is encoded once for the whole mixed batch. PET is encoded only for
  Full rows. Missing works with `pet=None`; even NaNs in Missing PET rows
  cannot enter either encoder/decoder route.
- A single shared GroupNorm decoder is used. Correction follows D4 projection
  and D3/D2/D1 fusion blocks, before the next upsampling/segmentation head.
  Decoder channels are **(512,256,128,64)**, not encoder skip channels.
- Full rows bypass correction. Missing rows read existing training memories.
  All reads happen before any writes from the current step.

## Knowledge definition

On **Full training rows only**, an additional CT-only decoder pass with
correction disabled supplies reference logits and corresponding decoder features.
It reuses encoded CT features and the same decoder; no second CT encoder call.
The CT reference receives an explicit auxiliary BCE+Dice loss, default weight
0.1. Main loss remains `0.5*Full + 0.5*Missing`. Thus the **total** loss is
`0.5*Full + 0.5*Missing + 0.1*CT_reference` by default.

For target y and probabilities p0/pF, benefit is
`max((p0-y)^2-(pF-y)^2, 0)`. Full must be correct at threshold 0.5, and benefit
must be at least `prototype_min_gain` (default 0.01). Foreground/background
benefit priorities default to 2/1. These priorities affect selection and
foreground/background mixture; a constant factor cancels in an isolated
normalized average, so they are **not** claimed to amplify every FG prototype.
No new foreground-weighted segmentation objective is silently introduced.

Each decoder scale has an independent fixed orthogonal channel projection,
dimension min(32,C), seeded locally and checkpointed. This prevents a separate
learned collector from changing the stored matrices' coordinate basis. The
encoder/decoder still evolve; centroid EMA mitigates, but cannot eliminate,
feature drift. Source Full/reference use the same sample-wise common RMS
normalization; the query uses its own CT-only RMS. Statistics run in float32.

**Spatial library** stores Euclidean centers of local Full descriptors:
center feature + 3x3 neighborhood mean + local mean absolute deviation.
Benefit-weighted pooling is performed in non-overlapping 5x5 windows
(smaller for tiny maps). Up to 16 candidates per sample/scale are retained,
prioritizing foreground windows and limiting pure background to 8. Area
downsampling of benefit/foreground masks preserves small-region mass. These
are transferable local content descriptors, not patient coordinates or full
spatial shape templates. Spatial center shape: `[K,3d]`.

**Channel library** stores centers of signed
`weighted_Gram(Full)-weighted_Gram(CT_reference)`, using the same benefit map,
basis and normalization for the two terms. One candidate per useful
sample/scale. Negative entries are retained; this is not a PSD covariance and
is not `Gram(Full-CT)`. Channel center shape: `[K,d*d]`.

Both are task-selected descriptors. Neither is an identified causal PET effect,
and the two branches do not establish strict statistical disentanglement.

## Correction and lifecycle

- Spatial Q: local descriptor of the current Missing decoder CT feature.
  K/V: learned projections of spatial centers. Each position reads its own
  prototype mixture. A learned spatial sigmoid gate weights the read residual.
- Channel Q: current CT projected mean + Gram statistics. K/V: learned
  projections of signed relation centers. The retrieved vector generates
  bounded channel scale/shift residuals (0.25*tanh). There is no direct
  query-to-output adapter bypass around the channel memory read.
- Both output heads are zero initialized. Empty libraries bypass exactly;
  populated libraries at initialization also give the original Add outputs.
- Memory warmup defaults to the first 100 **successful optimizer steps**.
  Only later candidates enter the FIFO. Each library needs 24 candidates
  and at least K distinct descriptors before becoming valid; K defaults to 6.
- Each bank has a 256-entry checkpointed ring buffer. Deterministic farthest
  seeding + Euclidean K-means (up to 15 iterations), mild within-cluster
  median/MAD filtering, one-to-one greedy center matching, then centroid EMA
  (0.999). Refresh on initialization and every 20 successful steps with new
  candidates. Empty/collapsed updates leave the previous library unchanged.
- Collection is detached/no-grad and restricted to Full training rows. The task
  holds small pending descriptors; `after_optimizer_step(True)` commits them,
  `False` discards them on overflow. There are no writes inside model forward.
- Validation/test never receive masks for knowledge construction and cannot
  update libraries. The training bank is available for held-out inference.
- If model EMA is enabled, network parameters use normal EMA, while prototype
  buffers (centers/queues/counts/basis/valid flags) are copied directly after
  centroid updates. They are not averaged a second time.
- Checkpoints contain model/EMA, all memory buffers, optimizer, scheduler,
  scaler, mixed-mask step, early-stop state and Python/NumPy/Torch/CUDA plus
  DataLoader generator states. Resume is supported at epoch boundaries.
  It validates scientific/scheduler settings before restoring state.
- Final evaluation uses the exact raw/EMA tag recorded when selecting best
  Joint, including when the best epoch precedes model EMA activation.

The uncorrected CT reference is a stable comparison path. Actual shallow-layer
Missing queries include upstream knowledge correction; this is a deliberate
distribution difference, not a guarantee of exact feature alignment.

## Train

Use your existing environment and pretrained paths. Replace `val_disjoint.txt`
with your patient-disjoint validation split; all data/split checks are inherited.

```bash
git switch exp/decoder-dual-prototype-add
python run_decoder_prototype_add.py \
  --hash add-mixed-dualproto-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file val_disjoint.txt --test_split_file test.txt \
  --batch_size 16 --epochs 60 --image_size_2d 512 \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --aug_mode cipa --norm_mode imagenet --hd95_mode cipa_official --random_state 2023
```

Keep the same optimizer/LR/EMA settings as your matched baseline; the command
above inherits them. Model EMA remains off by default, as on the source branch.
For EMA comparisons, explicitly use the same `--ema_enabled true` on both runs.
CUDA batch-16 memory consumption has not been measured in this environment.

`cipa` augmentation is independent of ImageNet input normalization.
Validation and final test inherit the same `cipa_official` HD95 mode, which
requires MedPy (`pip install MedPy`) and 512x512 masks. Dice/IoU keep the
aligned baseline's global pixel-count accumulation at sigmoid threshold 0.5.
HD95 follows its official center-pixel fallback without changing Dice masks;
empty GT with nonempty prediction raises as in the source baseline.
Small synthetic runner tests explicitly use `legacy`; this is not the formal
training default. Checkpoints predating PET alignment are not architecture-
compatible and should not be resumed for the corrected experiment.

Minimal ablations, using the same command/splits/seed:

| Experiment | Extra arguments |
| --- | --- |
| Original mixed Add objective | `--prototype_enabled false --prototype_ct_aux_weight 0` |
| Add + matched CT-reference supervision | `--prototype_enabled false --prototype_ct_aux_weight 0.1` |
| Spatial only | `--prototype_spatial true --prototype_channel false` |
| Channel only | `--prototype_spatial false --prototype_channel true` |
| Both | defaults |

Use unique `--hash` values. Interpretation requires the matched CT-reference
control: improvements over the original baseline alone cannot isolate the
prototype effect. Small changes in stochastic execution relative to the
original runner are possible; exact zero-head equivalence is tested in eval.

Resume with **the same settings and --hash**, adding:

```bash
--resume checkpoints_new/MDT/add-mixed-dualproto-s2023/ckpt.last.pth.tar
```

The original best checkpoint must remain alongside the last checkpoint.
Resume without local pretrained files is supported. Existing epoch logs are
preserved; a new run refuses to overwrite an existing last checkpoint.

Evaluate separately:

```bash
python eval_decoder_prototype_add.py \
  --checkpoint checkpoints_new/MDT/add-mixed-dualproto-s2023/ckpt.best_joint.pth.tar
```

Output: `train_log.csv`, `prototype_log.jsonl`, `ckpt.last.pth.tar`,
`ckpt.best_joint.pth.tar`, final missing-rate metrics and patient assignments.
`prototype_log.jsonl` reports validity/candidate/update counts per scale/branch;
check these before attributing any change to a library that never activated.

## Validation and source attribution

```bash
OMP_NUM_THREADS=2 python -m pytest tests -q
PYTHONPATH=. python tests/smoke_decoder_prototype.py --size 64 --backward
PYTHONPATH=. python tests/smoke_decoder_prototype.py --size 512
```

Tests cover exact zero-head Add equivalence, active Full bypass, Missing PET
isolation, output order, foreground/background gain correctness, signed matrix
symmetry, no-gain rejection, branch gradients, CPU bfloat16 autocast, centroid
EMA and serialization, post-success writes, eval immutability, model-EMA memory
copying, and interrupted-epoch-boundary resume versus uninterrupted training.
The smoke uses real encoders, random weights, synthetic inputs and populated
memories; it is not a patient-data training experiment.

Default model: 32,989,397 parameters, of which 546,756 belong to the prototype
module. No patient training data, pretrained checkpoints or CUDA device are
available here; clinical accuracy and RTX 3090 AMP/memory behavior are unverified.

Borrowed mechanisms, independently adapted (not a reproduction):

- [FedMEPD official implementation](https://github.com/ccarliu/FedMEPD):
  `models/attention.py` (feature queries, anchor K/V scaled attention) and
  `utils/fl_utils.py` (prototype pooling and EMA). Its FL infrastructure and
  original anchor extraction are not imported.
- [SAR-Net paper](https://papers.miccai.org/miccai-2026/paper/1730_paper.pdf):
  channel Gram statistics as a memory representation. This experiment uses
  task-selected signed relationship changes, not SAR-Net's style reconstruction.
- [PNDC](https://github.com/yansheng-qiu/PNDC): motivation to distinguish positive
  and negative fusion effects; the squared-error benefit formula here is our
  operational design, not a claimed implementation of PNDC's calibration.
