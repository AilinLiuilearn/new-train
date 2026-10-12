# Joint PET correction memory (2D)

Base: `exp/aligned-add-baseline@663f9786122365d8b2774e4497df92bc06b715d5`.
All baseline files/entry points are unchanged. No independent Full-only
pretraining is required. Local encoder pretrained weights remain enabled.

## What is implemented

- CT ConvNeXtV2-Nano and PET MiT-B1, independent existing channel alignments.
- Full uses the original **unweighted AddFusion**, never a new fusion block.
- GroupNorm decoder proj/fuse/seg_head are reused without architectural changes.
- Slice-level Full/Missing sampling: default 0.7/0.3. For B=16, deterministic
  11/5 and 12/4 allocations yield 112 Full rows per 160 rows over ten batches.
- Main group-mean losses have independent weights 0.7/0.3 (no double weighting).
- Independent supervised CT auxiliary decoder, all rows, default weight 0.1.
- Four shared stage readouts (S1 reuses the final seg_head; three new heads), reference/conditional-PET segmentation
  supervision averaged over scales, default weight 0.1.
- No distillation, PET reconstruction or K-means loss.

## Controlled local correction, not cumulative Full-minus-CT

At decoder scale s, project the CT/full skip using the original proj module.
Use the **same CT reference upstream** for the two block evaluations:

`H_C[s] = B_s(upsample(H_C[s+1]), proj_s(C_s))`

`H_plus[s] = B_s(upsample(H_C[s+1]), proj_s(C_s+P_s))`

`delta_s = H_plus[s] - H_C[s]`.

The deepest scale only uses proj4. The normal Full output still comes from
the original complete Full decode, not these conditional probes. The model
performs additional CT-reference/local-probe passes; compute overhead is real.
These are network interventions, not proof of a medical causal effect.

Encoder-aligned channels S1..S4: 64,128,320,512. Actual decoder Value channels
S1..S4: **64,128,256,512** with the default decoder. Never confuse these lists.

## PNDC-inspired region audit and calibration

Readout logits are interpolated to the ORIGINAL GT grid. Binary BCE gain is
`BCE(reference,GT)-BCE(plus,GT)`. Low entropy is not sufficient for admission.
Binary entropy includes both p and 1-p; confidently wrong CT predictions can
still be rescued and qualify. Empty-GT slices are supported.

GT foreground and background are reduced separately per feature cell, then
balanced. A cell containing foreground must also improve its foreground.
Admission defaults: gain >0.02, mean correct-class probability >=0.6, negative
fraction <=0.1. These thresholds are experimental settings, not proven optima.
Entropy-based confidence and gain determine the stored reliability. Sampling
reserves a foreground quota where eligible, then fills from other eligible
positions; max 16 entries per Full slice per scale. Values retain signed channels.

Negative regions are loss-increase regions plus confident wrong conditional
predictions where the CT reference was correct. Their BCE receives weight
`1+0.5*N`; the weighted BCE is normalized by weight sum, Dice is unchanged.
This is a **GT-adapted audit/checksum and loss reweighting**, not an exact copy
of PNDC's four-MRI Reverse Audit / Forward Checksum or its per-modality attribution.
No harmful-value bank is created. Masks/admission decisions are detached.

## Stable paired queues and inference

Four FIFO queues, 2048 entries each by default. Each entry stores a raw CT
descriptor, decoder delta Value, reliability, stable slice ID, optional source
group ID, write step and validity. No GT mask or image is stored.

The physical CT key is kept **before** its learned 32-D projection; query and
bank keys are reprojected using the SAME live projection at read time. This
avoids old projection coordinates becoming incompatible as the projection
learns. Source encoder/decoder drift is limited by momentum sourcing and FIFO
with a 128-successful-update TTL, not claimed to be eliminated.

With EMA active, its eval-mode CT encoder generates BOTH training queries and
bank source descriptors. Its PET encoder and controlled decoder produce values
from Full rows only, under no_grad. This costs extra source encoder/probe passes;
it is a stabilizer, not teacher-matching supervision. EMA-off uses the raw model
in temporary eval mode for coherent paired collection/querying.

At Missing inference, CT alone generates queries. Cosine TopK=4, temperature=.1,
query blocks of 256; never allocate B*H*W*queue dense attention. Readout is
softmax(similarity)*reliability weighted signed Values. A CT-conditioned
residual adapter (zero output projection) corrects the current feature BEFORE
the next upsampling. No available entry or zero readout gives exact identity.

Missing rows never enter a PET encoder. Full inference reads no queue. Ordinary
forward never writes memory. Only successful optimizer updates commit detached
candidates, after the batch has queried the old bank. Skipped AMP steps discard
candidates and do not age/write/update EMA. EMA queue buffers are copied exactly,
never averaged; raw/EMA checkpoints contain synchronized paired queues.

Slice identity uses stable BLAKE2b hashes of dataset image_id, unchanged under
augmentation and across processes. Same-slice exclusion is mandatory in training.
The existing loader's case_id is a filename-prefix group, not automatically a
verified patient ID. Group exclusion defaults OFF; enable only after verification.
No-case metadata is represented by group -1. All evaluation memory is train-only.

## Lifecycle and checkpoint protocol

Epochs 1..5: supervised joint training, no collection or retrieval.
Epoch 6: collect qualifying Full tokens. Epoch >=7: permit retrieval, but each
query still requires >=16 eligible/non-self/non-expired entries. Empty banks do
not fabricate knowledge. Best selection uses 0.5*(Full Dice+Missing Dice), from
one checkpoint for both states. HD95 remains `cipa_official` with MedPy required.
No metric implementation is changed.

Checkpoints include optimizer, scheduler, scaler, RNG, global batch step, queue
payloads/pointers/ages, activation flag and EMA source. Resume in the same run
directory with the original training settings. Per-epoch loader/augmentation
seeds make the epoch-boundary resume reproducible. If best Joint predates active
memory, final output explicitly reports it; do not claim retrieval gain.

Default additional trainable parameters (excluding original encoders/decoder):
CT auxiliary decoder 3,464,897; new stage heads 899; retrieval/adapters 556,208;
total **4,022,004**. Queue buffers ~15.7 MiB per raw/EMA model at default settings.

## Reference provenance

- PNDC author repository: <https://github.com/yansheng-qiu/PNDC>, inspected
  `train_PNDC.py` and `models_pndc.py`, commit
  `d52d1358e7cddf744019d770155761661148d96f`.
  Borrowed ideas: entropy confidence, region audit/checksum and targeted
  supervised calibration. Its entropy-only regions are NOT equated with our
  GT-measured PET benefit. Known variable/mask inconsistencies in the posted
  script are not copied. New audit code is independently implemented for binary 2D.
- SAR-Net: <https://papers.miccai.org/miccai-2026/paper/1730_paper.pdf>.
  Borrowed paired indexing, momentum memory and top-k lookup. No Gram/AdaIN or
  reconstruction is claimed. The linked FS2R repository contained README/LICENSE
  only when inspected, so there was no executable SAR code to transplant.
- FedMEPD: <https://arxiv.org/pdf/2603.04887>. Borrowed task-level decoder
  calibration, not federation, personalized filters or clustered class anchors.
- Original project encoder loading, GroupNorm decoder, optimizer/scheduler,
  CIPA normalization/augmentation/metrics and dataset conventions are reused.

## Run

```bash
python run_joint_pet_correction.py \
  --hash joint-pet-correction-add-r70-imagenet-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --image_size_2d 512 --batch_size 16 \
  --norm_mode imagenet --aug_mode cipa \
  --full_ratio 0.7 --full_loss_weight 0.7 --missing_loss_weight 0.3 \
  --ema_enabled True --ema_decay 0.999 \
  --hd95_mode cipa_official --random_state 2023 --epochs 60
```

Reusing test as validation is the user's exploratory protocol, not an independent
final test. Use val.txt and omit that opt-in for a held-out validation experiment.
If VRAM requires batch 8, the quota still maintains the configured long-run ratio.

Control with the same heads/auxiliary/calibration capacity: `--memory_enabled False`.
Disable region reweighting only: `--region_calibration False`. Adjust sample ratio
and main loss weights independently; keep Joint evaluation unweighted.

Resume: append `--resume checkpoints_new/MDT/<same-hash>/ckpt.last.pth.tar` to the
original command. Optional aligned-add initialization: `--init_checkpoint PATH`
(strict baseline keys required). Standalone reevaluation:

```bash
python run_joint_pet_correction.py --eval_checkpoint checkpoints_new/MDT/<hash>/ckpt.best_joint.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K --hash correction-reeval
```

Minimal CPU contract/runner tests (no real data):

```bash
python -m unittest discover -s tests -p 'test_*correction*.py' -v
```

One synthetic **real-backbone** smoke (no pretrained downloads):

```bash
python run_joint_pet_correction.py --smoke True --pretrained False --image_size_2d 128
```

The module is an experimental hypothesis, not a validated improvement. Compare
against the original Add baseline and the same-capacity no-memory control.

## Verification performed

2026-10-12: 9 CPU contract/runner tests passed, including a two-epoch synthetic
training run, checkpoint resume/evaluation, step-skipping without queue writes,
PET leakage exclusion, same-source conditional deltas, queue TTL/self-exclusion,
EMA buffer integrity, zero-init equivalence and retrieval projection gradients.
Actual ConvNeXtV2-Nano + MiT-B1 passed a CPU 128x128 synthetic optimization step
and Missing forward, generating 10 candidate tokens. Syntax/diff checks passed.
No PCLT20K data, local pretrained files or CUDA device were available here;
512x512 GPU memory and real-data accuracy/official HD95 were not measured.
