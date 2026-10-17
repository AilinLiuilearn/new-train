# Native MMSFormer FusionBlock in the current PET-CT model

This Stage-1 Full experiment is added to `exp/full-cmx-ffm` without changing
existing Add, CMX, SKFF or channel-calibration experiments. Both real CT and
real PET pass through the original encoders and aligners; four native
MMSFormer FusionBlock stages replace the Add boundary before the GN decoder.

## Source and license

- Paper: Md Kaykobad Reza, Ashley Prater-Bennette, M. Salman Asif,
  *MMSFormer: Multimodal Transformer for Material and Semantic Segmentation*,
  IEEE Open Journal of Signal Processing, 2024, 5:599-610.
  DOI: https://doi.org/10.1109/OJSP.2024.3389812
- Author project: https://csiplab.github.io/MMSFormer/
- Official repository: https://github.com/CSIPlab/MMSFormer
- Pinned commit: `b5287df092c39e95dd46a06c6ba842d501285619`.
- Full original `semseg/models/backbones/mmsformer.py` is archived unchanged
  in `third_party/MMSFormer/mmsformer.py`, git blob
  `47a8a834d7275414553c64951e311e0965fcc319`.
- Exact upstream Apache-2.0 license is `third_party/MMSFormer/LICENSE`, blob
  `ac4f24556b2e04cf30cc4c6fa294b2b0d77c7bf2`. No upstream NOTICE file exists
  at the pinned revision.

`models/components/mmsformer_source.py` extracts the original class
definitions from ChannelAttentionBlock through FusionBlock. The unrelated
encoder classes and their unused `semseg.models.layers.DropPath` import
are omitted. **Every included class body, calculation and constructor
initialization is unchanged.** This avoids installing the unrelated
upstream segmentation framework. Tests check AST identity and compare exact
weights, actual-scale outputs, input gradients and parameter gradients to
classes executed directly from the archived original file.

We reuse the source FusionBlock; this is not a reproduction of the entire
MMSFormer architecture or its published performance. The CT ConvNeXtV2-Nano,
PET MiT-B1 and the current decoder remain the project baseline components.

## Exact fusion computation and tensors

For stage i with aligned feature pairs CT/PET of shape B,D,H,W:

```
U = Linear_2D_to_D(concat(CT, PET))      # per spatial position
F = MixFFN(U) + ChannelAttentionBlock(U)
```

The native MixFFN performs linear projection, pointwise convolution+BN,
parallel depthwise convolutions with kernels 3/5/7, summation including the
unconvolved feature, pointwise convolution+BN, GELU and linear projection.
ChannelAttentionBlock is the native GAP + two linear layers + sigmoid SE
branch, with reduction 16. Its gated U is added to the MixFFN result.

| Stage | Aligned channels | Size at 512 input | Fusion parameters |
|---|---:|---:|---:|
| S1 | 64 | 128x128 | 31,168 |
| S2 | 128 | 64x64 | 112,512 |
| S3 | 320 | 32x32 | 657,600 |
| S4 | 512 | 16x16 | 1,654,272 |

The wrapper uses `FusionBlock([64,128,320,512], reduction=16, num_modals=2)`
and calls the corresponding `layer_idx` for each scale. Total added fusion
parameters: **2,455,552**. Default total model: **34,898,193**.
It is larger than the previous channel-only module; no GPU timing or memory
advantage is claimed. Spatial resizing, when necessary, follows baseline
bilinear PET alignment. Shapes, dtypes, devices, scale counts and finite
inputs/outputs are validated before downstream decoding.

**Enabled initialization is native, not Add-equivalent.** There is no new
zero initialization, bounded tanh, global output multiplier or extra CT+PET
residual. Native BN is retained inside the fusion; the decoder stays GN.
The whole baseline is constructed first, and new module construction is
inside a CPU RNG fork so shared base weights and subsequent CPU RNG remain
identical to Add. All original trainable parameters and fusion parameters
are optimized by the same AdamW optimizer at lr 4e-5.

With `--mmsformer_enabled false`, the wrapper has no fusion parameters and
returns exact CT+PET at all four scales. This is the matched control, not
the enabled module's initial behavior.

Only Stage-1 Full training is connected. Inherited Missing/auto behavior is
preserved: Missing samples bypass PET encoding and fusion, use aligned CT,
and cannot depend on supplied absent PET contents. No retrieval/memory bank
or Stage-2 compensation is connected by this change. The fused outputs feed
only the shared decoder, not subsequent backbone stages.

## AutoDL commands

On the current branch, from the repository root:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/check_mmsformer.py --device cuda --image_size 512 --batch_size 2
bash scripts/run_full_mmsformer_fusion.sh
```

Default run directory:
`checkpoints_new/MDT/full-mmsformer-fusion-lr4e5-60ep-ema-s2023/`.
The launcher uses the current Full protocol: image 512, batch 16,
train_original.txt, val/test test.txt with explicit overlap opt-in, CIPA
augmentation, imagenet normalization and cipa_official HD95. Training:
60 epochs, patience 60, warmup 3, lr 4e-5, minimum lr 1e-6, flat ratio 0.3,
weight decay 1e-4, clip 5, AMP enabled, EMA .999 starting after epoch 3.
Loss remains the existing main BCE+Dice only; no added supervision.

Matched Add control using the same entry, recipe and shared initialization:

```bash
bash scripts/run_full_mmsformer_fusion.sh --mmsformer_enabled false --hash full-mmsformer-add-control-lr4e5-60ep-ema-s2023
```

New seed example:

```bash
bash scripts/run_full_mmsformer_fusion.sh --hash full-mmsformer-fusion-lr4e5-60ep-ema-s2024 --random_state 2024
```

Always use a fresh hash. Nonempty run directories are rejected; resume is
not implemented. Existing Add/channel-calibration entries remain unchanged.
Logs retain encoder/aligner/decoder/fusion gradients, EMA updates and skipped
updates. Best and last checkpoints contain both raw and EMA weights.

## Evaluate a checkpoint

The independent evaluator reconstructs only the saved architecture, validates
source and initialization metadata, and loads weights strictly. Original
backbone pretraining files are not needed for evaluation.

```bash
python eval_full_mmsformer.py --checkpoint checkpoints_new/MDT/full-mmsformer-fusion-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar --output_dir diagnostics/mmsformer-ema --use_ema true
python eval_full_mmsformer.py --checkpoint checkpoints_new/MDT/full-mmsformer-fusion-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar --output_dir diagnostics/mmsformer-raw --use_ema false
```

Use different fresh evaluation directories. Omit `--use_ema` to select the
recorded evaluation weights. `--root`, `--batch_size`, `--num_workers` and
`--device` can be overridden; saved normalization and metric definitions are
retained. Other architecture checkpoints are rejected. Older plateau
diagnostic scripts retain their existing architecture scope; use the new
MMSFormer evaluator for this experiment.

## Hypothesis, verification and limits

Hypothesis: native joint channel mixing plus local multiscale convolution
produces more useful Full features than Add under the same recipe. This
tests a different mechanism from global channel reweighting. Feature spatial
alignment and learned channel alignment do not guarantee semantic alignment;
the block cannot explicitly move misregistered PET regions or guarantee
improved boundary accuracy. Larger capacity may also worsen overfitting.

Compare matched Add vs native FusionBlock first, then paired seeds
2023/2024/2025. Use the same model selection protocol and assess Dice, HD95
and per-patient FP/FN. Do not attribute improvement solely to multiscale
convolution: channel mixing, SE and native initialization also differ. Do
not keep expanding the module if gains are absent across paired runs or
HD95 consistently deteriorates. No theoretical-ceiling or novelty claim is
made. Published natural-image results are not PET-CT performance evidence.

Data-free checks:

```bash
python -m pytest -q tests/test_mmsformer_fusion.py
python scripts/check_mmsformer.py --device cpu --image_size 512 --batch_size 2
```

Synthetic checks use pretrained=False. They validate integration, not server
pretrained coverage, real-data gains or CUDA/batch-16 memory capacity.

### Verified on 2026-10-17

- **144 passed, 5 CUDA tests skipped, 1 existing dataset test deselected**:
  `python -m pytest -q tests --deselect tests/test_baseline_contract.py::test_ct_only_dataset_has_no_pet`.
  The deselected test previously triggered a native OpenCV import bus error
  in this local environment. No dataset code was changed to bypass it.
- PyTorch 2.5.1+cpu, real ConvNeXtV2-Nano + MiT-B1, 512x512, batch 2:
  shared base weights, exact disabled-Add logits, native enabled output,
  two BCE+Dice backward/AdamW updates and all-stage finite gradients passed.
  Fusion parameters 2,455,552; total model parameters 34,898,193.
- In this synthetic initialization, S1 SE's small bias-free ReLU bottleneck
  was inactive on both batches, giving zero SE parameter gradients. The
  linear/multiscale paths had nonzero gradients at every stage; S2-S4 SE
  gradients were also nonzero. This matches native source behavior and is
  reported by the preflight, not fixed by silently changing initialization.
- Source/license blob identity, extracted-class identity, exact native
  constructor weights, actual-feature-scale outputs and input/parameter
  gradients are tested. Python syntax, shell syntax and entry help passed.
- CUDA/AMP, batch-16 GPU memory, server pretraining coverage and real-data
  segmentation improvement remain unverified. Run the CUDA preflight first.
