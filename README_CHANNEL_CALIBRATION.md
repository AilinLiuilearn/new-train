# Joint channel calibration + addition (Stage-1 Full)

This isolated experiment is added to `exp/full-cmx-ffm`. Existing Add, CMX,
SKFF, training task, loss, decoder and diagnostic-script files are unchanged.
No new branch is needed. This is a **MMTM-inspired adaptation**, not an
unchanged official MMTM reproduction or a verified performance improvement.

## Source and adaptation

- Paper: *MMTM: Multimodal Transfer Module for CNN Fusion*, CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Joze_MMTM_Multimodal_Transfer_Module_for_CNN_Fusion_CVPR_2020_paper.html
- Author's public reimplementation: https://github.com/haamoon/mmtm
- Pinned commit: `1c81cfefad5532cfb39193b8af3840ac3346e897`.
- Exact unmodified `mmtm.py` is preserved in `third_party/MMTM/mmtm.py`,
  git blob `513b906450288beb9fef21f4d97b3d0e4c368e21`.
- Exact MIT license: `third_party/MMTM/LICENSE`,
  git blob `164301f9e9fe36407082079d91a8ec9071c1ab75`.
- The archived whole source includes action-recognition imports unavailable
  here and is intentionally not imported. The small MMTM mechanism is adapted
  in `models/components/channel_calibration.py` without those dependencies.

Borrowed: separate global average pooling for each modality, concatenation,
shared linear squeeze + ReLU, two modality-specific excitation heads.

Changed: a smaller hidden width `max(D//8,16)`, bounded residual tanh weights,
zero excitation initialization, and addition after calibrating both modalities.
The upstream class uses sigmoid scaling and returns two streams; we do not
claim to reproduce its full architecture, initialization or reported results.

## Tensor and training contract

At each stage, after the existing independent CT/PET Conv1x1+BN+ReLU aligners:

```
z = concat(mean_HW(CT), mean_HW(PET))           # B, 2D
h = ReLU(Linear(2D, max(D//8,16))(z))
w_CT  = 1 + 0.25*tanh(Linear_CT(h))            # B,D,1,1
w_PET = 1 + 0.25*tanh(Linear_PET(h))           # B,D,1,1
F = w_CT*CT + w_PET*PET                        # B,D,H,W
```

| Stage | Channels | Size at 512 input | Hidden | Trainable parameters |
|---|---:|---:|---:|---:|
| S1 | 64 | 128x128 | 16 | 4,240 |
| S2 | 128 | 64x64 | 16 | 8,464 |
| S3 | 320 | 32x32 | 40 | 51,880 |
| S4 | 512 | 16x16 | 64 | 132,160 |

Total additional parameters: **196,744**. Total default model: **32,639,385**.
Both modalities and all four scales are active. Weights are independent, in
`[0.75,1.25]` up to floating-point rounding, without modality softmax.
PET spatial resizing uses the baseline bilinear alignment if needed.

Each excitation head weight/bias starts at zero; enabled step-0 output is
exactly CT+PET. Baseline initialization and subsequent CPU RNG are preserved.
The first step updates excitation heads; the shared squeeze layer starts
receiving nonzero gradients after heads move. This is expected, not a dead
gate. All original trainable parameters remain trainable. AMP pools in FP32.

Only Stage-1 Full training is connected: real CT + real PET, no memory bank,
retrieval, auxiliary heads or added loss. The model inherits baseline Missing
and auto routing: Missing bypasses PET encoding and calibration entirely and
uses aligned CT. This does not implement or validate Stage-2 compensation.

The original GN decoder, BCE+Dice, AdamW and EMA implementation are reused.
Defaults: lr `4e-5`, 60 epochs, patience 60, warmup 3, minimum lr `1e-6`,
flat ratio 0.3, weight decay `1e-4`, EMA `.999` starting after epoch 3.
The launcher explicitly uses the current imagenet/CIPA/Full data protocol.

## Run on AutoDL

From the repository root on the existing branch:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/check_channel_calibration.py --device cuda --image_size 512 --batch_size 2
bash scripts/run_full_channel_calibration.sh
```

Default output:
`checkpoints_new/MDT/full-channel-calibration-lr4e5-60ep-ema-s2023/`.
Existing nonempty run directories are rejected; choose a fresh hash, e.g.:

```bash
bash scripts/run_full_channel_calibration.sh --hash full-channel-calibration-lr4e5-s2024 --random_state 2024
```

Matched Add control with identical new entry, recipe, and base initialization:

```bash
bash scripts/run_full_channel_calibration.sh --channel_calibration_enabled false --hash full-channel-add-control-lr4e5-60ep-ema-s2023
```

Disabled mode has zero fusion parameters and is exactly the original Add
forward, but uses new experiment metadata. No CMX/SKFF module is active.
The original `run_full_add_baseline.py` is still usable and unchanged.

Logs include `grad_enc_ct`, `grad_enc_pet`, `grad_ct_align`, `grad_pet_align`,
`grad_decoder`, `grad_fusion`, EMA updates and skipped optimizer updates.
Both raw and EMA weights are saved; checkpoint schema records stage order,
source identity, reduction, strength, initialization and output semantics.
No silent non-strict loading of old Add/CMX/SKFF checkpoints is performed.

## Evaluate raw and EMA without original pretraining files

```bash
python eval_full_channel_calibration.py --checkpoint checkpoints_new/MDT/full-channel-calibration-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar --output_dir diagnostics/channel-ema --use_ema true
python eval_full_channel_calibration.py --checkpoint checkpoints_new/MDT/full-channel-calibration-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar --output_dir diagnostics/channel-raw --use_ema false
```

Omit `--use_ema` to use the checkpoint's recorded evaluation weights.
Use new output directories. `--root`, `--batch_size`, `--num_workers` can
override data location/loader settings; normalization and metric semantics
come from the saved config. Old CMX/SKFF evaluation and plateau-diagnosis
scripts retain their existing architecture support; use this new evaluator
for channel-calibration checkpoints.

## Falsifiable hypothesis and limitations

Hypothesis: separately conditioning on both modality descriptors before
fusion improves held-out segmentation over Add under the same recipe.
This is channel selection, not new spatial information or registration.
It cannot directly correct local PET/CT misalignment; GAP can be dominated
by background. Bounds may also limit useful corrections. The current logs
do not prove channel selection is the bottleneck or guarantee an improvement.

Compare matched Add and calibrated Add first, then repeat with paired seeds
2023/2024/2025 before claiming stability. Assess patient-level Dice changes,
FP/FN and HD95, not only a single best epoch. Stop expanding this mechanism
if it has no repeatable gain or consistently damages HD95. A future matched
independent-channel control can isolate cross-modal conditioning; it is not
part of this first implementation. No novelty claim is established.

Validation commands (no dataset required):

```bash
python -m pytest -q tests/test_channel_calibration.py
python scripts/check_channel_calibration.py --device cpu --image_size 512 --batch_size 2
```

The synthetic preflight deliberately uses `pretrained=False`; it checks
integration and gradients, not real-data performance or server pretrained
coverage. CUDA checks require a real CUDA runtime.

### Verified on 2026-10-16

- PyTorch 2.5.1+cpu: real ConvNeXtV2-Nano + MiT-B1, 512x512, batch 2,
  exact initial Add logits, two BCE+Dice backward/optimizer updates passed.
- Regression suite: **127 passed, 4 CUDA checks skipped, 1 deselected**.
  Command: `python -m pytest -q tests --deselect tests/test_baseline_contract.py::test_ct_only_dataset_has_no_pet`.
  The deselected pre-existing dataset test hits a native OpenCV import bus
  error in this local environment; no dataset code was changed to bypass it.
- Source/license blob identity, shell syntax and new Python syntax checked.
- Real dataset training, server pretraining coverage, CUDA/AMP and batch-16
  GPU memory have **not** been verified here. Run the CUDA preflight first.
