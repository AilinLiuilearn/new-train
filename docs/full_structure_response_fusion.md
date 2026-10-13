# Full-only structure/response complementary residual fusion

Branch: exp/full-structure-response-fusion, based on aligned Add at
663f9786122365d8b2774e4497df92bc06b715d5. All baseline files and entry points
remain unchanged. The correction-memory branch is not modified or merged here.

## Exact design

For each aligned CT/PET pair C,P [B,D,H,W], D=64/128/320/512,
use the same architecture with independent parameters and inner width d=D/2:

1. CT 1x1 projection D->d, then fixed Scharr SEE: depthwise x/y Scharr,
   abs(x)+abs(y), BatchNorm, add to projected CT, d->16 1x1, SiLU,
   16->d 3x3. Kernels remain buffers and are copied exactly even under EMA.
2. PET contrasts P-AvgPool_k(P), k=3/5/7, stride1, excluding padding in means.
   Concatenate P and all signed contrasts, 4D->d 1x1, d-channel depthwise3x3,
   GELU. No lesion threshold, foreground mask or SUV interpretation.
3. Two independent DRWKV-style channel CrossAttention modules (four heads).
   Q from recipient, K/V from sender; 1x1+DW3x3, spatial L2 Q/K normalization,
   FP32 channel covariance/softmax using the released multiplicative learnable
   temperature, V aggregation, output1x1. This is not spatial correspondence.
4. Per branch concatenate its raw D-channel input, own enhancement and received
   d-channel interaction: (D+2d)->d 1x1, channel LayerNorm, full SAE/IEL.
   SAE retains expansion=2.66, DW3x3, split, two additional DW3x3,
   tanh plus per-branch residual, multiplication and output1x1.
5. Independent d->D final projections, zero initialized, with NO subsequent
   activation/norm. F=(C+delta_C)+(P+delta_P). Raw C/P are each added once.

First forward is exactly Add. Output heads learn first; most internal new
parameters have zero gradient on the first step until these heads move.
Only the Scharr buffers are frozen; all other fusion weights are trainable.
No gamma/gate, new loss, auxiliary decoder, missing reconstruction, memory or
CT-centric final baseline is introduced. Only real CT/PET Full is accepted;
Missing/mixed calls explicitly fail before either encoder runs.

Files:
- models/structure_response_fusion.py: standalone module, PyTorch only.
- models/full_structure_response.py: baseline-derived Full-only wrapper.
- configs/full_structure_response.py: independent parser.
- tasks/full_structure_response.py: same supervised task/EMA policy, copying
  fixed Scharr kernels exactly rather than numerically averaging constants.
- run_full_structure_response_fusion.py: independent runner.
- utils/structure_response_visualization.py: sparse logs and fixed-slice maps.
- tests/test_structure_response_fusion.py: bounded synthetic contract checks.
- third_party/DRWKV/: author Apache license and attribution/adaptation notice.

Fusion trainable parameters: 4,637,536, stages 60,544 / 215,144 / 1,243,424 /
3,118,424. Actual full model total/trainable: 37,080,177 with default encoders
and decoder, versus 32,442,641 for aligned Add. GPU 512x512 speed/peak memory
have not been measured; these counts are not an efficiency guarantee.

## Training and evaluation

All train/validation/test samples use real PET. Single BCE+Dice, original
AdamW, decoder, normalization/augmentation and metric implementations.
Validation selection is by Full Dice; final test strict-loads the selected
raw/EMA best Full checkpoint. No Full/Missing/Joint scores are reported here.
EMA retains baseline activation after epoch3 (epoch4) and decay/warmup policy.
New runner uses deterministic per-epoch model/loader RNG seeds; enabled/disabled
controls share this protocol. It does not change the original Add entry.

CUDA is required by default before any real data/model allocation, avoiding
accidental CPU fallback. Synthetic --smoke opts out; intentional CPU contracts
use --allow_cpu True. Default workers=2; adjust explicitly for the machine.
Resume preserves optimizer/scaler/scheduler/RNG and architecture/protocol; use
the same hash/epochs/settings. Visualization/log interval may change.

```bash
python run_full_structure_response_fusion.py \
  --hash full-structure-response-imagenet-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --image_size_2d 512 --batch_size 16 --num_workers 2 \
  --norm_mode imagenet --aug_mode cipa \
  --ema_enabled True --ema_start_epoch 3 --ema_decay 0.999 \
  --hd95_mode cipa_official --random_state 2023 --epochs 60 \
  --fusion_enabled True --fusion_log_every 100 \
  --fusion_visualize True --fusion_vis_every_epochs 5 --fusion_vis_samples 2
```

Current exploratory val=test opt-in is kept explicit; it is not an independent
final test. Formal experiments use a held-out val split under the shared loader.
Same-runner aligned Add control: change only --fusion_enabled False and --hash.
With False, no fusion parameters are created and the original AddFusion is used.
If batch16 exceeds VRAM, compare both experiments with the same smaller batch.

## Meaningful logs, not confidence claims

Startup: full state, actual parameter count, widths, heads, SAE expansion,
fixed kernels, PET neighborhoods, runtime/device/AMP, EMA policy.
First batch and every100: BCE/Dice loss, representative gradient connectivity,
unscaled gradient norm, per-stage rCT=RMS(delta_C)/RMS(C) and rPET,
actual residual RMS, CT projected-Scharr edge RMS, signed PET contrast RMS
at 3/5/7, channel-attention normalized entropy, learnable temperatures.
Residual cosine is saved when both residuals are nonzero; otherwise null.
These are network response measurements, not lesion confidence probabilities.

fusion_diagnostics.jsonl contains sparse actual forward measurements.
train_log.csv records sampled-batch means and sample counts, plus Full metrics.
Gradients are measured sparsely to avoid scanning/synchronizing every parameter
on every batch. Skipped updates are recorded; EMA/scheduler advance only after
successful optimization. The ordinary grad_norm is the finite-batch mean.

## Feature visualization

Disabled by default; --fusion_visualize True processes first2 fixed validation
slices at epochs1/2 and every5 thereafter, not a second whole-validation pass.
Outputs in checkpoints_new/MDT/<hash>/fusion_visualizations/epoch_XXX/:

- *_predictions.png: CT/PET/GT, same-model Add counterfactual probability,
  fusion probability, signed probability change. This is NOT an independently
  trained Add baseline: its encoders/decoder have learned with the new fusion.
- *_features.png: four scales CT structure, PET 5x5 contrast RMS, CT/PET
  enhanced feature RMS, and both actual final residual RMS.
- *_channel_attention.png: CT->PET/PET->CT channel attention matrices,
  averaged over heads for display; never described as lesion-position maps.
- *_maps.npz: exact native-resolution maps, including all3 signed contrast
  means/RMS, signed residual channel means, both head-wise attention matrices,
  enhanced features, prediction probabilities and GT.
- *_metadata.json: weight tag/settings and scale diagnostics.

Original-resolution GT contours overlay enlarged feature maps, preventing
GT lesions from disappearing through S3/S4 downsampling. Feature RMS uses
displayed color bars; consult numeric ranges before comparing epochs/modalities.
No GT enters fusion computation. Visualization restores mode, torch RNG and
loader generator and changes no model/checkpoint-selection state.

Checkpoint inspection without retraining:

```bash
python run_full_structure_response_fusion.py \
  --eval_checkpoint checkpoints_new/MDT/<hash>/ckpt.best_full.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K --hash structure-response-inspect \
  --fusion_visualize True --fusion_vis_samples 2
```

Resume: original command plus --resume <same run>/ckpt.last.pth.tar.

## Verification and limitations

Six grouped CPU contracts pass on torch2.5.1: all scales and rectangular shapes,
exact Add init, frozen Scharr including EMA, output-head then internal gradients,
AMP finiteness, diagnostics invariance, disabled Add equivalence, early Missing
rejection, save/resume/EMA checkpoint evaluation, and actual PNG/NPZ generation
without mode/buffer/RNG changes. Real ConvNeXtV2-Nano + MiT-B1 on torch2.14.1
CPU completed a 64x64 synthetic forward/backward/optimizer step; no pretrained
weights downloaded. These checks are not PCLT20K accuracy experiments.

No real-data training, CUDA test, 512x512 VRAM benchmark or numerical MedPy
verification performed here. References have different source/target tasks.
Scharr/contrast can amplify background; channel CA can be background-dominated;
the two residuals can interfere. No Dice/HD95 gain or top-conference novelty is
claimed before comparison with independently trained Add and component ablations.

```bash
python -m unittest discover -s tests -p 'test_structure_response_fusion.py' -v
python run_full_structure_response_fusion.py --smoke True --pretrained False
```
