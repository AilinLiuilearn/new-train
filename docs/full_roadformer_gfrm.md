# Full-only RoadFormer+ GFRM transplant

Base: `exp/aligned-add-baseline`, commit
`663f9786122365d8b2774e4497df92bc06b715d5`.
Experiment branch: `exp/full-roadformer-gfrm`.

Only new files are added. Original CT-only, Full Add and Full/Missing entries,
encoders, both alignment layers, decoder, criterion, metrics and EMA are untouched.
This runner uses all paired CT/PET training slices: no PET masking, memory bank,
retrieval, deep supervision, distillation or extra loss.

## Actual structure and provenance

Each of four aligned scales, D=64/128/320/512:

1. Flatten spatial axes; compute channel matrices `C @ P.T` and `P @ C.T`.
2. Row softmax separately; multiply by the opposite modality's channel values.
3. Add each message to its own stream with an independent learned scalar,
   initialized to zero (negative values permitted, as in the author code).
4. Concatenate two streams; apply LayerNorm over HW, exactly as author GFFM.
5. FFRM: GAP -> bias-free 1x1 -> GN32 -> ReLU -> sigmoid; `Z + Z * weights`.
6. Split 2D channels into two D-channel halves and sum once for the original
   GroupNorm UNet decoder. No raw CT/PET is added a second time.

All scales have the same structure and independent parameters. Spatial LN is
created for the configured image size / strides 4/8/16/32; no shape interpolation
or replacement by another normalization is hidden in the implementation.

Operators and forward/backward were compared to the original author class bodies,
including nonzero positive/negative cross scales. FP32 attention under AMP is a
numerical safeguard; no pooling, sampling, cosine replacement or attention
approximation is introduced. OpenMMLab wrappers are replaced by the same native
PyTorch operators. See `third_party/RoadFormer/NOTICE.md` and the pinned original
source snapshot for attribution and explicit differences.

This is the GFRM submodule experiment, not the full RoadFormer+ model. No GFE,
LFE, LFFM or FEIM is claimed. Author GFRM has 2D output; fixed split-add is the
explicit target-model interface adaptation.

At 512x512: GFRM parameters **1,587,720**, full model **34,030,361**
(baseline 32,442,641). Channel attention matrices are D×D, not (HW)×(HW).
Actual 512/batch16 GPU speed and memory must be measured on the training machine.

**Enabled initialization is not raw Add equivalence.** Zero scales disable only
cross messages. Spatial LN and FFRM are still present, initially often producing
FFRM weights near 0.5. Their normalization effects belong to this source-based
candidate. `--gfrm_enabled False` removes GFRM entirely and builds original
AddFusion, with no fusion parameters; it is the exact architectural control.

## Training

```bash
git fetch origin
git switch exp/full-roadformer-gfrm

python run_full_roadformer_gfrm.py \
  --hash full-roadformer-gfrm-imagenet-ema-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --image_size_2d 512 --batch_size 16 --num_workers 2 \
  --norm_mode imagenet --aug_mode cipa \
  --optimizer adamw --learning_rate 0.00008 --weight_decay 0.0001 \
  --ema_enabled True --ema_start_epoch 3 --ema_decay 0.999 --ema_decay_warmup True \
  --hd95_mode cipa_official --random_state 2023 --epochs 60 \
  --gfrm_enabled True \
  --fusion_log_every 100 --fusion_visualize True \
  --fusion_vis_every 5 --fusion_vis_samples 2
```

Existing baseline defaults also retain warmup3, cosine schedule/min1e-6,
flat_ratio0.3, gradient clipping5 and early_stop_patience10. This is up to 60
epochs, with the baseline's early stop; set the same patience on both experiments
if a fixed longer schedule is needed. Existing EMA starts after three epochs:
epochs1–3 evaluate raw; epoch4 syncs and starts EMA. Final evaluation strictly
uses the best checkpoint's recorded raw/EMA tag, not simply whether EMA exists.
CPU-only formal starts fail early; `--allow_cpu True` is for small smoke tests.

For Add control use the same command with a distinct hash and
`--gfrm_enabled False --fusion_visualize False`, or keep using the unchanged
`run_full_add_baseline.py`. Hold loss, data splits, normalization, training
schedule, EMA and HD95 mode constant. Reusing test.txt for model selection is the
existing explicit opt-in protocol, not an independent held-out final test.

## Logs and maps

Default first batch and every100 batches:

- real Full count, BCE, DiceLoss, EMA updates;
- per-scale CT/PET gamma, message/input RMS ratio;
- normalized channel-attention entropy (not spatial tumor confidence);
- mean FFRM weight (not PET reliability);
- fused-vs-raw-add RMS difference including LN/recalibration, not just interaction.

CSV stores gradient norms, raw/EMA validation tag, Dice/HD95 and the last sampled
per-scale diagnostics of that epoch. Disabled-module diagnostic cells read
`disabled`, not misleading zero. Parameter connectivity is checked only on the
first batch; finite AMP-overflow skips keep the baseline scaler policy.

Outputs under `checkpoints_new/MDT/<hash>/`:

- `train_log.csv`, `train_log_readable.txt`, `config_args.json`;
- `ckpt.best_full.pth.tar`, `ckpt.last.pth.tar`, `final_test_full.json`;
- `fusion_visualizations/epoch_XXX/diagnostics.json`, sample PNG and NPZ.

Visualize the first two validation samples at epochs1,2 and every5 epochs,
using the same raw/EMA model as validation. PNG shows CT, PET, GT, probability,
plus four scales of CT/PET, two signed-message RMS maps, fused and Add maps.
NPZ stores native-resolution channel RMS maps, input/GT/probability and channel
weights; it does not dump all feature channels. Heatmaps are independently scaled:
compare amplitudes using numbers, not colors. RMS maps show activation magnitude,
not signed direction, boundaries, or proof of causal tumor localization.
RNG, per-module train/eval modes and model state remain unchanged by visualization.
Disable visualization via `--fusion_visualize False` if preferred.

## Bounded validation completed

```bash
python -m unittest discover -s tests -p test_roadformer_gfrm.py -v
```

Five synthetic CPU checks cover author output/gradient parity, four scales,
AMP finiteness, true disabled Add equivalence, rejection before encoder calls,
optimizer membership, checkpoint roundtrip, raw/EMA selection, visualization
state preservation, and two one-epoch runner smoke runs. Actual ConvNeXtV2-Nano
and MiT-B1 + original decoder also passed a 2x3x64x64 CPU forward, BCE+Dice
backward and optimizer step without pretrained weights. Syntax checks passed.
No PCLT20K training or GPU-AMP/batch16 benchmark was run here. Effectiveness
remains unverified: assess Full Dice and HD95 at the same selected checkpoint.
