# Full CT+PET with official CMX FFM

Branch: `exp/full-cmx-ffm`, based on Add baseline `663f9786122365d8b2774e4497df92bc06b715d5`.

The same branch now also offers `--fusion_backend skff` and a standalone SKFF launcher. See [SKFF instructions](README_SKFF_FUSION.md). The existing CMX default, source and FFM/FRM behavior are retained.

Default path: CT/PET encoders → existing projection aligners → **official FFM at S1–S4** → existing decoder. Channels are `[64,128,320,512]`, heads `[1,2,5,8]`, reduction 1. At 512 input, spatial sizes are `[128,64,32,16]`. FFM replaces Add and retains its own internal residuals, native initialization, LayerNorm and BatchNorm2d. There is no external Add around FFM.

The original CMX source and MIT license are vendored byte-for-byte; see [source provenance](CMX_SOURCE.md). No Triton extension is introduced. This experiment is Full-only; it rejects missing PET and mixed availability before encoder execution. It does not connect CMX to retrieval/prototype paths.

## Switches and parameter counts

| Launcher arm | `cmx_ffm_enabled` | `cmx_frm_enabled` | Fusion at each scale | Fusion parameters | Total parameters |
| --- | --- | --- | --- | ---: | ---: |
| `add` | false | false | CT + PET | 0 | 32,442,641 |
| `ffm` (default) | true | false | FFM(CT, PET) | 6,572,032 | 39,014,673 |
| `frm_add` | false | true | Add after FRM | 10,019,848 | 42,462,489 |
| `frm_ffm` | true | true | FFM after FRM | 16,591,880 | 49,034,521 |

Counts use the default backbones/decoder. Only both switches off reproduces baseline Add exactly. Enabled modules use the original constructor initialization. Optional FRM is local to the aligned fusion seam, unlike the encoder-stage feedback in the complete original CMX model.

The training loss remains the baseline **BCE + Dice**. There are no lesion heads, auxiliary supervision, distillation or prototype losses in this runner. Data loading, augmentation, normalization, optimizer, scheduler, clipping, EMA and metric implementations are reused without editing baseline files. Logs additionally expose PET-aligner and fusion gradient norms.

## Run on the training server

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/full-cmx-ffm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Real backbones, 512 input, two optimizer updates; no dataset/pretraining needed.
python scripts/check_cmx.py --device cuda --image_size 512 --batch_size 2

# Start with FFM alone.
bash scripts/run_full_cmx_ablation.sh ffm
```

The preflight checks shared baseline initialization, exact Add behavior when disabled, original non-Add behavior when enabled, BCE+Dice, finite gradients in both encoders/aligners/decoder/every enabled fusion stage, and two optimizer updates. It is not a batch-16 GPU memory benchmark. Do not start training if preflight fails.

Control and optional follow-up runs:

```bash
bash scripts/run_full_cmx_ablation.sh add
bash scripts/run_full_cmx_ablation.sh frm_ffm
# FRM alone, if needed to separate mechanisms:
bash scripts/run_full_cmx_ablation.sh frm_add
```

The launcher fixes the existing protocol: 512 input, batch 16, CIPA augmentation, ImageNet normalization, official CIPA HD95, seed 2023, AdamW lr 8e-5/weight decay 1e-4, 60 epochs, warmup 3, min lr 1e-6, flat ratio 0.3, EMA 0.999 starting epoch 3, clipping 5, patience 10. Train split is `train_original.txt`; validation and test both use `test.txt`, with the existing explicit permission flag. Thus these results do not provide an independent held-out test estimate.

Data/pretrained paths are set in the launcher for the current server. Extra arguments at the end override launcher defaults, for example `--root /path/to/PCLT20K --hash my-unique-run`. Each arm has a separate default run directory. Use the same settings across arms for comparisons; reducing batch size also changes BatchNorm behavior.

CUDA is required by default and checked before experiment files are created. `--require_cuda false` is for CPU smoke checks. Existing runs containing checkpoints, epoch rows or unknown files are never overwritten or resumed. `--retry_failed_start true` archives only a verified startup-only directory. Use a fresh `--hash` for completed/partially trained runs.

## Checkpoints and evaluation

FFM default output directory: `checkpoints_new/MDT/full-cmx-ffm-imagenet-ema-s2023/`.

It contains `config_args.json`, `train_log.csv`, `ckpt.best_full.pth.tar`, `ckpt.last.pth.tar`, and `final_test_full.json`. Saved configuration records both switches, source blob, channel/head order, initialization and output mode. Final evaluation rebuilds the CMX model and strictly reloads the recorded raw/EMA weights.

```bash
python eval_full_cmx_fusion.py \
  --checkpoint checkpoints_new/MDT/full-cmx-ffm-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir cmx-eval-ffm
```

Standalone evaluation uses saved architecture/switches and the checkpoint's recorded raw/EMA choice; `--use_ema true|false` explicitly overrides the weight choice. It reconstructs without ImageNet weight files and refuses incompatible metadata. Use a new output directory. Optional overrides include `--root`, `--batch_size`, `--num_workers`, and `--hd95_mode`.

## Integration API and verification

`models.components.cmx_fusion.CMXFusion(ffm_enabled=True, frm_enabled=False)` accepts two four-element BCHW feature lists in S1–S4 order and returns four fused tensors. Each pair must have matching batch/channels/device/floating dtype. PET spatial resizing follows baseline bilinear alignment (`align_corners=False`). `models.full_cmx_fusion.FullCMXPETCT` attaches this wrapper to the unchanged baseline model.

See [verification results](CMX_VERIFICATION.md). CPU source/gradient/integration/checkpoint tests and actual 512 preflight passed; CUDA AMP and real-data training remain to be run on the server. No accuracy improvement or batch-16 GPU fit is claimed from these checks.

For rollback, switch back to `exp/aligned-add-baseline` and use its existing runner. This branch only adds files and never overwrites baseline code or baseline checkpoints.
