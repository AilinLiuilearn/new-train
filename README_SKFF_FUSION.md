# SKFF option on the current CMX branch

This change stays on `exp/full-cmx-ffm`. It adds `--fusion_backend skff` to the current model/configuration and a standalone `run_full_skff_fusion.py` entry point. The CMX runner still defaults to `--fusion_backend cmx` and preserves its previous FFM/FRM settings.

## Exact behavior

At S1–S4, the existing aligned CT and PET tensors enter the author's `SKFF(channels, height=2, reduction=8, bias=False)`. Channels are `[64,128,320,512]`. The original per-sample/channel softmax weighted sum is returned directly. There is **no multiplication by 2**, additional Add, extra normalization, custom initialization, auxiliary head or auxiliary loss. The decoder/backbones/aligners and BCE+Dice/EMA training loop are shared with the existing runner. Native initialization is not Add-equivalent.

Four SKFF modules add **144,384 parameters**; the default model totals **32,587,025 parameters**. This is a channel selection mechanism with spatially shared weights, not a lesion-localization or registration module. Original MIRNet-v2 uses it for branch fusion in image restoration; PET–CT fusion is our adaptation. Improvement in this task remains unverified.

The SKFF backend disables CMX FFM and FRM, records both flags as false, and refuses `--cmx_frm_enabled true`. The standalone Python entry point defaults to SKFF and both CMX flags off. For an Add control using that entry point, select `--fusion_backend cmx --cmx_ffm_enabled false --cmx_frm_enabled false` and a separate `--hash`.

## Run

```bash
cd /root/autodl-tmp/mkd-main/new-train
git pull --ff-only origin exp/full-cmx-ffm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Data-free real-backbone preflight; needs CUDA on the training server.
python scripts/check_cmx.py --fusion_backend skff --device cuda --image_size 512 --batch_size 2

# Standalone launch file with the existing server paths and Full protocol.
bash scripts/run_full_skff_fusion.sh
```

The launcher uses input 512, batch 16, seed 2023, the existing pretrained CT/PET files, CIPA augmentation, ImageNet normalization, official CIPA HD95, AdamW lr 8e-5/weight decay 1e-4, 60 epochs/warmup 3, EMA 0.999 starting epoch 3, clipping 5 and patience 10. The existing train/validation/test splits are retained (`train_original.txt`, `test.txt`, `test.txt`, with explicit overlap permission). Validation/test are therefore not independent. Trailing CLI arguments override defaults, e.g. `--hash my-skff-run` or alternate data/pretrained paths.

To use the current entry point with an existing full command, add `--fusion_backend skff` to `python run_full_cmx_fusion.py ...`. Alternatively, `python run_full_skff_fusion.py ...` defaults to that backend. The standalone launcher chooses a new run ID `full-skff-imagenet-ema-s2023`; it does not overwrite CMX results. Existing completed runs are rejected; the inherited startup-only archival mechanism is retained.

Default results: `checkpoints_new/MDT/full-skff-imagenet-ema-s2023/`. The saved configuration includes the selected backend, effective CMX flags, SKFF source hash, channels/order, two input branches, reduction 8, native initialization and output scale 1. The architecture/experiment family stays `full_cmx_fusion`; the effective `fusion_type` is `SKFFFusion`.

Evaluation uses the existing evaluator, which now selects the backend from the checkpoint and strictly checks its metadata/state dict:

```bash
python eval_full_cmx_fusion.py \
  --checkpoint checkpoints_new/MDT/full-skff-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir skff-eval
```

It uses the checkpoint's recorded raw/EMA weights unless `--use_ema true|false` is supplied. Older CMX checkpoints without `fusion_backend` still rebuild as CMX. This experiment remains Full-only; missing/mixed states are rejected before either encoder.

## Source and license

- Official repository: https://github.com/swz30/MIRNetv2
- Pinned commit: `9014049f051ad714eed8b816f23a571e986a46c7`
- Original: `basicsr/models/archs/mirnet_v2_arch.py`
- Local unmodified source: `models/components/skff_source.py`
- Identical source blob: `637424c6073422745d906eb632918460d5209181`
- Original license: `LICENSE.md`; local copy: `third_party/MIRNetv2_LICENSE.md`
- Identical license blob: `610e48de827921d0590f7709da8e358d1554de3f`

The original file is retained byte-for-byte for provenance; only its SKFF class is instantiated. MIRNet-v2 uses its **Academic Public License** for noncommercial research, not the MIT license of CMX. The upstream license and attribution are included unchanged.

```bash
git hash-object models/components/skff_source.py third_party/MIRNetv2_LICENSE.md
```

## Verification

Executed on Python 3.12.14 / PyTorch 2.5.1+cpu: **47 passed, 3 CUDA tests skipped** across the three test files below. This includes 18 new SKFF checks and 29 existing CMX checks. The actual-backbone SKFF CPU preflight at 512 input/batch 2 completed two optimizer updates (BCE+Dice 1.749540, 1.535897). Source/license blob hashes, standalone CLI help, launcher arguments, Python compilation and Bash syntax also passed. Original Add baseline files were not edited.

`tests/test_skff_fusion.py` checks source/license hashes, original constructor initialization, all actual feature scales, exact original forward/input-gradient parity, absence of x2/residual, interpolation, invalid inputs, shared baseline initialization/RNG, actual-backbone two-step optimization, BCE+Dice only, standalone training/validation/EMA/final reconstruction, strict raw/EMA reload, old CMX checkpoint compatibility and rejection of Missing before encoder calls. CUDA AMP is a separate GPU-only test.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_skff_fusion.py tests/test_cmx_fusion.py tests/test_cmx_integration.py
```

The CPU checks establish code-path correctness. Real-data Dice, pretrained-file loading and GPU batch-16 memory/AMP behavior must be evaluated on the server. Return to the CMX backend through `--fusion_backend cmx`; the original Add baseline files remain untouched.
