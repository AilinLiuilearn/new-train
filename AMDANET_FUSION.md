# Full-only AMDANet fusion comparison

Base: `exp/aligned-add-baseline`, commit
`663f9786122365d8b2774e4497df92bc06b715d5`.
Experiment: `exp/full-amdanet-fusion`.

## What this experiment measures

All samples use real CT and real PET. The original CT ConvNeXtV2-Nano,
PET MiT-B1, independent Conv1x1+BN+ReLU aligners, GroupNorm U-Net decoder,
BCE+Dice loss, AdamW, scheduler, augmentations, normalization choices and
best-Full-Dice selection are reused. No Missing simulation, retrieval,
prototype bank, SCI, image reconstruction, extra loss, or encoder-internal
interaction is introduced. All parameters train jointly from the same
encoder pretraining used by the Add reference; no old segmentation
checkpoint is loaded. Same-seed common weights initialize identically.

At scale `l`, after the existing channel aligners:

```python
local_ct, local_pet = Local_Eliminating(ct_l, pet_l)
delta = Global_Eliminating(local_ct, local_pet)
fused_l = ct_l + pet_l + delta
```

**Exact adaptation:** the original FDAM classes are copied unchanged, but
the outer `ct_l + pet_l` bypass is added for this experiment. "Dual residual"
means both original modalities have a direct identity path; this is ONE
native FDAM correction, not two newly invented correction heads. There is
no learned scale, zero initialization, gate, or additional normalization on
this outer sum. The new model is therefore NOT equal to Add at step zero.

The upstream global module uses two self-attention enhancements followed by
SEQUENTIAL cross-attention, then projects the second result. Its ordering
is preserved (x1=CT, x2=PET). It is not replaced by parallel cross-attention.
The local module's post-exchange nonlinear reweighting is also preserved.

This tests **adapted FDAM + both-modal residuals**, not a reproduction of
the complete AMDANet network, nor proof of AMDANet's original reported scores.
The source uses dual MiT-B4 branches and feeds locally updated features to
subsequent encoder stages; our existing encoders finish independently and
their four aligned outputs are fused before the existing decoder.

## Source and attribution

- Paper: Zhong et al., *AMDANet: Attention-Driven Multi-Perspective
  Discrepancy Alignment for RGB-Infrared Image Fusion and Segmentation*,
  ICCV 2025, pp. 10645-10655.
- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_AMDANet_Attention-Driven_Multi-Perspective_Discrepancy_Alignment_for_RGB-Infrared_Image_Fusion_and_ICCV_2025_paper.html
- Author repository: https://github.com/Zhonghaifeng6/AMDANet
- Pinned commit: `88e4562eacc41c6d61377d1a34da894decc5a994`.
- Source: `models/modules.py`, `From_Channel` through `Global_Eliminating`.
- Local copy: `models/components/amdanet_fdam.py`; class bodies unchanged.
- Original hyperparameters: reduction=4, heads=(1,2,5,8), sr_ratios=(8,4,2,1),
  qkv_bias=True at the Global_Eliminating constructor, attn/proj dropout=0.
  All inherited internal argument handling and initialization remain as upstream.

No new license is asserted for upstream code; retain the author attribution.

## Tensor and execution contract

Four CT and four PET feature maps in S1-to-S4 order:

| Scale | Channels | Spatial size at image size 512 | K/V reduction |
|---|---:|---:|---:|
| S1 | 64 | 128x128 | 8 |
| S2 | 128 | 64x64 | 4 |
| S3 | 320 | 32x32 | 2 |
| S4 | 512 | 16x16 | 1 |

Each output has exactly the paired input shape. No interpolation is added.
Finite floating-point inputs on matching devices/dtypes are required. Each
spatial dimension must be at least its K/V reduction. The model accepts
`forward_mode='full'` or `auto` with ALL states available; it rejects Missing
before either encoder runs. `pet=None` is rejected.

The fusion adds **11,644,360 parameters**. This is not a lightweight module.
Batch-16 CUDA memory and throughput require validation on the training GPU.
If memory requires a smaller batch, compare BOTH arms at the same batch size.

## Entry points and comparison

```bash
git switch exp/full-amdanet-fusion
git pull --ff-only origin exp/full-amdanet-fusion

# Full Add reference (same original model)
bash scripts/run_full_fusion_comparison.sh add

# Full AMDANet experiment
bash scripts/run_full_fusion_comparison.sh amdanet
```

The script uses ImageNet normalization and EMA in both arms, seed 2023,
batch 16, 60 epochs, LR 8e-5, warmup 3, weight decay 1e-4. These are explicit
comparison settings, not evidence that an older run used identical settings.
To override a setting, append the SAME flags to BOTH commands. Use a fresh
`--hash` for every new run; the inherited baseline runner reuses its run directory.

The commands retain the user's existing `val=test.txt` setup via explicit
`--allow_val_equals_test true`. This selects checkpoints on the test set,
so the reported comparison is not an independent held-out test. For a formal
evaluation, supply a disjoint validation split to BOTH arms, e.g.
`--val_split_file val_disjoint.txt --allow_val_equals_test false`.

Both runners use the SAME loop in `run_full_add_baseline.py`. Its added
builder/config/type arguments default to the original Add behavior. All CT,
Missing, loss, metric, EMA and data files are unchanged. The dedicated
AMDANet entry supplies the experiment builder at initial AND final creation.

Selection: validation Full Dice. Outputs in the run's checkpoint directory:

- `config_args.json`, with source/base commit, formula and architecture.
- `train_log.csv`, including per-epoch Full Dice/IoU/HD95.
- `ckpt.best_full.pth.tar`, `ckpt.last.pth.tar`.
- `final_test_full.json`: test metrics from best-Full checkpoint, following
  the checkpoint's recorded raw/EMA weight selection.

Standalone evaluation (do not use the Missing-rate evaluator for this model):

```bash
python eval_full_amdanet_fusion.py \
  --checkpoint <run_dir>/ckpt.best_full.pth.tar \
  --output_dir <new_eval_dir>
```

This rebuilds the AMDANet architecture and strictly loads its weights,
defaults to the recorded raw/EMA choice, and preserves saved normalization,
splits and HD95 mode. Encoder pretrained files are unnecessary. `--root`,
`--batch_size`, `--num_workers`, `--hd95_mode` and `--use_ema` can override
their saved values explicitly. Add checkpoints are intentionally rejected;
the Add baseline's own final evaluator remains available via its runner.

## Verification

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q -p no:cacheprovider

# Real 512x512 encoders, fusion, decoder and backward; random data only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/smoke_full_amdanet.py --device cpu --image_size 512 --batch_size 1

# On the training GPU, check the actual AMP/batch-16 memory before training
python tools/smoke_full_amdanet.py --device cuda --amp --image_size 512 --batch_size 16
```

The `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` setting is TEST-ONLY for the inherited
EMA test's plain `torch.load` of a trusted, self-created checkpoint. PyTorch
2.6+ otherwise defaults to weights_only=True, which rejects its NumPy RNG
state. Production checkpoint readers already explicitly use weights_only=False.
The new tests also pass without this setting. No loading policy is changed
in production, and the comparison script does not export this variable.

The new tests cover unchanged upstream source, exact local/global/outer-sum
composition, all scales and backward, real CT/PET encoder integration,
optimizer membership, gradients and updates in fusion, forbidden Missing
routes, PET sensitivity, same-seed baseline initialization, strict raw/EMA
checkpoint rebuilding, and a synthetic train/validate/save/rebuild/test loop.
Small synthetic tests use surface HD95 to permit 64x64; formal runs retain
512x512 and cipa_official HD95. Synthetic scores are not performance evidence.

Local CPU 512x512 smoke (batch 1, random initialization) produced logits
`[1,1,512,512]`, with finite BCE+Dice and gradients in every trainable
parameter. Total parameters: 44,087,001; fusion: 11,644,360. GPU/AMP batch-16
memory and real PET-CT Dice/IoU/HD95 are not validated in this environment.

Verification result: 74 tests passed (21 new AMDANet tests plus 53 inherited
baseline tests) with the test-only PyTorch loading compatibility setting.
The 21 new tests also passed under the runtime's default loading behavior.

Return to the original baseline with `git switch exp/aligned-add-baseline`.
No existing experiment outputs are migrated or rewritten by this change.
