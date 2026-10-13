# Full-only homogeneous encoder addition baseline

Branch derived from `exp/full-structure-response-fusion` at `09c09dde`.
Independent entry: `run_full_homogeneous_add.py`.

## Architecture

- `--encoder_pair mit_b1`: two independent MiT-B1 encoders, native channels
  64/128/320/512. Both default to the existing `pretrained/mit-b1` weights.
- `--encoder_pair convnextv2_nano`: two independent ConvNeXtV2-Nano encoders,
  native channels 80/160/320/640. Both default to the existing
  `pretrained/convnextv2_nano` weights.
- Each encoder has its own trainable parameters; architecture equality does not
  mean weight sharing. The same pretrained file initializes both branches.
- `ct_align` and `pet_align` are parameter-free identities: no pre-fusion
  projection, normalization or ReLU. Feature distributions need not be equal
  simply because architecture/initialization are equal.
- Four scales fuse with exact `CT + PET`. Decoder input widths follow native
  encoder widths. Existing decoder 1x1 projection blocks, GroupNorm, decoder
  output widths 512/256/128/64, and upsampling remain intact.
- Full-only train/val/test: real PET always present. No retrieval, mixed batches,
  CT auxiliary decoder, extra loss or deep supervision.
- Same loader, BCE+Dice, AdamW, scheduler, EMA activation and HD95 implementation
  as the Full-only fusion runner. Evaluation EMA starts at epoch 4 by default.
- `--fusion_enabled True` enables the existing complete structure-response
  fusion at native channels, providing an optional comparison after the Add run.
  Default is False; no fusion parameters are instantiated for Add.

## Training

```bash
python run_full_homogeneous_add.py \
  --encoder_pair mit_b1 \
  --hash full-mit-mit-native-add-imagenet-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --image_size_2d 512 --batch_size 16 --num_workers 2 \
  --norm_mode imagenet --aug_mode cipa \
  --fusion_enabled False \
  --ema_enabled True --ema_start_epoch 3 --ema_decay 0.999 \
  --hd95_mode cipa_official --random_state 2023 --epochs 60
```

For ConvNeXt replace `--encoder_pair` with `convnextv2_nano` and `--hash` with
`full-convnext-convnext-native-add-imagenet-s2023`; both default pretrained paths
switch automatically. Supply `--ct_pretrained_path` and `--pet_pretrained_path`
explicitly if your weights live elsewhere. MiT defaults to `mit-b1`, not `mit_b1`.
Same run configuration must be used for candidate/control comparisons; if batch
16 does not fit, use the same smaller batch for both. Do not compare GPU count
alone as an explanation for accuracy differences.

## Checkpoints and interpretation

- Use a new hash. Old heterogeneous/aligned checkpoints are not compatible and
  are not silently partially loaded. Resume checks encoder pair, backbone names,
  alignment policy and training settings; tensors load strictly.
- Resume with `--resume PATH/ckpt.last.pth.tar` and the original training flags.
- Reevaluate with `--eval_checkpoint PATH/ckpt.best_full.pth.tar`. The saved
  architecture and selected raw/EMA weights are restored; pretrained files are
  not needed for reevaluation. Runtime data root and output hash can be changed.
- Results: `train_log.csv`, `config_args.json`, `ckpt.best_full.pth.tar`,
  `ckpt.last.pth.tar`, `final_test_full.json` under the usual run directory.
  Startup prints both backbone names, native widths, no-alignment policy and
  parameter counts. Alignment/fusion gradient columns are zero for Identity/Add,
  which have no trainable parameters.
- Optional existing visualization is available with `--fusion_visualize True`.
  In pure Add mode the same-model Add counterfactual equals the actual output;
  enhancement/attention maps only exist with fusion enabled.
- Reusing test.txt for validation preserves the user's current comparison
  protocol but is not an independent final test estimate.
- This experiment changes both encoder composition and alignment. It cannot
  separately attribute an improvement to either change. In ConvNeXt mode the
  decoder input projections also differ because native widths differ.
- Minimal CPU checks cover native Add, branch independence, gradient connection,
  optional fusion zero-init equivalence, checkpoint resume and reevaluation.
  Real encoder 64x64 CPU smoke is separate from real-data/GPU accuracy testing.

Implementation checks: 9 focused tests passed, including existing fusion-entry
regression tests. Both real backbone pairs passed a 2x3x64x64 forward, backward
and optimizer update on CPU without pretrained weights. Pure Add parameter
counts are 29,767,745 (MiT-B1 pair) and 33,498,593 (ConvNeXtV2-Nano pair),
with the default decoder. No real PCLT20K/GPU accuracy run was performed here.

## Rollback

Run the original `run_full_structure_response_fusion.py` or
`run_full_add_baseline.py` to retain the previous architecture. The only edited
existing file is the fusion runner, generalized to accept an explicit config and
model builder. Its no-argument entry retains the original defaults. Shared
backbone, decoder, loader, losses and metric source files are unchanged.
