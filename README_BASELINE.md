# Clean Joint Full/Missing PET-CT Baseline

This branch defines a reproducible baseline with two encoders and one shared decoder.

## Data flow
- Full: CT -> CT encoder, PET -> PET encoder, four-scale add fusion, shared decoder.
- Missing: CT -> CT encoder, aligned CT features directly to the shared decoder.

## Why Missing does not use zero PET
Missing is controlled at the training/evaluation loop level. The code never creates zero PET tensors or zero PET features to simulate missingness.

## Shared decoder
Full and Missing use the same decoder object and the same decoder parameters.

## Joint Dice
`joint_dice = 0.5 * full_dice + 0.5 * missing_dice`

## Checkpoint selection
- `ckpt.best_joint.pth.tar` and `ckpt.best.pth.tar` follow `joint_dice`
- `ckpt.best_full.pth.tar` follows full validation Dice
- `ckpt.best_missing.pth.tar` follows missing validation Dice
- `ckpt.last.pth.tar` is updated every epoch

## Gradient diagnostics
Gradient norm logging is diagnostic only. Gradient cosine diagnostics, when enabled, are meant to observe possible direction disagreement between Full and Missing on shared modules; they do not change optimization.

## Training command
```bash
python run_mdt_seg.py --checkpoint_dir ./checkpoints_new/MDT/e1-clean-joint-baseline
```

## Evaluation command
```bash
python eval_baseline.py --checkpoint_dir ./checkpoints_new/MDT/e1-clean-joint-baseline
```

## Missing-rate tests
The final test script evaluates 0%, 25%, 50%, 75%, and 100% missing settings.

## No visualization
This baseline does not run visualization, Grad-CAM, overlay export, or feature map dumping.
