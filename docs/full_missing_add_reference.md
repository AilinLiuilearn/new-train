# Full/Missing Add reference (configurable ratio, 50/50 defaults)

Independent entry: `run_full_missing_add_reference.py`.
Base: exp/aligned-add-baseline @663f978.
Branch: exp/full-missing-add-reference. Existing entry/model/config/task untouched.

- Defaults: each batch16 exactly8 Full and8 Missing, identical old random assignment.
- `--full_ratio 0.7`: batch16 uses11/5 or12/4, deterministic cumulative quota;
  cumulative requested-vs-actual Full count differs by less than1 slice.
  Missing fraction = 1-full_ratio; both subsets must exist in every batch.
- Full: real CT/PET encoders -> both existing aligns -> Add -> original decoder.
- Missing: CT encoder -> ct_align -> same decoder; PET is never encoded for
  missing samples. No GFRM, retrieval, compensation, CT auxiliary decoder,
  extra supervision or losses.
- Loss defaults: 0.5 Full BCE+Dice + 0.5 Missing BCE+Dice. Explicit independent
  `--full_loss_weight` / `--missing_loss_weight` flags must be positive and sum to1.
  Each component is the subset mean. Shared task's new optional keywords retain
  exact0.5/0.5 defaults for every existing entry.
- Validation: evaluate all validation samples once as Full and once as Missing.
- Select ckpt.best_joint by (Full Dice + Missing Dice)/2. Full/Missing/HD95
  are reported from that SAME checkpoint, not separate best epochs.
- EMA policy inherited: first3 epochs raw, activate EMA in epoch4. Final testing
  uses the checkpoint's recorded eval_weights; existence of an inactive EMA
  copy cannot override raw weights. Reuse the model instead of loading new
  pretrained encoders for final testing.
- Final evaluation only rates0 and1; intermediate rates are intentionally omitted.
- Keep data/norm/augmentation/loss/optimizer/schedule/EMA/HD95 settings equal
  when testing later joint-trained fusion or compensation candidates.

```bash
python run_full_missing_add_reference.py \
  --hash joint-add-reference-r50-imagenet-ema-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --image_size_2d 512 --batch_size 16 --num_workers 2 \
  --norm_mode imagenet --aug_mode cipa \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --optimizer adamw --learning_rate 0.00008 --weight_decay 0.0001 \
  --ema_enabled True --ema_start_epoch 3 --ema_decay 0.999 --ema_decay_warmup True \
  --hd95_mode cipa_official --random_state 2023 --epochs 60 \
  --train_batch_mode mixed --final_test_missing_rates 0 1
```

50/50 is a reproducible balanced reference, not a proven optimal ratio.
For complete70/30 protocol use a NEW hash and add:

```bash
--full_ratio 0.7 --full_loss_weight 0.7 --missing_loss_weight 0.3
```

This changes sampling and loss weighting together. For a sampling-only ablation,
keep loss weights0.5/0.5 while setting full_ratio0.7. Joint validation selection
remains0.5/0.5 in either experiment; do not redefine the score to favor70% Full.
Subsequent70/30 compensation candidates need this matching70/30 Add control.
Joint-trained Missing results do not replace independently trained CT-only
results: their shared encoder/decoder have seen full-modality training.

Output: checkpoints_new/MDT/<hash>/train_log.csv, ckpt.best_joint.pth.tar,
ckpt.last.pth.tar, final_test_metrics.csv/json. Runtime prints Full/Missing
counts, requested ratio and loss weights, actual epoch Full fraction,
validation Dice/HD95, raw/EMA tag. CSV records eval_weights, requested/actual
Full fraction and both loss weights.
The explicit test.txt-as-validation opt-in retains the current comparison
protocol; it is not an independent held-out final test.

Targeted checks: python -m unittest discover -s tests -p test_full_missing_add_reference.py -v
Checks cover default-half exact assignments,70% cumulative quota/determinism,
invalid settings, weighted loss and gradient equations, and checkpoint raw/EMA
selection. Existing model and state-routing implementations are reused.
