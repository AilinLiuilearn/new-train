# Full/Missing Add reference (50/50)

Independent entry: `run_full_missing_add_reference.py`.
Base: exp/aligned-add-baseline @663f978.
Branch: exp/full-missing-add-reference. Existing entry/model/config/task untouched.

- Each batch16: exactly8 Full and8 Missing, assignment randomized each step.
- Full: real CT/PET encoders -> both existing aligns -> Add -> original decoder.
- Missing: CT encoder -> ct_align -> same decoder; PET is never encoded for
  missing samples. No GFRM, retrieval, compensation, CT auxiliary decoder,
  extra supervision or losses.
- Loss: 0.5 Full BCE+Dice + 0.5 Missing BCE+Dice, unchanged shared training task.
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

No --full_ratio or loss-weight flags: this entry fixes both to50/50 explicitly.
50/50 is a reproducible balanced reference, not a proven optimal ratio.
Changing to70/30 later requires a matching70/30 no-compensation control.
Joint-trained Missing results do not replace independently trained CT-only
results: their shared encoder/decoder have seen full-modality training.

Output: checkpoints_new/MDT/<hash>/train_log.csv, ckpt.best_joint.pth.tar,
ckpt.last.pth.tar, final_test_metrics.csv/json. Runtime prints Full/Missing
counts, validation Dice/HD95, raw/EMA tag. CSV records eval_weights.
The explicit test.txt-as-validation opt-in retains the current comparison
protocol; it is not an independent held-out final test.

Targeted checks: python -m unittest discover -s tests -p test_full_missing_add_reference.py -v
Only fixed ratio/config and recorded raw/EMA checkpoint selection are changed;
existing model/loss/state-routing implementations are reused.
