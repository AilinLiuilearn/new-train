# Fixed CT-only reference

Based on `exp/aligned-add-baseline` at `663f978`.

Architecture is unchanged: ConvNeXtV2-Nano -> current CT StageChannelAlign ->
GroupNorm UNetStyleDecoder. Uses all training CT slices, reads no PET files,
instantiates no PET encoder, fusion or memory. Loss is the same BCE+Dice.

Use the same normalization, augmentation, input size, batch size, optimizer,
learning rate, training budget and evaluation EMA policy as the main experiments.
This branch fixes two final-test issues only: selected weights must follow the
checkpoint's `eval_weights` tag (EMA merely existing is insufficient), and a
CT-only model must use the CT evaluation interface rather than the dual-modal
missing-rate interface. Final test is one pass, not five redundant missing rates.
Existing final_missing_rates CLI options are retained by the shared config but
are not used by this CT-only entry.

```bash
python run_ct_only_seg.py \
  --hash ct-only-reference-imagenet-ema-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test True \
  --ct_backbone convnextv2_nano \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --image_size_2d 512 --batch_size 16 --num_workers 2 \
  --norm_mode imagenet --aug_mode cipa \
  --optimizer adamw --learning_rate 0.00008 --weight_decay 0.0001 \
  --ema_enabled True --ema_start_epoch 3 --ema_decay 0.999 \
  --ema_decay_warmup True \
  --hd95_mode cipa_official --random_state 2023 --epochs 60
```

Epochs 1-3 select online weights; epoch 4 onward select EMA weights. The best
validation CT Dice checkpoint determines both reported Dice and HD95. No
separate HD95-best checkpoint is cherry-picked. Output under
`checkpoints_new/MDT/<hash>/`: `train_log.csv`, `config_args.json`,
`ckpt.best_ct.pth.tar`, `ckpt.last.pth.tar`, `final_test_ct.json`.
The final JSON records selected epoch, raw/EMA tag, Dice, IoU and HD95 mode.

Using test.txt as validation matches the user's existing comparison protocol,
but does not produce an independent final test estimate. If all experiments
later switch to separate validation, switch all together.

One seed produces a reproducible reference run, not a universal CT ceiling or
an estimate of run-to-run variance. This CT-only baseline measures independent
CT training; same-framework no-retrieval controls separately isolate retrieval.
No baseline architecture, EMA decay/schedule, loss or metric implementation was
changed. Targeted synthetic CT-only tests exercise both pre-EMA/raw and active
EMA final selection without any PET data.
