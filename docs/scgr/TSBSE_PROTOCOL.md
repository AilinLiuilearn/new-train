# Original T plus revised TSBSE (candidate third contribution)

Only final-logit supervision changes. Original M1, BioMedCLIP-conditioned postfusion refiner and GN decoder are unchanged. Generic ConvNeXtV2/MiT pretraining only; all task modules fresh, no originalT/experts/task checkpoints. All visual parameters trainable. Same checkpoint format, same CT/PET-only inference, no new inference parameter.

## Exact first-run objective
Original BCE+Dice plus 0.005 times the mean of cover and spill. Each image is independent; all sample risks including skipped-zero samples are averaged over full batch.

Current predicted foreground = detached logit>=0. GT foreground = target>0.5. Exact Euclidean distance to foreground (SciPy EDT on CPU). Predicted empty mask receives image-diagonal distance. GT scale r=max(1,sqrt(GT_area/pi)). Geometry weight=d/(d+r), detached, in [0,1).

Cover: mean over GT of predicted-distance weight times softplus(-logit).
Spill: background risks=GT-distance weight times softplus(logit); take mean of largest K risks where K=min(GT area,background count). Gradients pass through selected risk values. No straight-through estimator, instance labeling, random directions, probability normalization, or extra heads.
Empty GT skips added term; no background gives zero spill, retaining cover. Original BCE+Dice always retained.

## Weight audit and known limitations
Actual augmented 16x512 initial batch: provisional weight0.01 gave output-logit gradient norm19.38% of base; first run uses0.005, confirmed by repeated full preflight before launch (~9.52%, FP16 gradient quantization prevents exact half scaling). Selection uses numerical initial-batch scale only, no validation tuning. Initial added gradient is mostly spill; common coefficient does not imply balanced contributions. Log raw cover/spill risks, valid counts, K/area/radius, timing and both branch output gradients every100 training batches. Original per-scale feature/gate/residual and AMP parameter-gradient diagnostics retained.

Bounded geometric weights bound per-image absolute output-logit derivative sums, not total loss or parameter gradients. Geometry threshold is nonsmooth. Small K concentrates gradients; whole-slice radius mixes regions; very far weights saturate. Top-K may select ambiguous/noisy regions and can favor distant diffuse background over near-boundary FP. No claim to recover visually absent evidence or directly optimize HD95.

## Evidence and comparison
CPU checks: exact implementation covers empty/full GT, empty/full prediction, +/-1000 logits, stable-rank finite difference and zero-skipped batch averaging. Real preflight: full forward/backward/AdamW discarded step, no checkpoint loaded/saved, decoder verified GN and all parameters trainable. Preflight Scaler1024 only; formal training retains original Scaler settings.

One authorized full candidate, no automatic sweep/control queue. Seed2023,512,batch16,lr8e-5,AdamW wd1e-4, original augmentation/scheduler/EMA, max60 and patience3, myenv, best-only. One full validation each epoch. Same selected best reports Dice/HD95/precision/sensitivity, zero-overlap/empty prediction counts, FP/FN spatial partitions, GT area group pooled Dice. Original confirmed T78.949269%/16.169107 stays unchanged. Reused development set, not independent test; same-seed nondeterminism means single-run change is not isolated causal evidence.

Candidate innovation only. Prior Guzzi LAHsym source already has GT-area normalization norm3 and prediction distance outside no_grad. Our review and CPU prototypes are stored in analysis/tsbse_review_20261117. Later controls needed: same hard-negative risk without geometry, closest existing distance loss. Training request does not authorize publishing claims of novelty or efficacy.
