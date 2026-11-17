# OriginalT + TSSDA candidate

User-approved2026-11-17. Parent architecture: originalT training8cd9ce2,
selected epoch6EMA Dice78.949269%,HD9516.169107. Preserve original checkpoints.

Only new training factor: final-logit TSSDA coefficient0.01. No mass/KL/boundary
term, no architecture change or added text supervision. Same seed2023,
generic encoder pretraining only, fresh task modules, all visual parameters
trainable,512/batch16,AdamW8e-5,WD1e-4,original augmentation/scheduler/EMA/AMP,
GN decoder,60 epochs maximum,patience3,best-only.

16 fixed theta=k*pi/16, unit PNG pixel coordinates. Normalize sigmoid with
logsigmoid/logsumexp, preserving unit mass without an epsilon deficit. Each
direction uses one-pixel linear interpolation into two bins, preserving mass
and first moment. Integrate CDF absolute difference over one-pixel intervals.
Divide each direction by max(1,GT projected standard deviation). Compute each
valid image separately, then average. EmptyGT contributes0 to new term.
No stochastic angle, hard prediction selection, downsampling, bounded
compression, connected-component filtering or extra prediction heads.

Fixed-order sorted segment summation avoids CUDA atomic scatter accumulation
inside the new loss. It does not guarantee bitwise determinism of inherited
backbone/AMP. New loss runs FP32 outside autocast. Existing originalT inference
loader and checkpoint format remain valid; loss adds no model parameters.

Audit: CPU mass/first-moment preservation, segment/scatter equivalence, finite
differences, shifts, confidence changes, thin/irregular/separate foreground,
empty/single-pixel GT, extreme logits and diffuse background. One real
augmented batch16/512 passed full finite backward/optimizer update, all visual
parameters trainable and GN decoder. Preflight weights discarded. GPU peak
allocation15298.33MiB; first-batch base1.910844,TSSDA21.089790,weighted0.210898;
weighted/base output-gradient norm ratio0.0395673. This is not efficacy evidence.
Machine evidence: management/tssda_20261117/preflight.

Known limits: background probability mass, saturation, finite-direction
ambiguity, whole-slice variance includes separation between foreground
regions. Not per-instance scale, registration, clinical size or mm geometry.

Entry: bash scripts/myenv.sh -m variants.scgr.train_tssda --run scgr_T_tssda_20261117

One full fusion validation per epoch, original threshold/HD95. Log original
and new losses, scale stats, output-gradient ratio/cosine, per-scale visual
and AMP-safe parameter-gradient descriptors. No extra on/off evaluation.
Save best.pt only; no optimizer,scaler,RNG packages or last.pt. Select validation
Dice best and use all other metrics from the same weights. Compare with originalT;
retain incomplete repeat78.5703/16.5099 as incomplete, not a completed baseline.
All development-set reuse remains exploratory, not independent testing.

This launch is one candidate only, no automatic sweeps, LR changes or successors.
Plain SW1 and uniform-radius normalization remain planned controls, not queued.
Pixel errors/empty predictions use existing evaluation; final patient/area/
zero-overlap diagnosis can be a separate offline analysis without reselection.

Natural early stop,max epochs,userSTOP or explicit failure ends the run.
Supervisor records exits and final comparison. Interruptions are not complete;
best-only prevents exact continuation. Improvement and novelty remain unproven.
