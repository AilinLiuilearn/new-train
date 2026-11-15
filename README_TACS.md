# TACS: Text-guided Adaptive Channel-Spatial Sparse Fusion

This branch adds a separate Full-only fusion experiment. It does not modify the
clean Add baseline branch or the existing CT/PET encoders and GN decoder.

## Design

For each of the four aligned feature scales (channels 64/128/320/512):

1. **Channel routing (BCSI-inspired adaptation).** CT and PET each produce a
   global channel descriptor and a modality-specific channel score. The scores
   are combined with a smooth maximum, then Top-K selects a shared channel
   subset. The initial ratio is 25%, giving K = [16, 32, 80, 128]. This is
   deliberately a per-scale extension for this PET-CT model; it is not a claim
   that the original BCSI applies its router at every scale.
2. **Spatial sampling (DSCF-inspired adaptation).** CT and PET independently
   predict bounded 2-D offsets from an 8x8 reference grid (or the largest grid
   supported by a small smoke-test feature map). The union of both deformed
   point sets is used to sample both modalities.
3. **Bidirectional channel interaction (BCSI-inspired).** Each selected
   channel is represented by its response profile over the sparse union of
   spatial points. CT channel tokens retrieve PET channel context and PET
   channel tokens retrieve CT channel context.
4. **Bidirectional spatial interaction (DSCF-inspired).** CT reference queries
   retrieve channel-refined PET features sampled at the union positions; PET
   reference queries retrieve channel-refined CT features. Both directions
   operate only in the selected channel subspace.
5. **Residual reinsertion.** The sparse correction is projected to K channels,
   upsampled to the feature resolution, and scattered back only into the
   selected channel indices:

   F_out = CT + PET + Delta_selected.

   Unselected channels bypass the correction. The output projection is
   zero-initialized, so the initial module output is exactly the original
   Add fusion.

## Text prior

Text is optional and is not required to run TACS. When enabled, one frozen
BioMedCLIP text embedding is loaded once and projected into each scale's
channel-router score space. It biases **channel selection only**; it does not
directly predict spatial offsets or segmentation masks. The spatial selector
remains driven by image features.

Default short prompt:

> Lung tumor: focal PET uptake guided by CT anatomy and lesion boundaries, distinct from physiological uptake.

Create the embedding once:

    pip install open_clip_torch
    python tools/encode_tacs_text.py --output /root/autodl-tmp/mkd-main/new-train/tacs_text_embedding.pt

Then enable it:

    bash scripts/train_tacs.sh --tacs_use_text true --tacs_text_embedding_path /root/autodl-tmp/mkd-main/new-train/tacs_text_embedding.pt

The segmentation model stores the fixed embedding in its state dict and does
not run the text encoder during training or inference.

## Controlled experiments

Keep the same data, split, encoders, decoder, loss, optimizer, LR, EMA, batch
size, and training budget.

| Run | tacs_channel_ratio | Text | Purpose |
|---|---:|---|---|
| Add baseline | — | off | Existing reference |
| Spatial-only TACS | 1.0 | off | Disable channel routing/attention; retain sparse spatial interaction |
| Channel + spatial TACS | 0.25 | off | Main module |
| Channel + spatial + text | 0.25 | on | Measure text-prior contribution |

Recommended order: first compare Add vs spatial-only vs channel+spatial. Only
if channel+spatial is promising, run the text-guided variant. A later ratio
ablation can compare 0.125/0.25/0.5; do not tune all ratios at once.

## Run and tests

    PYTHONPATH=. python -m pytest tests/test_tacs_fusion.py -q
    bash scripts/train_tacs.sh

The runner retains the baseline training defaults: 512x512, batch 16,
AdamW, LR 8e-5, 60 epochs, EMA 0.999, and BCE+Dice. TACS is only called on
the Full CT+PET path. The inherited Missing path bypasses PET and bypasses
TACS, so Missing inference remains CT-only.

**Evaluation warning:** the supplied script mirrors the current project
configuration that explicitly reuses test.txt for validation and testing.
Those results are development/model-selection results, not an independent
held-out test estimate. Use a patient-disjoint validation split for formal
reporting.

## Interpretation checks

Track the per-scale tacs_delta_ratio_s* and tacs_offset_abs_s* values in
train_log.csv. A zero delta at initialization is expected. During training,
check that:
- the correction becomes non-zero rather than remaining inactive;
- all four scales receive finite gradients;
- the learned offsets remain within the configured local range;
- the channel+spatial variant improves over spatial-only, not merely over Add;
- text guidance improves over the same channel+spatial checkpoint recipe.

This is a testable design hypothesis, not a guaranteed performance gain.
