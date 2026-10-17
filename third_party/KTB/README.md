# KTB source provenance

- Paper: Jiaxin Cai et al., *Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X Semantic Segmentation*, CVPR 2025, pp. 10587–10598.
- Paper URL: https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html
- Official repository: https://github.com/imcjx/KTB
- Pinned commit: `75370dd584fb9d8571bced1aabade198dfbdc80f`.
- `swin.py`: exact upstream `semseg/models/backbones/swin.py`, Git blob `e333b8f4f881b1f4846682d4470f305ac0158291`.
- `LICENSE`: exact upstream root Apache-2.0 license, Git blob `ac4f24556b2e04cf30cc4c6fa294b2b0d77c7bf2`. No upstream NOTICE file exists at this pin.

`models/components/dscf_source.py` extracts the unchanged class bodies
`LayerNormProxy`, `conv_bn_relu`, `DAttentionMM`, and `DeformMPGBlock`.
It omits unrelated Swin/MAPA code and framework imports; all executable
integration changes are in `models/components/dscf_residual.py`.
Tests verify the archived blob hashes, exact extracted-class AST identity,
and forward/input-gradient/parameter-gradient parity against classes
executed directly from this archive at all four real feature sizes.

This experiment borrows DSCF, not the complete KTB network or training recipe.
Supported attention path is the upstream default sampled relative-position
bias path (`no_off=False`, `offset_range_factor=-1`, `fixed_pe=False`,
`dwc_pe=False`, `log_cpb=False`, one group, two heads). Unused upstream
alternative branches are retained in the archive/extraction and not exposed
as experiment options.

Explicit adaptations:

1. Independent low-rank CT/PET projections with ranks `[16,16,40,64]`.
2. Per-stage query geometry and sampling stride from `image_size_2d`;
   at 512px these are `[128,64,32,16]` and `[16,8,4,2]`, respectively.
3. Only the final `U_fc1` weights and bias are zero-initialized.
4. The correction runs in FP32 with AMP disabled locally, retaining the
   source attention/grid-sampling arithmetic; output is cast to the Add dtype.
5. The external output is `CT + PET + correction`, instead of replacing Add.
6. Input validation and detached residual-size diagnostics surround the core.

The native internal identity/deform weights, grouped offset convolutions,
six sampling operations, modality sample weighting, relative-position bias,
and attention equations are preserved. The reference-grid coordinates and
clamping are also preserved to retain parity with this selected source path.
