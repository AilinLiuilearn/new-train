# CMX integration verification

Environment: Python 3.12.14, PyTorch 2.5.1+cpu, timm 1.0.11, transformers 4.44.2, torchvision 0.20.1+cpu. No CUDA device is available in this development environment. Tests use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Executed checks

| Group | Result | Coverage |
| --- | --- | --- |
| `tests/test_cmx_fusion.py` | 12 passed, 2 CUDA tests skipped | Exact source/license hashes; official constructors/heads/norm; all four arms at actual feature scales; independent original-module output/input-gradient parity; first-step parameter gradients; FFM initialization consistency across FRM toggle; interpolation and invalid inputs |
| `tests/test_cmx_integration.py` | 17 passed | Actual backbones; shared baseline initialization; disabled exact Add equality; four arms with two optimizer updates; no auxiliary loss; training/validation/EMA/final reconstruction; raw/EMA strict reloads; Full-only rejection before encoders; startup/retry/checkpoint safety; metadata validation |
| Original baseline regression | 53 passed | `test_baseline_contract.py`, `test_ema.py`, `test_full_add_runner.py`, `test_mixed_batch_training.py` |
| Actual 512 preflight | passed | Default FFM only, CPU batch 2, real CT/PET encoders/aligners/decoder, two complete forward/backward/optimizer updates |
| Launcher argument check | passed | Four arm flag mappings, seed/lr/input/batch/CUDA protocol, distinct run IDs, invalid arm rejected with exit 2 |
| Python compile and Bash syntax | passed | All newly added Python files; `bash -n scripts/run_full_cmx_ablation.sh` |

Total pytest results across the three groups: **82 passed, 2 skipped**. Integration fixtures use synthetic inputs, not PCLT20K patients or pretrained weights. Existing deprecation and scheduler-test warnings do not change the results.

512 preflight output:

```text
[PASS] shared baseline weights equal; disabled fusion exactly Add; enabled fusion uses official init
[PASS] optimizer step=1 main_BCE_Dice=1.742681
[PASS] optimizer step=2 main_BCE_Dice=1.481983
status=passed, device=cpu, image_size=512, batch_size=2
cmx_ffm_enabled=true, cmx_frm_enabled=false
params_fusion=6572032, params_total=39014673
peak_cuda_gib=null
```

## Reproduce

With the project's normal dependencies installed:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q tests/test_cmx_fusion.py tests/test_cmx_integration.py
python -m pytest -q tests/test_baseline_contract.py tests/test_ema.py \
  tests/test_full_add_runner.py tests/test_mixed_batch_training.py
python scripts/check_cmx.py --device cpu --image_size 512 --batch_size 2
git hash-object models/components/cmx_source.py third_party/CMX_LICENSE
```

The official source contains upstream trailing whitespace. Keep the original bytes; apply whitespace checks to the integration files while excluding `models/components/cmx_source.py`.

## Remaining server checks

Run `python scripts/check_cmx.py --device cuda --image_size 512 --batch_size 2` before launching training. CUDA AMP, GPU peak memory at batch 16, original pretrained-file loading and real-data performance were not tested here. The optional FRM paths were tested at actual feature scales in unit tests and with real backbones at smaller input size; only default FFM received the complete 512 real-backbone preflight.

Retain the Add control and compare like-for-like runs. The tests establish integration correctness, not a Dice improvement. Do not infer cross-modal utility from gradient magnitudes alone.

## Rollback and invariants

All tracked baseline files at `663f9786122365d8b2774e4497df92bc06b715d5` remain byte-identical. Only new CMX files, tests and documentation are added. The original Add runner/model remain available. Use `exp/aligned-add-baseline` to return to the baseline experiment.

No extra FFM residual, zero initialization, decoder auxiliary supervision, custom fusion loss, CT-only PET fabrication or Missing-path access is introduced. Source and license Git blobs match the pinned upstream copies; see `CMX_SOURCE.md`.
