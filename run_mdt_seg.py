# -*- coding: utf-8 -*-
"""Compatibility entry for the old training script.

Forwarding rule (printed at startup, always):
  --train_batch_mode=mixed -> delegates to run_full_missing_baseline.main
  any other value         -> raises: alternating/full whole-batch switching
                             was removed from the clean baselines.

Use run_full_missing_baseline.py / run_ct_only_seg.py directly for new runs.
"""
from configs.seg_mdt import SegMDTConfig


def main():
    import run_full_missing_baseline as mixed_runner
    cfg = SegMDTConfig.parse_arguments()
    mode = str(getattr(cfg, 'train_batch_mode', 'alternating'))
    if mode != 'mixed':
                    raise ValueError(
            f'run_mdt_seg.py no longer supports train_batch_mode={mode!r}: '
            'alternating/full whole-batch switching was removed. '
            'Run the Full/Missing mixed baseline with --train_batch_mode mixed, '
            'or use run_ct_only_seg.py for the CT-only baseline.')
    print('[INFO] run_mdt_seg.py compatibility entry: actual training mode is '
          'the Full/Missing mixed baseline (run_full_missing_baseline)', flush=True)
    mixed_runner.main()


if __name__ == '__main__':
    main()