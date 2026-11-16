"""One full-model evaluation of a selected checkpoint, without training."""
from pathlib import Path
import argparse
import importlib
import json
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv, resolve_config, verify_resources, environment, sha

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['tsbci', 'a2', 'concat_single', 'v31', 'ccscmi', 'scgr'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base-checkpoint',help='Relocated v31 joint best.pt; SHA must match correction checkpoint')
    p.add_argument('--paths', default='paths.local.json')
    p.add_argument('--config')
    p.add_argument('--output', required=True, help='New evaluation report path')
    args = p.parse_args()
    require_myenv()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    default_config='rerun_concat_single_p3.json' if args.variant=='concat_single' else args.variant+'.json'
    if args.variant=='ccscmi':default_config='ccscmi_visual.json'
    if args.variant=='scgr':default_config='scgr_V.json'
    args.config = args.config or str(Path(__file__).resolve().parents[1] / 'configs' / default_config)
    report_path = Path(args.output).resolve()
    if report_path.exists():raise FileExistsError(report_path)
    args.output = None;args.run = 'evaluation';args.arm = 'full'
    cfg, _, locations = resolve_config(args, args.variant)
    cfg._evaluation_only = True
    verify_resources(locations)
    module = importlib.import_module('variants.' + args.variant + '.inference')
    m = module.load_model(args.checkpoint, device='cuda',base_checkpoint=args.base_checkpoint) if args.variant=='v31' else module.load_model(args.checkpoint,device='cuda')
    from datasets.staged import loaders
    from utils.metrics_seg import SegmentationMetricsCIPA
    metric = SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode)
    with tempfile.TemporaryDirectory() as scratch:
        _, val, _ = loaders(cfg, Path(scratch))
        with torch.no_grad():
            for batch in val:
                ct,pet,y = [batch[k].to('cuda',non_blocking=True) for k in ['ct','pet','mask']]
                metric.update(m(ct,pet)['logits'],y)
    report = {'metrics':metric.compute(),'checkpoint_sha256':sha(args.checkpoint),
              'environment':environment(),'scope':'Reused development set, full fusion path only'}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
