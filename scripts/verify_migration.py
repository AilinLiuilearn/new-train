"""Optional local migration audit against an untouched historical snapshot."""
from pathlib import Path
import argparse
import ast
import importlib
import importlib.util
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, require_myenv, sha


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['tsbci','a2'],required=True)
    p.add_argument('--historical-snapshot',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();require_myenv()
    import torch
    torch.set_num_threads(4)
    old=Path(args.historical_snapshot).resolve()
    for folder in ['models','datasets','utils']:
        for f in (old/'source'/folder).rglob('*.py'):
            assert sha(f)==sha(REPO/'src'/f.relative_to(old/'source')),str(f)
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before=functions(old/'run.py');after=functions(REPO/'variants'/args.variant/'train.py')
    preserved=['stats','gradients','diagnostics','batch_to','objective','update','evaluate']
    assert all(before[n]==after[n] for n in preserved)
    newmod=importlib.import_module('variants.'+args.variant+'.model')
    sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('historical_model_for_audit',old/'model.py')
    oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
    cls='SHCIPETCT' if args.variant=='tsbci' else 'Model'
    cfg=json.loads((old/'CONFIG.json').read_text())
    torch.manual_seed(2023)
    new=getattr(newmod,cls)(cfg,load_generic=False)
    torch.manual_seed(2023)
    original=getattr(oldmod,cls)(cfg,load_generic=False)
    ns,os=new.state_dict(),original.state_dict()
    assert ns.keys()==os.keys()
    assert all(torch.equal(ns[k],os[k]) for k in ns),'Fresh initialization changed'
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    new.load_state_dict(ck['weights'],strict=True)
    original.load_state_dict(ck['weights'],strict=True)
    new.cuda().eval();original.cuda().eval()
    torch.manual_seed(64)
    # At 128, the deepest 4x4 map cannot use the unchanged radius-4 Gaussian kernel.
    ct=torch.randn(2,3,256,256,device='cuda');pet=torch.randn_like(ct)
    with torch.no_grad():
        a=new(ct,pet)['logits'];b=original(ct,pet)['logits']
    difference=float((a-b).abs().max())
    assert torch.equal(a,b),difference
    del original,ck,ns,os
    new.train();new.zero_grad(set_to_none=True)
    from utils.seg_losses import BCEDiceLoss
    from utils.amp import autocast_context
    with autocast_context(torch.device('cuda'),True):out=new(ct,pet)
    target=(torch.rand_like(out['logits'])>.9).float()
    loss,_=BCEDiceLoss()(out['logits'].float(),target);loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v.grad).all() for v in new.parameters() if v.grad is not None)
    report={'variant':args.variant,'checkpoint_sha256':sha(args.checkpoint),'myenv':sys.executable,
            'shared_source_identical':True,'numeric_trainer_functions_unchanged':preserved,
            'fresh_initialization_bitwise_equal':True,'strict_checkpoint_load':True,
            'synthetic_eval_max_abs_difference':difference,'synthetic_AMP_backward_finite':True,
            'parameters':sum(v.numel() for v in new.parameters()),
            'scope':'256x256 synthetic migration check only; not full evaluation or a formal training run'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
