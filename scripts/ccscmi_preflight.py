"""Actual 512/batch16 checks, all temporary models discarded before formal training."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import tempfile
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,resolve_config,sha
from variants.tsbci.model import SHCIPETCT
from variants.ccscmi.model import Model
from datasets.staged import loaders
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from utils.amp import make_grad_scaler,autocast_context


def hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--output',required=True)
    args=p.parse_args();require_myenv();torch.set_num_threads(4)
    args.run='preflight';args.arm='full';args.config=str(REPO/'configs/ccscmi_visual.json')
    cfg,_,_=resolve_config(args,'ccscmi_visual');cfg.num_workers=0
    with tempfile.TemporaryDirectory() as d:
        _,val,_=loaders(cfg,Path(d));b=next(iter(val))
    ct,pet,y=[b[k].cuda() for k in ['ct','pet','mask']]
    assert len(y)==16 and ct.shape[-2:]==(512,512)
    seed_everything(2023);base=SHCIPETCT(vars(cfg),arm='m1',load_generic=True).cuda().eval()
    reference_hashes=hashes(base);reference_rng=torch.get_rng_state().clone()
    with torch.no_grad():reference=base(ct[:1],pet[:1])['logits']
    del base;gc.collect();torch.cuda.empty_cache()
    results={};common_visual=None
    for text in [False,True]:
        cfg.text_enabled=text;seed_everything(2023)
        model=Model(vars(cfg),load_generic=True)
        h=hashes(model)
        assert all(h[k]==v for k,v in reference_hashes.items())
        assert torch.equal(torch.get_rng_state(),reference_rng)
        if common_visual is None:common_visual=h
        else:assert all(h[k]==v for k,v in common_visual.items())
        model.cuda().eval()
        with torch.no_grad():initial=model(ct[:1],pet[:1])['logits']
        assert torch.equal(initial,reference),'FP32 initial output changed'
        model.train();scaler=make_grad_scaler(torch.device('cuda'),True)
        opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();success=False
        for attempt in range(8):
            opt.zero_grad(set_to_none=True)
            with autocast_context(torch.device('cuda'),True):o=model(ct,pet,capture=True)
            loss,_=BCEDiceLoss()(o['logits'].float(),y);scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            finite=bool(torch.isfinite(norm));scaler.step(opt);scaler.update()
            if finite:success=True;break
            del o,loss;gc.collect();torch.cuda.empty_cache()
        assert success,'Eight consecutive AMP overflows'
        from variants.tsbci import train as engine
        from variants.ccscmi.train import diagnostics
        with tempfile.TemporaryDirectory() as scratch:
            engine.OUT=Path(scratch)
            diagnostics(model,o,0,1,{'success':True,'backward_scale':scaler.get_scale()})
            assert json.loads((engine.OUT/'diagnostics.jsonl').read_text())['scales']
        results['text' if text else 'visual']={'passed':True,'parameters':sum(v.numel() for v in model.parameters()),
             'shared_initialization_identical':True,'loader_rng_identical':True,'initial_output_exact':True,
             'AMP_batch16_forward_backward_finite':True,'AMP_attempts':attempt+1,
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3}
        del model,opt,scaler,o,loss,initial;gc.collect();torch.cuda.empty_cache()
    results['scope']='Temporary preflight, no experiment performance metrics or reusable trained weights'
    files=list((REPO/'variants/ccscmi').glob('*.py'))+list((REPO/'third_party').glob('*.py'))
    files+=[REPO/'variants/tsbci/train.py',REPO/'configs/ccscmi_visual.json',REPO/'configs/ccscmi_text.json']
    results['source_sha256']={str(p.relative_to(REPO)):sha(p) for p in files}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
