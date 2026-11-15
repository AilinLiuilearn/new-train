"""Run only after the current GPU campaign: one real sample, no experiment metrics."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv,resolve_config,sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--init',required=True);p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--output',required=True)
    args=p.parse_args();require_myenv()
    import torch
    from datasets.staged import loaders
    from variants.v31.model import JointModel,CorrectedModel
    from variants.v31.losses import collate,objective
    from variants.v31.train import fingerprint
    from utils.amp import autocast_context
    from utils.run_common import seed_everything
    torch.set_num_threads(4);seed_everything(2023)
    args.config=str(REPO/'configs/v31.json');args.run='preflight';args.arm='full'
    cfg,_,_=resolve_config(args,'v31');cfg.batch_size=1;cfg.num_workers=0
    with tempfile.TemporaryDirectory() as scratch:
        _,val,_=loaders(cfg,Path(scratch));b=collate([val.dataset[0]])
    b={k:v.cuda() for k,v in b.items() if torch.is_tensor(v)}
    m=JointModel(vars(cfg));ck=torch.load(args.init,map_location='cpu',weights_only=False,mmap=True);m.import_m1(ck);del ck
    m.cuda().eval()
    with torch.no_grad():
        reference=m.base(b['ct'],b['pet'])['logits'];new=m(b['ct'],b['pet'])['logits']
        assert torch.equal(reference,new)
    m.train()
    with autocast_context(torch.device('cuda'),True):o=m(b['ct'],b['pet'])
    loss,_=objective(o,b,'joint');loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.ct_expert.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.pet_expert.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.zero_grad(set_to_none=True);del o,loss,reference,new
    corrected=CorrectedModel(m).cuda();before=fingerprint(m);corrected.train()
    with autocast_context(torch.device('cuda'),True):o=corrected(b['ct'],b['pet'])
    assert torch.equal(o['logits'],o['base_logits'])
    loss,_=objective(o,b,'correction');loss.backward()
    assert all(p.grad is None for p in m.parameters())
    assert all(not module.training for module in m.modules())
    assert all(torch.isfinite(p.grad).all() for p in corrected.corrector.parameters() if p.grad is not None)
    opt=torch.optim.AdamW(corrected.corrector.parameters(),lr=8e-5);opt.step()
    assert fingerprint(m)==before
    report={'passed':True,'historical_M1_strict_EMA_import':True,'joint_fusion_initially_identical':True,
            'independent_experts_receive_finite_gradients':True,'zero_corrector_exact_identity':True,
            'frozen_parameters_and_buffers_unchanged_after_update':True,'no_expert_gradient_in_correction':True,
            'input_shape':list(b['ct'].shape),'init_sha256':sha(args.init),
            'scope':'One actual validation sample for forward/backward only; not a performance evaluation'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
