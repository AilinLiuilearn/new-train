"""Finite synthetic checks plus seed/architecture controls; no formal experiment."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO,require_myenv


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();require_myenv()
    import torch
    from variants.tsbci.model import SHCIPETCT
    from variants.concat_single.model import ConcatSingle
    from models.components.channel_align import StageChannelAlign
    from utils.run_common import seed_everything
    from utils.seg_losses import BCEDiceLoss
    from utils.amp import autocast_context
    torch.set_num_threads(4)
    cfg=json.loads((REPO/'configs/rerun_m1_p3.json').read_text())
    def hashes(m):
        return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}
    seed_everything(2023);ref=SHCIPETCT(cfg,arm='m1',load_generic=False)
    reference=hashes(ref);rng=torch.get_rng_state().clone();del ref
    report={}
    for variant in ['m1','concat_single','tsbci']:
        seed_everything(2023)
        model=ConcatSingle(cfg,load_generic=False) if variant=='concat_single' else SHCIPETCT(cfg,arm='m1' if variant=='m1' else 'full',load_generic=False)
        current=hashes(model)
        shared=[k for k in reference if k.startswith(('enc_ct.','enc_pet.','ct_align.','decoder_full.'))]
        assert all(current[k]==reference[k] for k in shared)
        assert torch.equal(rng,torch.get_rng_state()),'Loader RNG stream changed'
        alignment_count=sum(isinstance(m,StageChannelAlign) for m in model.modules())
        assert alignment_count==(1 if variant=='concat_single' else 2)
        assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.decoder_full.modules())
        if variant=='concat_single':assert not hasattr(model,'pet_align')
        model.cuda().train();ct=torch.randn(2,3,256,256,device='cuda');pet=torch.randn_like(ct)
        with autocast_context(torch.device('cuda'),True):output=model(ct,pet,capture=True)
        loss,_=BCEDiceLoss()(output['logits'].float(),(torch.rand_like(output['logits'])>.9).float())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        report[variant]={'parameters':sum(v.numel() for v in model.parameters()),'alignment_modules':alignment_count,
                         'shared_seeded_parameters_identical':True,'subsequent_RNG_identical':True,
                         'decoder_GN':True,'synthetic_AMP_forward_backward_finite':True,
                         'output_shape':list(output['logits'].shape),'patience':3}
        del model,ct,pet,output,loss;gc.collect();torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
