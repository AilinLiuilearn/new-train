"""Disposable full512/batch16 checks for V -> T -> S; no formal training."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from runtime import REPO,require_myenv,resolve_config,sha,gpu_lock_path
from datasets.staged import loaders
from variants.tsbci.model import SHCIPETCT
from variants.scgr.model import Model
from variants.scgr.losses import objective
from variants.scgr.inference import load_model
from utils.seg_losses import BCEDiceLoss
from utils.ema import ModelEMA
from utils.run_common import seed_everything,module_grad_norm
from utils.amp import make_grad_scaler,autocast_context


def hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}


def main(args):
    require_myenv();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;torch.set_num_threads(4)
    args.run='scgr_preflight';args.arm='full';args.config=str(REPO/'configs/scgr_V.json')
    cfg,_,_=resolve_config(args,'scgr_V');cfg.num_workers=0
    with open(gpu_lock_path(),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory() as scratch:
            _,val,_=loaders(cfg,Path(scratch));batch=next(iter(val))
        ct,pet,y=[batch[k].cuda() for k in ['ct','pet','mask']]
        assert len(y)==16 and ct.shape[-2:]==(512,512)
        seed_everything(2023);base=SHCIPETCT(vars(cfg),arm='m1',load_generic=True).cuda().eval()
        base_hash=hashes(base);rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        with torch.no_grad():reference=base(ct[:1],pet[:1])['logits']
        del base;gc.collect();torch.cuda.empty_cache()
        common=None;reports={}
        for recipe in ['V','T','S']:
            args.config=str(REPO/'configs'/f'scgr_{recipe}.json');cfg,_,_=resolve_config(args,'scgr_'+recipe)
            seed_everything(2023);model=Model(vars(cfg),load_generic=True)
            initial=hashes(model)
            assert all(initial[k]==v for k,v in base_hash.items())
            assert torch.equal(rng,torch.get_rng_state())
            assert torch.equal(cuda_rng,torch.cuda.get_rng_state()),'New CPU module changed common CUDA RNG'
            if common is None:common=initial
            else:assert initial==common
            assert all(p.requires_grad for p in model.parameters())
            assert not hasattr(model,'decoder_sep') and not hasattr(model,'decoder_ct')
            model.cuda().eval()
            with torch.no_grad():out=model(ct[:1],pet[:1])
            assert torch.equal(out['logits'],reference)
            del out;gc.collect();torch.cuda.empty_cache()
            model.train();ema=ModelEMA(model,.999,warmup=True,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
            scaler=make_grad_scaler(torch.device('cuda'),True);criterion=BCEDiceLoss()
            torch.cuda.reset_peak_memory_stats();attempts=successful=0;seen={};parts={}
            while successful<3 and attempts<10:
                attempts+=1;opt.zero_grad(set_to_none=True)
                with autocast_context(torch.device('cuda'),True):out=model(ct,pet,capture=True)
                loss,parts=objective(model,out,y,criterion);assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                finite=bool(torch.isfinite(norm))
                if finite:
                    for s,b in enumerate(model.refiner.blocks,1):
                        for name,sub in b.named_children():
                            key=f'S{s}/{name}';seen[key]=max(seen.get(key,0),module_grad_norm(sub))
                scaler.step(opt);scaler.update()
                if finite:successful+=1;ema.update(model)
                print('PREFLIGHT',recipe,attempts,'successful',successful,'gradient',float(norm),flush=True)
                del out,loss;gc.collect()
            assert successful==3
            assert all(v>0 for k,v in seen.items() if not (recipe=='V' and k.endswith('/text_projection'))),seen
            if recipe=='V':assert all(v==0 for k,v in seen.items() if k.endswith('/text_projection'))
            selected=ema.model.eval()
            with torch.no_grad():expected=selected(ct[:1],pet[:1])['logits']
            with tempfile.TemporaryDirectory() as scratch:
                ck=Path(scratch)/'best.pt'
                torch.save({'format':'scgr_support_conditioned_v1','config':vars(cfg),'weights':selected.state_dict()},ck)
                deployed=load_model(ck,'cuda')
                with torch.no_grad():actual=deployed(ct[:1],pet[:1])['logits']
                assert torch.equal(actual,expected)
                del deployed,actual
            reports[recipe]={'passed':True,'parameters':sum(p.numel() for p in model.parameters()),
                'common_initialization_rng_identical':True,'initial_identity_exact':True,
                'single_GN_decoder':True,'all_visual_parameters_trainable':True,
                'batch16_512_AMP_successes':successful,'AMP_attempts':attempts,
                'max_observed_gradients':seen,'temporary_last_losses':parts,
                'checkpoint_inference_exact':True,'text_tower_at_inference':False,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
                'memory_includes_optimizer_EMA_and_temporary_deployed_copy':True}
            del model,selected,ema,opt,scaler,expected;gc.collect();torch.cuda.empty_cache()
        files=list((REPO/'variants/scgr').glob('*.py'))+[REPO/f for f in ['variants/tsbci/train.py',
            'scripts/scgr_preflight.py','scripts/run_scgr.py','scripts/scgr_examples.py','resources/scgr_text.json']]
        files += [REPO/'configs'/f'scgr_{r}.json' for r in ['V','T','S']]
        reports['source_sha256']={str(p.relative_to(REPO)):sha(p) for p in files}
        reports['scope']='Disposable implementation checks, not experiment metrics. No temporary model retained.'
        Path(args.output).write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--gpu',default='0')
    p.add_argument('--output',default=str(REPO/'docs/scgr/GPU_PREFLIGHT.json'));main(p.parse_args())
