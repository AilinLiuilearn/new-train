"""Check all three fresh recipes on actual512/batch16; discard temporary weights."""
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
from variants.gt_vcl.model import Model
from variants.gt_vcl.losses import objective
from variants.gt_vcl.inference import load_model
from utils.seg_losses import BCEDiceLoss
from utils.ema import ModelEMA
from utils.run_common import seed_everything,module_grad_norm
from utils.amp import make_grad_scaler,autocast_context


def hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k,v in model.state_dict().items()}


def main(args):
    require_myenv();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;torch.set_num_threads(4)
    args.run='gt_vcl_preflight';args.arm='full';args.config=str(REPO/'configs/gt_vcl_E1.json')
    cfg,_,_=resolve_config(args,'gt_vcl_E1');cfg.num_workers=0
    with open(gpu_lock_path(),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory() as scratch:
            _,val,_=loaders(cfg,Path(scratch));batch=next(iter(val))
        ct,pet,y=[batch[k].cuda() for k in ['ct','pet','mask']]
        assert len(y)==16 and ct.shape[-2:]==(512,512)
        seed_everything(2023);base=SHCIPETCT(vars(cfg),arm='m1',load_generic=True).cuda().eval()
        base_hash=hashes(base);rng=torch.get_rng_state().clone()
        with torch.no_grad():reference=base(ct[:1],pet[:1])['logits']
        del base;gc.collect();torch.cuda.empty_cache()
        first_hash=None;reports={}
        for recipe in ['E1','E2','E4']:
            args.config=str(REPO/'configs'/('gt_vcl_'+recipe+'.json'))
            cfg,_,_=resolve_config(args,'gt_vcl_'+recipe)
            seed_everything(2023);model=Model(vars(cfg),load_generic=True)
            initial_hash=hashes(model)
            assert all(initial_hash[k]==v for k,v in base_hash.items())
            assert torch.equal(rng,torch.get_rng_state())
            if first_hash is None:first_hash=initial_hash
            else:assert all(initial_hash[k]==v for k,v in first_hash.items())
            assert not ({p.data_ptr() for p in model.decoder_sep.parameters()}&{p.data_ptr() for p in model.decoder_full.parameters()})
            model.cuda().eval()
            with torch.no_grad():out=model(ct[:1],pet[:1])
            assert torch.equal(out['logits'],reference)
            del out;gc.collect();torch.cuda.empty_cache()
            model.train();ema=ModelEMA(model,.999,warmup=True,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
            scaler=make_grad_scaler(torch.device('cuda'),True);criterion=BCEDiceLoss()
            torch.cuda.reset_peak_memory_stats();attempts=successful=0;seen={};last_parts={}
            while successful<3 and attempts<10:
                attempts+=1;opt.zero_grad(set_to_none=True)
                with autocast_context(torch.device('cuda'),True):out=model(ct,pet,capture=True)
                loss,parts=objective(model,out,y,criterion)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                finite=bool(torch.isfinite(norm));scaler.step(opt);scaler.update()
                if finite:
                    successful+=1;ema.update(model);last_parts=parts
                    for name,module in model.named_children():seen[name]=max(seen.get(name,0),module_grad_norm(module))
                print('PREFLIGHT',recipe,attempts,'successful',successful,'gradient',float(norm),flush=True)
                del out,loss;gc.collect()
            assert successful==3 and all(v>0 for v in seen.values()),seen
            selected=ema.model.eval()
            with torch.no_grad():expected=selected(ct[:1],pet[:1])['logits']
            with tempfile.TemporaryDirectory() as scratch:
                checkpoint=Path(scratch)/'best.pt'
                torch.save({'format':'gt_vcl_inference_v1','config':vars(cfg),'weights':selected.state_dict()},checkpoint)
                deployed=load_model(checkpoint,'cuda')
                assert not hasattr(deployed,'decoder_sep') and not hasattr(deployed,'semantic')
                with torch.no_grad():actual=deployed(ct[:1],pet[:1])['logits']
                assert torch.equal(actual,expected),'Pruned inference differs from selected fusion output'
                del deployed,actual
            reports[recipe]={'passed':True,'parameters':sum(p.numel() for p in model.parameters()),
                'common_initialization_and_rng_identical':True,'initial_full_output_exact':True,
                'shared_CT_PET_decoder_no_fusion_sharing':True,'all_visual_parameters_trainable':True,
                'batch16_AMP_three_updates_finite':True,'AMP_attempts':attempts,
                'max_observed_gradient_groups':seen,'last_temporary_losses':last_parts,
                'pruned_inference_exact':True,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
                'memory_includes_optimizer_EMA_and_temporary_deployment_copy':True}
            del model,selected,ema,opt,scaler,expected;gc.collect();torch.cuda.empty_cache()
        files=list((REPO/'variants/gt_vcl').glob('*.py'))
        files += [REPO/name for name in ['variants/tsbci/train.py','src/models/components/group_unet_decoder.py',
                 'scripts/gt_vcl_preflight.py','resources/gt_vcl_text.json']]
        files += [REPO/'configs'/('gt_vcl_'+r+'.json') for r in ['E1','E2','E4']]
        reports['source_sha256']={str(p.relative_to(REPO)):sha(p) for p in files}
        reports['scope']='Temporary preflight only; no trained weights retained or performance evaluation'
        Path(args.output).write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'));p.add_argument('--gpu',default='0')
    p.add_argument('--output',default=str(REPO/'docs/GT_VCL_GPU_PREFLIGHT.json'));main(p.parse_args())
