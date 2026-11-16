"""Temporary full-model 512/batch16 checks before the single K50 experiment."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from runtime import REPO, require_myenv, resolve_config, sha, gpu_lock_path
from variants.tsbci.model import SHCIPETCT
from variants.bcsi_adapt.model import Model
from variants.bcsi_adapt.train import diagnostics
from variants.bcsi_adapt.inference import load_model
from variants.tsbci import train as engine
from datasets.staged import loaders
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from utils.amp import make_grad_scaler, autocast_context
from utils.ema import ModelEMA


def hashes(model):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k,v in model.state_dict().items()}


def main(args):
    require_myenv();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;torch.set_num_threads(4)
    args.run='bcsi_adapt_preflight';args.arm='full';args.config=str(REPO/'configs/bcsi_adapt.json')
    cfg,_,_=resolve_config(args,'bcsi_adapt');cfg.num_workers=0
    with open(gpu_lock_path(),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory() as scratch:
            _,val,_=loaders(cfg,Path(scratch));batch=next(iter(val))
        ct,pet,y=[batch[k].cuda() for k in ['ct','pet','mask']]
        assert len(y)==16 and ct.shape[-2:]==(512,512)
        seed_everything(2023)
        base=SHCIPETCT(vars(cfg),arm='m1',load_generic=True).cuda().eval()
        common_hashes=hashes(base);rng=torch.get_rng_state().clone()
        with torch.no_grad():reference=base(ct[:1],pet[:1])['logits']
        del base;gc.collect();torch.cuda.empty_cache()
        seed_everything(2023)
        model=Model(vars(cfg),load_generic=True)
        candidate_hashes=hashes(model)
        assert all(candidate_hashes[k]==v for k,v in common_hashes.items())
        assert torch.equal(torch.get_rng_state(),rng)
        model.cuda().eval()
        with torch.no_grad():
            initial=model(ct[:1],pet[:1],capture=True)
        initial_logit_delta=float((initial['logits']-reference).abs().max())
        initial_feature_deltas={};initial_feature_rms={}
        for i,info in enumerate(initial['capture']['interaction']):
            for modality in ['ct','pet']:
                raw=initial['capture'][modality+'_aligned'][i]
                updated=initial['capture'][modality+'_interacted'][i]
                ids=info[modality+'_indices'];expected=raw.shape[1]//2
                assert ids.shape[1]==expected
                mask=torch.zeros(raw.shape[:2],device=raw.device,dtype=torch.bool).scatter(1,ids,True)
                assert torch.equal(raw[~mask],updated[~mask])
                initial_feature_deltas[f'{modality}_s{i+1}']=float((updated-raw).float().square().mean().sqrt()
                                                                /(raw.float().square().mean().sqrt()+1e-8))
                initial_feature_rms[f'{modality}_s{i+1}']=float(raw.float().square().mean().sqrt())
        del initial,reference;gc.collect();torch.cuda.empty_cache()
        model.train();ema=ModelEMA(model,.999,warmup=True,device='cuda')
        criterion=BCEDiceLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
        scaler=make_grad_scaler(torch.device('cuda'),True)
        torch.cuda.reset_peak_memory_stats();attempts=successful=0;last_snapshot=None;first_snapshot=None
        while successful<3 and attempts<10:
            attempts+=1;optimizer.zero_grad(set_to_none=True);scale_before=scaler.get_scale()
            with autocast_context(torch.device('cuda'),True):output=model(ct,pet,capture=True)
            loss,_=criterion(output['logits'].float(),y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            finite=bool(torch.isfinite(norm));scaler.step(optimizer);scaler.update()
            if finite:
                successful+=1;ema.update(model)
                with tempfile.TemporaryDirectory() as scratch:
                    engine.OUT=Path(scratch)
                    diagnostics(model,output,0,successful,{'success':True,'backward_scale':scale_before})
                    last_snapshot=json.loads((engine.OUT/'diagnostics.jsonl').read_text())
                    if first_snapshot is None:first_snapshot=last_snapshot
            print('PREFLIGHT',attempts,'successful_steps',successful,'gradient',float(norm),flush=True)
            del output,loss;gc.collect()
        assert successful==3,'Could not complete three finite AMP task-only updates'
        gradient_summary={}
        for key,row in last_snapshot['scales'].items():
            assert all(value>0 for value in row['gradient_groups'].values()),row['gradient_groups']
            for modality in ['ct','pet']:
                assert row[modality+'_router_score_gradient']['selected_abs_mean']>0
                assert row[modality+'_router_score_gradient']['unselected_abs_mean']>0
            gradient_summary[key]={'parameter_groups':row['gradient_groups'],
                                   'ct_scores':row['ct_router_score_gradient'],
                                   'pet_scores':row['pet_router_score_gradient']}
        with tempfile.TemporaryDirectory() as scratch:
            p=Path(scratch)/'best.pt'
            torch.save({'format':'bcsi_adapt_k50_inference_v1','weights':ema.model.state_dict(),'config':vars(cfg)},p)
            restored=load_model(p,device='cpu')
            assert hashes(restored)==hashes(ema.model)
            del restored
        result={'passed':True,'parameters':sum(p.numel() for p in model.parameters()),
                'new_parameters':sum(p.numel() for p in model.front.parameters()),
                'selected_channels':[32,64,160,256],
                'shared_initialization_identical':True,'loader_rng_identical':True,
                'direct_residual_initialization':'Intentional nonidentity; no zero exit or learned alpha',
                'initial_max_abs_logit_delta':initial_logit_delta,
                'initial_relative_feature_deltas':initial_feature_deltas,
                'initial_eval_feature_rms':initial_feature_rms,
                'first_train_mode_relative_deltas':{
                    modality+'_s'+scale:row[modality+'_relative_delta_rms']
                    for scale,row in first_snapshot['scales'].items() for modality in ['ct','pet']},
                'first_train_mode_feature_rms':{
                    modality+'_s'+scale:row[modality+'_aligned']['rms']
                    for scale,row in first_snapshot['scales'].items() for modality in ['ct','pet']},
                'third_train_mode_relative_deltas':{
                    modality+'_s'+scale:row[modality+'_relative_delta_rms']
                    for scale,row in last_snapshot['scales'].items() for modality in ['ct','pet']},
                'unselected_channels_unchanged':True,'AMP_batch16_three_task_updates_finite':True,
                'AMP_attempts':attempts,'task_gradient_summary':gradient_summary,
                'best_only_roundtrip':True,'EMA_and_optimizer_included':True,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
                'scope':'Temporary verification only; all updated weights discarded; not performance results'}
        files=list((REPO/'variants/bcsi_adapt').glob('*.py'))
        files += [REPO/name for name in ['variants/tsbci/train.py','configs/bcsi_adapt.json','scripts/bcsi_adapt_preflight.py']]
        result['source_sha256']={str(p.relative_to(REPO)):sha(p) for p in files}
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',default=str(REPO/'paths.local.json'))
    p.add_argument('--gpu',default='0');p.add_argument('--output',default=str(REPO/'docs/BCSI_ADAPT_GPU_PREFLIGHT.json'))
    main(p.parse_args())
