"""Two explicit stages; best-only checkpoints, immutable reference, original validation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from runtime import parser,prepare_run,gpu_lock_path,snapshot_sources,sha
from datasets.staged import loaders
from utils.run_common import seed_everything,module_grad_norm
from utils.amp import make_grad_scaler,autocast_context
from utils.optimization import get_cosine_scheduler
from utils.ema import ModelEMA
from utils.metrics_seg import SegmentationMetricsCIPA
from .model import JointModel,CorrectedModel
from .inference import load_joint
from .losses import objective,collate
from .metrics import RoleMetrics


def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def log(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')


def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def stopped(out):
    if (out/'STOP').exists() or (out.parent/'STOP').exists():
        dump(out/'status.json',{'state':'stopped_by_request'});raise SystemExit(0)


def move(batch,device):
    if not batch['is_full'].bool().all():raise ValueError('Full PET/CT only')
    return {k:v.to(device,non_blocking=True) for k,v in batch.items() if torch.is_tensor(v)}


def stats(x):
    x=x.detach().float()
    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite diagnostic')
    return {'shape':list(x.shape),'mean':float(x.mean()),'rms':float(x.square().mean().sqrt()),
            'min':float(x.min()),'max':float(x.max())}


@torch.no_grad()
def evaluate(model,loader,cfg,phase,out,selected_corrector=None):
    model.eval();metric=SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode)
    before=SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode) if phase=='correction' else None
    roles=RoleMetrics();totals={};n=0;forward_ms=0.;timed_images=0
    for step,b in enumerate(loader,1):
        stopped(out);b=move(b,'cuda')
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        o=model(b['ct'],b['pet'],corrector=selected_corrector) if before else model(b['ct'],b['pet'])
        end.record();end.synchronize()
        if step>1:forward_ms+=begin.elapsed_time(end);timed_images+=len(b['mask'])
        if not torch.isfinite(o['logits']).all():raise RuntimeError('Nonfinite validation output')
        _,parts=objective(o,b,phase);metric.update(o['logits'],b['mask']);roles.update(o,b)
        if before:before.update(o['base_logits'],b['mask'])
        count=len(b['mask']);n+=count
        for k,v in parts.items():totals[k]=totals.get(k,0)+v*count
        if step==1 or step%50==0 or step==len(loader):
            dump(out/'progress.json',{'activity':'validation','step':step,'steps':len(loader)})
            print('EVAL',phase,step,'/',len(loader),flush=True)
    r={'full':metric.compute(),'losses':{k:v/n for k,v in totals.items()},'roles':roles.compute(),
       'batched_forward_ms_per_image':forward_ms/max(timed_images,1),
       'timing_scope':'CUDA forward only, original batch size, excluding first batch; not batch1 latency'}
    if before:
        r['uncorrected']=before.compute();r['dice_gain_pp']=100*(r['full']['dice']-r['uncorrected']['dice'])
    return r


def main(args):
    out,cfg,manifest=prepare_run(args,'v31_'+args.phase)
    args.init='generic' if args.init=='generic' else str(Path(args.init).resolve());base_sha='generic_encoders_only' if args.init=='generic' else sha(args.init)
    dump(out/'status.json',{'state':'waiting_for_gpu','phase':args.phase})
    with open(gpu_lock_path(),'a') as lock:
        while True:
            stopped(out)
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(5)
        seed_everything(cfg.random_state);torch.set_num_threads(4)
        if args.phase=='joint':
            model=JointModel(vars(cfg))
            if args.init=='generic':
                from models.components.backbones import load_local_weights_safe
                load_local_weights_safe(model.base.enc_ct,cfg.ct_pretrained_path,name='CT_Encoder')
                load_local_weights_safe(model.base.enc_pet,cfg.pet_pretrained_path,name='PET_Encoder')
            else:
                bundle=torch.load(args.init,map_location='cpu',weights_only=False,mmap=True)
                model.import_m1(bundle);del bundle
            trainable=model
        else:
            base,_=load_joint(args.init);model=CorrectedModel(base);trainable=model.corrector
        model.cuda();frozen=fingerprint(model.base) if args.phase=='correction' else None
        train,val,_=loaders(cfg,out)
        train.collate_fn=collate;val.collate_fn=collate
        dump(out/'protocol.json',{'phase':args.phase,'config':vars(cfg),'input_checkpoint':args.init,
             'input_sha256':base_sha,'selection':'Fusion/final validation Dice; same-epoch complete selected state',
             'train_batches':len(train),'val_batches':len(val),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'total_parameters':sum(p.numel() for p in model.parameters()),'frozen_fingerprint':frozen,
             'target_rules':'PET disk dilation3; SDF clip16; CT band3 weight5; protect foreground and outer GT band3',
             'scope':'Full candidate only, reused development set, no isolated component efficacy claim'})
        snapshot_sources(out)
        opt=torch.optim.AdamW(trainable.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        scheduler=get_cosine_scheduler(opt,cfg.epochs,warmup_steps=cfg.cosine_warmup*len(train),
             min_lr=cfg.cosine_min_lr,steps_per_epoch=len(train),flat_ratio=cfg.lr_flat_ratio)
        scaler=make_grad_scaler(torch.device('cuda'),cfg.mixed_precision)
        ema=ModelEMA(trainable,cfg.ema_decay,warmup=True,device='cuda');active=False;best=-1.;best_epoch=0;rows=[];overflow_streak=0
        for epoch in range(1,cfg.epochs+1):
            stopped(out)
            if epoch>cfg.ema_start_epoch and not active:ema.reset(trainable);active=True
            model.train();totals={};n=0;skips=0
            if frozen and (any(v.training for v in model.base.modules()) or any(p.requires_grad for p in model.base.parameters())):
                raise RuntimeError('Frozen base changed mode or trainability')
            dump(out/'status.json',{'state':'training','phase':args.phase,'epoch':epoch,'best':best,'best_epoch':best_epoch})
            for step,b in enumerate(train,1):
                stopped(out);b=move(b,'cuda');opt.zero_grad(set_to_none=True)
                capture=step==1 or step%100==0 or step==len(train)
                with autocast_context(torch.device('cuda'),cfg.mixed_precision):o=model(b['ct'],b['pet'],capture=capture)
                loss,parts=objective(o,b,args.phase)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(trainable.parameters(),cfg.grad_clip)
                finite=bool(torch.isfinite(norm))
                if not finite:
                    bad=[k for k,p in trainable.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                    if not scaler.is_enabled() or not bad:raise RuntimeError('Nonrecoverable nonfinite gradient norm')
                    overflow_streak+=1;skips+=1
                    log(out/'amp_events.jsonl',{'epoch':epoch,'step':step,'scale':old_scale,'bad':bad})
                    if overflow_streak>=8:raise RuntimeError('Eight consecutive AMP overflows')
                else:overflow_streak=0
                groups={k:module_grad_norm(v) for k,v in trainable.named_children()} if capture and finite else {}
                if capture and finite and args.phase=='joint':
                    groups.update({'base.'+k:module_grad_norm(v) for k,v in model.base.named_children()})
                scaler.step(opt);scaler.update()
                if finite:
                    scheduler.step()
                    if active:ema.update(trainable)
                if capture:
                    scalestats={k:[stats(x) for x in v] for k,v in o.get('scales',{}).items()}
                    detail={'epoch':epoch,'step':step,'losses':parts,'gradient_norm':float(norm) if finite else None,
                            'gradient_groups':groups,'AMP_scale':scaler.get_scale(),'optimizer_step':finite,'scales':scalestats,
                            'outputs':{k:stats(o[k]) for k in ['logits','pet_logits','ct_distance','gate','delta','change','consensus'] if k in o}}
                    detail['effective_pixels']={'GT_foreground':int((b['mask']>.5).sum()),
                        'PET_target_foreground':int((b['pet_target']>.5).sum()),'CT_boundary_band':int((b['ct_weight']>1).sum())}
                    if args.phase=='correction':
                        correct=(o['base_logits'].sigmoid()>.5)==(b['mask']>.5)
                        detail['effective_pixels'].update(errors=int((~correct).sum()),
                            protected_fg=int((correct&(b['mask']>.5)).sum()),
                            protected_near_bg=int((correct&(b['mask']<=.5)&(b['near_bg']>.5)).sum()))
                    log(out/'diagnostics.jsonl',detail);dump(out/'progress.json',{'activity':'training','epoch':epoch,'step':step,'steps':len(train),'losses':parts})
                    print('TRAIN',args.phase,epoch,step,'/',len(train),parts,'grad',float(norm) if finite else None,flush=True)
                count=len(b['mask']);n+=count
                for k,v in parts.items():totals[k]=totals.get(k,0)+v*count
            dump(out/'status.json',{'state':'validation','phase':args.phase,'epoch':epoch})
            selected=ema.model if active else trainable
            validation=evaluate(selected if args.phase=='joint' else model,val,cfg,args.phase,out,
                                selected_corrector=selected if args.phase=='correction' else None)
            if frozen and fingerprint(model.base)!=frozen:raise RuntimeError('Frozen parameters/BN buffers changed')
            score=validation['full']['dice'];improved=score>best
            if improved:
                best=score;best_epoch=epoch
                bundle={'format':'v31_joint_v1' if args.phase=='joint' else 'v31_correction_v1',
                        'weights':selected.state_dict(),'config':vars(cfg),'epoch':epoch,'eval_weights':'ema' if active else 'raw',
                        'validation':validation,'base_checkpoint':args.init,'base_sha256':base_sha}
                if __import__('shutil').disk_usage(out).free<800*1024**2:raise RuntimeError('Insufficient checkpoint space')
                tmp=out/'best.tmp';torch.save(bundle,str(tmp));tmp.replace(out/'best.pt')
            row={'epoch':epoch,'weights':'ema' if active else 'raw','train':{k:v/n for k,v in totals.items()},
                 'validation':validation,'best_epoch':best_epoch,'AMP_skips':skips,'frozen_unchanged':True if frozen else None}
            rows.append(row);log(out/'train.jsonl',row)
            bestrow=max(rows,key=lambda r:r['validation']['full']['dice'])
            summary={'complete':False,'phase':args.phase,'best_epoch':bestrow['epoch'],'metrics':bestrow['validation']['full'],
                     'validation':bestrow['validation'],'scope':'Single seed, reused development set, full candidate'}
            dump(out/'RESULTS.json',summary)
            lines=['|Epoch|Weights|Dice %|Same-epoch HD95|','|---|---|---:|---:|']
            for r in rows:
                m=r['validation']['full'];lines.append(f"|{r['epoch']}|{r['weights']}|{100*m['dice']:.4f}|{m['hd95']:.4f}|")
            (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
            print('EPOCH_RESULT',json.dumps(row),flush=True)
            if epoch-best_epoch>=cfg.early_stopping_patience:break
        summary['complete']=True;dump(out/'RESULTS.json',summary)
        dump(out/'status.json',{'state':'complete','phase':args.phase,'best_epoch':best_epoch,'best':best})


if __name__=='__main__':
    p=parser('v31');p.add_argument('--phase',choices=['joint','correction'],required=True);p.add_argument('--init',required=True)
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        traceback.print_exc()
        # Failures before prepare_run are retained in the pipeline's per-stage log.
        from runtime import resolve_config
        _,out,_=resolve_config(args,'v31')
        if out.exists():dump(out/'status.json',{'state':'failed','error':repr(error)})
        raise SystemExit(1)
