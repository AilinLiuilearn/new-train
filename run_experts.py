"""2D CT/PET expert pretraining, paired cooperation, resume and evaluation.

Uses official PlainConvUNet; custom PNG pipeline and trainer, not full nnU-Net.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from models.simmlm2d import Expert2D, DualExpert2D
from datasets.pclt_experts import loaders, split_signature
from utils.ema import ModelEMA
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.seg_losses import BCEDiceLoss
from utils.optimization import get_cosine_scheduler
from utils.run_common import seed_everything


def load(path):
    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')

def atomic_save(value,path):
    tmp=str(path)+'.tmp'; torch.save(value,tmp); os.replace(tmp,path)

def build(cfg):
    if cfg.mode=='joint': return DualExpert2D(tuple(cfg.features),cfg.gate_hidden)
    return Expert2D(tuple(cfg.features))

def selected(ckpt):
    tag=ckpt['eval_weights']
    if tag not in ('raw','ema'): raise ValueError('invalid eval_weights')
    value=ckpt['model_ema'] if tag=='ema' else ckpt['model']
    if value is None: raise ValueError('selected checkpoint weights absent')
    return value

def import_expert(model,path,modality,cfg,signature):
    ckpt=load(path); old=ckpt['config']
    if old['mode']!=modality: raise ValueError(f'{path} is not a {modality} expert')
    for key in ['features','image_size','norm']:
        if old[key]!=getattr(cfg,key): raise ValueError(f'expert {key} mismatch')
    if ckpt['splits']['train']!=signature['train'] or ckpt['splits']['val']!=signature['val']:
        raise ValueError('expert training/validation split mismatch')
    model.load_state_dict(selected(ckpt),strict=True)
    return {'path':str(Path(path).resolve()),'epoch':ckpt['epoch'],
            'weights':ckpt['eval_weights'],'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def forward_loss(model,batch,cfg,device,criterion):
    y=batch['mask'].to(device,non_blocking=True)
    if cfg.mode=='joint':
        out=model(batch['ct'].to(device,non_blocking=True),batch['pet'].to(device,non_blocking=True))
        lf=criterion(out['logits'],y)[0]; lm=criterion(out['ct_logits'],y)[0]
        return .5*(lf+lm), {'full_loss':float(lf.detach()),'missing_loss':float(lm.detach()),
                'w_ct':float(out['weights'][:,0].detach().mean())}
    logits=model(batch[cfg.mode].to(device,non_blocking=True))
    loss=criterion(logits,y)[0]
    return loss, {'expert_loss':float(loss.detach())}

@torch.inference_mode()
def evaluate(model,loader,cfg,device,criterion,tag):
    model.eval(); names=['full','missing'] if cfg.mode=='joint' else [cfg.mode]
    metrics={k:SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode) for k in names}
    sums={k:0. for k in names}; count=0; wsum=0.
    print(f'[{tag}] start batches={len(loader)} modes={names}',flush=True)
    for step,batch in enumerate(loader,1):
        y=batch['mask'].to(device); n=len(y)
        if cfg.mode=='joint':
            out=model(batch['ct'].to(device),batch['pet'].to(device))
            logits={'full':out['logits'],'missing':out['ct_logits']}
            wsum+=float(out['weights'][:,0].sum())
        else: logits={cfg.mode:model(batch[cfg.mode].to(device))}
        for k,z in logits.items():
            if not torch.isfinite(z).all(): raise RuntimeError(f'nonfinite {tag} {k} logits')
            metrics[k].update(z,y); sums[k]+=float(criterion(z,y)[0])*n
        count+=n
        if step==1 or step%cfg.log_interval==0 or step==len(loader):
            print(f'[{tag}] batch {step}/{len(loader)}',flush=True)
    result={k:{**metrics[k].compute(),'loss':sums[k]/count} for k in names}
    if cfg.mode=='joint':
        result['joint_dice']=.5*(result['full']['dice']+result['missing']['dice'])
        result['w_ct']=wsum/count
    return result

def rng_state(train):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader':train.generator.get_state()}

def restore_rng(state,train):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])
    train.generator.set_state(state['loader'])

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ct','pet','joint'])
    p.add_argument('--root',default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--train_split',default='train_original.txt'); p.add_argument('--val_split',default='val.txt')
    p.add_argument('--test_split',default='test.txt'); p.add_argument('--allow_val_equals_test',action='store_true')
    p.add_argument('--output',default=None); p.add_argument('--ct_checkpoint'); p.add_argument('--pet_checkpoint')
    p.add_argument('--resume',help='Epoch-boundary resume; saved configuration is restored')
    p.add_argument('--evaluate',help='Evaluate selected raw/EMA weights; no training')
    p.add_argument('--image_size',type=int,default=512); p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--num_workers',type=int,default=4); p.add_argument('--features',type=int,nargs='+',default=[32,64,128,256,512,512])
    p.add_argument('--gate_hidden',type=int,default=64); p.add_argument('--norm',choices=['zscore','unit'],default='zscore')
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--learning_rate',type=float,default=8e-5)
    p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--warmup_epochs',type=int,default=3)
    p.add_argument('--min_lr',type=float,default=1e-6); p.add_argument('--flat_ratio',type=float,default=.3)
    p.add_argument('--seed',type=int,default=2023); p.add_argument('--log_interval',type=int,default=50)
    p.add_argument('--grad_clip',type=float,default=5.); p.add_argument('--no_amp',action='store_true')
    p.add_argument('--no_ema',action='store_true'); p.add_argument('--ema_decay',type=float,default=.999)
    p.add_argument('--ema_start_epoch',type=int,default=3); p.add_argument('--no_augment',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--hd95_mode',choices=['surface','cipa_official'],default='cipa_official')
    return p

def main():
    args=parser().parse_args()
    if args.resume and args.evaluate: raise ValueError('choose resume or evaluate')
    checkpoint=load(args.resume or args.evaluate) if args.resume or args.evaluate else None
    cfg=argparse.Namespace(**checkpoint['config']) if checkpoint else args
    if args.evaluate:
        # Evaluation uses saved model/data protocol; only root and device may be overridden.
        import sys
        if '--root' in sys.argv: cfg.root=args.root
        cfg.device=args.device
    if not cfg.mode or not cfg.output: raise ValueError('--mode and --output are required for new runs')
    if cfg.epochs<1 or cfg.batch_size<1 or cfg.log_interval<1 or len(cfg.features)<2 or any(x<1 for x in cfg.features):
        raise ValueError('invalid epochs/batch/log interval/features')
    div=2**(len(cfg.features)-1)
    if cfg.image_size%div or cfg.image_size//div<2: raise ValueError('image size incompatible with encoder depth')
    if not 0<cfg.learning_rate or not 0<=cfg.warmup_epochs<cfg.epochs or not 0<=cfg.flat_ratio<1:
        raise ValueError('invalid learning rate or schedule')
    output=Path(cfg.output); output.mkdir(parents=True,exist_ok=True)
    if not checkpoint and (output/'config.json').exists(): raise FileExistsError('existing run; use --resume or a new --output')
    if args.resume and not (output/'best.pt').exists(): raise FileNotFoundError('resume requires original run directory including best.pt')
    seed_everything(cfg.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else cfg.device)
    signature=split_signature(cfg)
    if checkpoint and signature!=checkpoint['splits']: raise ValueError('dataset split contents changed')
    print(f'[START] mode={cfg.mode} device={device} size={cfg.image_size} batch={cfg.batch_size} lr={cfg.learning_rate}',flush=True)
    print('[DATA] scanning split files and image paths',flush=True)
    train,val,test=loaders(cfg,output)
    print(f'[DATA] train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)}',flush=True)
    model=build(cfg)
    sources={}
    if checkpoint: model.load_state_dict(selected(checkpoint) if args.evaluate else checkpoint['model'],strict=True)
    elif cfg.mode=='joint':
        if not cfg.ct_checkpoint or not cfg.pet_checkpoint: raise ValueError('joint requires both expert checkpoints')
        sources={m:import_expert(getattr(model,m),getattr(cfg,m+'_checkpoint'),m,cfg,signature) for m in ['ct','pet']}
    model.to(device)
    print(f'[MODEL] official PlainConvUNet features={cfg.features} params={sum(p.numel() for p in model.parameters()):,}',flush=True)
    criterion=BCEDiceLoss()
    if args.evaluate:
        result={'epoch':checkpoint['epoch'],'weights':checkpoint['eval_weights'],
                'metrics':evaluate(model,test,cfg,device,criterion,'test')}
        path=output/f'reeval_{time.time_ns()}.json'; path.write_text(json.dumps(result,indent=2)); print(result); return
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=get_cosine_scheduler(optimizer,cfg.epochs,cfg.warmup_epochs*len(train),
        min_lr=cfg.min_lr,steps_per_epoch=len(train),flat_ratio=cfg.flat_ratio)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and not cfg.no_amp)
    ema=None if cfg.no_ema else ModelEMA(model,decay=cfg.ema_decay,warmup=True,device=device)
    active=False; best=-1.; best_epoch=0; start_epoch=1
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler']); best=checkpoint['best']; best_epoch=checkpoint['best_epoch']
        active=checkpoint['ema_active']; start_epoch=checkpoint['epoch']+1; sources=checkpoint['sources']
        if ema:
            ema.model.load_state_dict(checkpoint['model_ema']); ema.updates=checkpoint['ema_updates']
        restore_rng(checkpoint['rng'],train)
    else:
        (output/'config.json').write_text(json.dumps(vars(cfg),indent=2))
        import importlib.metadata as md
        versions={k:md.version(k) for k in ['torch','dynamic-network-architectures','numpy','opencv-python-headless']}
        (output/'environment.json').write_text(json.dumps(versions,indent=2))
    for epoch in range(start_epoch,cfg.epochs+1):
        if ema and not active and epoch>cfg.ema_start_epoch: ema.reset(model); active=True
        model.train(); totals={}; samples=skipped=0; begin=time.time()
        print(f'[TRAIN {epoch}/{cfg.epochs}] start batches={len(train)}',flush=True)
        for step,batch in enumerate(train,1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                loss,stats=forward_loss(model,batch,cfg,device,criterion)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip if cfg.grad_clip>0 else float('inf'))
            if not scaler.is_enabled() and not torch.isfinite(grad): raise RuntimeError('nonfinite gradients')
            before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            success=scaler.get_scale()>=before
            if success:
                scheduler.step()
                if active: ema.update(model)
            else: skipped+=1
            n=len(batch['mask']); samples+=n
            for k,v in {'loss':float(loss.detach()),**stats}.items(): totals[k]=totals.get(k,0)+v*n
            if step==1 or step%cfg.log_interval==0 or step==len(train):
                elapsed=time.time()-begin; eta=elapsed/step*(len(train)-step)
                print(f'[TRAIN {epoch}] {step}/{len(train)} loss={float(loss.detach()):.4f} lr={optimizer.param_groups[0]["lr"]:.3g} elapsed={elapsed:.0f}s eta={eta:.0f}s',flush=True)
        evaluation=ema.model if active else model
        result=evaluate(evaluation,val,cfg,device,criterion,'val')
        score=result['joint_dice'] if cfg.mode=='joint' else result[cfg.mode]['dice']
        if not np.isfinite(score): raise RuntimeError('nonfinite selection metric')
        improved=score>best
        if improved: best,best_epoch=score,epoch
        payload={'config':vars(cfg),'splits':signature,'sources':sources,'epoch':epoch,
            'model':model.state_dict(),'model_ema':ema.state_dict() if ema else None,
            'ema_updates':ema.updates if ema else 0,'ema_active':active,'eval_weights':'ema' if active else 'raw',
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'best':best,'best_epoch':best_epoch,'val':result,'rng':rng_state(train)}
        if improved: atomic_save(payload,output/'best.pt')
        atomic_save(payload,output/'last.pt')
        row={'epoch':epoch,**{'train_'+k:v/samples for k,v in totals.items()},'selection_dice':score,
             'best':best,'best_epoch':best_epoch,'skipped':skipped,'epoch_seconds':time.time()-begin,
             'lr':optimizer.param_groups[0]['lr'],'ema_updates':ema.updates if ema else 0}
        for k,v in result.items():
            if isinstance(v,dict): row.update({k+'_'+a:b for a,b in v.items()})
            else: row[k]=v
        with (output/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        csvpath=output/'train.csv'; exists=csvpath.exists()
        with csvpath.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)
        print(f'[EPOCH {epoch}] score={score:.4f} best={best:.4f}@{best_epoch} result={result}',flush=True)
    best_ckpt=load(output/'best.pt'); model.load_state_dict(selected(best_ckpt))
    final={'epoch':best_ckpt['epoch'],'weights':best_ckpt['eval_weights'],
           'metrics':evaluate(model,test,cfg,device,criterion,'final_test')}
    (output/'final_test.json').write_text(json.dumps(final,indent=2)); print(final,flush=True)

if __name__=='__main__': main()
