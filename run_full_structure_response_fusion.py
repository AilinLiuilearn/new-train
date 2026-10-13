"""Independent Full-only fusion experiment. Original entry points are untouched."""
import copy
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch

from configs.full_structure_response import FullStructureConfig
from models.full_structure_response import build_structure_model
from models.structure_response_fusion import StructureResponseFusion
from tasks.mdt_seg import MDTSegTeacher
from tasks.full_structure_response import FullStructureTask
from run_full_add_baseline import _loaders
from utils.optimization import get_cosine_scheduler
from utils.run_common import count_parameters,module_grad_norm,optimizer_step_succeeded,seed_everything
from utils.train_logger import append_epoch_log,init_train_log
from utils.structure_response_visualization import log_fusion,visualize_fusion


def core_gradient_check(model):
    names=['enc_ct','enc_pet','ct_align','pet_align','decoder']
    for name in names:
        p=next(p for p in getattr(model,name).parameters() if p.requires_grad and p.ndim>=2)
        if p.grad is None:raise RuntimeError(f'Disconnected core gradient: {name}')
    if isinstance(model.fusion,StructureResponseFusion):
        for s,stage in enumerate(model.fusion.stages):
            for name in ('ct_output','pet_output'):
                if getattr(stage,name).weight.grad is None:raise RuntimeError(f'Disconnected S{s+1}.{name}')
    print('[GRAD CHECK] encoders/alignments/decoder and enabled residual outputs connected',flush=True)


def train_step(task,batch,diagnose=False):
    ct=batch['ct'].to(task.device);pet=batch['pet'].to(task.device);gt=batch['mask'].to(task.device).float()
    out=task.model(ct,pet,forward_mode='full',return_diagnostics=diagnose)
    loss,parts=task._finish_step(out['logits'].float(),gt,'full_structure_response')
    return loss,out,parts


def restore(task,path):
    ckpt=MDTSegTeacher.load_state_dicts(path)
    fields=('fusion_enabled','fusion_heads','sae_expansion','decoder_channels','norm_mode','aug_mode',
            'epochs','batch_size','random_state','learning_rate','weight_decay','ema_enabled','ema_decay',
            'ema_start_epoch','ema_decay_warmup','train_split_file','val_split_file','test_split_file','hash',
            'image_size_2d','hd95_mode','ct_backbone','pet_backbone','cosine_warmup','cosine_min_lr','lr_flat_ratio','grad_clip')
    for key in fields:
        if ckpt['config'].get(key)!=getattr(task.config,key):raise ValueError(f'Resume config mismatch: {key}')
    task.model.load_state_dict(ckpt['model'],strict=True)
    if task.ema is not None:
        task.ema.model.load_state_dict(ckpt['model_ema'],strict=True)
        task.ema.updates=int(ckpt['ema_updates']);task.ema_active=ckpt['eval_weights']=='ema'
    task.optimizer.load_state_dict(ckpt['optimizer']);task.scheduler.load_state_dict(ckpt['scheduler'])
    task.scaler.load_state_dict(ckpt['scaler']);task.global_batch_step=int(ckpt['global_batch_step'])
    random.setstate(ckpt['random_state_python']);np.random.set_state(ckpt['random_state_numpy'])
    torch.set_rng_state(ckpt['random_state_torch'])
    if torch.cuda.is_available() and 'random_state_cuda' in ckpt:torch.cuda.set_rng_state_all(ckpt['random_state_cuda'])
    return ckpt


def final_eval(task,loader,path):
    ckpt=MDTSegTeacher.load_state_dicts(path);tag=ckpt['eval_weights']
    # Reuse the allocated model, strict-load selected weights; no pretrained
    # reload or third full-model replica at final test.
    task.model.load_state_dict(ckpt['model_ema'] if tag=='ema' else ckpt['model'],strict=True)
    task.model.eval()
    if task.ema is not None:
        task.ema_active=tag=='ema'
        if tag=='ema':task.ema.model.load_state_dict(ckpt['model_ema'],strict=True)
    score=task.evaluate(loader,eval_mode='full',model=task.model)
    result={'experiment_type':'full_structure_response','weights':tag,'epoch':ckpt['epoch'],
            'fusion_enabled':task.config.fusion_enabled,'hd95_mode':task.metrics.hd95_mode,
            'checkpoint':os.path.abspath(path),**{k:score[k] for k in ('dice','iou','hd95','total_loss')}}
    with open(os.path.join(task.config.checkpoint_dir,'final_test_full.json'),'w') as f:json.dump(result,f,indent=2)
    print('[FINAL FULL]',json.dumps(result),flush=True)
    return int(ckpt['epoch']),tag


def smoke(cfg):
    cfg=copy.copy(cfg);cfg.pretrained=False;cfg.mixed_precision=False;cfg.ema_enabled=False
    task=FullStructureTask(build_structure_model(cfg),cfg);task.model.train()
    batch={'ct':torch.randn(2,3,64,64),'pet':torch.randn(2,3,64,64),'mask':torch.ones(2,1,64,64)}
    loss,_,_=train_step(task,batch,True);loss.backward();core_gradient_check(task.model)
    task.optimizer.step();print(f'[SMOKE PASS] loss={float(loss.detach()):.4f} device={task.device} '
                               f'params={count_parameters(task.model)}',flush=True)


def main():
    cfg=FullStructureConfig.parse_arguments()
    if cfg.fusion_visualize and importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Visualization requires matplotlib; install it or --fusion_visualize False')
    if cfg.smoke:seed_everything(cfg.random_state);smoke(cfg);return
    if not torch.cuda.is_available() and not cfg.allow_cpu:
        raise RuntimeError('CUDA is unavailable: check nvidia-smi and torch.cuda.is_available before training. Explicit CPU tests use --allow_cpu True.')
    if cfg.eval_checkpoint:
        eval_path=os.path.abspath(cfg.eval_checkpoint)
        ckpt=MDTSegTeacher.load_state_dicts(eval_path)
        saved=dict(ckpt['config']);saved.pop('checkpoint_dir',None)
        if saved.get('model_arch')!='full_structure_response':raise ValueError('Expected this experiment checkpoint')
        saved.update(root=cfg.root,pretrained=False,ct_pretrained_path=None,pet_pretrained_path=None,
                     hash=cfg.hash,checkpoint_root=cfg.checkpoint_root,
                     fusion_visualize=cfg.fusion_visualize,fusion_vis_samples=cfg.fusion_vis_samples)
        cfg=FullStructureConfig(args=saved);task=FullStructureTask(build_structure_model(cfg),cfg)
        _,_,test=_loaders(cfg);epoch,tag=final_eval(task,test,eval_path)
        if cfg.fusion_visualize:
            files=visualize_fusion(task.model,test,cfg,task.device,epoch,tag);print(f'[VIS] saved {len(files)} files',flush=True)
        return
    cfg.experiment_type='full_structure_response';cfg.pet_missing_rate=0.;seed_everything(cfg.random_state)
    train,val,test=_loaders(cfg);task=FullStructureTask(build_structure_model(cfg),cfg)
    task.scheduler=get_cosine_scheduler(task.optimizer,epochs=cfg.epochs,warmup_steps=cfg.cosine_warmup*len(train),
        min_lr=cfg.cosine_min_lr,steps_per_epoch=len(train),flat_ratio=cfg.lr_flat_ratio)
    best,best_epoch,start=-1.,0,1
    if cfg.resume:
        ckpt=restore(task,cfg.resume);start=int(ckpt['epoch'])+1;best=float(ckpt['best']);best_epoch=int(ckpt['best_epoch'])
    with open(os.path.join(cfg.checkpoint_dir,'config_args.json'),'w') as f:json.dump(vars(cfg),f,indent=2)
    total,trainable=count_parameters(task.model);fusion_params=sum(p.numel() for p in task.model.fusion.parameters())
    print(f'[MODEL] Full-only; params={total}/{trainable} fusion_params={fusion_params} fusion_enabled={cfg.fusion_enabled}',flush=True)
    print(f'[RUNTIME] torch={torch.__version__} device={task.device} AMP={bool(task.scaler.is_enabled())} workers={cfg.num_workers}',flush=True)
    print(f'[DESIGN] channels=64/128/320/512 internal=32/64/160/256 heads={cfg.fusion_heads} '
          f'SAE={cfg.sae_expansion} fixed_Scharr=True PETcontrast=3/5/7 residual_outputs=zero_init; no memory/missing/extra loss',flush=True)
    print(f'[EMA] enabled={cfg.ema_enabled} start_epoch={cfg.ema_start_epoch} decay={cfg.ema_decay}; '
          f'[DIAG] every={cfg.fusion_log_every} visualize={cfg.fusion_visualize}',flush=True)
    headers=['train_samples','train_batches','best_full','best_full_epoch','ema_updates','skipped_updates',
             'epoch_time','grad_sampled_batches','fusion_sampled_batches']
    gradient_modules=('enc_ct','enc_pet','ct_align','pet_align','fusion','decoder')
    headers += [f'grad_{name}' for name in gradient_modules]
    diagnostic_keys=('ct_residual_ratio','pet_residual_ratio','ct_residual_rms','pet_residual_rms',
                     'edge_rms','pet_contrast3_rms','pet_contrast5_rms','pet_contrast7_rms',
                     'ct_ca_entropy','pet_ca_entropy','ct_temperature','pet_temperature')
    headers += [f's{s+1}_{key}' for s in range(4) for key in diagnostic_keys]
    log=os.path.join(cfg.checkpoint_dir,'train_log.csv');diag_log=os.path.join(cfg.checkpoint_dir,'fusion_diagnostics.jsonl')
    if not cfg.resume:init_train_log(log,headers);open(diag_log,'w').close()
    last=os.path.join(cfg.checkpoint_dir,'ckpt.last.pth.tar');best_path=os.path.join(cfg.checkpoint_dir,'ckpt.best_full.pth.tar')
    stale=max(0,start-1-best_epoch)
    for epoch in range(start,cfg.epochs+1):
        seed_everything(cfg.random_state+epoch*100003)
        if getattr(train,'generator',None) is not None:train.generator.manual_seed(cfg.random_state+epoch*100003)
        task.model.train();task.begin_epoch(epoch);begin=time.time()
        loss_sum=0.;samples=batches=skipped=norm_count=0;norm_sum=0.;grad_samples=0
        grad_sums={name:0. for name in gradient_modules};diag_sums={};diag_count=0
        for i,batch in enumerate(train):
            diagnose=i==0 or (i+1)%cfg.fusion_log_every==0
            task.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=task.device.type,enabled=bool(task.scaler.is_enabled())):
                loss,out,parts=train_step(task,batch,diagnose)
            if task.scaler.is_enabled():task.scaler.scale(loss).backward();task.scaler.unscale_(task.optimizer)
            else:loss.backward()
            if i==0:core_gradient_check(task.model)
            if diagnose:
                grad_samples+=1
                for name in gradient_modules:grad_sums[name]+=module_grad_norm(getattr(task.model,name))
            if cfg.grad_clip>0:gn=torch.nn.utils.clip_grad_norm_(task.trainable_parameters(),cfg.grad_clip)
            else:gn=torch.stack([p.grad.detach().float().norm() for p in task.trainable_parameters() if p.grad is not None]).norm()
            if not torch.isfinite(gn):
                if not task.scaler.is_enabled():raise RuntimeError('Non-finite gradients')
                task.scaler.update(new_scale=max(task.scaler.get_scale()/2,1.));skipped+=1
            else:
                norm_sum+=float(gn);norm_count+=1
                if optimizer_step_succeeded(task):task.scheduler.step();task.update_ema()
                else:skipped+=1
            if diagnose:
                print(f'[BATCH {i+1}] Full={batch["ct"].shape[0]} loss={float(loss.detach()):.4f} '
                      f'BCE={float(parts["loss_bce"]):.4f} DiceLoss={float(parts["loss_dice"]):.4f} '
                      f'grad_norm={float(gn):.4f} EMAupdates={task.ema.updates if task.ema else 0}',flush=True)
                values=log_fusion(out.get('fusion_diagnostics',{}),diag_log,epoch,i+1)
                if values:
                    diag_count+=1
                    for s,v in values.items():
                        for key in diagnostic_keys:diag_sums[f's{int(s)+1}_{key}']=diag_sums.get(f's{int(s)+1}_{key}',0.)+v[key]
            n=batch['ct'].shape[0];loss_sum+=float(loss.detach())*n;samples+=n;batches+=1;task.global_batch_step+=1
        score=task.evaluate(val,eval_mode='full',model=task.eval_model());improved=score['dice']>best
        if improved:best=float(score['dice']);best_epoch=epoch;stale=0
        else:stale+=1
        if improved:task.save_checkpoint(best_path,epoch,best,best_epoch,score)
        task.save_checkpoint(last,epoch,best,best_epoch,score)
        extra={'train_samples':samples,'train_batches':batches,'best_full':best,'best_full_epoch':best_epoch,
               'ema_updates':task.ema.updates if task.ema else 0,'skipped_updates':skipped,'epoch_time':time.time()-begin,
               'grad_sampled_batches':grad_samples,'fusion_sampled_batches':diag_count}
        extra.update({f'grad_{name}':grad_sums[name]/max(1,grad_samples) for name in gradient_modules})
        extra.update({f's{s+1}_{key}':diag_sums[f's{s+1}_{key}']/diag_count if diag_count else 'n/a'
                      for s in range(4) for key in diagnostic_keys})
        append_epoch_log(log,epoch,loss_sum/max(1,samples),score,task.optimizer.param_groups[0]['lr'],norm_sum/max(1,norm_count),extra)
        print(f'[EPOCH {epoch}] Full Dice={score["dice"]:.4f} HD95={score["hd95"]:.3f} '
              f'weights={task.eval_weights_tag()} best={best:.4f}@{best_epoch}',flush=True)
        if cfg.fusion_visualize and (epoch in (1,2) or epoch%cfg.fusion_vis_every_epochs==0):
            files=visualize_fusion(task.eval_model(),val,cfg,task.device,epoch,task.eval_weights_tag())
            print(f'[VIS] saved={len(files)} under {cfg.checkpoint_dir}/fusion_visualizations/epoch_{epoch:03d}',flush=True)
        if stale>=cfg.early_stop_patience:break
    if not os.path.isfile(best_path):raise RuntimeError('No best Full checkpoint')
    final_eval(task,test,best_path)


if __name__=='__main__':main()
