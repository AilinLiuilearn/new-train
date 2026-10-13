"""Independent entry: 70/30 slice-level Full/Missing, audited correction memory.

No Full-only first stage is required. Original baseline runners are untouched.
Checkpoints include all queues, separate bank/evaluation EMA, optimizer/scaler/RNG states.
  python run_joint_pet_correction.py --help
  python run_joint_pet_correction.py --eval_checkpoint <best_joint>
"""
import copy
import json
import os
import random
import time

import numpy as np
import torch

from configs.pet_correction import PETCorrectionConfig
from models.pet_correction_model import build_correction_model
from tasks.pet_correction import PETCorrectionTask, batch_ids, build_ratio_state, assert_core_gradients
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import count_parameters, optimizer_step_succeeded, seed_everything
from utils.train_logger import append_epoch_log, init_train_log
from run_full_missing_baseline import _loaders


def load_training_state(task,path):
    ckpt=MDTSegTeacher.load_state_dicts(path)
    saved=ckpt['config']
    # Prevent changing the meaning/shape of saved memory while resuming.
    fields=('full_ratio','full_loss_weight','missing_loss_weight','decoder_channels',
            'memory_capacity','memory_key_dim','memory_exclude_group','memory_max_age',
            'memory_warmup_epochs','retrieval_start_epoch','norm_mode','ema_enabled',
            'epochs','learning_rate','ct_aux_weight','scale_loss_weight','negative_region_weight',
            'memory_enabled','region_calibration','memory_min_entries','memory_gain_margin',
            'memory_min_support','memory_max_negative_fraction','memory_tokens_per_full',
            'memory_temperature','memory_topk','ema_decay','ema_start_epoch','ema_decay_warmup',
            'bank_ema_enabled','bank_ema_start_epoch','bank_ema_decay','bank_ema_decay_warmup',
            'random_state','batch_size','image_size_2d','aug_mode','train_split_file',
            'val_split_file','test_split_file','hash')
    for key in fields:
        if saved.get(key)!=getattr(task.config,key):
            raise ValueError(f'Resume config mismatch for {key}; use the original run settings')
    task.model.load_state_dict(ckpt['model'],strict=True)
    if task.ema is not None and ckpt.get('model_ema') is not None:
        task.ema.model.load_state_dict(ckpt['model_ema'],strict=True)
        task.ema.updates=int(ckpt.get('ema_updates',0))
        task.ema_active=ckpt.get('eval_weights')=='ema'
    task.restore_bank_ema(ckpt,required=True)
    task.optimizer.load_state_dict(ckpt['optimizer'])
    if ckpt.get('scheduler') is not None:
        task.scheduler.load_state_dict(ckpt['scheduler'])
    task.scaler.load_state_dict(ckpt.get('scaler',{}))
    task.global_batch_step=int(ckpt.get('global_batch_step',0))
    random.setstate(ckpt['random_state_python']); np.random.set_state(ckpt['random_state_numpy'])
    torch.set_rng_state(ckpt['random_state_torch'])
    if torch.cuda.is_available() and 'random_state_cuda' in ckpt:
        torch.cuda.set_rng_state_all(ckpt['random_state_cuda'])
    return ckpt


def initialize_from_add(task,path):
    ckpt=MDTSegTeacher.load_state_dicts(path)
    use_ema=ckpt.get('eval_weights')=='ema' and ckpt.get('model_ema') is not None
    state=ckpt['model_ema'] if use_ema else ckpt['model']
    expected={k for k in task.model.state_dict() if k.startswith(
              ('enc_ct.','enc_pet.','ct_align.','pet_align.','decoder.','fusion.'))}
    if set(state)!=expected:
        raise ValueError('init_checkpoint must be the aligned pure-add baseline, no silent partial load')
    task.model.load_state_dict(state,strict=False)
    task.model.ct_aux_decoder.load_state_dict(task.model.decoder.state_dict(),strict=True)
    if task.ema is not None:
        task.ema.reset(task.model)
    if task.bank_ema is not None:
        task.bank_ema.reset(task.model)


def final_eval(task,loader,checkpoint,path):
    ckpt=MDTSegTeacher.load_state_dicts(checkpoint)
    use_ema=ckpt.get('eval_weights')=='ema'
    state=ckpt.get('model_ema') if use_ema else ckpt['model']
    if state is None:
        raise ValueError('Selected EMA checkpoint has no EMA state')
    # No dependency on pretrained files after a checkpoint exists.
    task.model.load_state_dict(state,strict=True)
    task.restore_bank_ema(ckpt,required=True)
    task.model.eval()
    full=task.evaluate(loader,'full',model=task.model)
    missing=task.evaluate(loader,'missing',model=task.model)
    weights_tag='ema' if use_ema else 'raw'
    full['weights']=missing['weights']=weights_tag
    result={'checkpoint':os.path.abspath(checkpoint),'epoch':ckpt['epoch'],
            'weights':weights_tag,'full':full,'missing':missing,
            'joint_dice':.5*(full['dice']+missing['dice']),
            'hd95_mode':task.metrics.hd95_mode,'missing_unit':'slice',
            'memory_source':'training-only paired local tokens',
            'bank_ema_active':task.bank_ema_active,
            'bank_ema_updates':task.bank_ema.updates if task.bank_ema is not None else 0,
            'retrieval_active':bool(task.model.retrieval_active),
            'memory_entries':[int(r.memory.allowed(-1,-1,int(task.model.memory_step)).sum())
                              for r in task.model.retrievers]}
    with open(path,'w') as f:
        json.dump(result,f,indent=2)
    print(f'[TEST] full={full["dice"]:.4f} missing={missing["dice"]:.4f} '
          f'joint={result["joint_dice"]:.4f} weights={result["weights"]}',flush=True)
    if not result['retrieval_active'] or not any(result['memory_entries']):
        print('[NOTE] Selected checkpoint has no active/usable correction memory; '
              'these scores do not demonstrate retrieval benefit.',flush=True)


def smoke(cfg):
    """Synthetic real-backbone optimization and strict queue roundtrip, no data."""
    cfg=copy.copy(cfg); cfg.pretrained=False; cfg.ema_start_epoch=0
    cfg.ema_enabled=True; cfg.hd95_mode='legacy'
    task=PETCorrectionTask(build_correction_model(cfg),cfg)
    b=2; size=max(64,min(cfg.image_size_2d,128))
    batch={'ct':torch.randn(b,3,size,size),'pet':torch.randn(b,3,size,size),
           'mask':torch.zeros(b,1,size,size),'image_id':['smoke_a','smoke_b'],
           'case_id':['smoke_a','smoke_b']}
    batch['mask'][:,:,size//4:size//2,size//4:size//2]=1
    task.model.train(); task.begin_epoch(cfg.retrieval_start_epoch)
    loss,_,_,stats=task.train_step_mixed(batch,torch.tensor([1,0],device=task.device))
    loss.backward(); task.optimizer.step(); task.after_successful_step()
    for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder','ct_aux_decoder','stage_heads'):
        grads=[p.grad for p in getattr(task.model,name).parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f'Smoke gradient failure: {name}')
    task.model.eval()
    with torch.no_grad():
        result=task.model(batch['ct'].to(task.device),forward_mode='missing',
                          sample_ids=torch.tensor([10,11],device=task.device))['logits']
    if not torch.isfinite(result).all():
        raise RuntimeError('Smoke logits are not finite')
    print(f'[SMOKE PASS] real backbones device={task.device} loss={float(loss.detach()):.6f} '
          f'candidates={stats["candidate_tokens"]}',flush=True)


def main():
    cfg=PETCorrectionConfig.parse_arguments()
    if cfg.smoke:
        seed_everything(cfg.random_state); smoke(cfg); return
    if cfg.eval_checkpoint:
        eval_path=os.path.abspath(cfg.eval_checkpoint)
        ckpt=MDTSegTeacher.load_state_dicts(eval_path)
        saved=dict(ckpt['config']); saved.pop('checkpoint_dir',None)
        # Old single-source checkpoints remain reevaluable under their original
        # query behavior; do not silently invent an independent trained source.
        saved.setdefault('bank_ema_enabled',False)
        saved.setdefault('bank_ema_start_epoch',0)
        saved.setdefault('bank_ema_decay',.999)
        saved.setdefault('bank_ema_decay_warmup',True)
        # Eval takes shape/normalization/source settings from the checkpoint.
        saved.update(root=cfg.root,pretrained=False,ct_pretrained_path=None,pet_pretrained_path=None,
                     checkpoint_root=cfg.checkpoint_root,hash=cfg.hash)
        cfg=PETCorrectionConfig(args=saved)
        task=PETCorrectionTask(build_correction_model(cfg),cfg)
        _,_,test=_loaders(cfg)
        final_eval(task,test,eval_path,
                   os.path.join(cfg.checkpoint_dir,'reeval_test.json'))
        return
    seed_everything(cfg.random_state)
    if cfg.resume and cfg.init_checkpoint:
        raise ValueError('Choose resume OR init_checkpoint')
    os.makedirs(cfg.checkpoint_dir,exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir,'config_args.json'),'w') as f:
        json.dump(vars(cfg),f,indent=2,default=str)
    train,val,test=_loaders(cfg)
    if not train.drop_last:
        raise ValueError('Joint ratio training requires drop_last=True')
    task=PETCorrectionTask(build_correction_model(cfg),cfg)
    task.scheduler=get_cosine_scheduler(task.optimizer,epochs=cfg.epochs,
                    warmup_steps=cfg.cosine_warmup*len(train),min_lr=cfg.cosine_min_lr,
                    steps_per_epoch=len(train),flat_ratio=cfg.lr_flat_ratio)
    best,best_epoch,start=-1.,0,1
    if cfg.resume:
        ckpt=load_training_state(task,cfg.resume)
        start=int(ckpt['epoch'])+1; best=float(ckpt['best']); best_epoch=int(ckpt['best_epoch'])
    elif cfg.init_checkpoint:
        initialize_from_add(task,cfg.init_checkpoint)
    print(f'[MODEL] params={count_parameters(task.model)} Full=AddFusion '
          f'slice_ratio={cfg.full_ratio:.3f} main_weights={cfg.full_loss_weight}/{cfg.missing_loss_weight}',flush=True)
    print(f'[RUNTIME] torch={torch.__version__} device={task.device} '
          f'amp={bool(cfg.mixed_precision and task.device.type=="cuda")}',flush=True)
    print('[MEMORY] source=independent bank EMA when active; sample self-exclusion=True '
          f'group_exclusion={cfg.memory_exclude_group}; train-only; no distillation; '
          f'warmup_epochs={cfg.memory_warmup_epochs} retrieval_start_epoch={cfg.retrieval_start_epoch}',flush=True)
    print(f'[EMA] bank_enabled={cfg.bank_ema_enabled} bank_start_epoch={cfg.bank_ema_start_epoch} '
          f'bank_decay={cfg.bank_ema_decay}; eval_enabled={cfg.ema_enabled} '
          f'eval_start_epoch={cfg.ema_start_epoch} eval_decay={cfg.ema_decay}',flush=True)
    headers=['train_full_loss','train_missing_loss','train_ct_aux_loss','train_scale_loss',
             'train_full_samples','train_missing_samples','positive_fraction','negative_fraction',
             'candidate_tokens','val_full_dice','val_missing_dice','val_full_hd95','val_missing_hd95',
             'best_joint','best_joint_epoch','ema_updates','bank_ema_updates','bank_ema_active',
             'skipped_updates','epoch_time',
             'hd95_mode','retrieval_active']+[f'memory_s{i+1}_entries' for i in range(4)]
    log=os.path.join(cfg.checkpoint_dir,'train_log.csv')
    if not cfg.resume or not os.path.isfile(log):
        init_train_log(log,headers)
    last=os.path.join(cfg.checkpoint_dir,'ckpt.last.pth.tar')
    best_path=os.path.join(cfg.checkpoint_dir,'ckpt.best_joint.pth.tar')
    no_improve=max(0,start-1-best_epoch)
    for epoch in range(start,cfg.epochs+1):
        # Reproducible per-epoch loader/augmentation stream, including resume.
        if getattr(train,'generator',None) is not None:
            train.generator.manual_seed(cfg.random_state+epoch*100003)
        seed_everything(cfg.random_state+epoch*100003)
        task.model.train(); task.begin_epoch(epoch); t=time.time()
        sums={k:0. for k in ('loss_full','loss_missing','loss_ct_aux','loss_scale',
                             'positive_fraction','negative_fraction','candidate_tokens')}
        total=norm=0.; nf=nm=skipped=steps=0
        for batch in train:
            state=build_ratio_state(batch['ct'].shape[0],task.global_batch_step,
                                    cfg.random_state,cfg.full_ratio,task.device)
            task.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=task.device.type,
                                 enabled=bool(cfg.mixed_precision and task.device.type=='cuda')):
                loss,_,_,stats=task.train_step_mixed(batch,state)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward(); task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            if steps==0:
                checked=assert_core_gradients(task.model)
                print('[GRAD CHECK] connected='+','.join(checked),flush=True)
            if cfg.grad_clip>0:
                gn=torch.nn.utils.clip_grad_norm_(task.trainable_parameters(),cfg.grad_clip)
            else:
                norms=[p.grad.detach().float().norm() for p in task.trainable_parameters() if p.grad is not None]
                gn=torch.stack(norms).norm() if norms else torch.tensor(0.,device=task.device)
            # One scalar check, not hundreds of per-parameter CUDA synchronizations.
            finite_grad=bool(torch.isfinite(gn))
            if not finite_grad:
                task.pending=[None]*4
                if task.scaler.is_enabled():
                    task.scaler.update(new_scale=max(task.scaler.get_scale()/2,1.))
                    skipped+=1
                else:
                    raise RuntimeError('Non-finite unscaled gradients; optimizer/memory were not updated')
            elif optimizer_step_succeeded(task):
                task.scheduler.step(); task.after_successful_step()
            else:
                task.pending=[None]*4; skipped+=1
            nf+=stats['num_full']; nm+=stats['num_missing']
            for k in sums:
                factor=stats['num_full'] if k=='loss_full' else stats['num_missing'] if k=='loss_missing' else 1
                sums[k]+=float(stats[k])*factor
            total+=float(loss.detach()); norm+=float(gn); steps+=1; task.global_batch_step+=1
            if steps%100==0:
                print(f'[BATCH {steps}] loss={float(loss.detach()):.4f} '
                      f'Full/Missing={stats["num_full"]}/{stats["num_missing"]} '
                      f'accepted={stats["candidate_tokens"]} '
                      f'full_loss={float(stats["loss_full"]):.4f} '
                      f'missing_loss={float(stats["loss_missing"]):.4f} '
                      f'ct_aux_loss={float(stats["loss_ct_aux"]):.4f} '
                      f'scale_loss={float(stats["loss_scale"]):.4f}',flush=True)
        vf=task.evaluate(val,'full'); vm=task.evaluate(val,'missing')
        joint=.5*(vf['dice']+vm['dice'])
        if joint>best:
            best,best_epoch,no_improve=joint,epoch,0
            task.save_checkpoint(best_path,epoch,best,best_epoch,{'full':vf,'missing':vm})
        else:
            no_improve+=1
        task.save_checkpoint(last,epoch,best,best_epoch,{'full':vf,'missing':vm})
        avg={k:v/max(1,nf if k=='loss_full' else nm if k=='loss_missing' else steps) for k,v in sums.items()}
        extra={'train_full_loss':avg['loss_full'],'train_missing_loss':avg['loss_missing'],
               'train_ct_aux_loss':avg['loss_ct_aux'],'train_scale_loss':avg['loss_scale'],
               'train_full_samples':nf,'train_missing_samples':nm,
               'positive_fraction':avg['positive_fraction'],'negative_fraction':avg['negative_fraction'],
               'candidate_tokens':avg['candidate_tokens'],'val_full_dice':vf['dice'],
               'val_missing_dice':vm['dice'],'val_full_hd95':vf['hd95'],'val_missing_hd95':vm['hd95'],
               'best_joint':best,'best_joint_epoch':best_epoch,
               'ema_updates':task.ema.updates if task.ema else 0,'skipped_updates':skipped,
               'bank_ema_updates':task.bank_ema.updates if task.bank_ema else 0,
               'bank_ema_active':int(task.bank_ema_active),
               'epoch_time':time.time()-t,'hd95_mode':cfg.hd95_mode,
               'retrieval_active':int(task.model.retrieval_active),
               **{f'memory_s{i+1}_entries':int(r.memory.allowed(-1,-1,int(task.model.memory_step)).sum())
                  for i,r in enumerate(task.model.retrievers)}}
        metrics={k:.5*(vf[k]+vm[k]) for k in ('total_loss','dice','iou','acc','acc_pixel','hd95')}
        append_epoch_log(log,epoch,total/max(1,steps),metrics,task.optimizer.param_groups[0]['lr'],
                         norm/max(1,steps),extra)
        print(f'[EPOCH {epoch}] Full={vf["dice"]:.4f} Missing={vm["dice"]:.4f} Joint={joint:.4f} '
              f'HD95={vf["hd95"]:.3f}/{vm["hd95"]:.3f} active={bool(task.model.retrieval_active)}',flush=True)
        # Do not stop before memory has had a chance to train.
        if epoch>=cfg.retrieval_start_epoch+3 and no_improve>=cfg.early_stop_patience:
            break
    if not os.path.isfile(best_path):
        raise RuntimeError('No best_joint checkpoint; resume in the same run directory')
    final_eval(task,test,best_path,os.path.join(cfg.checkpoint_dir,'final_test_metrics.json'))


if __name__=='__main__':
    main()
