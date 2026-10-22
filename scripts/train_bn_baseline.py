"""Single-factor BN-decoder baseline from source-run CONFIG, not task weights."""
import argparse,json,shlex,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_staged


def command(a):
    source=Path(a.source_run).resolve();config=json.loads((source/'config.json').read_text())
    if (config.get('stage')!=2 or not config.get('joint_from_pretrained') or config.get('no_pretrained')
        or config.get('stage1_checkpoint') or config.get('decoder_norm','GN')!='GN'
        or config.get('crop_guidance','none')!='none' or config.get('copy_paste','none')!='none'):
        raise ValueError('Need a GN paired generic-pretrained baseline without experimental augmentations')
    from models.components.backbones import _resolve_weight_file
    init=json.loads((source/'initialization.json').read_text())
    for name in ('ct','pet'):
        path=_resolve_weight_file(config[name+'_pretrained_path'],name)
        if run_staged.file_sha(path)!=init[name]['sha256']:raise ValueError('Generic encoder changed: '+name)
    actions={x.dest:x for x in run_staged.parser()._actions}
    excluded={'output','resume','evaluate','stage1_checkpoint','stage1_weights','joint_from_pretrained',
              'decoder_norm','early_stopping_patience','save_last_checkpoint','device'}
    cmd=[]
    for name,value in config.items():
        if name not in actions or name in excluded or value is None:continue
        action=actions[name]
        if isinstance(value,bool):
            if isinstance(action,argparse.BooleanOptionalAction):cmd.append(('--' if value else '--no-')+name)
            elif value:cmd.append('--'+name)
        else:cmd+=['--'+name,str(value)]
    cmd+=['--joint_from_pretrained','--decoder_norm','BN','--output',str(Path(a.output).resolve()),
          '--early_stopping_patience','5','--no-save_last_checkpoint','--device','cuda']
    cfg=run_staged.parser().parse_args(cmd);run_staged.validate(cfg)
    from datasets.staged import split_signature
    if split_signature(cfg)!=json.loads((source/'splits.json').read_text()):raise ValueError('Source splits changed')
    plan={'source_config':str(source/'config.json'),'source_config_sha256':run_staged.file_sha(source/'config.json'),
          'task_checkpoint_loaded':False,'initialization':init,'command':[sys.executable,'-u','run_staged.py']+cmd,
          'changes':{'decoder_norm':['GN','BN'],'early_stopping_patience':[config.get('early_stopping_patience'),5],
                     'save_last_checkpoint':[config.get('save_last_checkpoint','legacy default'),False]},
          'fixed':{k:getattr(cfg,k) for k in ('batch_size','learning_rate','image_size_2d','epochs','norm_mode','aug_mode',
                     'weight_decay','cosine_warmup','cosine_min_lr','lr_flat_ratio','ema_enabled','ema_decay','ema_start_epoch',
                     'mixed_precision','grad_clip','random_state','train_split_file','val_split_file','test_split_file')}}
    return cmd,cfg,plan


def preflight(cfg,review):
    import torch
    from datasets.pclt20k_seg import PCLT20KSegDataset,_make_loader
    from datasets.staged import split_signature
    from utils.dual_state_maps import train_records
    from utils.run_common import seed_everything
    from utils.ema import ModelEMA
    from utils.amp import make_grad_scaler
    from scripts.check_staged import preflight_update
    seed_everything(cfg.random_state);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
    records=train_records(vars(cfg),split_signature(cfg)['train'])
    ds=PCLT20KSegDataset(records,image_size=cfg.image_size_2d,train=True,aug_mode=cfg.aug_mode,norm_mode=cfg.norm_mode)
    loader=_make_loader(ds,cfg.batch_size,cfg.num_workers,True,True,cfg.random_state+11,True)
    device=torch.device('cuda');model=run_staged.build_fresh_model(cfg).to(device).train()
    batch=next(iter(loader));optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    ema=ModelEMA(model,cfg.ema_decay,device=device);start=time.monotonic()
    _,stats,grads,update=preflight_update(model,batch,2,device,optimizer,make_grad_scaler(device,cfg.mixed_precision),ema)
    def forbidden(*args):raise AssertionError('Missing entered PET/Full module')
    hooks=[m.register_forward_pre_hook(forbidden) for m in (model.enc_pet,model.pet_align,model.fusion,model.decoder_full)]
    model.eval()
    try:
        with torch.inference_mode():
            x=batch['ct'][:2].to(device)
            assert torch.equal(model(x,mode='missing')['logits'],model(x,torch.full_like(x,float('nan')),mode='missing')['logits'])
    finally:
        for h in hooks:h.remove()
    result={'real_png':True,'batch_size':cfg.batch_size,'size':cfg.image_size_2d,'decoder_norm':cfg.decoder_norm,
            'parameters':sum(p.numel() for p in model.parameters()),'stats':stats,'gradients':grads,'update':update,
            'missing_isolated':True,'seconds':time.monotonic()-start,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'initialization':'generic pretrained only'}
    (review/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[BN PREFLIGHT] '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_run',default='checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1')
    p.add_argument('--output',required=True);p.add_argument('--review',required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--dry_run',action='store_true')
    a=p.parse_args();cmd,cfg,plan=command(a);review=Path(a.review);review.mkdir(parents=True,exist_ok=True)
    (review/'launch.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('[BN BASELINE] '+shlex.join(plan['command']),flush=True)
    if a.preflight:preflight(cfg,review)
    elif not a.dry_run:
        status={'state':'running','pid':os.getpid(),'output':cfg.output,'started_unix':time.time(),
                'decoder_norm':'BN','only_this_experiment':True}
        (review/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        try:
            result=run_staged.main(cmd)
            source_rows=[json.loads(x) for x in (Path(a.source_run)/'train.jsonl').read_text().splitlines()]
            gn=max(source_rows,key=lambda r:r['joint_dice'])
            bn_rows=[json.loads(x) for x in (Path(cfg.output)/'train.jsonl').read_text().splitlines()]
            bn=max(bn_rows,key=lambda r:r['joint_dice'])
            comparison={'GN':{k:gn[k] for k in ('epoch','full_dice','missing_dice','joint_dice')},
                        'BN':{k:bn[k] for k in ('epoch','full_dice','missing_dice','joint_dice')},
                        'BN_minus_GN_joint':bn['joint_dice']-gn['joint_dice'],
                        'limits':'Single seed; reused pilot val/test; not independent final evidence.'}
            (review/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            status.update(state='complete',ended_unix=time.time(),comparison=comparison)
        except BaseException as error:
            status.update(state='failed',ended_unix=time.time(),error=repr(error));raise
        finally:
            (review/'status.json').write_text(json.dumps(status,indent=2)+'\n')
