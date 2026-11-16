"""Full-only revised TSBCI conditioned ST selection + receiver Null + M1 fresh task training; one full validation per epoch."""
from pathlib import Path
import argparse,fcntl,gc,hashlib,json,os,random,shutil,sys,time,traceback
from collections import defaultdict
import numpy as np
import torch
from variants.tsbci.model import SHCIPETCT,ROOT
from datasets.staged import loaders,split_signature
from utils.run_common import seed_everything,module_grad_norm
from utils.amp import make_grad_scaler,autocast_context
from utils.ema import ModelEMA
from utils.optimization import get_cosine_scheduler
from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA
from runtime import prepare_run, gpu_lock_path, snapshot_sources
OUT=None;DEVICE=torch.device('cuda')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def dump(name,data):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def log(name,data):
    with (OUT/name).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')

def status(state,**kw):
    d={'state':state,'time_epoch':time.time(),**kw};dump('status.json',d);print('STATUS',json.dumps(d),flush=True)

def stopped():
    if (ROOT/'STOP').exists() or (OUT/'STOP').exists():
        status('stopped_by_request');raise SystemExit(0)

def save(name,data):
    if shutil.disk_usage(OUT).free<1200*1024**2:raise RuntimeError('Less than 1200MiB free; stopping without removing history')
    if name!='best.pt':raise RuntimeError('Only best.pt may be saved')
    path=OUT/name;tmp=path.with_suffix('.tmp');torch.save(data,str(tmp));os.replace(tmp,path)

def stats(x):
    x=x.detach().float()
    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite diagnostic tensor')
    return {'shape':list(x.shape),'mean':float(x.mean()),'std':float(x.std(unbiased=False)),
            'rms':float(x.square().mean().sqrt()),'abs_max':float(x.abs().max())}

def gradients(m):
    out={n:module_grad_norm(v) for n,v in m.named_children()}
    if hasattr(m,'shci'):
        out['SHCI_scales']={str(i+1):{**{n:module_grad_norm(v) for n,v in block.named_children()}, **{n:float(p.grad.detach().float().norm()) if p.grad is not None else 0. for n,p in block.named_parameters(recurse=False)}} for i,block in enumerate(m.shci.blocks)}
    return out

def diagnostics(m,o,epoch,step,u):
    cap=o['capture'];scales={}
    for i in range(4):
        sc={n:stats(cap[n][i]) for n in ['ct_aligned','pet_aligned','ct_interacted','pet_interacted','fused']}
        if str(i+1) in cap['interaction']:
            info=cap['interaction'][str(i+1)]
            sc['SHCI']={n:stats(x) for n,x in info.items()}
            sc['selected_channel_indices']={n:info[n][0].tolist() for n in ['ct_indices','pet_indices']}
            for n in ['ct','pet']:
                a=info[n+'_relation'].float().clamp_min(1e-12)
                sc[n+'_normalized_relation_entropy']=float((-(a*a.log()).sum(-1)/np.log(a.shape[-1])).mean())
                inp=cap[n+'_aligned'][i].detach().float()
                sc[n+'_relative_residual_rms']=float(info[n+'_delta'].float().square().mean().sqrt()/(inp.square().mean().sqrt()+1e-8))
                sc[n+'_relation_top1']=float(a.max(-1).values.mean())
                sc[n+'_uniform_relation_top1']=1/a.shape[-1]
                sc[n+'_conditioned_score_std']=float(info[n+'_score'].float().std(unbiased=False))
                score=info[n+'_score'];grad=score.grad
                if grad is not None and u['success']:
                    grad=grad.detach().float()/u['backward_scale']
                    chosen=torch.zeros_like(score,dtype=torch.bool).scatter(1,info[n+'_indices'],True)
                    sc[n+'_score_gradient']={'norm':float(grad.norm()),'unselected_nonzero_fraction':float((grad[~chosen].abs()>0).float().mean()),'selected_nonzero_fraction':float((grad[chosen].abs()>0).float().mean())}
                idx=info[n+'_indices'];delta=info[n+'_actual_delta'].float()
                msg=delta.gather(1,idx[:,:,None,None].expand(-1,-1,*delta.shape[-2:])).flatten(2)
                sc[n+'_message_relative_diversity']=float((msg-msg.mean(1,keepdim=True)).square().mean().sqrt()/(msg.square().mean().sqrt()+1e-8))
        scales[str(i+1)]=sc
    log('diagnostics.jsonl',{'epoch':epoch,'step':step,'AMP_gradient':u,'scales':scales})

def batch_to(batch):
    stopped()
    if not batch['is_full'].bool().all():raise RuntimeError('Missing modality sample in full-only training')
    return [batch[k].to(DEVICE,non_blocking=True) for k in ('ct','pet','mask')]

def objective(o,y,criterion):
    loss,parts=criterion(o['logits'].float(),y)
    return loss,{'total':float(loss.detach()),'bce':float(parts['loss_bce']),'dice_loss':float(parts['loss_dice'])}

def update(loss,m,opt,sch,scaler,ema,active,capture):
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5)
    if not torch.isfinite(norm):
        bad=[n for n,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if not scaler.is_enabled() or not bad:raise RuntimeError('Nonfinite global norm with no recoverable AMP overflow')
        old=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        m._overflow_streak=getattr(m,'_overflow_streak',0)+1
        log('amp_events.jsonl',{'old_scale':old,'new_scale':scaler.get_scale(),'bad_parameters':bad,'optimizer_skipped':True})
        if m._overflow_streak>=8:raise RuntimeError('Eight consecutive AMP overflows')
        return {'success':False,'norm':None,'scale':scaler.get_scale()}
    m._overflow_streak=0
    g=gradients(m) if capture else {}
    scaler.step(opt);scaler.update();sch.step()
    if active:ema.update(m)
    return {'success':True,'norm':float(norm),'groups_after_clip':g,'scale':scaler.get_scale(),'backward_scale':scale_before}

@torch.no_grad()
def evaluate(m,loader,cfg,criterion,epoch,objective_fn=None,extra_metrics_factory=None):
    m.eval();metric=SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode);sums=defaultdict(float);n=0
    compute_objective=objective_fn or objective
    extra_metric=extra_metrics_factory() if extra_metrics_factory else None
    for step,batch in enumerate(loader,1):
        ct,pet,y=batch_to(batch);o=m(ct,pet)
        if not torch.isfinite(o['logits']).all():raise RuntimeError('Nonfinite validation logits')
        _,parts=compute_objective(o,y,criterion);metric.update(o['logits'],y)
        if extra_metric is not None:extra_metric.update(o['logits'],y)
        for k,v in parts.items():sums[k]+=v*len(y)
        n+=len(y)
        if step==1 or step%50==0 or step==len(loader):
            print('EVAL',epoch,step,'/',len(loader),flush=True)
            dump('progress.json',{'activity':'validation','epoch':epoch,'step':step,'steps':len(loader)})
    result={'full':metric.compute(),'losses':{k:v/n for k,v in sums.items()},'validation':'one complete fusion, no module-on/off or single-direction evaluations'}
    if extra_metric is not None:result['auxiliary_diagnostics']=extra_metric.compute()
    return result

def results(rows,complete=False):
    if not rows:return
    best=max(rows,key=lambda x:x['validation']['full']['dice']);m=best['validation']['full']
    historical={'dice':.7836444556934279,'hd95':17.48659619401712}
    dump('RESULTS.json',{'complete':complete,'best_epoch':best['epoch'],'metrics':m,'historical_M1':historical,
        'historical_dice_difference_pp':100*(m['dice']-historical['dice']),'dice79_met':m['dice']>=.79,
        'previous_text_reference':json.loads((ROOT/'TEXT_REFERENCE.json').read_text()),
        'previous_K50_reference':json.loads((ROOT/'COMPARISON_REFERENCES.json').read_text())['previous_K50'],
        'candidate_minus_K50_dice_pp':100*(m['dice']-json.loads((ROOT/'COMPARISON_REFERENCES.json').read_text())['previous_K50']['metrics']['dice']),
        'candidate_minus_previous_text_dice_pp':100*(m['dice']-json.loads((ROOT/'TEXT_REFERENCE.json').read_text())['metrics']['dice']),
        'scope':'Single seed exploratory fresh task training from generic encoder weights only. New complete candidate: shared text-conditioned selection, ST routing, receiver Null and zero diagonal exit. Multiple mechanism changes, no isolated efficacy or semantic alignment claim.'})
    lines=['# Revised TSBCI text-conditioned ST selection + receiver Null + unchanged M1, fresh task training','',
        '|Epoch|Weights|Dice %|HD95|Sensitivity %|Precision %|','|---|---|---:|---:|---:|---:|']
    for row in rows:
        v=row['validation']['full'];lines.append(f"|{row['epoch']}|{row['weights']}|{100*v['dice']:.4f}|{v['hd95']:.4f}|{100*v['sensitivity']:.4f}|{100*v['precision']:.4f}|")
    lines+=['','Primary M1 rerun reference: Dice78.3644%, HD9517.4866. All visual parameters trainable, generic encoder pretraining only; aligners, M1, SHCI and GN decoder freshly initialized.','Only one full validation per epoch. Fixed text semantic contribution is not proven by this single run.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main(args,variant='tsbci',model_factory=None,results_fn=None,diagnostics_fn=None,
         objective_fn=None,protocol_extra=None,extra_metrics_factory=None):
    global OUT
    OUT,cfg,manifest=prepare_run(args,variant=variant)
    factory=model_factory or SHCIPETCT
    write_results=results_fn or results
    write_diagnostics=diagnostics_fn or diagnostics
    compute_objective=objective_fn or objective
    dump('process.json',{'pid':os.getpid(),'args':vars(args)});status('waiting_for_gpu')
    with open(gpu_lock_path(), 'a') as lock:
        while True:
            stopped()
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(3)
        cfg.arm=args.arm;seed_everything(cfg.random_state);torch.set_num_threads(4)
        if getattr(cfg,'initial_checkpoint',None) or cfg.stage1_checkpoint:raise RuntimeError('Task checkpoints forbidden for this experiment')
        status('loading_model')
        m=factory(vars(cfg),arm=args.arm,load_generic=True)
        dump('initialization.json',{'generic_encoders':manifest['generic_weights'],'task_checkpoint_loaded':False,'seed':cfg.random_state,'task_modules':'fresh'})
        m.to(DEVICE)
        if not all(p.requires_grad for p in m.parameters()):raise RuntimeError('Unexpected freezing')
        train,val,_=loaders(cfg,OUT);criterion=BCEDiceLoss()
        dump('protocol.json',{'config':vars(cfg),'manifest':manifest,'split_hashes':split_signature(cfg),
            'train_batches':len(train),'val_batches':len(val),'initialization':'Generic encoder pretraining only; all task modules fresh. No historical segmentation or expert weights.',
            'loss':'Original BCE + per-sample Dice only; no auxiliary losses',
            'validation':'One full model evaluation per epoch, original CIPA FP32 and pixel HD95',
            'scope':'Development set reused; full candidate only, no independent text efficacy or isolated module efficacy claims',
            **(protocol_extra or {})})
        dump('trainable.json',{'all_trainable':True,'parameters':sum(p.numel() for p in m.parameters()),
            'groups':{n:sum(p.numel() for p in v.parameters()) for n,v in m.named_children()},
            'GN_decoder':sum(isinstance(v,torch.nn.GroupNorm) for v in m.decoder_full.modules()),
            'BN_decoder':sum(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in m.decoder_full.modules()),
            'text':('frozen512D role buffers concatenated with visual32D descriptors; shared544->64->1 selector per scale' if hasattr(m,'shci') else
                    'frozen BioMedCLIP role buffers condition sample-specific routing only' if getattr(getattr(m,'front',None),'text_enabled',False) else 'none')})
        snapshot_sources(OUT)
        opt=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        sch=get_cosine_scheduler(opt,cfg.epochs,warmup_steps=cfg.cosine_warmup*len(train),min_lr=cfg.cosine_min_lr,
            steps_per_epoch=len(train),flat_ratio=cfg.lr_flat_ratio)
        scaler=make_grad_scaler(DEVICE,cfg.mixed_precision);ema=ModelEMA(m,cfg.ema_decay,warmup=True,device=DEVICE)
        active=False;start=1;best=-1.;best_epoch=0;global_step=0;rows=[]
        for epoch in range(start,cfg.epochs+1):
            stopped()
            if best_epoch and epoch-1-best_epoch>=cfg.early_stopping_patience:break
            if epoch>cfg.ema_start_epoch and not active:ema.reset(m);active=True
            m.train();status('training',epoch=epoch,best=best,best_epoch=best_epoch);sums=defaultdict(float);n=0;skips=0;begin=time.time()
            for step,batch in enumerate(train,1):
                ct,pet,y=batch_to(batch);opt.zero_grad(set_to_none=True);capture=step==1 or step%100==0 or step==len(train)
                with autocast_context(DEVICE,cfg.mixed_precision):o=m(ct,pet,capture=capture)
                loss,parts=compute_objective(o,y,criterion);u=update(loss,m,opt,sch,scaler,ema,active,capture)
                global_step+=int(u['success']);skips+=int(not u['success'])
                for k,v in parts.items():sums[k]+=v*len(y)
                n+=len(y)
                if capture:
                    write_diagnostics(m,o,epoch,step,u)
                    dump('progress.json',{'activity':'training','epoch':epoch,'step':step,'steps':len(train),'losses':parts,'AMP_gradient':u})
                    print('TRAIN',epoch,step,'/',len(train),'loss',round(parts['total'],5),'grad',u['norm'],flush=True)
            status('validation',epoch=epoch);selected=ema.model if active else m
            validation=evaluate(selected,val,cfg,criterion,epoch,objective_fn=compute_objective,
                                extra_metrics_factory=extra_metrics_factory);score=validation['full']['dice'];improved=score>best
            if improved:best=score;best_epoch=epoch
            row={'epoch':epoch,'weights':'ema' if active else 'raw','train':{k:v/n for k,v in sums.items()},'validation':validation,
                'AMP_skips':skips,'seconds':time.time()-begin,'best':best,'best_epoch':best_epoch,'lr':opt.param_groups[0]['lr'],'global_step':global_step}
            if improved:save('best.pt',{'format':getattr(cfg,'checkpoint_format','tsbci_conditioned_null_inference_v1'),'weights':selected.state_dict(),'config':vars(cfg),'epoch':epoch,'validation':validation,'manifest':manifest})
            log('train.jsonl',row);rows.append(row);write_results(rows);print('EPOCH_RESULT',json.dumps(row),flush=True)
            if epoch-best_epoch>=cfg.early_stopping_patience:break
        write_results(rows,True);status('complete',best=best,best_epoch=best_epoch)
        dump('COMPLETION.json',{'time_epoch':time.time(),'best_checkpoint':str(OUT/'best.pt'),'results':str(OUT/'RESULTS.json'),
            'next_experiments':[],'reason':'User requested only the complete candidate, no automatic ablation/tuning queue'})

if __name__=='__main__':
    from runtime import parser
    args=parser('tsbci').parse_args()
    try:main(args)
    except Exception as e:
        traceback.print_exc()
        if OUT is not None:status('failed',error=repr(e))
        sys.exit(1)
