"""Read-only, train-only snapshot diagnosis of paired Full/Missing gradients.

No optimizer, weight updates, modality dropout or gradient surgery. Train-mode
buffers are restored after each batch; eval-mode measurements are separate.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pclt20k_seg import PCLT20KSegDataset
from models.staged_petct import StagedPETCT
from run_staged import FORMAT, file_sha, load, selected
from utils.dual_state_maps import atomic_json, commit, train_records
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss


def geometry(dot, missing_sq, full_sq):
    """Unweighted task gradients; the actual shared objective averages them."""
    if not all(math.isfinite(v) for v in (dot, missing_sq, full_sq)):
        raise ValueError('Nonfinite gradient geometry')
    nm, nf = math.sqrt(missing_sq), math.sqrt(full_sq)
    return dict(dot=dot, missing_norm=nm, full_norm=nf,
                cosine=dot/(nm*nf) if nm*nf else None,
                full_to_missing_norm=nf/nm if nm else None,
                # Positive means a local decrease under plain infinitesimal SGD.
                missing_descent_dot=.5*(missing_sq+dot),
                full_descent_dot=.5*(full_sq+dot),
                cancellation_ratio=(math.sqrt(max(0., missing_sq+full_sq+2*dot))/(nm+nf)
                                    if nm+nf else None))


def group_name(name):
    if name.startswith('ct_align.'):
        return 'ct_align'
    parts = name.split('.')
    if len(parts)>2 and parts[1]=='model':
        parts=parts[:1]+parts[2:]
    # timm feature-only ConvNeXt uses stages_0.*, etc.
    if len(parts)>1 and parts[1].startswith('stages_'):
        return 'enc_ct.'+parts[1]
    if len(parts)>2 and parts[1]=='stages':
        return '.'.join(parts[:3])
    return 'enc_ct.stem_other'


def summarize(rows):
    result={}
    for name in rows[0]['groups']:
        vals=[r['groups'][name] for r in rows]
        valid=[v for v in vals if v['cosine'] is not None]
        result[name]={'batches':len(vals),'zero_gradient_batches':len(vals)-len(valid)}
        if valid:
            result[name].update(
                cosine_mean=float(np.mean([v['cosine'] for v in valid])),
                cosine_median=float(np.median([v['cosine'] for v in valid])),
                cosine_min=min(v['cosine'] for v in valid),
                conflict_fraction=float(np.mean([v['dot']<0 for v in valid])),
                missing_adverse_fraction=float(np.mean([v['missing_descent_dot']<0 for v in valid])),
                full_adverse_fraction=float(np.mean([v['full_descent_dot']<0 for v in valid])),
                full_to_missing_norm_median=float(np.median([v['full_to_missing_norm'] for v in valid])),
                cancellation_ratio_mean=float(np.mean([v['cancellation_ratio'] for v in valid])))
    return result


def diagnose(a):
    output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    if a.batches<1 or a.batch_size<1: raise ValueError('Positive batches/batch_size required')
    output.mkdir(parents=True)
    started=time.monotonic();seed_everything(a.seed);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ck=load(a.checkpoint);cfg=ck['config']
    if ck.get('format')!=FORMAT or cfg.get('stage')!=2 or cfg.get('training_protocol')!='paired_full_missing':
        raise ValueError('Need paired stage-2 baseline checkpoint')
    if cfg.get('crop_guidance','none')!='none' or cfg.get('copy_paste','none')!='none':
        raise ValueError('This diagnosis requires an unmodified augmentation baseline')
    records=train_records(cfg,ck['splits']['train'])
    order=np.random.RandomState(a.seed).permutation(len(records))[:a.batches*a.batch_size]
    chosen=[records[int(i)] for i in order]
    if len(chosen)!=a.batches*a.batch_size: raise ValueError('Requested sample count exceeds train split')
    device=torch.device(a.device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    model=StagedPETCT(2,decoder_norm=cfg.get('decoder_norm','GN')).float().to(device)
    named=[(n,p) for n,p in model.named_parameters() if n.startswith(('enc_ct.','ct_align.'))]
    params=[p for _,p in named];criterion=BCEDiceLoss();summaries={};isolation={}
    for mode in a.modes:
        train_mode=mode=='train_raw';tag='ema' if mode=='eval_ema' else 'raw'
        state,_=selected(ck,tag);model.load_state_dict(state,strict=True);model.train(train_mode)
        before={n:v.detach().clone() for n,v in model.named_buffers()}
        seed_everything(a.seed)
        ds=PCLT20KSegDataset(chosen,image_size=cfg['image_size_2d'],train=train_mode,
                            aug_mode=cfg['aug_mode'] if train_mode else 'none',norm_mode=cfg['norm_mode'])
        loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,drop_last=False)
        rows=[]
        for step,batch in enumerate(loader,1):
            x,p,y=[batch[k].to(device) for k in ('ct','pet','mask')]
            out=model(x,p,mode='full')
            if not all(torch.isfinite(out[k]).all() for k in ('ct_logits','full_logits')):
                raise ValueError('Nonfinite logits')
            lm=criterion(out['ct_logits'],y)[0];lf=criterion(out['full_logits'],y)[0]
            gm=torch.autograd.grad(lm,params,retain_graph=True)
            gf=torch.autograd.grad(lf,params,retain_graph=step==1)
            combined_error=None
            if step==1:
                gc=torch.autograd.grad(.5*(lm+lf),params)
                errors=[];err_sq=0.;reference_sq=0.
                for m,f,c in zip(gm,gf,gc):
                    expected=.5*(m+f)
                    if not torch.allclose(c,expected,rtol=3e-4,atol=3e-6):
                        raise AssertionError('Shared gradient fails FP32 elementwise linearity check')
                    err_sq+=float((c.double()-expected.double()).square().sum())
                    reference_sq+=float(expected.double().square().sum())
                    errors.append(float((c-expected).abs().max()))
                combined_error=max(errors)
                relative_error=math.sqrt(err_sq/max(reference_sq,1e-30))
                print(f'[LINEARITY] max_abs={combined_error:.6g} relative_L2={relative_error:.6g}',flush=True)
                if relative_error>1e-4:
                    raise AssertionError('Shared gradient does not match equal-weight loss within FP32 relative L2 tolerance')
                del gc
            accum={}
            for (name,_),m,f in zip(named,gm,gf):
                if not (torch.isfinite(m).all() and torch.isfinite(f).all()):
                    raise ValueError('Nonfinite gradients')
                d=float((m.double()*f.double()).sum());ms=float(m.double().square().sum());fs=float(f.double().square().sum())
                groups=set(('shared_all','enc_ct' if name.startswith('enc_ct.') else 'ct_align',group_name(name)))
                for group in groups:
                    arr=accum.setdefault(group,[0.,0.,0.])
                    for i,v in enumerate((d,ms,fs)):arr[i]+=v
            row={'step':step,'image_ids':batch['image_id'],'missing_loss':float(lm.detach()),
                 'full_loss':float(lf.detach()),'combined_gradient_max_error':combined_error,
                 'combined_gradient_relative_l2_error':relative_error if step==1 else None,
                 'groups':{k:geometry(*v) for k,v in accum.items()}}
            rows.append(row)
            with (output/(mode+'.jsonl')).open('a') as stream:stream.write(json.dumps(row)+'\n')
            del out,lm,lf,gm,gf
            with torch.no_grad():
                for n,v in model.named_buffers():v.copy_(before[n])
            if step==1 or step%8==0 or step==len(loader):
                print(f'[GRAD] {mode} {step}/{len(loader)} cosine={row["groups"]["shared_all"]["cosine"]:.4f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if any(not torch.equal(v,before[n]) for n,v in model.named_buffers()):raise AssertionError('Buffer changed')
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Diagnostic accumulated .grad')
        if any(not torch.equal(v.detach().cpu(),state[n]) for n,v in model.named_parameters()):raise AssertionError('Weight changed')
        summaries[mode]=summarize(rows)
        if not train_mode:
            model.eval()
            def forbidden(*args):raise AssertionError('Missing invoked PET/Full module')
            hooks=[m.register_forward_pre_hook(forbidden) for m in (model.enc_pet,model.pet_align,model.fusion,model.decoder_full)]
            try:
                with torch.inference_mode():
                    first=model(x,mode='missing')['ct_logits'];second=model(x,torch.zeros_like(p),mode='missing')['ct_logits']
                    isolation[mode]=bool(torch.equal(first,second))
                    if not isolation[mode]:raise AssertionError('Missing depends on supplied PET')
            finally:
                for h in hooks:h.remove()
    report={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha(a.checkpoint),
            'epoch':ck['epoch'],'checkpoint_eval_weights':ck.get('eval_weights'),'commit':commit(),
            'seed':a.seed,'batches':a.batches,'batch_size':a.batch_size,'samples':len(chosen),
            'train_split_sha256':ck['splits']['train'],'precision':'FP32; TF32 disabled; no autocast/scaler',
            'train_raw':'baseline CIPA augmentation, train-mode stochastic layers/BN; buffers restored every batch',
            'eval_modes':'no augmentation, eval-mode BN/stochastic layers; gradients enabled',
            'optimizer_steps':0,'parameters_unchanged':True,'buffers_unchanged':True,
            'missing_isolation':isolation,'summary':summaries,'seconds':time.monotonic()-started,
            'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'limits':['Snapshot at best checkpoint, not full trajectory.','Batch geometry depends on batch size and sampled slices, not patient-level independent evidence.',
                      'SGD directional derivatives are not actual AdamW/clipped/AMP updates or causal evidence of negative transfer.',
                      'EMA eval gradients are deployment-state sensitivity, not historical optimizer gradients.']}
    atomic_json(output/'report.json',report)
    print('[DONE] '+str(output/'report.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--batches',type=int,default=32);p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--modes',nargs='+',choices=['train_raw','eval_raw','eval_ema'],default=['train_raw','eval_raw','eval_ema'])
    diagnose(p.parse_args())
