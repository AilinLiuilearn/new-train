"""Compare strict raw/EMA states from the same checkpoint on the same split.

FP32 baseline metrics, no updates or BN recalibration. Saved best checkpoints
are selected by their original rule, so this is not a raw-best vs EMA-best run.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from datasets.pclt20k_seg import PCLT20KSegDataset, _records_from_ids, _check_patient_disjoint
from datasets.staged import split_signature
from models.staged_petct import StagedPETCT
from run_staged import FORMAT, load, selected, file_sha
from utils.dual_state_maps import atomic_json, checked_ids, commit
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss


def state_difference(model,raw,ema):
    groups={}
    param_names=set(dict(model.named_parameters()))
    for name,r in raw.items():
        e=ema[name]
        if r.shape!=e.shape:raise ValueError('Shape mismatch: '+name)
        if not (torch.isfinite(r).all() and torch.isfinite(e).all()):raise ValueError('Nonfinite state: '+name)
        kind='parameters' if name in param_names else 'floating_buffers' if r.is_floating_point() else 'integer_buffers'
        g=groups.setdefault(kind,{'elements':0,'changed_elements':0,'squared_difference':0.,'raw_squared_norm':0.})
        g['elements']+=r.numel();g['changed_elements']+=int((r!=e).sum())
        g['squared_difference']+=float((r.double()-e.double()).square().sum())
        g['raw_squared_norm']+=float(r.double().square().sum())
    return groups


@torch.inference_mode()
def evaluate_state(model,loader,cfg,device,tag,output):
    metrics={k:SegmentationMetricsCIPA(hd95_mode=cfg.hd95_mode) for k in ('full','missing')}
    criterion=BCEDiceLoss();losses={'full':0.,'missing':0.};count=0;start=time.monotonic()
    buffers={n:v.clone() for n,v in model.named_buffers()}
    with (output/(tag+'-samples.jsonl')).open('w') as stream:
        for step,batch in enumerate(loader,1):
            x,p,y=[batch[k].to(device,non_blocking=True) for k in ('ct','pet','mask')]
            out=model(x,p,mode='full');rows=[{'image_id':i,'case_id':c} for i,c in zip(batch['image_id'],batch['case_id'])]
            for mode,key in [('full','full_logits'),('missing','ct_logits')]:
                logits=out[key]
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
                loss=criterion(logits,y)[0]
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                losses[mode]+=float(loss)*len(y);metrics[mode].update(logits,y)
                pred=logits.sigmoid()>.5;gt=y.bool();dims=(1,2,3)
                counts=torch.stack([(pred&gt).sum(dims),(pred&~gt).sum(dims),(~pred&gt).sum(dims),(~pred&~gt).sum(dims)],1).cpu().tolist()
                hd=metrics[mode].hd95_list[-len(y):]
                for row,nums,h in zip(rows,counts,hd):row[mode]=dict(zip(('tp','fp','fn','tn'),nums),hd95=h)
            for row in rows:stream.write(json.dumps(row)+'\n')
            count+=len(y)
            if step==1 or step%25==0 or step==len(loader):
                elapsed=time.monotonic()-start
                print(f'[{tag}] {step}/{len(loader)} samples={count} elapsed={elapsed:.1f}s eta={elapsed/step*(len(loader)-step):.1f}s',flush=True)
    if count!=len(loader.dataset):raise AssertionError('Tail batch lost')
    if any(not torch.equal(v,buffers[n]) for n,v in model.named_buffers()):raise AssertionError('BN buffers changed')
    result={mode:{**m.compute(),'total_loss':losses[mode]/count,'tp_pixels':int(m.tp),'fp_pixels':int(m.fp),'fn_pixels':int(m.fn),'tn_pixels':int(m.tn)} for mode,m in metrics.items()}
    result.update(joint_dice=.5*(result['full']['dice']+result['missing']['dice']),samples=count,seconds=time.monotonic()-start)
    atomic_json(output/(tag+'.json'),result)
    return result


def compare(a):
    rootout=Path(a.output)
    if rootout.exists():raise FileExistsError(rootout)
    rootout.mkdir(parents=True);all_reports=[];torch.set_num_threads(1)
    for index,path in enumerate(a.checkpoint):
        start=time.monotonic();output=rootout/f'{index:02d}-{Path(path).parent.name}';output.mkdir()
        print('[START] '+str(path),flush=True)
        ck=load(path);cfg=SimpleNamespace(**ck['config'])
        if ck.get('format')!=FORMAT or cfg.stage!=2:raise ValueError('Need stage2 paired checkpoint')
        if split_signature(cfg)!=ck['splits']:raise ValueError('Split identity changed')
        recs={}
        for split in ('train','val','test'):
            ids=checked_ids([x.strip() for x in (Path(cfg.root)/getattr(cfg,split+'_split_file')).read_text().splitlines() if x.strip()])
            recs[split]=_records_from_ids(cfg.root,ids)
        _check_patient_disjoint(recs['train'],recs['val'],recs['test'],cfg.allow_val_equals_test)
        reused=set(r['image_id'] for r in recs['val'])==set(r['image_id'] for r in recs['test'])
        if reused:print('[WARNING] val/test are reused pilot data; not independent final test.',flush=True)
        seed_everything(cfg.random_state);device=torch.device(a.device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        model=StagedPETCT(2,decoder_norm=getattr(cfg,'decoder_norm','GN')).float().to(device).eval().requires_grad_(False)
        raw,_=selected(ck,'raw');ema,_=selected(ck,'ema')
        if not ck.get('ema_active'):raise ValueError('Checkpoint EMA was not active')
        difference=state_difference(model,raw,ema)
        ds=PCLT20KSegDataset(recs[a.split],image_size=cfg.image_size_2d,train=False,aug_mode='none',norm_mode=cfg.norm_mode)
        loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.num_workers,shuffle=False,drop_last=False,pin_memory=device.type=='cuda')
        results={}
        for tag,state in [('raw',raw),('ema',ema)]:
            seed_everything(cfg.random_state);model.load_state_dict(state,strict=True)
            results[tag]=evaluate_state(model,loader,cfg,device,tag,output)
            if any(not torch.equal(v.detach().cpu(),state[n]) for n,v in model.state_dict().items()):raise AssertionError('State changed')
        delta={mode:{k:results['ema'][mode][k]-results['raw'][mode][k] for k in results['raw'][mode]} for mode in ('full','missing')}
        delta['joint_dice']=results['ema']['joint_dice']-results['raw']['joint_dice']
        report={'checkpoint':str(Path(path).resolve()),'sha256':file_sha(path),'epoch':ck['epoch'],
                'checkpoint_selected_by':ck['eval_weights'],'ema_updates':ck['ema_updates'],'ema_decay':cfg.ema_decay,'ema_start_epoch':cfg.ema_start_epoch,
                'split':a.split,'split_sha256':ck['splits'][a.split],'val_equals_test':reused,'precision':'FP32 baseline evaluation; no AMP',
                'batch_size':a.batch_size,'optimizer_steps':0,'buffers_recalibrated':False,'state_difference':difference,
                'results':results,'ema_minus_raw':delta,'commit':commit(),'seconds':time.monotonic()-start,
                'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                'limits':['Same saved epoch comparison; not independently selected raw-best vs EMA-best.',
                          'Saved checkpoints were selected by EMA, so selection favors EMA.',
                          'EMA includes floating buffers, including BN running statistics; this comparison does not isolate parameter vs buffer contributions.',
                          'EMA is a detached side copy, not a gradient source; removing it can change evaluation, selection and early stopping.']}
        atomic_json(output/'report.json',report);all_reports.append(report)
        atomic_json(rootout/'comparison.json',all_reports)
        print('[DONE] '+json.dumps({'checkpoint':path,'epoch':ck['epoch'],'ema_minus_raw_joint':delta['joint_dice'],'seconds':report['seconds']}),flush=True)
        del model,ck,raw,ema
    return all_reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--device',default='cuda')
    p.add_argument('--batch_size',type=int,default=16);p.add_argument('--num_workers',type=int,default=4)
    compare(p.parse_args())
