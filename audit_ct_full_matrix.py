"""One-pass encoders, frozen multi-intervention audit of a matrix checkpoint."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
import numpy as np
import torch
from utils.matrix_audit import audit_scale, audit_findings


def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def run_audit(model, loader, device, hd95_mode, threshold, output_dir, samples=128,
              small_area=100, tolerance=.001, max_batches=0):
    from scipy.ndimage import label
    from utils.metrics_seg import compute_hd95_pair
    os.makedirs(output_dir,exist_ok=True)
    model.eval(); before=fingerprint(model)
    names=['off','normal','uniform','shuffle','g010','g030','g100']
    names += [f'scale{i}' for i in range(1,5)] + [f'full_s{i}' for i in range(1,5)] + ['full_all']
    cases=defaultdict(lambda:dict(tp=0,fp=0,fn=0,hd_sum=0.,slices=0,probability_change=0.,
        changed_pixels=0,pixels=0,lesions=0,detected=0,small_lesions=0,small_detected=0))
    feature=defaultdict(lambda:defaultdict(float))
    full_check=0.; count=0
    for step,batch in enumerate(loader,1):
        ct=batch['ct'].to(device);pet=batch['pet'].to(device);mask=batch['mask'].to(device).float()
        if mask.ndim==3:mask=mask[:,None]
        cf=model._encode_ct(ct);pf=model._encode_pet(pet);full=model.fusion(cf,pf)
        variants=[]
        for i,(scale,c,f) in enumerate(zip(model.matrix.scales,cf,full),1):
            outputs,rows=audit_scale(scale,c,f,mask,samples)
            variants.append(outputs)
            for row in rows:
                key=(str(batch['case_id'][row.pop('batch_index')]),f's{i}',row.pop('region'))
                for k,v in row.items():feature[key][k]+=v
        probability={}
        # Encoders are reused; only decoder is repeated. No candidate collection or bank writes.
        for name in names:
            if name=='off': features=cf
            elif name=='full_all':features=full
            elif name.startswith('full_s'):
                chosen=int(name[-1])-1;features=[f if i==chosen else c for i,(c,f) in enumerate(zip(cf,full))]
            elif name.startswith('scale'):
                chosen=int(name[-1])-1;features=[v['normal'] if i==chosen else c for i,(c,v) in enumerate(zip(cf,variants))]
            else:features=[v[name] for v in variants]
            probability[name]=model._decode(features,ct.shape[-2:])['logits'].float().sigmoid().cpu().numpy()[:,0]
        if step==1:
            # Cached paths must match public model forward before interpreting interventions.
            for mode,name in [('off','off'),('normal','normal')]:
                old=model.matrix_read_mode;model.matrix_read_mode=mode
                direct=model(ct,forward_mode='missing')['logits'].float().sigmoid().cpu().numpy()[:,0]
                model.matrix_read_mode=old
                if not np.allclose(direct,probability[name],rtol=1e-4,atol=1e-5):
                    raise RuntimeError(f'audit path does not match public forward: {name}')
            full_direct=model(ct,pet,forward_mode='full')['logits'].float().sigmoid().cpu().numpy()[:,0]
            full_check=float(np.max(np.abs(full_direct-probability['full_all'])))
            if full_check>1e-5:raise RuntimeError('Full cached path mismatch')
        gt=mask.cpu().numpy()[:,0]>=.5
        for b,cid in enumerate(batch['case_id']):
            components,n=label(gt[b]); sizes=np.bincount(components.ravel())
            small_ids=[j for j in range(1,n+1) if sizes[j]<=small_area]
            for name in names:
                p=probability[name][b]>threshold; g=gt[b];record=cases[(name,str(cid))]
                record['tp']+=int((p&g).sum());record['fp']+=int((p&~g).sum());record['fn']+=int((~p&g).sum())
                record['hd_sum']+=compute_hd95_pair(p,g,hd95_mode);record['slices']+=1
                record['probability_change']+=float(np.abs(probability[name][b]-probability['off'][b]).sum())
                record['changed_pixels']+=int((p!=(probability['off'][b]>threshold)).sum());record['pixels']+=p.size
                hit=set(np.unique(components[p]))-{0}
                record['lesions']+=n;record['detected']+=len(hit)
                record['small_lesions']+=len(small_ids);record['small_detected']+=sum(j in hit for j in small_ids)
        count+=len(ct)
        if step%10==0:print(f'[audit] batch {step}/{len(loader)} samples={count}',flush=True)
        if max_batches and step>=max_batches:break
    if not count:raise ValueError('empty evaluation loader')
    if fingerprint(model)!=before:raise RuntimeError('audit mutated checkpoint state')
    per_case=[];summary={}
    for (name,cid),r in cases.items():
        den=2*r['tp']+r['fp']+r['fn']
        per_case.append(dict(mode=name,case_id=cid,dice=2*r['tp']/den if den else 1.,
            hd95=r['hd_sum']/r['slices'],recall=r['tp']/(r['tp']+r['fn']) if r['tp']+r['fn'] else None,
            mean_probability_change=r['probability_change']/r['pixels'],**r))
    for name in names:
        rows=[r for r in per_case if r['mode']==name];s={k:sum(r[k] for r in rows) for k in cases[next(iter(cases))]}
        den=2*s['tp']+s['fp']+s['fn']
        summary[name]=dict(dice=2*s['tp']/den if den else 1.,hd95=s['hd_sum']/s['slices'],
            fp_pixels=s['fp'],fn_pixels=s['fn'],changed_pixels=s['changed_pixels'],
            mean_probability_change=s['probability_change']/s['pixels'],
            lesion_recall=s['detected']/s['lesions'] if s['lesions'] else None,
            small_lesion_recall=s['small_detected']/s['small_lesions'] if s['small_lesions'] else None)
    delta=[];off={r['case_id']:r for r in per_case if r['mode']=='off'}
    for r in per_case:
        b=off[r['case_id']]
        delta.append(dict(mode=r['mode'],case_id=r['case_id'],dice_delta=r['dice']-b['dice'],hd95_delta=r['hd95']-b['hd95']))
    feature_rows=[]
    for (cid,scale,region),r in feature.items():
        feature_rows.append(dict(case_id=cid,scale=scale,region=region,**r))
    aggregate=[]
    for scale in ['s1','s2','s3','s4']:
        for region in ['foreground','background']:
            rows=[r for r in feature_rows if r['scale']==scale and r['region']==region]
            if not rows:continue
            r={k:sum(v[k] for v in rows) for k in rows[0] if k not in ['case_id','scale','region']}
            r.update(scale=scale,region=region)
            for label_,key in [('ct_mse','ct_error_sum'),('normal_mse','normal_error_sum'),('oracle_mse','oracle_error_sum')]:r[label_]=r[key]/r['mass']
            r['attention_entropy']=r['attention_entropy_sum']/r['mass']
            r['attention_l1_from_uniform']=r['attention_deviation_sum']/r['mass']
            r['value_variance']=r['value_variance_sum']/r['mass']
            r['correction_rms_ratio']=(r['correction_energy_sum']/r['ct_energy_sum'])**.5 if r['ct_energy_sum'] else None
            aggregate.append(r)
    findings=audit_findings(summary,aggregate,tolerance)
    nd=[r['dice_delta'] for r in delta if r['mode']=='normal']
    distribution=dict(improved=sum(v>tolerance for v in nd),worsened=sum(v< -tolerance for v in nd),within_tolerance=sum(abs(v)<=tolerance for v in nd))
    report=dict(summary=summary,findings=findings,case_distribution=distribution,feature_summary=aggregate,
        metadata=dict(samples=count,partial=bool(max_batches),hd95_mode=hd95_mode,threshold=threshold,
            small_component_area=small_area,feature_samples_per_image_region=samples,dice_tolerance=tolerance,
            state_sha256=before,full_cached_path_max_probability_difference=full_check,
            note='Full interventions and feature oracle use real PET: diagnosis only. Lesions are 2D components, detected by any overlapping predicted pixel. Case Dice aggregates counts; global Dice aggregates all counts. Feature region samples are bounded, not exhaustive.'))
    write_csv(os.path.join(output_dir,'per_case.csv'),per_case);write_csv(os.path.join(output_dir,'case_deltas.csv'),delta)
    write_csv(os.path.join(output_dir,'feature_per_case.csv'),feature_rows);write_csv(os.path.join(output_dir,'feature_summary.csv'),aggregate)
    write_csv(os.path.join(output_dir,'interventions.csv'),[dict(mode=k,**v) for k,v in summary.items()])
    with open(os.path.join(output_dir,'audit.json'),'w') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    with open(os.path.join(output_dir,'report.md'),'w') as f:
        f.write('# 冻结 checkpoint 审查\n\n')
        f.write('以下为诊断线索，不是自动证明的唯一原因。真实 Full／oracle 对照不可作为部署结果。\n\n')
        f.write('\n'.join('- '+s for s in findings));f.write('\n\n病例变化：'+str(distribution)+'\n')
        f.write('\n|模式|Dice|HD95|\n|---|---:|---:|\n')
        for name,r in summary.items():f.write(f"|{name}|{r['dice']:.6f}|{r['hd95']:.4f}|\n")
    print(json.dumps(dict(summary=summary,findings=findings),ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    from run_ct_full_matrix_add import CTFullMatrixConfig, _loaders
    from models.ct_full_matrix_add_model import build_matrix_model
    from tasks.ct_full_matrix_seg import CTFullMatrixSegTask
    from utils.run_common import seed_everything
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--root')
    p.add_argument('--output_dir',required=True);p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--feature_samples',type=int,default=128);p.add_argument('--small_area',type=int,default=100)
    p.add_argument('--dice_tolerance',type=float,default=.001);p.add_argument('--max_batches',type=int,default=0)
    p.add_argument('--weights',choices=['raw','ema'],default='raw');args=p.parse_args()
    if args.batch_size<1 or args.feature_samples<1 or args.small_area<1 or args.max_batches<0 or args.dice_tolerance<0:p.error('invalid audit limits')
    ckpt=CTFullMatrixSegTask.load_state_dicts(args.checkpoint)
    if ckpt.get('experiment_type')!='ct_full_kv_matrix_add':raise ValueError('wrong checkpoint type')
    saved=dict(ckpt['config']);saved.pop('checkpoint_dir',None)
    saved.update(pretrained=False,ct_pretrained_path=None,pet_pretrained_path=None,ema_enabled=False,aug_mode='none',batch_size=args.batch_size)
    if args.root:saved['root']=args.root
    cfg=CTFullMatrixConfig(args=saved);seed_everything(cfg.random_state)
    task=CTFullMatrixSegTask(build_matrix_model(cfg),cfg)
    state=ckpt['model'] if args.weights=='raw' else ckpt.get('model_ema')
    if state is None:raise ValueError('requested weights unavailable')
    task.model.load_state_dict(state,strict=True)
    if not task.model.matrix_enabled:raise ValueError('matrix disabled in checkpoint')
    if not all(bool(b.valid) for s in task.model.matrix.scales for b in s.banks):raise ValueError('audit requires all eight banks valid')
    _,_,loader=_loaders(cfg)
    report=run_audit(task.model,loader,task.device,task.metrics.hd95_mode,task.metrics.threshold,
        args.output_dir,args.feature_samples,args.small_area,args.dice_tolerance,args.max_batches)
    with open(os.path.join(args.output_dir,'checkpoint.json'),'w') as f:
        json.dump(dict(path=os.path.abspath(args.checkpoint),epoch=ckpt.get('epoch'),weights=args.weights),f,indent=2)


if __name__=='__main__':main()
