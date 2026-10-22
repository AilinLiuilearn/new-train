"""Train-only sensitivity probe, not a performance ablation or retraining."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pclt20k_seg import PCLT20KSegDataset
from models.staged_petct import StagedPETCT
from run_staged import load,selected,file_sha
from utils.dual_state_maps import train_records
from utils.run_common import seed_everything


@torch.inference_mode()
def probe(a):
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1);seed_everything(2023)
    ck=load(a.checkpoint);cfg=ck['config'];state,tag=selected(ck)
    model=StagedPETCT(2,decoder_norm=cfg.get('decoder_norm','GN')).float().to(a.device).eval().requires_grad_(False)
    model.load_state_dict(state,strict=True)
    records=train_records(cfg,ck['splits']['train'])
    indices=np.random.RandomState(2023).permutation(len(records))[:8]
    ds=PCLT20KSegDataset([records[i] for i in indices],image_size=cfg['image_size_2d'],train=False,aug_mode='none',norm_mode=cfg['norm_mode'])
    buffers={n:v.clone() for n,v in model.named_buffers()};results=[]
    for b in DataLoader(ds,batch_size=4,shuffle=False):
        x,p=b['ct'].to(a.device),b['pet'].to(a.device)
        c=model.encode_ct(x);pet=model.pet_align(model.enc_pet(p));f=model.fusion(c,pet)
        base=model.decoder_full(f,x.shape[-2:])['logits'];prob=base.sigmoid()
        cases={'uniform_times_2':[2*v for v in f],
               'per_scale_scalar':[s*v for s,v in zip((.5,2,.5,2),f)],
               'relative_pet_times_half':[v+.5*w for v,w in zip(c,pet)]}
        for name,features in cases.items():
            out=model.decoder_full(features,x.shape[-2:])['logits'];pr=out.sigmoid()
            if not torch.isfinite(out).all():raise ValueError('Nonfinite probe')
            results.append({'intervention':name,'image_ids':b['image_id'],
                'logit_relative_l2':float((out-base).norm()/base.norm().clamp_min(1e-12)),
                'probability_mean_abs_change':float((pr-prob).abs().mean()),
                'probability_max_abs_change':float((pr-prob).abs().max()),
                'binary_changed_fraction':float(((pr>.5)!=(prob>.5)).float().mean())})
    if any(not torch.equal(v,buffers[n]) for n,v in model.named_buffers()):raise AssertionError('Buffer mutated')
    if any(not torch.equal(v.detach().cpu(),state[n]) for n,v in model.state_dict().items()):raise AssertionError('State mutated')
    report={'checkpoint':str(Path(a.checkpoint).resolve()),'sha256':file_sha(a.checkpoint),'weights':tag,
            'samples':8,'split':'train only, fixed seed2023, no augmentation','optimizer_steps':0,
            'results':results,'limitations':['Sensitivity, not performance or causal training ablation.',
                'Pure positive scalar scaling is approximately removed by Conv(no bias)+GN; this does not imply arbitrary channel/spatial attention is removed.',
                'Effects depend on GN epsilon, input variance and finite precision.']}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    probe(p.parse_args())
