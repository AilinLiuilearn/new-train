"""One-network encoder-pretrained preflight with the original BCE+Dice objective."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from configs.base import str2bool
from models.full_local_reassembly import FullLocalReassemblyPETCT
from utils.seg_losses import BCEDiceLoss


def grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--image_size',type=int,default=512)
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--pretrained',type=str2bool,default=True)
    parser.add_argument('--ct_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    parser.add_argument('--pet_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU does not validate CUDA')
    if args.batch_size<2 or args.image_size<32 or args.image_size%32:
        raise ValueError('Use batch_size>=2 and image_size>=32 divisible by 32')
    device=torch.device(args.device)
    torch.manual_seed(2023)
    model=FullLocalReassemblyPETCT(pretrained=args.pretrained,
        ct_pretrained_path=args.ct_pretrained_path,pet_pretrained_path=args.pet_pretrained_path).to(device).eval()
    ct=torch.randn(args.batch_size,3,args.image_size,args.image_size,device=device)
    pet=torch.randn_like(ct)
    with torch.no_grad():
        out=model(ct,pet,forward_mode='full')
        # Preflight-only Add comparison; absent from training and model API.
        model.fusion.enabled=False
        added=model(ct,pet,forward_mode='full')
        model.fusion.enabled=True
        torch.testing.assert_close(out['logits'],added['logits'],atol=0,rtol=0)
        del added
    del out
    print('[PASS] one model; fused logits exactly match Add at initialization; training uses BCE+Dice only',flush=True)
    calls=dict(ct=0,pet=0,decoder=0)
    def count(name):
        def hook(*args): calls[name]+=1
        return hook
    hooks=[model.enc_ct.register_forward_hook(count('ct')),model.enc_pet.register_forward_hook(count('pet')),
           model.decoder.register_forward_hook(count('decoder'))]
    mask=torch.zeros(args.batch_size,1,args.image_size,args.image_size,device=device)
    mask[:,:,args.image_size//3:args.image_size//2,args.image_size//3:args.image_size//2]=1
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=128.)
    criterion=BCEDiceLoss()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            out=model(ct,pet,forward_mode='full')
            main_loss,_=criterion(out['logits'],mask)
            loss=main_loss
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder','fusion'):
            grads=[p.grad for p in getattr(model,name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads),name
            assert grad_sum(getattr(model,name))>0,name
        stages=[]
        for index,block in enumerate(model.fusion.blocks):
            values={name:grad_sum(getattr(block,name)) for name in
                    ('up_ct','up_pet','ct_project','pet_project','ct_query','pet_query','ct_key','pet_key','pet_to_ct','ct_to_pet')}
            assert values['up_ct']>0 and values['up_pet']>0
            for name in ('ct_project','pet_project','ct_query','pet_query','ct_key','pet_key','pet_to_ct','ct_to_pet'):
                assert (values[name]==0) if step==0 else (values[name]>0),(step,index,name)
            stages.append(dict(stage=index+1,**values))
        assert calls==dict(ct=step+1,pet=step+1,decoder=step+1)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt); scaler.update()
        print(json.dumps(dict(step=step+1,main_loss=float(main_loss.detach()),
            calls=calls,gradients=stages,
            diagnostics=model.fusion.pop_diagnostics())),flush=True)
    for hook in hooks: hook.remove()
    print(json.dumps(dict(status='passed',device=str(device),image_size=args.image_size,
        batch_size=args.batch_size,encoder_pretrained=args.pretrained,segmentation_checkpoint=False,
        extra_reference_model=False,loss_method='BCE+Dice',
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None),indent=2))


if __name__=='__main__': main()
