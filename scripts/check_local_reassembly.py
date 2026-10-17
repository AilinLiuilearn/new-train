"""Real-backbone two-step preflight, including training-only frozen reference."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_local_reassembly import FullLocalReassemblyPETCT
from utils.relative_protection import initialize_from_add,relative_protection_loss
from utils.seg_losses import BCEDiceLoss


def grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--image_size',type=int,default=512)
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--baseline_checkpoint')
    parser.add_argument('--baseline_weights',choices=('auto','raw','ema'),default='auto')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; a CPU run cannot validate CUDA')
    if args.batch_size<2 or args.image_size<32 or args.image_size%32:
        raise ValueError('Use batch_size>=2 and image_size>=32 divisible by 32')
    device=torch.device(args.device)
    torch.manual_seed(2023)
    model=FullLocalReassemblyPETCT(pretrained=False).eval()
    if args.baseline_checkpoint:
        cfg=SimpleNamespace(baseline_checkpoint=args.baseline_checkpoint,baseline_weights=args.baseline_weights,
            protect_enabled=True,image_size_2d=args.image_size,norm_mode='imagenet',
            ct_backbone='convnextv2_nano',pet_backbone='mit_b1',decoder_channels=[512,256,128,64])
        reference=initialize_from_add(model,cfg)
        source=dict(path=args.baseline_checkpoint,sha256=cfg.baseline_sha256,weights=cfg.baseline_weights_resolved)
    else:
        torch.manual_seed(2023)
        reference=DualSharedAddPETCTBaseline(pretrained=False).eval().requires_grad_(False)
        for key,value in reference.state_dict().items():
            assert torch.equal(value,model.state_dict()[key]),key
        source=dict(path=None,weights='synthetic_random_shared_baseline')
    reference.eval().requires_grad_(False)
    ct=torch.randn(args.batch_size,3,args.image_size,args.image_size,device=device)
    pet=torch.randn_like(ct)
    model.to(device); reference.to(device)
    with torch.no_grad():
        expected=reference(ct,pet,forward_mode='full')['logits']
        actual=model(ct,pet,forward_mode='full')['logits']
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        fusion=model.fusion
        model.fusion=AddFusion()
        torch.testing.assert_close(model(ct,pet,forward_mode='full')['logits'],expected,atol=0,rtol=0)
        model.fusion=fusion
    del expected,actual
    frozen_before={k:v.detach().cpu().clone() for k,v in reference.state_dict().items()}
    print('[PASS] shared Add weights, enabled zero init, and disabled output match exactly',flush=True)
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
            outputs=model(ct,pet,forward_mode='full')
            main_loss,_=criterion(outputs['logits'],mask)
        with torch.no_grad():
            ref_logits=reference(ct,pet,forward_mode='full')['logits']
        protect,counts=relative_protection_loss(outputs['logits'],ref_logits,mask)
        loss=main_loss+.1*protect
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
        assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt); scaler.update()
        print(json.dumps(dict(step=step+1,main_loss=float(main_loss.detach()),
            protection=float(protect.detach()),gradients=stages,
            diagnostics=model.fusion.pop_diagnostics())),flush=True)
    for key,value in reference.state_dict().items():
        assert torch.equal(value.cpu(),frozen_before[key]),key
    print(json.dumps(dict(status='passed',device=str(device),image_size=args.image_size,
        batch_size=args.batch_size,source=source,reference_unchanged=True,
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None),indent=2))


if __name__=='__main__': main()
