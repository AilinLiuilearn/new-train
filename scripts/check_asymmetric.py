"""Synthetic real-backbone preflight: initialization, backward, bypass, reload.

No dataset or segmentation checkpoint is required. Encoder pretraining defaults
to ON; --pretrained false explicitly opts out for local random-init smoke tests.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from configs.base import str2bool
from configs.full import FullConfig
from models.build_model import build_model
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss


def check(args):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cfg=FullConfig.parse_arguments('asymmetric',[
        '--pretrained',str(args.pretrained),'--image_size_2d',str(args.image_size),
        '--batch_size',str(args.batch_size),'--ct_pretrained_path',args.ct_pretrained_path,
        '--pet_pretrained_path',args.pet_pretrained_path])
    seed_everything(2023)
    model=build_model(cfg).to(device)
    assert type(model.enc_ct).__name__=='TimmFeatureBackbone'
    assert type(model.enc_pet).__name__=='SegformerFeatureBackbone'
    assert {p.data_ptr() for p in model.enc_ct.parameters()}.isdisjoint(
        {p.data_ptr() for p in model.enc_pet.parameters()})
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    ct=torch.randn(args.batch_size,3,args.image_size,args.image_size,device=device)
    pet=torch.randn_like(ct)
    mask=torch.zeros(args.batch_size,1,args.image_size,args.image_size,device=device)
    mask[:,:,args.image_size//3:args.image_size//2,args.image_size//3:args.image_size//2]=1
    model.eval()
    with torch.no_grad():
        c,p=model._encode_ct(ct),model._encode_pet(pet)
        for i,(x,y) in enumerate(zip(c,p)):
            assert tuple(x.shape)==(args.batch_size,(64,128,320,512)[i],
                                      args.image_size//(4*2**i),args.image_size//(4*2**i))
            assert x.shape==y.shape
        ref=model._decode([x+y for x,y in zip(c,p)],ct.shape[-2:])['logits']
        initial=model(ct,pet,forward_mode='full')['logits']
        assert torch.equal(initial,ref),'Zero-init must equal Add'
        model.fusion.enabled=False
        disabled=model(ct,pet,forward_mode='full')['logits']
        assert torch.equal(disabled,ref),'Disabled must equal Add'
        model.fusion.enabled=True
    del c,p,ref,initial,disabled
    opt=torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=128.)
    criterion=BCEDiceLoss()
    losses=[]
    model.train()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            out=model(ct,pet,forward_mode='full')
            loss,_=criterion(out['logits'].float(),mask)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
        for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder'):
            assert sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters())>0,name
        for stage in model.fusion.stages:
            assert stage.up_ct.weight.grad.abs().sum()>0
            assert stage.up_pet.weight.grad.abs().sum()>0
            if step:
                assert stage.attention.qkv.weight.grad.abs().sum()>0
                assert stage.region_head.weight.grad.abs().sum()>0
                assert stage.ega.fusion_conv[0].weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
    # Missing output cannot depend on PET, and forbidden modules must not run.
    def forbidden(*unused): raise AssertionError('Missing invoked PET/fusion')
    h1=model.enc_pet.register_forward_pre_hook(forbidden)
    h2=model.fusion.register_forward_pre_hook(forbidden)
    model.eval()
    with torch.no_grad():
        a=model(ct,None,forward_mode='missing')['logits']
        b=model(ct,pet*100,forward_mode='missing')['logits']
        assert torch.equal(a,b)
    h1.remove();h2.remove()
    # Reload this trained architecture, without reloading encoder pretraining.
    fresh=build_model(cfg,load_encoder_pretrained=False).to(device).eval()
    fresh.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        a=model(ct,pet,forward_mode='full')['logits']
        b=fresh(ct,pet,forward_mode='full')['logits']
        assert torch.equal(a,b)
    result=dict(status='passed',device=str(device),ct='convnextv2_nano',pet='mit_b1',
        encoder_pretrained=args.pretrained,image_size=args.image_size,batch_size=args.batch_size,
        init_equals_add=True,disabled_equals_add=True,missing_bypasses_pet_and_fusion=True,
        strict_reload=True,losses=losses,learning_rate=4e-5,region_loss_weight=0,
        params_total=sum(p.numel() for p in model.parameters()),
        params_fusion=sum(p.numel() for p in model.fusion.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None)
    print(json.dumps(result,indent=2),flush=True)
    del fresh,model,opt,scaler,out,loss,ct,pet,mask
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--image_size',type=int,default=512)
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--pretrained',type=str2bool,default=True)
    p.add_argument('--ct_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    p.add_argument('--pet_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    check(p.parse_args())


if __name__=='__main__': main()
