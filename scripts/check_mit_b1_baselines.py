"""Two-step Full-Add / CT-only MiT-B1 preflight without a dataset or task checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.base import str2bool
from models.build_mdt_seg import build_ct_only_model, build_dual_model
from models.components.add_fusion import AddFusion
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss


def check(args, mode, *, backbone='mit_b1', pretrained_path=None):
    seed_everything(2023)
    device = torch.device(args.device)
    encoder_types = {'mit_b1': 'SegformerFeatureBackbone',
                     'convnextv2_nano': 'TimmFeatureBackbone'}
    if backbone not in encoder_types:
        raise ValueError(f'Unsupported preflight backbone: {backbone}')
    if pretrained_path is None:
        if backbone != 'mit_b1':
            raise ValueError('Provide the matching encoder pretrained_path explicitly')
        pretrained_path = args.mit_pretrained_path
    cfg = SimpleNamespace(ct_backbone=backbone, pet_backbone=backbone,
        ct_pretrained_path=pretrained_path, pet_pretrained_path=pretrained_path,
        pretrained=args.pretrained, decoder_channels=(512,256,128,64),
        use_deep_supervision=False, deep_supervision=False)
    model = (build_dual_model(cfg) if mode == 'full' else build_ct_only_model(cfg))['model'].to(device)
    assert type(model.enc_ct).__name__ == encoder_types[backbone]
    if mode == 'full':
        assert isinstance(model.fusion, AddFusion)
        assert type(model.enc_ct) is type(model.enc_pet)
        assert model.enc_ct is not model.enc_pet
        assert {p.data_ptr() for p in model.enc_ct.parameters()}.isdisjoint(
            {p.data_ptr() for p in model.enc_pet.parameters()})
    else:
        assert not hasattr(model, 'enc_pet') and not hasattr(model, 'fusion')
    calls = dict(ct=0, decoder=0)
    if mode == 'full': calls['pet'] = 0
    expected_channels = [64,128,320,512]
    hooks = []
    def count(name):
        def hook(module, inputs, output): calls[name] += 1
        return hook
    def check_features(module, inputs, output):
        for feature, channel, stride in zip(output, expected_channels, [4,8,16,32]):
            assert tuple(feature.shape) == (args.batch_size, channel,
                                            args.image_size//stride, args.image_size//stride)
    hooks.append(model.enc_ct.register_forward_hook(count('ct')))
    hooks.append(model.ct_align.register_forward_hook(check_features))
    hooks.append(model.decoder.register_forward_hook(count('decoder')))
    if mode == 'full':
        hooks.append(model.enc_pet.register_forward_hook(count('pet')))
        hooks.append(model.pet_align.register_forward_hook(check_features))
    ct = torch.randn(args.batch_size,3,args.image_size,args.image_size,device=device)
    pet = torch.randn_like(ct) if mode == 'full' else None
    mask = torch.zeros(args.batch_size,1,args.image_size,args.image_size,device=device)
    mask[:,:,args.image_size//3:args.image_size//2,args.image_size//3:args.image_size//2] = 1
    optimizer = torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=128.)
    criterion = BCEDiceLoss()
    model.train()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            out = model(ct,pet=pet,forward_mode='full') if mode == 'full' else model(ct)
            assert tuple(out['logits'].shape) == tuple(mask.shape)
            loss,_ = criterion(out['logits'],mask)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        for name in ('enc_ct','ct_align','decoder') + (('enc_pet','pet_align') if mode=='full' else ()):
            assert sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters()) > 0, name
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        assert all(value == step+1 for value in calls.values()), calls
    for hook in hooks: hook.remove()
    result = dict(status='passed',mode=mode,device=str(device),encoder=backbone,
        independent_encoders=mode=='full',encoder_pretrained=args.pretrained,
        encoder_pretrained_path=pretrained_path if args.pretrained else None,
        encoder_channels=list(model.enc_ct.feature_info.channels()),
        aligned_channels=expected_channels,
        learning_rate=4e-5,loss='BCE+Dice',image_size=args.image_size,batch_size=args.batch_size,
        calls=calls,losses=losses,params_total=sum(p.numel() for p in model.parameters()),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None)
    print(json.dumps(result,indent=2),flush=True)
    del out, loss, model, optimizer, scaler, ct, pet, mask
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('both','full','ct_only'),default='both')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--image_size',type=int,default=512)
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--pretrained',type=str2bool,default=True)
    parser.add_argument('--mit_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    args = parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.batch_size < 2 or args.image_size < 32 or args.image_size % 32:
        raise ValueError('Use batch_size>=2 and image_size>=32 divisible by 32')
    for mode in (('full','ct_only') if args.mode=='both' else (args.mode,)):
        check(args,mode)


if __name__ == '__main__':
    main()
