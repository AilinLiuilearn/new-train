"""Real 512px forward/backward/optimizer preflight; no data or pretrained weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.simmlm2d import DualExpert2D
from utils.seg_losses import BCEDiceLoss

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--image_size',type=int,default=512)
    p.add_argument('--batch_size',type=int,default=1)
    a=p.parse_args();device=torch.device(a.device)
    torch.set_num_threads(1)
    model=DualExpert2D().to(device);opt=torch.optim.AdamW(model.parameters(),lr=8e-5)
    x=torch.randn(a.batch_size,1,a.image_size,a.image_size,device=device)
    y=(torch.rand_like(x)>.98).float(); criterion=BCEDiceLoss()
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
        out=model(x,x);loss=.5*(criterion(out['logits'],y)[0]+criterion(out['ct_logits'],y)[0])
    scaler.scale(loss).backward();scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    def forbidden(*args):raise AssertionError('Missing called PET or gate')
    h1=model.pet.register_forward_pre_hook(forbidden);h2=model.gate.register_forward_pre_hook(forbidden)
    model.eval()
    with torch.no_grad(): missing=model(x,mode='missing')['logits']
    h1.remove();h2.remove()
    print({'shape':list(missing.shape),'loss':float(loss.detach()),'params':sum(p.numel() for p in model.parameters()),
           'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None})
