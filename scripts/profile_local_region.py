"""Target-batch CUDA measurements, AMP gradient audit, strict raw/EMA replay."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from run_staged import parser, normalize_fusion_config, build_fresh_model, forward_loss
from utils.amp import autocast_context, make_grad_scaler
from utils.ema import ModelEMA
from utils.seg_losses import BCEDiceLoss
from scripts.check_staged import preflight_update

torch.set_num_threads(1)
torch.manual_seed(2023)
cfg=parser().parse_args(['--stage','2','--joint_from_pretrained','--fusion_type','local_region_residual'])
normalize_fusion_config(cfg)
m=build_fresh_model(cfg).cuda().eval()
shape=(16,3,512,512)
c,p=torch.randn(shape,device='cuda'),torch.randn(shape,device='cuda')
report={'batch':16,'image_size':512,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'fusion_config':cfg.fusion_config,
        'total_params':sum(t.numel() for t in m.parameters()),'scales':[]}
handles=[]
for i,block in enumerate(m.fusion.blocks):
    entry={'scale':i+1,'params':sum(t.numel() for t in block.parameters())}; report['scales'].append(entry)
    def capture(mod,args,entry=entry):
        C,P=args
        entry.update(ct=list(C.shape),pet=list(P.shape),region_tokens=(C.shape[-2]//4)*(C.shape[-1]//4))
    handles.append(block.register_forward_pre_hook(capture))
with torch.inference_mode():
    # Compare the exact same features under AMP and FP32 at target batch.
    for amp in (False,True):
        with autocast_context('cuda',enabled=amp):
            cs=m.encode_ct(c); ps=m.pet_align(m.enc_pet(p)); ys=m.fusion(cs,ps)
            report['amp_initial_max_error' if amp else 'fp32_initial_max_error']=max(float((y-(C+P)).abs().max()) for y,C,P in zip(ys,cs,ps))
    del cs,ps,ys
    for h in handles:h.remove()
    for amp in (False,True):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        with autocast_context('cuda',enabled=amp): out=m(c,p,mode='full')
        torch.cuda.synchronize()
        report['inference_amp' if amp else 'inference_fp32']={'seconds':time.perf_counter()-start,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
        del out
m.train(); ema=ModelEMA(m,device='cuda'); opt=torch.optim.AdamW(m.parameters(),lr=8e-5,weight_decay=1e-4)
scaler=make_grad_scaler('cuda',True)
batch={'ct':c,'pet':p,'mask':(torch.rand(16,1,512,512,device='cuda')>.98).float(),'is_full':torch.ones(16,dtype=torch.bool)}
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
updates=[]
for step in range(3):
    loss,stats,grads,info=preflight_update(m,batch,2,torch.device('cuda'),opt,scaler,ema)
    updates.append(info)
    if step>=1:
        for i,block in enumerate(m.fusion.blocks):
            for name in ('stem','local_block','region_block','contrast'):
                gs=[v.grad for v in getattr(block,name).parameters() if v.grad is not None]
                assert gs and all(torch.isfinite(g).all() for g in gs)
                assert sum(float(g.abs().sum()) for g in gs)>0,(i,name)
torch.cuda.synchronize()
report['training_amp']={'successful_updates':3,'updates':updates,'seconds':time.perf_counter()-start,
 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
 'all_internal_branches_have_finite_nonzero_gradients_after_projection_update':True}
opt.zero_grad(set_to_none=True)
# Strict raw and EMA reconstruction on CPU: no source pretrained files needed.
from models.staged_petct import StagedPETCT
probe=torch.randn(1,3,256,256)
for name,source in [('raw',m),('ema',ema.model)]:
    clone=StagedPETCT(2,fusion_type='local_region_residual',fusion_config=cfg.fusion_config).eval()
    clone.load_state_dict({k:v.detach().cpu() for k,v in source.state_dict().items()},strict=True)
    with torch.inference_mode():
        expected=clone(probe,probe,mode='full')
        reload=StagedPETCT(2,fusion_type='local_region_residual',fusion_config=cfg.fusion_config).eval()
        reload.load_state_dict(clone.state_dict(),strict=True)
        actual=reload(probe,probe,mode='full')
        for k in expected:torch.testing.assert_close(expected[k],actual[k],atol=0,rtol=0)
    report[name+'_strict_reconstruction']=True
    del clone,reload,expected,actual
Path('docs/local_region_validation/performance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
