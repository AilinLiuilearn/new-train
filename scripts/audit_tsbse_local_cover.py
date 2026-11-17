"""Coverage-scale repair invariants plus fresh real peak-strength step."""
import json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,parser,resolve_config,REPO
require_myenv()
import torch
from variants.scgr.tsbse import TSBSE
from variants.scgr.model import Model
from variants.scgr import train_tsbse as base
from variants.scgr.train_tsbse_ramp import epoch_start
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from datasets.staged import loaders

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);base.engine.OUT=out
torch.set_num_threads(4)
namespace={};source=subprocess.check_output(['git','show','345c4bc:variants/scgr/tsbse.py'],cwd=REPO,text=True)
exec(compile(source,'parent_TSBSE.py','exec'),namespace)
parent=namespace['TSBSE']();global_loss=TSBSE();local_loss=TSBSE('gt_local_thickness')
y=torch.zeros(1,1,24,24);y[:,:,4:20,4:20]=1
cases={}
fragmented=torch.zeros_like(y);fragmented[:,:,3,3]=1;fragmented[:,:,9:11,9:11]=1;fragmented[:,:,18,18:21]=1
for name,target,z in [('correct',y,torch.where(y.bool(),20.,-20.)),('all_missed',y,torch.full_like(y,-1000.)),
    ('all_foreground',y,torch.full_like(y,1000.)),('emptyGT',torch.zeros_like(y),torch.zeros_like(y)),
    ('fullGT',torch.ones_like(y),torch.full_like(y,-20.)),('fragmentedGT',fragmented,torch.full_like(y,-20.)),('near_threshold',y,torch.full_like(y,-.001)),('edgeGT',torch.roll(y,(-4,-4),(-2,-1)),torch.full_like(y,-5.))]:
    z=z.clone().requires_grad_(True)
    old,_,old_terms=parent(z,target);unchanged,_,_=global_loss(z,target)
    assert torch.equal(old,unchanged)
    go=torch.autograd.grad(old,z,retain_graph=True)[0];gg=torch.autograd.grad(unchanged,z,retain_graph=True)[0]
    assert torch.equal(go,gg)
    new,stats,new_terms=local_loss(z,target)
    assert torch.equal(old_terms['spill'],new_terms['spill'])
    a=torch.autograd.grad(old_terms['spill'],z,retain_graph=True)[0]
    b=torch.autograd.grad(new_terms['spill'],z,retain_graph=True)[0]
    assert torch.equal(a,b)
    grad=torch.autograd.grad(new,z,retain_graph=True)[0]
    assert torch.isfinite(new) and torch.isfinite(grad).all() and grad.abs().sum()<=1.00001
    cases[name]={'old':float(old),'new':float(new),'gradient_l1':float(grad.abs().sum()),**stats}
# A one-pixel missing edge must receive stronger cover; background risk stays exact.
z=torch.where(y.bool(),20.,-20.);z[:,:,4,4:20]=-20.;z.requires_grad_(True)
a,_,_=global_loss(z,y);b,stats,_=local_loss(z,y);assert b>a
cases['missed_one_pixel_edge']={'global':float(a),'local':float(b),**stats}
# Stable masks and ranks, finite-difference audit.
z=torch.linspace(-5,-.1,576).reshape_as(y).requires_grad_(True);direction=torch.linspace(.01,.05,576).reshape_as(y)
v=local_loss(z,y)[0];g=torch.autograd.grad(v,z)[0];eps=.005
numerical=(local_loss(z.detach()+eps*direction,y)[0]-local_loss(z.detach()-eps*direction,y)[0])/(2*eps)
analytic=(g*direction).sum();assert abs(float(numerical-analytic))<.0001
# Config is one factor relative preserved ramp parent, not latest failed half.
old=json.loads((REPO/'configs/scgr_T_tsbse_ramp.json').read_text());new=json.loads((REPO/'configs/scgr_T_tsbse_local_cover.json').read_text())
diff={k:[old.get(k),new.get(k)] for k in set(old)|set(new) if old.get(k)!=new.get(k)}
assert set(diff)=={'tsbse_cover_scale','campaign_loss','parent_experiment','single_factor_change'}
result={'synthetic_passed':True,'cases':cases,'finite_difference':float(numerical),'analytic':float(analytic),'config_diff':diff,'global_default_matches_parent_values_and_gradients':True,'spill_values_and_gradients_unchanged':True}
(out/'AUDIT.json').write_text(json.dumps(result,indent=2))
args=parser('scgr_T_tsbse_local_cover').parse_args(['--run','tsbse_local_cover_preflight'])
cfg,_,_=resolve_config(args,'scgr_T_tsbse_local_cover');seed_everything(2023)
model=Model(vars(cfg),load_generic=True).cuda().train();epoch_start(model,model,6,cfg)
train,_,_=loaders(cfg,out);batch=next(iter(train));ct,pet,target=[batch[k].cuda() for k in ('ct','pet','mask')]
opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
scaler=torch.cuda.amp.GradScaler(init_scale=1024.);torch.cuda.reset_peak_memory_stats();start=time.time()
with torch.autocast('cuda',dtype=torch.float16):output=model(ct,pet)
primary=BCEDiceLoss()(output['logits'].float(),target)[0]
value,parts=base.objective(model,output,target,BCEDiceLoss())
assert base.LOSS.coverage_scale=='gt_local_thickness'
assert all(p.requires_grad for p in model.parameters())
scaler.scale(value).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
assert torch.isfinite(norm);scaler.step(opt);scaler.update();torch.cuda.synchronize()
result['gpu']={'batch':len(target),'image_shape':list(ct.shape),'parts':parts,'unscaled_grad_norm':float(norm),
    'all_parameters_trainable':all(p.requires_grad for p in model.parameters()),'peak_mib':torch.cuda.max_memory_allocated()/1024**2,
    'seconds':time.time()-start,'scope':'Fresh generic-encoder initial batch tested at peak coefficient; discarded optimizer step; not efficacy. Formal AMP scaler unchanged.'}
result['passed']=True;(out/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
