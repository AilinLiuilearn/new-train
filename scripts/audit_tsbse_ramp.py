"""Schedule/EMA/gradient checks plus discarded real first-batch step."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,parser,resolve_config
require_myenv()
import torch
from types import SimpleNamespace
from variants.scgr import train_tsbse as parent
from variants.scgr.train_tsbse_ramp import coefficient,epoch_start
from variants.scgr.model import Model
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from datasets.staged import loaders

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);parent.engine.OUT=out
cfg=SimpleNamespace(tsbse_weight=.005,tsbse_delay_epochs=3,tsbse_ramp_epochs=3)
expected=[0.,0.,0.,.005/3,.01/3,.005,.005]
a=SimpleNamespace();b=SimpleNamespace()
for e,w in enumerate(expected,1):
    epoch_start(a,b,e,cfg)
    assert abs(a._tsbse_effective_weight-w)<1e-12 and b._tsbse_effective_weight==a._tsbse_effective_weight
# Same objective and exact gradient as base loss while extra coefficient=0.
y=torch.zeros(2,1,16,16);y[:,0,4:8,4:8]=1
z=torch.linspace(-2,2,y.numel()).reshape_as(y).requires_grad_(True)
m=SimpleNamespace(cfg={'tsbse_weight':.005},_tsbse_effective_weight=0.,_tsbse_epoch=1)
criterion=BCEDiceLoss();base=criterion(z,y)[0];gb=torch.autograd.grad(base,z,retain_graph=True)[0]
full,parts=parent.objective(m,{'logits':z},y,criterion);gf=torch.autograd.grad(full,z,retain_graph=True)[0]
assert torch.equal(base,full) and torch.equal(gb,gf)
m._tsbse_effective_weight=.005
full,_=parent.objective(m,{'logits':z},y,criterion)
gf=torch.autograd.grad(full,z)[0];assert torch.isfinite(gf).all() and not torch.equal(gb,gf)

args=parser('scgr_T_tsbse_ramp').parse_args(['--run','tsbse_ramp_preflight'])
cfg,_,_=resolve_config(args,'scgr_T_tsbse_ramp');seed_everything(2023);torch.set_num_threads(4)
model=Model(vars(cfg),load_generic=True).cuda().train();epoch_start(model,model,1,cfg)
train,_,_=loaders(cfg,out);batch=next(iter(train));ct,pet,y=[batch[k].cuda() for k in ('ct','pet','mask')]
opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
torch.cuda.reset_peak_memory_stats();begin=time.time()
with torch.autocast('cuda',dtype=torch.float16):o=model(ct,pet)
base=criterion(o['logits'].float(),y)[0]
base_gradient=torch.autograd.grad(base,o['logits'],retain_graph=True)[0]
value,parts=parent.objective(model,o,y,criterion)
full_gradient=torch.autograd.grad(value,o['logits'],retain_graph=True)[0]
assert torch.equal(value,base) and torch.equal(base_gradient,full_gradient)
scaler.scale(value).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
assert torch.isfinite(norm);scaler.step(opt);scaler.update();torch.cuda.synchronize()
result={'passed':True,'schedule_epochs1to7':expected,'raw_and_ema_attributes_match':True,
    'zero_weight_loss_and_logit_gradient_exactly_match_base':True,'nonzero_weight_changes_gradient':True,
    'gpu_batch':len(y),'image_shape':list(ct.shape),'base_loss':float(base),'total_loss':float(value),
    'unscaled_parameter_grad_norm':float(norm),'all_visual_parameters_trainable':all(p.requires_grad for p in model.parameters()),
    'max_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'seconds':time.time()-begin,
    'scope':'One discarded real-batch update; no checkpoint; formal training retains original AMP scaler settings.'}
(out/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
