"""Verify single-factor coefficient change and one discarded real peak-strength step."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv,parser,resolve_config,REPO
require_myenv()
import torch
from variants.scgr.model import Model
from variants.scgr import train_tsbse as base
from variants.scgr.train_tsbse_ramp import coefficient,epoch_start
from utils.run_common import seed_everything
from utils.seg_losses import BCEDiceLoss
from datasets.staged import loaders

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);base.engine.OUT=out
parent=json.loads((REPO/'configs/scgr_T_tsbse_ramp.json').read_text())
child=json.loads((REPO/'configs/scgr_T_tsbse_ramp_half.json').read_text())
diff={k:[parent.get(k),child.get(k)] for k in set(parent)|set(child) if parent.get(k)!=child.get(k)}
assert set(diff)=={'tsbse_weight','campaign_loss','parent_experiment','single_factor_change'}
assert child['tsbse_weight']==parent['tsbse_weight']/2
schedule=[coefficient(e,.0025,3,3) for e in range(1,8)]
assert all(abs(a-b)<1e-12 for a,b in zip(schedule,[0.,0.,0.,.0025/3,.005/3,.0025,.0025]))
# Parent saved file must correspond to reported same-epoch best.
path=Path('/root/autodl-tmp/ct_pet_research/outputs/scgr_T_tsbse_ramp_20261117/best.pt')
bundle=torch.load(path,map_location='cpu',weights_only=False)
assert bundle['epoch']==11 and bundle['format']=='scgr_support_conditioned_v1'
parent_validation=bundle['validation']['full']
assert abs(parent_validation['dice']-.786314135438005)<1e-12
assert abs(parent_validation['hd95']-16.214441468768772)<1e-12
del bundle
args=parser('scgr_T_tsbse_ramp_half').parse_args(['--run','tsbse_half_preflight'])
cfg,_,_=resolve_config(args,'scgr_T_tsbse_ramp_half')
seed_everything(2023);torch.set_num_threads(4)
model=Model(vars(cfg),load_generic=True).cuda().train();epoch_start(model,model,6,cfg)
train,_,_=loaders(cfg,out);batch=next(iter(train));ct,pet,y=[batch[k].cuda() for k in ('ct','pet','mask')]
opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
torch.cuda.reset_peak_memory_stats();start=time.time()
with torch.autocast('cuda',dtype=torch.float16):o=model(ct,pet)
criterion=BCEDiceLoss();primary=criterion(o['logits'].float(),y)[0]
loss,parts=base.objective(model,o,y,criterion)
gb=torch.autograd.grad(primary,o['logits'],retain_graph=True)[0].float()
gfull=torch.autograd.grad(loss,o['logits'],retain_graph=True)[0].float()
# Check exact scalar relationship on float32 leaf avoids fp16 derivative rounding.
leaf=o['logits'].detach().float().requires_grad_(True);extra,_,_=base.TSBSE()(leaf,y)
g1=torch.autograd.grad(.005*extra,leaf,retain_graph=True)[0]
g2=torch.autograd.grad(.0025*extra,leaf)[0]
assert torch.equal(g2,.5*g1)
scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
assert torch.isfinite(norm);scaler.step(opt);scaler.update();torch.cuda.synchronize()
result={'passed':True,'only_config_differences':diff,'coefficient_epochs1to7':schedule,'parent_best_metadata_matches':True,
    'float32_aux_gradient_halves_exactly':True,'batch':len(y),'image_shape':list(ct.shape),
    'base_loss':float(primary),'total_loss':float(loss),'parts':parts,
    'combined_base_logit_gradient_ratio':float(gfull.norm()/gb.norm()),'unscaled_parameter_grad_norm':float(norm),
    'all_parameters_trainable':all(p.requires_grad for p in model.parameters()),
    'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'seconds':time.time()-start,
    'scope':'Fresh real augmented batch at peak coefficient, discarded optimizer step; parent task weights used only for metadata check, never loaded into model.'}
(out/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
