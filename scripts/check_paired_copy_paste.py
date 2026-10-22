"""Real training PNG batch + generic pretrained model + one verified CUDA update."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.launch_joint_aug import arguments
from scripts.check_staged import preflight_update
from datasets.pclt20k_seg import PCLT20KSegDataset,_make_loader
from utils.paired_copy_paste import PasteBank
from utils.run_common import seed_everything
from utils.amp import make_grad_scaler
from utils.ema import ModelEMA
from run_staged import parser,build_fresh_model,validate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True);p.add_argument('--paste_index',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    _,command=arguments([a.map_dir,'checkpoints_new/CHECK_PASTE_ONLY','--crop_guidance','none',
        '--copy_paste','dual_state','--paste_index',a.paste_index,'--copy_paste_prob','1','--dry_run','--device','cuda'])
    cfg=parser().parse_args(command);validate(cfg);seed_everything(cfg.random_state);torch.set_num_threads(1)
    device=torch.device('cuda');torch.cuda.reset_peak_memory_stats()
    bank=PasteBank(a.paste_index,a.map_dir,vars(cfg))
    ds=PCLT20KSegDataset(bank.records,image_size=cfg.image_size_2d,train=True,norm_mode=cfg.norm_mode,
                        copy_paste='dual_state',copy_paste_prob=1,paste_bank=bank)
    loader=_make_loader(ds,cfg.batch_size,cfg.num_workers,True,True,cfg.random_state+11,True)
    model=build_fresh_model(cfg).to(device).train()
    batch=next(iter(loader));success=float(batch['paste_stats']['success'].sum())
    if success==0:raise AssertionError('No actual paste in real preflight batch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    ema=ModelEMA(model,cfg.ema_decay,device=device)
    started=time.monotonic()
    _,stats,grads,report=preflight_update(model,batch,2,device,optimizer,make_grad_scaler(device,True),ema)
    result={'real_png':True,'paste_attempt_probability':1,'training_probability':.3,
            'batch':cfg.batch_size,'successful_pastes':success,'parameters':sum(v.numel() for v in model.parameters()),
            'stats':stats,'gradient_modules':grads,'update':report,'seconds':time.monotonic()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu':torch.cuda.get_device_name(),'initialization':'generic_pretrained_only'}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print('[PASTE CHECK DONE] '+json.dumps(result),flush=True)
