"""Extract source baseline CONFIG only; task checkpoint is never opened here."""
import argparse
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_staged
from utils.dual_state_maps import MODEL, SCHEMA, sha256


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('map_dir')
    p.add_argument('output')
    p.add_argument('--crop_guidance',choices=['none','foreground','hard','dual_state'],default='dual_state')
    p.add_argument('--guided_crop_prob',type=float,default=.5)
    p.add_argument('--copy_paste',choices=['none','random','hard','dual_state'],default='none')
    p.add_argument('--copy_paste_prob',type=float,default=.3)
    p.add_argument('--paste_index')
    p.add_argument('--paste_selection',choices=['overlap','dominance'],default='overlap')
    p.add_argument('--guidance_region_sampling',choices=['uniform','sqrt_area'],default='uniform')
    p.add_argument('--early_stopping_patience',type=int,default=5)
    p.add_argument('--save_last_checkpoint',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--num_workers',type=int)
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args(argv)
    m=json.loads((Path(args.map_dir)/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA or not m.get('complete') or m.get('partial') or m.get('model')!=MODEL:
        raise ValueError('Need a complete matching v1 cache')
    config=m['baseline_config']
    if config.get('stage')!=2 or config.get('training_protocol')!='paired_full_missing' or config.get('no_pretrained'):
        raise ValueError('Source must use paired generic pretraining')
    # Verify identical generic encoder initialization, using files only, no task weights.
    from models.components.backbones import _resolve_weight_file
    for name in ('ct','pet'):
        path=_resolve_weight_file(config[name+'_pretrained_path'],name)
        if sha256(path)!=m['baseline_source'][name]['sha256']:
            raise ValueError(f'{name} generic pretraining changed from baseline')
    source_parser=run_staged.parser()
    actions={action.dest:action for action in source_parser._actions}
    excluded={'help','output','resume','evaluate','stage1_checkpoint','stage1_weights',
              'joint_from_pretrained','crop_guidance','guidance_map_dir','guided_crop_prob',
              'early_stopping_patience','device','num_workers','no_pretrained','guidance_region_sampling','save_last_checkpoint',
              'copy_paste','copy_paste_prob','paste_index','paste_selection'}
    command=[]
    for name,value in config.items():
        if name not in actions or name in excluded or value is None: continue
        if isinstance(value,bool):
            action=actions[name]
            if isinstance(action,argparse.BooleanOptionalAction): command.append(('--' if value else '--no-')+name)
            elif value: command.append('--'+name)
        else: command.extend(['--'+name,str(value)])
    command+=['--joint_from_pretrained','--output',str(Path(args.output).resolve()),
              '--crop_guidance',args.crop_guidance,'--guided_crop_prob',str(args.guided_crop_prob),
              '--guidance_region_sampling',args.guidance_region_sampling,
              '--early_stopping_patience',str(args.early_stopping_patience),'--device',args.device,
              '--num_workers',str(config['num_workers'] if args.num_workers is None else args.num_workers)]
    if args.crop_guidance!='none' or args.copy_paste!='none': command+=['--guidance_map_dir',str(Path(args.map_dir).resolve())]
    command+=['--copy_paste',args.copy_paste,'--copy_paste_prob',str(args.copy_paste_prob)]
    command+=['--paste_selection',args.paste_selection]
    if args.paste_index:command+=['--paste_index',str(Path(args.paste_index).resolve())]
    command.append('--save_last_checkpoint' if args.save_last_checkpoint else '--no-save_last_checkpoint')
    print('[LAUNCH CONFIG ONLY] '+shlex.join(['python','-u','run_staged.py']+command),flush=True)
    print('[BUDGET] source epochs/scheduler unchanged; early stopping patience='+str(args.early_stopping_patience),flush=True)
    return args,command


if __name__=='__main__':
    args,command=arguments()
    if not args.dry_run: run_staged.main(command)
