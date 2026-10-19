"""Real heterogeneous encoders and shared Full runner; synthetic data only."""
import csv
import json
from pathlib import Path

import pytest
import torch

from configs.full import FullConfig
from models.build_model import build_model
from tasks.segmentation import SegmentationTask
from train_full import train_step_full


def config(tmp_path=None, fusion='asymmetric', *extra):
    args=['--fusion',fusion,'--pretrained','false','--device','cpu',
          '--image_size_2d','64','--batch_size','2','--hd95_mode','surface',
          '--epochs','1','--cosine_warmup','0','--ema_start_epoch','0']
    if tmp_path:args+=['--checkpoint_root',str(tmp_path),'--hash','synthetic']
    return FullConfig.parse_arguments(args=args+list(extra))


def batch():
    return dict(ct=torch.randn(2,3,64,64),pet=torch.randn(2,3,64,64),
                mask=(torch.rand(2,1,64,64)>.7).float())


def test_config_heterogeneous_lr_loss_and_legacy_rejection(tmp_path):
    cfg=config(tmp_path)
    assert cfg.ct_backbone=='convnextv2_nano' and cfg.pet_backbone=='mit_b1'
    assert cfg.learning_rate==4e-5 and cfg.region_loss_weight==0
    assert not Path(cfg.checkpoint_dir).exists()
    for args in (['--baseline_checkpoint','anything'],['--ct_backbone','mit_b1'],
                 ['--train_batch_mode','mixed']):
        with pytest.raises(SystemExit):FullConfig.parse_arguments(args=args)
    with pytest.raises(ValueError):config(tmp_path,'add','--region_loss_weight','.1')


def test_baseline_equivalence_rng_and_state_after_training():
    torch.manual_seed(23)
    add=build_model(config(fusion='add')).eval()
    state=torch.get_rng_state()
    torch.manual_seed(23)
    asym=build_model(config()).eval()
    assert torch.equal(state,torch.get_rng_state())
    for key,value in add.state_dict().items():
        assert torch.equal(value,asym.state_dict()[key]),key
    b=batch()
    with torch.no_grad():
        a=add(b['ct'],b['pet'],forward_mode='full')['logits']
        f=asym(b['ct'],b['pet'])['logits']
    assert torch.equal(a,f)
    # Disabled remains exact addition even after residual weights change.
    for stage in asym.fusion.stages:torch.nn.init.normal_(stage.up_ct.weight)
    asym.fusion.enabled=False
    with torch.no_grad():
        assert torch.equal(a,asym(b['ct'],b['pet'])['logits'])


def test_missing_does_not_encode_pet_or_call_fusion():
    model=build_model(config()).eval()
    def forbidden(*a):raise AssertionError('Forbidden Missing dependency')
    h1=model.enc_pet.register_forward_pre_hook(forbidden)
    h2=model.fusion.register_forward_pre_hook(forbidden)
    b=batch()
    with torch.no_grad():
        a=model(b['ct'],None,forward_mode='missing')['logits']
        f=model(b['ct'],b['pet'],forward_mode='missing')['logits']
    assert torch.equal(a,f)
    h1.remove();h2.remove()


@pytest.mark.parametrize('weight',[0.,.1])
def test_only_requested_losses_reach_model_and_optional_head(weight):
    cfg=config(None,'asymmetric','--region_loss_weight',str(weight))
    task=SegmentationTask({'model':build_model(cfg)},cfg)
    b=batch()
    loss,out,stats=train_step_full(task,b)
    main=task.criterion(out['logits'].float(),b['mask'])[0]
    if not weight:
        assert torch.equal(loss,main)
        assert stats['train_region_weighted']==0
    else:
        assert stats['train_region_raw']>0
        assert torch.allclose(loss,main+weight*stats['train_region_raw'])
    loss.backward()
    assert task.model.fusion.stages[0].region_head.weight.grad is not None
    if weight:assert task.model.fusion.stages[0].region_head.weight.grad.abs().sum()>0


def test_ema_activation_checkpoint_and_strict_architecture(tmp_path):
    cfg=config(tmp_path,'asymmetric','--ema_start_epoch','3')
    task=SegmentationTask({'model':build_model(cfg)},cfg)
    assert task.eval_weights_tag()=='raw'
    task.begin_epoch(3)
    assert task.eval_weights_tag()=='raw'
    task.begin_epoch(4)
    task.update_ema()
    assert task.eval_weights_tag()=='ema' and task.ema.updates==1
    path=str(tmp_path/'ckpt.pth.tar')
    task.save_checkpoint(path,4,.7,4,{'dice':.7})
    ckpt=task.load_state_dicts(path)
    fresh=build_model(cfg,load_encoder_pretrained=False)
    fresh.load_state_dict(ckpt['model_ema'],strict=True)
    assert ckpt['eval_weights']=='ema'
    assert ckpt['config']['learning_rate']==4e-5
    assert ckpt['config']['region_loss_weight']==0
    assert all(not p.requires_grad for p in task.ema.model.parameters())
    ids={id(p) for group in task.optimizer.param_groups for p in group['params']}
    assert ids.isdisjoint({id(p) for p in task.ema.model.parameters()})


@pytest.mark.parametrize('fusion',['add','asymmetric'])
def test_one_epoch_train_select_reload_and_final_eval(tmp_path,monkeypatch,fusion):
    import train_full
    cfg=config(tmp_path,fusion)
    b=batch()
    monkeypatch.setattr(train_full.FullConfig,'parse_arguments',lambda *a:cfg)
    monkeypatch.setattr(train_full,'_loaders',lambda cfg:([b],[b],[b]))
    train_full.main(default_fusion=fusion)
    folder=Path(cfg.checkpoint_dir)
    assert (folder/'ckpt.best_full.pth.tar').is_file()
    assert (folder/'ckpt.last.pth.tar').is_file()
    final=json.loads((folder/'final_test_full.json').read_text())
    assert final['fusion']==fusion and final['weights']=='ema' and final['best_epoch']==1
    with (folder/'train_log.csv').open() as f:row=list(csv.DictReader(f))[0]
    assert float(row['train_region_weighted'])==0 and float(row['skipped_updates'])==0
    assert float(row['ema_updates'])==1
    with pytest.raises(FileExistsError):train_full.main(default_fusion=fusion)


def test_pretrained_paths_are_strict(tmp_path):
    cfg=config(None,'add','--pretrained','true','--ct_pretrained_path',str(tmp_path/'missing'))
    with pytest.raises(FileNotFoundError):build_model(cfg)


def test_cpu_bfloat16_autocast_is_finite():
    model=build_model(config()).eval()
    b=batch()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(b['ct'],b['pet'])
    assert torch.isfinite(out['logits']).all()
    out['logits'].float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
