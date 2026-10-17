"""Contracts for local transport, abstention, additive init and relative protection."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.components.local_reassembly import (
    LocalReassemblyFusion, LocalTransport, STAGE_CHANNELS, STAGE_RANKS)
from models.full_local_reassembly import FullLocalReassemblyPETCT, build_full_local_model
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from utils.relative_protection import relative_protection_loss, initialize_from_add


def features(actual=False, dtype=torch.float32):
    sizes = (128, 64, 32, 16) if actual else (8, 4, 2, 1)
    cs = [torch.randn(2, c, s, s, dtype=dtype, requires_grad=True)
          for c, s in zip(STAGE_CHANNELS, sizes)]
    return cs, [torch.randn_like(c, requires_grad=True) for c in cs]


@pytest.mark.parametrize('enabled', [True, False])
def test_all_scales_exact_add_and_roundtrip(enabled):
    fusion = LocalReassemblyFusion(enabled=enabled)
    cs, ps = features(actual=True)
    for a, c, p in zip(fusion(cs, ps), cs, ps):
        torch.testing.assert_close(a, c+p, atol=0, rtol=0)
    other = copy.deepcopy(fusion)
    other.load_state_dict(fusion.state_dict(), strict=True)
    for a, b in zip(fusion(cs, ps), other(cs, ps)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if not enabled:
        assert not list(fusion.parameters())
    assert all(v == 0 for k, v in fusion.pop_diagnostics().items() if 'delta' in k)


def test_transport_border_mask_empty_candidate_and_neighbor_selection():
    transport = LocalTransport(rank=2, heads=1)
    q = torch.ones(1, 2, 2, 2)
    k = torch.zeros_like(q)
    v = torch.arange(8.).reshape(1, 2, 2, 2)
    # Null dominates: values are fixed zero, not an unrelated learned noise value.
    with torch.no_grad():
        transport.null_key.fill_(100)
    out, null_mass, center_mass = transport(q, k, v)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-20, rtol=0)
    torch.testing.assert_close(null_mass, torch.ones_like(null_mass))
    assert center_mass.max() < 1e-20
    # With null disabled, invalid padded keys must not dilute the valid mean.
    with torch.no_grad():
        transport.null_key.fill_(-100)
    out, _, _ = transport(q, k, v)
    torch.testing.assert_close(out, v.mean((-2, -1), keepdim=True).expand_as(v))
    # Pick the right neighbor at the top-left query; this is not scalar gating.
    k.zero_(); k[:, :, 0, 1] = 100
    out, _, _ = transport(q, k, v)
    torch.testing.assert_close(out[:, :, 0, 0], v[:, :, 0, 1])


def test_two_steps_open_both_directions_and_null_keys():
    torch.manual_seed(2023)
    fusion = LocalReassemblyFusion()
    cs, ps = features()
    target = [torch.randn_like(c) for c in cs]
    opt = torch.optim.AdamW(fusion.parameters(), lr=4e-5)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        sum(F.mse_loss(a, y) for a, y in zip(fusion(cs, ps), target)).backward()
        for block in fusion.blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            for name in ('up_ct', 'up_pet'):
                assert getattr(block, name).weight.grad.abs().sum() > 0
            for name in ('ct_project', 'pet_project', 'ct_query', 'pet_query',
                         'ct_key', 'pet_key', 'pet_to_ct', 'ct_to_pet'):
                total = sum(float(p.grad.abs().sum()) for p in getattr(block, name).parameters())
                assert (total == 0) if step == 0 else (total > 0), (step, name)
        opt.step()
    diagnostics = fusion.pop_diagnostics()
    assert all(diagnostics[f'local_delta_ratio_s{i}'] > 0 for i in range(1, 5))
    assert all(0 <= v <= 1 for k, v in diagnostics.items() if 'mass' in k)


def test_different_pet_spatial_size_and_diagnostics_do_not_change_in_eval():
    fusion=LocalReassemblyFusion().eval()
    cs,ps=features()
    ps=[p[...,::2,::2] for p in ps]
    for out,c,p in zip(fusion(cs,ps),cs,ps):
        expected=c+F.interpolate(p,c.shape[-2:],mode='bilinear',align_corners=False)
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
    assert all(value==0 for value in fusion.pop_diagnostics().values())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_input_fp32_core(dtype):
    fusion = LocalReassemblyFusion()
    cs, ps = features(dtype=dtype)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = fusion(cs, ps)
        assert all(a.dtype == dtype for a in out)
        for a, c, p in zip(out, cs, ps):
            torch.testing.assert_close(a, c+p, atol=0, rtol=0)
        sum(a.float().square().mean() for a in out).backward()
    assert all(torch.isfinite(x.grad).all() for x in cs+ps)
    with pytest.raises(ValueError, match='master'):
        fusion.half()(cs, ps)


@pytest.mark.parametrize('bad', ['count', 'channel', 'batch', 'dtype', 'empty', 'nan'])
def test_invalid_features(bad):
    cs, ps = features()
    if bad == 'count': ps.pop()
    if bad == 'channel': ps[0] = ps[0][:, :63]
    if bad == 'batch': ps[0] = ps[0][:1]
    if bad == 'dtype': ps[0] = ps[0].double()
    if bad == 'empty': ps[0] = ps[0][..., :0]
    if bad == 'nan': ps[0] = torch.full_like(ps[0], float('nan'))
    with pytest.raises((ValueError, RuntimeError)):
        LocalReassemblyFusion()(cs, ps)


def test_reference_loss_protects_correct_pixels_balances_classes_and_detaches_teacher():
    target = torch.tensor([[[[1., 0., 1., 0.]]]])
    base = torch.tensor([[[[2., -2., -2., 2.]]]], requires_grad=True)
    # Regress on correct FG/BG; improve both original mistakes, which are unprotected.
    pred = torch.tensor([[[[1., -1., 4., -4.]]]], requires_grad=True)
    loss, stats = relative_protection_loss(pred, base, target)
    expected = F.softplus(torch.tensor(-1.)) - F.softplus(torch.tensor(-2.))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert base.grad is None
    assert pred.grad[0, 0, 0, 0] < 0 and pred.grad[0, 0, 0, 1] > 0
    assert torch.equal(pred.grad[..., 2:], torch.zeros_like(pred.grad[..., 2:]))
    assert stats['protect_correct_fg'] == 1 and stats['protect_correct_bg'] == 1
    same, _ = relative_protection_loss(base.detach(), base, target)
    assert same == 0
    better, _ = relative_protection_loss(base.detach()*2, base, target)
    assert better == 0
    # No correct pixels must return a finite graph-connected zero.
    wrong, _ = relative_protection_loss(pred, -base.detach(), target)
    assert torch.isfinite(wrong)
    zero, _ = relative_protection_loss(pred, -base.detach(), 1-target)
    assert torch.isfinite(zero)
    all_wrong = -torch.ones_like(pred, requires_grad=True)
    zero, _ = relative_protection_loss(pred, all_wrong, torch.ones_like(pred))
    assert zero == 0
    many_bg = torch.cat([pred.detach()[..., :1], pred.detach()[..., 1:2].expand(1,1,1,100)], -1)
    many_ref = torch.cat([base.detach()[..., :1], base.detach()[..., 1:2].expand(1,1,1,100)], -1)
    labels = torch.zeros_like(many_bg); labels[..., 0] = 1
    balanced, _ = relative_protection_loss(many_bg, many_ref, labels)
    torch.testing.assert_close(balanced, expected)


@pytest.mark.parametrize('enabled', [True, False])
def test_real_model_add_equivalence_rng_and_missing(enabled):
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullLocalReassemblyPETCT(pretrained=False, local_enabled=enabled).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    ct, pet = torch.randn(2,3,64,64), torch.randn(2,3,64,64)
    with torch.no_grad():
        torch.testing.assert_close(model(ct, pet, forward_mode='full')['logits'],
                                   base(ct, pet, forward_mode='full')['logits'], atol=0, rtol=0)
    def forbidden(*args): raise AssertionError('Missing cannot call PET/fusion')
    hooks = [m.register_forward_pre_hook(forbidden) for m in (model.enc_pet, model.fusion)]
    with torch.no_grad():
        a = model(ct, pet=None, forward_mode='missing')['logits']
        b = model(ct, pet=torch.full_like(pet, float('nan')), pet_available=[0,0], forward_mode='auto')['logits']
    for h in hooks: h.remove()
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    with torch.no_grad():
        a = model(ct, pet, pet_available=[1,0], forward_mode='auto')['logits']
        pet[1].fill_(float('nan'))
        b = model(ct, pet, pet_available=[1,0], forward_mode='auto')['logits']
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def baseline_checkpoint(tmp_path):
    baseline = DualSharedAddPETCTBaseline(pretrained=False).eval()
    config = dict(model_arch='dual_shared_add_baseline', experiment_type='full_add_baseline',
                  fusion_type='AddFusion', image_size_2d=64, norm_mode='imagenet',
                  ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                  decoder_channels=[512,256,128,64], use_deep_supervision=False,
                  deep_supervision=False)
    path = tmp_path/'add.pth'
    torch.save(dict(config=config, model=baseline.state_dict(), model_ema=baseline.state_dict(),
                    eval_weights='ema', epoch=11), path)
    return path, baseline, config


def test_warm_start_strict_reference_and_source_rejection(tmp_path):
    from types import SimpleNamespace
    path, base, saved = baseline_checkpoint(tmp_path)
    cfg = SimpleNamespace(**saved, baseline_checkpoint=str(path), baseline_weights='auto', protect_enabled=True)
    student = FullLocalReassemblyPETCT(pretrained=False)
    teacher = initialize_from_add(student, cfg)
    assert teacher is not None and not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not any(k.startswith('reference') for k in student.state_dict())
    for k,v in base.state_dict().items():
        assert torch.equal(v, student.state_dict()[k]) and torch.equal(v,teacher.state_dict()[k])
    assert len(cfg.baseline_sha256) == 64 and cfg.baseline_weights_resolved == 'ema'
    cfg.norm_mode = 'cipa'
    with pytest.raises(ValueError, match='norm_mode'): initialize_from_add(student,cfg)
    cfg.norm_mode = 'imagenet'
    payload = torch.load(path, weights_only=False)
    payload['model_ema']['extra.weight'] = torch.ones(1)
    torch.save(payload,path)
    with pytest.raises(ValueError, match='state'): initialize_from_add(student,cfg)


def test_train_step_keeps_reference_state_fixed_and_out_of_optimizer(tmp_path):
    from types import SimpleNamespace
    from tasks.mdt_seg import MDTSegTeacher
    import run_full_local_reassembly as runner
    path,_,saved=baseline_checkpoint(tmp_path)
    cfg=SimpleNamespace(**saved,baseline_checkpoint=str(path),baseline_weights='auto',
        protect_enabled=True,protect_weight=.1,learning_rate=4e-5,weight_decay=1e-4,
        mixed_precision=False,ema_enabled=True,ema_decay=.999,ema_decay_warmup=True,ema_start_epoch=0)
    student=FullLocalReassemblyPETCT(pretrained=False)
    teacher=initialize_from_add(student,cfg)
    task=MDTSegTeacher({'model':student},cfg)
    task.reference_model=teacher.to(task.device)
    task.model.train()
    frozen={k:v.clone() for k,v in teacher.state_dict().items()}
    optimizer_ids={id(p) for group in task.optimizer.param_groups for p in group['params']}
    assert not optimizer_ids.intersection(id(p) for p in teacher.parameters())
    batch=dict(ct=torch.randn(2,3,64,64),pet=torch.randn(2,3,64,64),mask=torch.ones(2,1,64,64))
    loss,_,_=runner.train_step_full(task,batch)
    loss.backward(); task.optimizer.step(); task.update_ema()
    for key,value in teacher.state_dict().items(): assert torch.equal(value,frozen[key])
    assert all(p.grad is None for p in teacher.parameters())
    assert not teacher.training
    assert not any('reference' in key for key in task.ema.state_dict())
    cfg.protect_enabled=False
    task.reference_model=None
    loss,out,_=runner.train_step_full(task,batch)
    main,_=task.criterion(out['logits'],batch['mask'].to(task.device))
    torch.testing.assert_close(loss,main,atol=0,rtol=0)


def test_parser_guards_and_scratch_initialization(monkeypatch,tmp_path):
    from configs.full_local_reassembly import FullLocalConfig
    from types import SimpleNamespace
    monkeypatch.setattr(sys,'argv',['x','--pretrained','false','--require_cuda','false',
        '--checkpoint_root',str(tmp_path),'--hash','scratch','--protect_enabled','false'])
    cfg=FullLocalConfig.parse_arguments()
    assert cfg.learning_rate==4e-5 and cfg.norm_mode=='imagenet'
    model=FullLocalReassemblyPETCT(pretrained=False)
    assert initialize_from_add(model,cfg) is None
    assert cfg.initialization_mode=='from_scratch_or_encoder_pretrained'
    with pytest.raises(ValueError,match='baseline_checkpoint'):
        initialize_from_add(model,SimpleNamespace(baseline_checkpoint=None,protect_enabled=True))
    (Path(cfg.checkpoint_dir)/'keep.txt').write_text('original')
    with pytest.raises(FileExistsError): FullLocalConfig.parse_arguments()
    assert (Path(cfg.checkpoint_dir)/'keep.txt').read_text()=='original'
    monkeypatch.setattr(sys,'argv',['x','--require_cuda','false','--protect_enabled','true'])
    with pytest.raises(ValueError,match='Protection requires'): FullLocalConfig.parse_arguments()


@pytest.mark.parametrize('local',[True,False])
def test_default_fresh_training_main_loss_only_and_no_reference(monkeypatch,tmp_path,local):
    """Primary ablation starts both arms afresh, with no segmentation checkpoint."""
    from configs.full_local_reassembly import FullLocalConfig
    from eval_full_local_reassembly import load_full_checkpoint
    import run_full_local_reassembly as runner
    from utils.seg_losses import BCEDiceLoss
    monkeypatch.setattr(sys,'argv',['x','--pretrained','false','--require_cuda','false',
        '--checkpoint_root',str(tmp_path),'--hash','fresh-default','--local_enabled',str(local),
        '--image_size_2d','64','--batch_size','2','--epochs','1','--cosine_warmup','0',
        '--mixed_precision','false','--hd95_mode','surface'])
    cfg=FullLocalConfig.parse_arguments()
    assert not cfg.protect_enabled and cfg.baseline_checkpoint is None
    assert cfg.learning_rate==4e-5 and cfg.ema_enabled and cfg.ema_decay==.999
    cfg.ema_start_epoch=0
    monkeypatch.setattr(FullLocalConfig,'parse_arguments',classmethod(lambda cls:cfg))
    mask=torch.zeros(2,1,64,64); mask[:,:,20:32,20:32]=1
    batch=dict(ct=torch.randn(2,3,64,64),pet=torch.randn(2,3,64,64),mask=mask)
    monkeypatch.setattr(runner,'_loaders',lambda cfg:([batch,batch],[batch],[batch]))
    actual_step=runner.train_step_full
    seen=[]
    def audited_step(task,batch):
        assert task.reference_model is None
        loss,out,stats=actual_step(task,batch)
        main,_=BCEDiceLoss()(out['logits'],batch['mask'].to(task.device))
        torch.testing.assert_close(loss,main,atol=0,rtol=0)
        assert stats['loss_protect']==0 and stats['loss_protect_weighted']==0
        seen.append(True)
        return loss,out,stats
    monkeypatch.setattr(runner,'train_step_full',audited_step)
    runner.main()
    assert len(seen)==2
    folder=Path(cfg.checkpoint_dir)
    model,saved,checkpoint,tag=load_full_checkpoint(folder/'ckpt.best_full.pth.tar')
    assert saved.initialization_mode=='from_scratch_or_encoder_pretrained'
    assert saved.baseline_sha256 is None and not saved.protect_enabled
    assert tag=='ema' and saved.local_enabled==local


def test_builder_disabled_pretraining_never_reads_encoder_files(monkeypatch):
    from types import SimpleNamespace
    import models.dual_shared_add_baseline as baseline_module
    calls=[]
    def no_file_read(model,path,name,required):
        assert path is None and required is False
        calls.append(name)
    monkeypatch.setattr(baseline_module,'load_local_weights_safe',no_file_read)
    cfg=SimpleNamespace(local_enabled=True,ct_backbone='convnextv2_nano',pet_backbone='mit_b1',
        ct_pretrained_path='/must/not/open/ct',pet_pretrained_path='/must/not/open/pet',
        pretrained=False,decoder_channels=[512,256,128,64],use_deep_supervision=False,deep_supervision=False)
    build_full_local_model(cfg)
    assert calls==['CT_Encoder','PET_Encoder']


@pytest.mark.parametrize('local,protect', [(False,False),(False,True),(True,False),(True,True)])
def test_runner_and_checkpoint_roundtrip(monkeypatch,tmp_path,local,protect):
    from configs.full_local_reassembly import FullLocalConfig
    import run_full_local_reassembly as runner
    from eval_full_local_reassembly import load_full_checkpoint
    path,_,_ = baseline_checkpoint(tmp_path)
    monkeypatch.setattr(sys,'argv',['x','--baseline_checkpoint',str(path),
        '--pretrained','false','--require_cuda','false','--checkpoint_root',str(tmp_path),
        '--hash','local-test','--local_enabled',str(local),'--protect_enabled',str(protect),
        '--image_size_2d','64','--batch_size','2','--epochs','1','--cosine_warmup','0',
        '--mixed_precision','false','--hd95_mode','surface'])
    cfg = FullLocalConfig.parse_arguments()
    cfg.ema_start_epoch=0
    monkeypatch.setattr(FullLocalConfig,'parse_arguments',classmethod(lambda cls:cfg))
    mask=torch.zeros(2,1,64,64); mask[:,:,20:32,20:32]=1
    batch=dict(ct=torch.randn(2,3,64,64),pet=torch.randn(2,3,64,64),mask=mask)
    monkeypatch.setattr(runner,'_loaders',lambda cfg:([batch,batch],[batch],[batch]))
    runner.main()
    folder=Path(cfg.checkpoint_dir)
    result=json.loads((folder/'final_test_full.json').read_text())
    assert result['local_enabled']==local and result['weights']=='ema'
    header=(folder/'train_log.csv').read_text().splitlines()[0]
    assert 'train_protect_loss' in header and 'local_null_mass_s4' in header
    path.unlink()  # Inference must not reopen the training reference/pretrained paths.
    for choice in (None,False):
        model,saved,checkpoint,tag=load_full_checkpoint(folder/'ckpt.best_full.pth.tar',use_ema=choice)
        assert saved.local_enabled==local and saved.protect_enabled==protect
        assert tag==('raw' if choice is False else 'ema')
        assert not any('reference' in k for k in model.state_dict())
    checkpoint['config']['local_window']=5
    torch.save(checkpoint,folder/'tampered.pth')
    with pytest.raises(ValueError,match='architecture'): load_full_checkpoint(folder/'tampered.pth')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_amp():
    fusion=LocalReassemblyFusion().cuda()
    cs,ps=features(actual=True,dtype=torch.float16)
    cs=[c.detach().cuda().requires_grad_() for c in cs]
    ps=[p.detach().cuda().requires_grad_() for p in ps]
    with torch.autocast('cuda',dtype=torch.float16):
        out=fusion(cs,ps)
        sum(x.float().square().mean() for x in out).backward()
    assert all(torch.isfinite(x.grad).all() for x in cs+ps)
