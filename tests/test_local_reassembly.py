"""Contracts for pretrained encoder initialization and fusion-only ablation."""
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


@pytest.mark.parametrize('enabled',[True,False])
def test_random_model_matches_random_add_and_never_loads_weights(monkeypatch,enabled):
    import models.dual_shared_add_baseline as baseline_module
    torch.manual_seed(2023)
    base=DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng=torch.get_rng_state()
    def forbidden(*args,**kwargs): raise AssertionError('Initialization must not read weights')
    monkeypatch.setattr(torch,'load',forbidden)
    torch.manual_seed(2023)
    model=FullLocalReassemblyPETCT(local_enabled=enabled,pretrained=False,
        ct_pretrained_path='must-not-read',pet_pretrained_path='must-not-read').eval()
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in base.state_dict().items(): assert torch.equal(value,model.state_dict()[key]),key
    assert not any('reference' in key or 'teacher' in key for key in model.state_dict())
    ct,pet=torch.randn(2,3,64,64),torch.randn(2,3,64,64)
    with torch.no_grad():
        torch.testing.assert_close(model(ct,pet,forward_mode='full')['logits'],
                                   base(ct,pet,forward_mode='full')['logits'],atol=0,rtol=0)


def test_pretraining_loads_encoders_only_and_preserves_add_initialization(monkeypatch):
    import models.dual_shared_add_baseline as baseline_module
    calls=[]
    def load_encoder(module,path,name,required):
        calls.append((name,path,required))
        assert name in ('CT_Encoder','PET_Encoder') and required is True
        with torch.no_grad():
            next(module.parameters()).fill_(.0123)
    monkeypatch.setattr(baseline_module,'load_local_weights_safe',load_encoder)
    kwargs=dict(pretrained=True,ct_pretrained_path='ct-source',pet_pretrained_path='pet-source')
    torch.manual_seed(2023)
    base=DualSharedAddPETCTBaseline(**kwargs).eval()
    rng=torch.get_rng_state()
    torch.manual_seed(2023)
    model=FullLocalReassemblyPETCT(**kwargs).eval()
    assert calls==[('CT_Encoder','ct-source',True),('PET_Encoder','pet-source',True)]*2
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in base.state_dict().items(): assert torch.equal(value,model.state_dict()[key]),key
    ct,pet=torch.randn(2,3,64,64),torch.randn(2,3,64,64)
    with torch.no_grad():
        torch.testing.assert_close(model(ct,pet,forward_mode='full')['logits'],
                                   base(ct,pet,forward_mode='full')['logits'],atol=0,rtol=0)


def test_actual_encoder_files_load_and_missing_sources_fail(tmp_path):
    model=FullLocalReassemblyPETCT(pretrained=False)
    ct_path=tmp_path/'ct.pth'; pet_path=tmp_path/'pet.pth'
    torch.save(model.enc_ct.state_dict(),ct_path)
    torch.save(model.enc_pet.state_dict(),pet_path)
    other=FullLocalReassemblyPETCT(pretrained=True,ct_pretrained_path=str(ct_path),
                                  pet_pretrained_path=str(pet_path))
    for name in ('enc_ct','enc_pet'):
        for key,value in getattr(model,name).state_dict().items():
            torch.testing.assert_close(value,getattr(other,name).state_dict()[key],atol=0,rtol=0)
    with pytest.raises(ValueError,match='pretrained path'):
        FullLocalReassemblyPETCT(pretrained=True)
    with pytest.raises(FileNotFoundError):
        FullLocalReassemblyPETCT(pretrained=True,ct_pretrained_path=str(tmp_path/'absent'))


def test_one_encoder_pass_one_decoder_pass_and_no_add_output():
    model=FullLocalReassemblyPETCT(pretrained=False).train()
    calls=dict(ct=0,pet=0,decoder=0)
    def count(name):
        def hook(*args): calls[name]+=1
        return hook
    hooks=[model.enc_ct.register_forward_hook(count('ct')),
           model.enc_pet.register_forward_hook(count('pet')),
           model.decoder.register_forward_hook(count('decoder'))]
    ct,pet=torch.randn(2,3,64,64),torch.randn(2,3,64,64)
    out=model(ct,pet,forward_mode='full')
    assert calls==dict(ct=1,pet=1,decoder=1)
    assert out['logits'].requires_grad and 'add_logits' not in out
    for hook in hooks: hook.remove()
    with pytest.raises(TypeError): model(ct,pet,forward_mode='full',return_add_logits=True)


def test_missing_and_auto_bypass_pet_and_fusion():
    model=FullLocalReassemblyPETCT(pretrained=False).eval()
    ct,pet=torch.randn(2,3,64,64),torch.randn(2,3,64,64)
    def forbidden(*args): raise AssertionError('Missing cannot call PET/fusion')
    hooks=[m.register_forward_pre_hook(forbidden) for m in (model.enc_pet,model.fusion)]
    with torch.no_grad():
        a=model(ct,pet=None,forward_mode='missing')['logits']
        b=model(ct,pet=torch.full_like(pet,float('nan')),pet_available=[0,0],forward_mode='auto')['logits']
    for hook in hooks: hook.remove()
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    with torch.no_grad():
        a=model(ct,pet,pet_available=[1,0],forward_mode='auto')['logits']
        pet[1].fill_(float('nan'))
        b=model(ct,pet,pet_available=[1,0],forward_mode='auto')['logits']
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_config_defaults_match_encoder_pretrained_baseline_and_remove_correction(monkeypatch,tmp_path):
    from configs.full_local_reassembly import FullLocalConfig
    args=['x','--require_cuda','false','--checkpoint_root',str(tmp_path),'--hash','pretrained']
    monkeypatch.setattr(sys,'argv',args)
    cfg=FullLocalConfig.parse_arguments()
    assert cfg.pretrained is True and cfg.initialization_mode=='encoder_pretrained_fresh_segmentation'
    assert cfg.loss_method=='BCE+Dice' and cfg.learning_rate==4e-5
    assert cfg.ct_pretrained_path.endswith('pretrained/convnextv2_nano')
    assert cfg.pet_pretrained_path.endswith('pretrained/mit-b1')
    for name in ('baseline_checkpoint','reference_checkpoint','protect_enabled','protect_weight'):
        assert not hasattr(cfg,name)
    (Path(cfg.checkpoint_dir)/'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError): FullLocalConfig.parse_arguments()
    for flag in ('--baseline_checkpoint','--reference_checkpoint','--protect_enabled','--protect_weight'):
        monkeypatch.setattr(sys,'argv',args+[flag,'anything'])
        with pytest.raises(SystemExit): FullLocalConfig.parse_arguments()


@pytest.mark.parametrize('local,pretrained',[(True,True),(False,False)])
def test_single_model_runner_main_loss_and_checkpoint_without_encoder_files(monkeypatch,tmp_path,local,pretrained):
    from configs.full_local_reassembly import FullLocalConfig
    from eval_full_local_reassembly import load_full_checkpoint
    from utils.seg_losses import BCEDiceLoss
    import models.dual_shared_add_baseline as baseline_module
    import run_full_local_reassembly as runner
    monkeypatch.setattr(sys,'argv',['x','--require_cuda','false','--checkpoint_root',str(tmp_path),
        '--hash','single','--local_enabled',str(local),'--pretrained',str(pretrained),
        '--image_size_2d','64','--batch_size','2','--epochs','1','--cosine_warmup','0',
        '--mixed_precision','false','--hd95_mode','surface'])
    cfg=FullLocalConfig.parse_arguments(); cfg.ema_start_epoch=0
    monkeypatch.setattr(FullLocalConfig,'parse_arguments',classmethod(lambda cls:cfg))
    # Source weights are simulated; actual encoder files live on the training server.
    sources=[]
    def load_encoder(module,path,name,required):
        if required:
            sources.append(name)
            with torch.no_grad(): next(module.parameters()).fill_(.0123)
        else:
            assert path is None  # checkpoint rebuild must never touch encoder source files
    monkeypatch.setattr(baseline_module,'load_local_weights_safe',load_encoder)
    mask=torch.zeros(2,1,64,64); mask[:,:,20:32,20:32]=1
    batch=dict(ct=torch.randn(2,3,64,64),pet=torch.randn(2,3,64,64),mask=mask)
    monkeypatch.setattr(runner,'_loaders',lambda cfg:([batch,batch],[batch],[batch]))
    actual=runner.train_step_full
    def audited(task,batch):
        assert not hasattr(task,'reference_model')
        calls=[]
        hook=task.model.decoder.register_forward_hook(lambda *args:calls.append(1))
        loss,out,stats=actual(task,batch)
        hook.remove()
        assert calls==[1] and 'add_logits' not in out
        assert not any('protect' in key for key in stats)
        main,_=BCEDiceLoss()(out['logits'],batch['mask'].to(task.device))
        torch.testing.assert_close(loss,main,atol=0,rtol=0)
        return loss,out,stats
    monkeypatch.setattr(runner,'train_step_full',audited)
    runner.main()
    assert sources==(['CT_Encoder','PET_Encoder'] if pretrained else [])
    folder=Path(cfg.checkpoint_dir)
    for choice in (None,False):
        model,saved,checkpoint,tag=load_full_checkpoint(folder/'ckpt.best_full.pth.tar',use_ema=choice)
        assert saved.pretrained==pretrained and saved.local_enabled==local
        assert saved.loss_method=='BCE+Dice'
        assert tag==('ema' if choice is None else 'raw')
    assert sources==(['CT_Encoder','PET_Encoder'] if pretrained else [])
    result=json.loads((folder/'final_test_full.json').read_text())
    assert result['pretrained']==pretrained and result['loss_method']=='BCE+Dice'
    assert 'protect' not in (folder/'train_log.csv').read_text().splitlines()[0]
    checkpoint['config']['local_schema_version']=2
    torch.save(checkpoint,folder/'old_protocol.pth')
    with pytest.raises(ValueError,match='architecture'): load_full_checkpoint(folder/'old_protocol.pth')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_amp_single_model_two_steps():
    from utils.seg_losses import BCEDiceLoss
    model=FullLocalReassemblyPETCT(pretrained=False).cuda().train()
    ct,pet=torch.randn(2,3,64,64,device='cuda'),torch.randn(2,3,64,64,device='cuda')
    mask=torch.zeros(2,1,64,64,device='cuda'); mask[:,:,20:32,20:32]=1
    optimizer=torch.optim.AdamW(model.parameters(),lr=4e-5)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            out=model(ct,pet,forward_mode='full')
            loss,_=BCEDiceLoss()(out['logits'],mask)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        scaler.step(optimizer); scaler.update()
