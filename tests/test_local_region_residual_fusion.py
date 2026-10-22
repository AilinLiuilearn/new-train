import copy
import gc
import io
import pytest
import torch
from models.components.local_region_residual_fusion import (
    LocalRegionResidualFusion, MultiScaleLocalRegionResidualFusion,
    GaussianScaleContrast, window_partition, window_reverse, broadcast_regions)
from models.staged_petct import StagedPETCT
from utils.ema import ModelEMA
import run_staged as runner

torch.set_num_threads(1)


def test_rectangular_coordinates_and_broadcast():
    x = torch.arange(20*28).reshape(1, 1, 20, 28).float()
    t = window_partition(x)
    torch.testing.assert_close(t[1, :, 0], x[0, 0, :4, 4:8].flatten(), atol=0, rtol=0)
    torch.testing.assert_close(window_reverse(t, (1, 20, 28)), x, atol=0, rtol=0)
    r = torch.arange(35).reshape(1, 1, 5, 7).float()
    y = broadcast_regions(r)
    for i in range(5):
        for j in range(7):
            assert (y[0, 0, i*4:(i+1)*4, j*4:(j+1)*4] == r[0, 0, i, j]).all()
    with pytest.raises(ValueError, match='divisible'):
        LocalRegionResidualFusion(16)(torch.randn(2,16,19,28), torch.randn(2,16,19,28))


def test_gaussian_fixed_signed_normalized_and_impulse():
    g = GaussianScaleContrast(16)
    assert all('kernel' not in n for n, _ in g.named_parameters())
    for name, k in g.named_buffers():
        torch.testing.assert_close(k.sum(), torch.tensor(1.), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(k, k.flip(0), atol=0, rtol=0)
    for d in g.differences(torch.ones(2,16,20,28)):
        assert d.abs().max() < 3e-7
    impulse = torch.zeros(1,16,20,28); impulse[:,:,10,14] = 1
    for d in g.differences(impulse):
        assert d[0,0].argmax() == 10*28+14
        assert d.min() < 0 and d.max() > 0
    with pytest.raises(ValueError, match='reflect'):
        g(torch.ones(2,16,4,4))


@pytest.mark.parametrize('use_gaussian', [True, False])
def test_zero_equivalence_inputs_gradients_roundtrip(use_gaussian):
    torch.manual_seed(4)
    model = LocalRegionResidualFusion(16, use_gaussian=use_gaussian)
    C = torch.randn(2,16,20,28, requires_grad=True)
    P = torch.randn_like(C, requires_grad=True)
    before = [x.detach().clone() for x in (C,P)]
    out = model(C,P)
    torch.testing.assert_close(out, C+P, atol=1e-6, rtol=1e-5)
    for x, b in zip((C,P),before): torch.testing.assert_close(x,b,atol=0,rtol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    out.square().mean().backward()
    for branch in (model.out_ct, model.out_pet):
        assert branch.weight.grad.isfinite().all() and branch.weight.grad.abs().sum() > 0
    assert model.stem.inner_convs[0].weight.grad.abs().sum() == 0
    assert C.grad.abs().sum() > 0 and P.grad.abs().sum() > 0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model(C,P).square().mean().backward()
    for branch in [model.stem.inner_convs, model.stem.outer_convs, model.local_block,
                   model.region_block] + ([model.contrast.project] if use_gaussian else []):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    model.eval(); expected = model(C,P)
    clone = LocalRegionResidualFusion(16, use_gaussian=use_gaussian).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(C,P),expected,atol=0,rtol=0)


def test_multiscale_alignment_and_validation():
    m = MultiScaleLocalRegionResidualFusion(channels=(16,16,16,16)).eval()
    cs = [torch.randn(1,16,h,w) for h,w in [(32,40),(24,32),(20,28),(16,20)]]
    ps = [torch.randn_like(c) for c in cs]
    for y,c,p in zip(m(cs,ps),cs,ps):
        assert y.dtype == c.dtype and y.device == c.device and y.shape == c.shape
        torch.testing.assert_close(y,c+p,atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError,match='four'): m(cs[:3],ps)
    with pytest.raises(ValueError,match='batch'): m(cs,[p[:,:,:,:].repeat(2,1,1,1) for p in ps])
    with pytest.raises(ValueError,match='dtype'): m(cs,[p.double() for p in ps])
    smaller = [p[:,:,::2,::2] for p in ps]
    for y,c,p in zip(m(cs,smaller),cs,smaller):
        torch.testing.assert_close(y,c+torch.nn.functional.interpolate(p,size=c.shape[-2:],mode='bilinear',align_corners=False))


def test_real_model_initialization_missing_ct_once_optimizer_ema_stage1():
    torch.manual_seed(31); add = StagedPETCT(2).eval()
    torch.manual_seed(31); new = StagedPETCT(2, fusion_type='local_region_residual').eval()
    common = add.state_dict()
    for k,v in common.items(): torch.testing.assert_close(new.state_dict()[k],v,atol=0,rtol=0)
    result = new.load_state_dict(common, strict=False)
    assert not result.unexpected_keys and result.missing_keys
    assert all(k.startswith('fusion.') for k in result.missing_keys)
    c=torch.randn(1,3,256,256); p=torch.randn_like(c)
    counts=[]; decoded=[]; shapes=[]
    handles=[new.enc_ct.register_forward_hook(lambda *a: counts.append(1)),
             new.decoder_ct.register_forward_pre_hook(lambda m,args: decoded.append(args[0])),
             new.fusion.register_forward_pre_hook(lambda m,args: shapes.append([list(t.shape) for t in args[0]]))]
    with torch.no_grad():
        a=add(c,p,mode='full'); b=new(c,p,mode='full')
    assert counts == [1] and len(decoded)==1
    assert shapes[0] == [[1,64,64,64],[1,128,32,32],[1,320,16,16],[1,512,8,8]]
    for k in a: torch.testing.assert_close(a[k],b[k],atol=1e-6,rtol=1e-5)
    for h in handles: h.remove()
    before={k:v.clone() for k,v in new.fusion.named_buffers()}
    def forbidden(*args): raise AssertionError('Missing touched Full branch')
    handles=[m.register_forward_pre_hook(forbidden) for m in (new.enc_pet,new.pet_align,new.fusion,new.decoder_full)]
    new.train()
    with torch.no_grad(): z=new(c,p,mode='missing')
    assert z['logits'].shape == (1,1,256,256)
    for k,v in new.fusion.named_buffers(): torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    for h in handles: h.remove()
    optimizer=torch.optim.AdamW(new.parameters(),lr=8e-5)
    ids={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in new.fusion.parameters())
    assert not any(id(b) in ids for b in new.fusion.buffers())
    ema=ModelEMA(new)
    ema.update(new)
    assert set(ema.model.state_dict())==set(new.state_dict())
    for source in [new,ema.model]:
        new.load_state_dict(source.state_dict(),strict=True)
    del add,common,ema,optimizer; gc.collect()
    stage1=StagedPETCT(1)
    new.import_stage1_state(stage1.state_dict())


def test_serialized_configuration_and_legacy_defaults():
    import argparse
    cfg=runner.parser().parse_args(['--stage','2','--output','unused','--fusion_type','local_region_residual','--no-fusion_gaussian'])
    runner.normalize_fusion_config(cfg)
    assert not cfg.fusion_config['use_gaussian']
    cfg.fusion_gaussian=True
    runner.normalize_fusion_config(cfg)
    assert not cfg.fusion_config['use_gaussian']
    legacy=argparse.Namespace(); runner.normalize_fusion_config(legacy)
    assert legacy.fusion_type=='add'
    m=StagedPETCT(2,**runner.fusion_kwargs(cfg))
    clone=StagedPETCT(2,**runner.fusion_kwargs(cfg))
    clone.load_state_dict(m.state_dict(),strict=True)


def test_real_runner_fusion_resume_and_evaluate(tmp_path, monkeypatch):
    from test_staged import make_pngs
    root=tmp_path/'data'; root.mkdir(); make_pngs(root,train=2)
    output=tmp_path/'joint'
    argv=['--stage','2','--joint_from_pretrained','--no_pretrained','--fusion_type','local_region_residual',
          '--root',str(root),'--output',str(output),'--image_size_2d','256','--batch_size','2',
          '--num_workers','0','--epochs','2','--cosine_warmup','0','--ema_start_epoch','0',
          '--device','cpu','--aug_mode','none','--hd95_mode','surface','--save_last_checkpoint']
    original=runner.atomic_save
    class Stop(Exception): pass
    def stop_at_epoch1(value,path):
        original(value,path)
        if str(path).endswith('last.pt') and value['epoch']==1: raise Stop()
    monkeypatch.setattr(runner,'atomic_save',stop_at_epoch1)
    with pytest.raises(Stop): runner.main(argv)
    monkeypatch.setattr(runner,'atomic_save',original)
    cfg=runner.load(output/'last.pt')['config']
    assert cfg['fusion_type']=='local_region_residual' and cfg['fusion_config']['window_size']==4
    result=runner.main(['--resume',str(output/'last.pt'),'--device','cpu'])
    ckpt=runner.load(output/'last.pt')
    assert ckpt['epoch']==2 and ckpt['global_step']==2
    assert 'fusion.blocks.0.out_ct.weight' in ckpt['model']
    assert ckpt['model']['fusion.blocks.0.out_ct.weight'].abs().sum()>0
    assert 'fusion.blocks.3.contrast.kernel_3' in ckpt['model_ema']
    assert runner.main(['--evaluate',str(output/'best.pt'),'--device','cpu'])==result


def test_frozen_launcher_preserves_protocol():
    from scripts.train_local_region import training_argv
    cfg=runner.parser().parse_args(training_argv('unused',use_gaussian=False))
    runner.normalize_fusion_config(cfg); runner.validate(cfg)
    assert cfg.batch_size==16 and cfg.image_size_2d==512 and cfg.learning_rate==8e-5
    assert cfg.decoder_norm=='GN' and cfg.epochs==60 and cfg.early_stopping_patience==5
    assert cfg.joint_from_pretrained and not cfg.save_last_checkpoint
    assert cfg.copy_paste=='none' and cfg.crop_guidance=='none'
    assert not cfg.fusion_config['use_gaussian']
