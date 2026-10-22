"""V1 contracts. Real encoders for map building and existing training regression."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pytest
import torch
import cv2
import run_staged as runner
from datasets.pclt20k_seg import PCLT20KSegDataset, _records_from_ids
from models.staged_petct import StagedPETCT
from scripts.build_dual_state_maps import build
from utils.dual_state_maps import (difference_map, map_model, train_records, MapCache,
                                   sha256, atomic_json)
from utils.dual_state_augmentation import choose_window, crop_with_guidance, augment, components

torch.set_num_threads(1)


def test_four_classes_threshold_nonfinite_and_background():
    y=torch.tensor([[[[0.,0.,0.,1.,0.]]]])
    m=torch.tensor([[[[-1.,1.,-1.,-1.,0.]]]])
    f=torch.tensor([[[[-1.,-1.,1.,-1.,0.]]]])
    assert difference_map(m,f,y).tolist()==[[[0,1,2,3,0]]]
    for value in [float('nan'),float('inf'),float('-inf')]:
        bad=m.clone();bad[...,0]=value
        with pytest.raises(ValueError,match='Nonfinite'):difference_map(bad,f,y)
    with pytest.raises(ValueError,match='binary'):difference_map(m,f,y+.1)
    assert not difference_map(torch.zeros_like(y),torch.zeros_like(y),torch.zeros_like(y)).any()


@pytest.fixture(scope='module')
def cache_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('dual_state')
    ids=[f'p{i}_001' for i in range(5)]
    for i,image_id in enumerate(ids):
        directory=root/f'p{i}';directory.mkdir()
        a=np.tile(np.arange(64,dtype=np.uint8),(64,1))*3
        for suffix,array in [('CT',a),('PET',255-a),('mask',((a>80)&(a<110)).astype(np.uint8)*255)]:
            Image.fromarray(array).save(directory/f'{image_id}_{suffix}.png')
    (root/'train.txt').write_text('\n'.join(ids)+'\n')
    (root/'val.txt').write_text('forbidden_val_001\n')
    (root/'test.txt').write_text('forbidden_test_001\n')
    cfg=runner.parser().parse_args(['--stage','2','--root',str(root),'--train_split_file','train.txt',
                                  '--image_size_2d','64','--hd95_mode','surface','--joint_from_pretrained'])
    cfg.training_protocol='paired_full_missing'
    model=StagedPETCT(2).eval()
    state=model.state_dict()
    raw=dict(state);raw['decoder_ct.seg_head.bias']=state['decoder_ct.seg_head.bias']+5
    checkpoint={'format':runner.FORMAT,'config':vars(cfg),'splits':{
        key:sha256(root/f'{key}.txt') for key in ('train','val','test')},'source':{},
        'epoch':6,'eval_weights':'ema','model':raw,'model_ema':state}
    path=root/'best.pt';torch.save(checkpoint,path)
    return root,path,checkpoint


def test_map_model_selection_strict_frozen_and_no_pretraining(cache_fixture,monkeypatch):
    _,_,checkpoint=cache_fixture
    def forbidden(*a,**k):raise AssertionError('Generic pretrained loading is forbidden')
    monkeypatch.setattr('models.staged_petct.load_local_weights_safe',forbidden)
    for tag in ('raw','ema'):
        c=dict(checkpoint,eval_weights=tag)
        model,chosen=map_model(c,'cpu')
        assert chosen==tag and not model.training and not any(p.requires_grad for p in model.parameters())
        expected=c['model' if tag=='raw' else 'model_ema']['decoder_ct.seg_head.bias']
        assert torch.equal(model.decoder_ct.seg_head.bias,expected)
        before={k:v.clone() for k,v in model.named_buffers()}
        with torch.inference_mode():model(torch.randn(2,3,64,64),torch.randn(2,3,64,64),mode='full')
        assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
        assert all(p.grad is None for p in model.parameters())
    for bad in [dict(checkpoint,eval_weights=None),dict(checkpoint,model_ema=None)]:
        with pytest.raises(ValueError):map_model(bad,'cpu')
    bad=dict(checkpoint,model_ema=dict(checkpoint['model_ema']))
    bad['model_ema'].pop('decoder_ct.seg_head.bias')
    with pytest.raises(RuntimeError):map_model(bad,'cpu')


def test_build_all_train_tail_and_cache_identity(cache_fixture,monkeypatch):
    root,path,checkpoint=cache_fixture
    import datasets.pclt20k_seg as data
    original=data._imread_grayscale
    def checked(path):
        assert 'forbidden_' not in str(path)
        return original(path)
    monkeypatch.setattr(data,'_imread_grayscale',checked)
    args=argparse.Namespace(checkpoint=str(path),output=str(root/'maps'),root=None,
        device='cpu',batch_size=2,num_workers=0,log_interval=1,limit=None,content_hash=True)
    m=build(args)
    assert m['count']==5 and m['complete'] and not m['partial'] and m['buffers_unchanged']
    assert m['source']['weights']=='ema' and m['source']['sha256']==sha256(path)
    cfg=checkpoint['config']
    cache=MapCache(root/'maps',cfg)
    assert cache.read('p4_001').shape==(64,64)
    with pytest.raises(FileExistsError):build(args)
    with pytest.raises(ValueError,match='identity'):MapCache(root/'maps',cfg,expected_identity={})
    for key,value in [('norm_mode','cipa'),('image_size_2d',128)]:
        with pytest.raises(ValueError,match='preprocessing'):MapCache(root/'maps',dict(cfg,**{key:value}))
    for change,match in [({'partial':True},'partial'),({'complete':False},'Incomplete'),
                          ({'image_ids':['oops']},'identity')]:
        modified=dict(m,**change);atomic_json(root/'maps'/'manifest.json',modified)
        with pytest.raises(ValueError,match=match):MapCache(root/'maps',cfg)
    atomic_json(root/'maps'/'manifest.json',m)
    image_path=root/'p0'/'p0_001_CT.png'
    stat=image_path.stat()
    import os
    os.utime(image_path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
    with pytest.raises(ValueError,match='fingerprint'):MapCache(root/'maps',cfg)
    os.utime(image_path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    split=root/'train.txt';saved=split.read_bytes();split.write_bytes(saved+b'\n')
    with pytest.raises(ValueError,match='SHA256'):MapCache(root/'maps',cfg)
    split.write_bytes(saved)
    changed=m['samples']['p0_001']['map_sha256'];m['samples']['p0_001']['map_sha256']='wrong'
    atomic_json(root/'maps'/'manifest.json',m)
    with pytest.raises(ValueError,match='hash'):MapCache(root/'maps',cfg)
    m['samples']['p0_001']['map_sha256']=changed;atomic_json(root/'maps'/'manifest.json',m)
    from utils.dual_state_maps import checked_ids
    for ids in [['../x'],['a','a'],['/absolute']]:
        with pytest.raises(ValueError):checked_ids(ids)
    partial=copy.copy(args);partial.output=str(root/'partial');partial.limit=3
    assert build(partial)['count']==3
    with pytest.raises(ValueError,match='partial'):MapCache(root/'partial',cfg)


def oracle(name):
    path=Path(__file__).parent/'oracles'/f'baseline_{name}.py'
    spec=importlib.util.spec_from_file_location('baseline_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_none_and_probability_zero_exact_baseline_rng(cache_fixture):
    root,_,_=cache_fixture
    old_aug=oracle('image_augmentation');old_data=oracle('pclt20k_seg')
    # The immutable original dataset must use the immutable original transforms.
    for name in ('randomShiftScaleRotate','randomHorizontalFlip','randomcrop'):
        setattr(old_data,name,getattr(old_aug,name))
    records=_records_from_ids(str(root),['p0_001','p1_001'])
    baseline=old_data.PCLT20KSegDataset(records,image_size=64,train=True)
    class Forbidden:
        def read(self,*a):raise AssertionError('none may not read map')
    off=PCLT20KSegDataset(records,image_size=64,train=True,guidance_cache=Forbidden())
    zero=PCLT20KSegDataset(records,image_size=64,train=True,crop_guidance='foreground',guided_crop_prob=0)
    for seed in range(24):
        np.random.seed(seed);expected=baseline[seed%2];rng=np.random.get_state()
        for dataset in (off,zero):
            np.random.seed(seed);actual=dataset[seed%2];after=np.random.get_state()
            for key in ('ct','pet','mask'):assert torch.equal(actual[key],expected[key]),(seed,key)
            assert rng[0]==after[0] and np.array_equal(rng[1],after[1]) and rng[2:]==after[2:]
            assert set(actual['mask'].unique().tolist())<={0.,1.}


def test_shared_geometry_nearest_and_pairing(monkeypatch):
    from utils.image_augmentation import randomShiftScaleRotate,randomHorizontalFlip
    side=64
    ct=np.arange(side*side,dtype=np.float32).reshape(side,side)%255
    image=np.stack([ct,ct],-1)
    d=np.zeros((side,side),np.uint8);d[16:25,20:29]=2
    calls=[];warp=cv2.warpPerspective
    def tracked(a,mat,size,**kw):
        calls.append((mat.copy(),kw['flags'],kw['borderValue']))
        return warp(a,mat,size,**kw)
    monkeypatch.setattr(cv2,'warpPerspective',tracked)
    np.random.seed(12)
    im,mask,dd=randomShiftScaleRotate(image,d*80,u=1,auxiliary=d)
    assert np.array_equal(im[...,0],im[...,1])
    assert len(calls)==3 and all(np.array_equal(calls[0][0],c[0]) for c in calls)
    assert calls[-1][1]==cv2.INTER_NEAREST and calls[-1][2]==0
    assert set(np.unique(dd))<={0,2}
    flipped=randomHorizontalFlip(im,mask,u=1,auxiliary=dd)
    assert np.array_equal(flipped[2],dd[:,::-1])
    cropped,gt,d_after,stats,_=crop_with_guidance(flipped[0],flipped[1],flipped[2],'dual_state',1,u=1)
    assert np.array_equal(cropped[...,0],cropped[...,1]) and cropped.shape==image.shape
    assert set(np.unique(d_after))<={0,2}


def test_regions_fairness_jitter_boundaries_and_fallback():
    diagonal=np.zeros((3,3),np.uint8);diagonal[0,0]=1;diagonal[1,1]=1
    assert components(diagonal)[0]==2  # Explicit 8-connectivity.
    mask=np.zeros((100,100),np.uint8)
    d=np.zeros_like(mask);d[20:22,20:22]=1;d[65:75,65:75]=2
    chosen=[];windows=[]
    np.random.seed(2023)
    for _ in range(600):
        window,region,info=choose_window(mask,d,'dual_state',80)
        chosen.append(info['selection']);windows.append(window)
        x,y,w,h=window
        assert 0<=x<=20 and 0<=y<=20 and region[y:y+h,x:x+w].sum()==region.sum()
    assert .43<chosen.count('1')/len(chosen)<.57
    assert len(set(windows))>20  # Not fixed-centered.
    d[d==2]=0
    assert choose_window(mask,d,'dual_state',80)[2]['selection']=='1'
    d[:]=3
    assert choose_window(mask,d,'dual_state',80)[2]['fallback']=='no_candidate'
    assert choose_window(mask,d,'hard',80)[2]['fallback']=='target_too_large'
    d[:]=0;d[0,0]=1
    window,_,_=choose_window(mask,d,'dual_state',80)
    assert window[:2]==(0,0)
    mask[:]=255
    assert choose_window(mask,d,'dual_state',80)[2]['fallback']=='target_too_large'
    image=np.zeros((100,100,2),np.uint8)
    _,_,_,stats,_=crop_with_guidance(image,mask,d,'dual_state',1,u=1)
    assert stats['guided_success']==0 and stats['random_crop']==1 and stats['fallback_target_too_large']==1
    empty=np.zeros_like(d)
    _,_,_,stats,_=crop_with_guidance(image,empty,empty,'dual_state',1,u=1)
    assert stats['fallback_no_candidate']==1 and stats['guided_success']==0
    # Equal component chance despite a 100x pixel-area difference.
    d=empty.copy();d[10,10]=1;d[50:60,50:60]=1
    areas=[choose_window(empty,d,'dual_state',80)[2]['target_area_sum'] for _ in range(500)]
    assert .4<areas.count(1)/500<.6


def test_worker_counts_are_returned_and_aggregatable(cache_fixture):
    root,_,_=cache_fixture
    records=_records_from_ids(str(root),[f'p{i}_001' for i in range(5)])
    ds=PCLT20KSegDataset(records,image_size=64,train=True,crop_guidance='foreground')
    loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=2,drop_last=False)
    total=0
    for batch in loader:
        total+=int(batch['augmentation_stats']['samples'].sum())
        assert set(batch['mask'].unique().tolist())<={0.,1.}
    assert total==5


def test_candidate_vanishes_under_shared_geometry(monkeypatch):
    import utils.dual_state_augmentation as aug
    from utils.image_augmentation import randomShiftScaleRotate
    real_crop=aug.crop_with_guidance
    def outside(image,mask,auxiliary):
        return randomShiftScaleRotate(image,mask,auxiliary=auxiliary,u=1,
            shift_limit=(2,2),rotate_limit=(0,0),scale_limit=(0,0),aspect_limit=(0,0))
    monkeypatch.setattr(aug,'randomShiftScaleRotate',outside)
    monkeypatch.setattr(aug,'crop_with_guidance',lambda *a,**k:real_crop(*a,**k,u=1))
    mask=np.zeros((64,64),np.uint8);d=mask.copy();d[20:24,20:24]=1
    image=np.stack([d,d],-1)
    out,gt,stats=aug.augment(image,mask,d,'dual_state',1)
    assert stats['fallback_vanished']==1 and stats['guided_success']==0 and stats['random_crop']==1
    assert not out.any() and not gt.any()


def test_config_compatibility_and_incompatible_modes():
    cfg=runner.parser().parse_args(['--stage','2','--output','unused','--joint_from_pretrained'])
    runner.validate(cfg)
    assert cfg.crop_guidance=='none' and cfg.early_stopping_patience==5
    cfg.crop_guidance='dual_state'
    with pytest.raises(ValueError,match='guidance_map_dir'):runner.validate(cfg)
    with pytest.raises(ValueError):PCLT20KSegDataset([],train=False,crop_guidance='foreground')
    with pytest.raises(ValueError):PCLT20KSegDataset([],train=True,ct_only=True,crop_guidance='foreground')
    assert not runner.patience_exhausted(15,6,10)
    assert runner.patience_exhausted(16,6,10)
    assert not runner.patience_exhausted(16,16,10)
    assert not runner.patience_exhausted(60,1,0)
    assert not runner.patience_exhausted(10,6,5)
    assert runner.patience_exhausted(11,6,5)
    assert not runner.patience_exhausted(11,11,5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_legacy_cuda_amp_overflow_and_effective_update():
    from scripts.check_staged import preflight_update
    from utils.amp import make_grad_scaler
    from utils.ema import ModelEMA
    device=torch.device('cuda')
    model=StagedPETCT(2).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
    ema=ModelEMA(model,decay=.999,device=device)
    batch={'ct':torch.randn(2,3,64,64,device=device),'pet':torch.randn(2,3,64,64,device=device),
           'mask':(torch.rand(2,1,64,64,device=device)>.97).float(),
           'is_full':torch.ones(2,dtype=torch.bool,device=device)}
    parameter=next(model.enc_ct.parameters());original=parameter.detach().clone();attempts=[]
    def overflow_once(gradient):
        attempts.append(1)
        return torch.full_like(gradient,float('inf')) if len(attempts)==1 else gradient
    handle=parameter.register_hook(overflow_once)
    _,stats,grads,report=preflight_update(model,batch,2,device,optimizer,make_grad_scaler(device,True),ema)
    handle.remove()
    assert report['skipped_updates']>=1 and report['attempts']==report['skipped_updates']+1
    assert ema.updates==1 and not torch.equal(original,parameter)
    assert stats['ct_samples']==stats['full_samples']==stats['missing_samples']==2
    assert all(np.isfinite(v) and v>0 for v in grads.values())
    optimizer.zero_grad(set_to_none=True)
    handle=parameter.register_hook(lambda g:torch.full_like(g,float('inf')))
    with pytest.raises(RuntimeError,match='No finite optimizer update'):
        preflight_update(model,batch,2,device,optimizer,make_grad_scaler(device,True),ema,max_attempts=2)
    assert ema.updates==1
    handle.remove()
    torch.cuda.empty_cache()


def test_augmented_checkpoint_resume_and_evaluate_without_maps(tmp_path,monkeypatch):
    from test_staged import make_pngs
    root=tmp_path/'data';root.mkdir();make_pngs(root)
    cfg=runner.parser().parse_args(['--stage','2','--root',str(root),'--image_size_2d','64',
                                  '--hd95_mode','surface','--joint_from_pretrained'])
    cfg.training_protocol='paired_full_missing'
    model=StagedPETCT(2)
    path=tmp_path/'map_teacher.pt'
    torch.save({'format':runner.FORMAT,'config':vars(cfg),
                'splits':{name:sha256(root/getattr(cfg,name+'_split_file')) for name in ('train','val','test')},
                'source':{},'epoch':0,'eval_weights':'raw','model':model.state_dict()},path)
    del model
    maps=tmp_path/'maps'
    build(argparse.Namespace(checkpoint=str(path),output=str(maps),root=None,device='cpu',
          batch_size=3,num_workers=0,log_interval=50,limit=None,content_hash=False))
    common=['--save_last_checkpoint','--stage','2','--joint_from_pretrained','--no_pretrained','--root',str(root),
            '--image_size_2d','64','--hd95_mode','surface','--batch_size','2','--num_workers','0',
            '--epochs','2','--cosine_warmup','0','--ema_start_epoch','0','--device','cpu',
            '--crop_guidance','dual_state','--guidance_map_dir',str(maps),'--guided_crop_prob','1',
            '--guidance_region_sampling','sqrt_area']
    real_load=runner.load
    def guarded_load(file):
        assert Path(file).resolve()!=path.resolve(), 'New model loaded map teacher task weights'
        return real_load(file)
    monkeypatch.setattr(runner,'load',guarded_load)
    reference=tmp_path/'reference'
    result=runner.main([*common,'--output',str(reference)])
    expected=real_load(reference/'last.pt')
    interrupted=tmp_path/'interrupted';real_save=runner.atomic_save
    class Stop(Exception):pass
    def interrupt(payload,file):
        real_save(payload,file)
        if Path(file).name=='last.pt' and payload['epoch']==1:raise Stop()
    monkeypatch.setattr(runner,'atomic_save',interrupt)
    with pytest.raises(Stop):runner.main([*common,'--output',str(interrupted)])
    monkeypatch.setattr(runner,'atomic_save',real_save)
    # Resume rejects a changed cache identity even if all files still parse.
    manifest=json.loads((maps/'manifest.json').read_text());modified=dict(manifest,extra='tampered')
    atomic_json(maps/'manifest.json',modified)
    with pytest.raises(ValueError,match='identity'):runner.main(['--resume',str(interrupted/'last.pt'),'--device','cpu'])
    atomic_json(maps/'manifest.json',manifest)
    runner.main(['--resume',str(interrupted/'last.pt'),'--device','cpu'])
    actual=real_load(interrupted/'last.pt')
    assert actual['config']['guidance_region_sampling']=='sqrt_area'
    for group in ('model','model_ema'):
        for key,value in expected[group].items():assert torch.equal(value,actual[group][key]),key
    rows=[json.loads(line) for line in (interrupted/'augmentation.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==[1,2] and all(r['samples']==4 for r in rows)
    assert rows==[json.loads(line) for line in (reference/'augmentation.jsonl').read_text().splitlines()]
    with pytest.raises(ValueError,match='unsupported overrides'):
        runner.main(['--resume',str(interrupted/'last.pt'),'--crop_guidance','hard'])
    with pytest.raises(ValueError,match='unsupported overrides'):
        runner.main(['--resume',str(interrupted/'last.pt'),'--guidance_region_sampling','uniform'])
    maps.rename(tmp_path/'maps-unavailable')
    for pet in root.rglob('*_PET.png'):pet.unlink()
    missing=runner.main(['--evaluate',str(reference/'best.pt'),'--eval_mode','missing','--device','cpu'])
    assert missing['metrics']['missing']==result['metrics']['missing']
