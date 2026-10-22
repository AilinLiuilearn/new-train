"""V2 changes only within-class component probabilities, retaining v1 defaults."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
import run_staged as runner
from utils.dual_state_augmentation import choose_window, augment
from datasets.pclt20k_seg import PCLT20KSegDataset


def test_square_root_distribution_and_class_balance():
    mask=np.zeros((100,100),np.uint8)
    d=mask.copy();d[10,10]=1;d[50:60,50:60]=1
    np.random.seed(20)
    small=0
    for _ in range(2400):
        window,region,info=choose_window(mask,d,'dual_state',80,region_sampling='sqrt_area')
        small+=info['target_area_sum']==1
        x,y,w,h=window
        assert region[y:y+h,x:x+w].sum()==region.sum()
    # sqrt(1):sqrt(100) -> 1:10, not uniform 1:1 or area-linear 1:100.
    assert .07 < small/2400 < .115
    d[50:60,50:60]=2
    selections=[choose_window(mask,d,'dual_state',80,'sqrt_area')[2]['selection'] for _ in range(1600)]
    assert .46 < selections.count('1')/1600 < .54
    d[:]=0
    assert choose_window(mask,d,'dual_state',80,'sqrt_area')[2]['fallback']=='no_candidate'
    d[:]=3
    assert choose_window(mask,d,'hard',80,'sqrt_area')[2]['fallback']=='target_too_large'


def test_uniform_matches_frozen_v1_and_zero_probability_rng():
    path=Path(__file__).parent/'oracles/v1_dual_state_augmentation.py'
    spec=importlib.util.spec_from_file_location('frozen_v1',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    image=np.zeros((100,100,2),np.uint8);image[...,0]=np.arange(100)[None,:];image[...,1]=image[...,0]
    mask=np.zeros((100,100),np.uint8);mask[30:40,30:40]=255
    d=np.zeros_like(mask);d[28:32,28:32]=1;d[38:42,38:42]=2
    for seed in range(24):
        np.random.seed(seed);expected=old.augment(image.copy(),mask.copy(),d.copy(),'dual_state',.5,return_trace=True)
        expected_rng=np.random.get_state()
        np.random.seed(seed);actual=augment(image.copy(),mask.copy(),d.copy(),'dual_state',.5,return_trace=True)
        for i in (0,1,2):assert np.array_equal(actual[i],expected[i])
        assert actual[3:5]==expected[3:5]
        actual_rng=np.random.get_state()
        assert np.array_equal(expected_rng[1],actual_rng[1]) and expected_rng[2:]==actual_rng[2:]
        np.random.seed(seed);zero_uniform=augment(image.copy(),mask.copy(),d.copy(),'dual_state',0)
        uniform_rng=np.random.get_state()
        np.random.seed(seed);zero_weighted=augment(image.copy(),mask.copy(),d.copy(),'dual_state',0,region_sampling='sqrt_area')
        assert all(np.array_equal(a,b) for a,b in zip(zero_uniform[:2],zero_weighted[:2]))
        weighted_rng=np.random.get_state()
        assert np.array_equal(uniform_rng[1],weighted_rng[1]) and uniform_rng[2:]==weighted_rng[2:]


def test_sampling_config_guards_and_resume_override():
    cfg=runner.parser().parse_args(['--stage','2','--output','unused','--joint_from_pretrained',
                                  '--crop_guidance','dual_state','--guidance_map_dir','maps',
                                  '--guidance_region_sampling','sqrt_area'])
    runner.validate(cfg)
    cfg.crop_guidance='none'
    with pytest.raises(ValueError,match='requires hard/dual_state'):runner.validate(cfg)
    with pytest.raises(ValueError):PCLT20KSegDataset([],guidance_region_sampling='unknown')
    with pytest.raises(ValueError):choose_window(np.zeros((64,64)),np.zeros((64,64)),'hard',50,'unknown')


def test_weighted_augmentation_real_model_cpu_update():
    from models.staged_petct import StagedPETCT
    from utils.seg_losses import BCEDiceLoss
    torch.set_num_threads(1)
    model=StagedPETCT(2).train()
    image=np.zeros((64,64,2),np.uint8);image[10:40,:,0]=128;image[...,1]=image[...,0]
    mask=np.zeros((64,64),np.uint8);mask[25:30,25:30]=255
    d=np.zeros((64,64),np.uint8);d[23:26,23:26]=1;d[29:35,29:35]=2
    samples=[]
    for seed in (3,4):
        np.random.seed(seed)
        x,y,_=augment(image.copy(),mask.copy(),d.copy(),'dual_state',1,region_sampling='sqrt_area')
        samples.append((torch.tensor(x[...,0].copy()).float().repeat(3,1,1)/255,
                        torch.tensor((y>=127.5).copy()).float().unsqueeze(0)))
    ct=torch.stack([s[0] for s in samples]);target=torch.stack([s[1] for s in samples])
    out=model(ct,ct,mode='full');criterion=BCEDiceLoss()
    lc=criterion(out['ct_logits'],target)[0];lf=criterion(out['full_logits'],target)[0]
    loss,stats=runner.forward_loss(model,{'ct':ct,'pet':ct,'mask':target,
        'is_full':torch.ones(2,dtype=torch.bool)},2,torch.device('cpu'),criterion)
    # Drop-path is stochastic in train mode; algebraic equality is verified by the
    # returned branch losses from this same forward, not two different forwards.
    assert abs(float(loss)-.5*(stats['ct_loss']+stats['full_loss']))<1e-6
    original=model.decoder_full.seg_head.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(original,model.decoder_full.seg_head.weight)
    assert stats['ct_samples']==stats['full_samples']==stats['missing_samples']==2


def test_post_run_interpretation_is_written_from_same_epoch(tmp_path):
    import json
    from scripts.run_crop_pilots import interpretation
    maps=tmp_path/'maps';maps.mkdir()
    (maps/'manifest.json').write_text(json.dumps({'source':{'path':str(tmp_path/'historical/best.pt')}}))
    def make_run(name, score):
        directory=tmp_path/name;directory.mkdir()
        row={'epoch':6,'selection_dice':score,'joint_dice':score,
             'full_dice':score+.01,'missing_dice':score-.01,'weights':'ema'}
        (directory/'train.jsonl').write_text(json.dumps(row)+'\n')
        (directory/'final_test.json').write_text('{}')
        return str(directory)
    reference={'dual_state':make_run('old_a3',.767),'hard':make_run('old_a2',.768),
               'none':make_run('b0',.78)}
    path=tmp_path/'reference.json';path.write_text(json.dumps({'runs':reference}))
    current={'dual_state':make_run('new_a3',.775),'hard':make_run('new_a2',.776)}
    result=interpretation(maps,current,path)
    assert result['same_strategy_joint_delta_from_v1']['dual_state']==pytest.approx(.008)
    assert 'No observed dual-state advantage' in result['conclusion']
    # A3 beats A2 but still loses to B0: do not label it a successful baseline change.
    row=json.loads((Path(current['hard'])/'train.jsonl').read_text());row['joint_dice']=.774;row['selection_dice']=.774
    (Path(current['hard'])/'train.jsonl').write_text(json.dumps(row)+'\n')
    assert 'still does not beat B0' in interpretation(maps,current,path)['conclusion']
