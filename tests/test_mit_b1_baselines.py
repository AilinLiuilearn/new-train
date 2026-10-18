"""Real MiT-B1 models, source loading, launch contracts and final CT evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from models.build_mdt_seg import build_ct_only_model, build_dual_model
from models.components.add_fusion import AddFusion
from models.components.backbones import SegformerFeatureBackbone
from tasks.mdt_seg import MDTSegTeacher

ROOT = Path(__file__).resolve().parents[1]


def cfg(**overrides):
    values = dict(ct_backbone='mit_b1',pet_backbone='mit_b1',pretrained=False,
        ct_pretrained_path='must-not-read',pet_pretrained_path='must-not-read',
        decoder_channels=[512,256,128,64],use_deep_supervision=False,deep_supervision=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('mode',['full_add','ct_only'])
def test_launcher_uses_original_protocol_and_correct_pretraining(monkeypatch,tmp_path,mode):
    # Capture the actual launcher arguments without starting a training job.
    capture=tmp_path/'args.json'; shim=tmp_path/'python'
    shim.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(dict(argv=sys.argv[1:],omp=os.environ["OMP_NUM_THREADS"],mkl=os.environ["MKL_NUM_THREADS"])))\n')
    shim.chmod(0o755)
    env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],CAPTURE_ARGS=str(capture))
    subprocess.run(['bash',str(ROOT/'scripts'/f'run_{mode}_mitb1_lr4e5.sh'),'--hash','test-override'],env=env,check=True)
    data=json.loads(capture.read_text())
    assert data['omp']==data['mkl']=='1'
    assert data['argv'][0]==('run_full_add_baseline.py' if mode=='full_add' else 'run_ct_only_seg.py')
    monkeypatch.setattr(sys,'argv',['x']+data['argv'][1:])
    if mode=='full_add':
        from run_full_add_baseline import FullAddConfig
        parsed=FullAddConfig.parse_arguments()
        assert parsed.train_batch_mode=='full'
        assert parsed.pet_backbone=='mit_b1' and parsed.pet_pretrained_path.endswith('/pretrained/mit-b1')
    else:
        from configs.seg_mdt import SegMDTConfig
        parsed=SegMDTConfig.parse_arguments()
    assert parsed.hash=='test-override'
    assert parsed.ct_backbone=='mit_b1' and parsed.pretrained
    assert parsed.ct_pretrained_path.endswith('/pretrained/mit-b1')
    assert parsed.learning_rate==4e-5 and parsed.epochs==60 and parsed.batch_size==16
    assert parsed.ema_enabled and parsed.ema_decay==.999 and parsed.ema_start_epoch==3
    assert parsed.aug_mode=='cipa' and parsed.norm_mode=='imagenet' and parsed.hd95_mode=='cipa_official'
    assert not parsed.deep_supervision and not parsed.use_deep_supervision
    assert parsed.val_split_file==parsed.test_split_file=='test.txt' and parsed.allow_val_equals_test
    assert not hasattr(parsed,'protect_enabled') and not hasattr(parsed,'local_enabled')


def test_homogeneous_is_two_independent_pretrained_encoders_and_ct_only_has_no_pet(tmp_path):
    source=build_ct_only_model(cfg())['model']
    path=tmp_path/'mit.pth'
    torch.save(source.enc_ct.state_dict(),path)
    dual=build_dual_model(cfg(pretrained=True,ct_pretrained_path=str(path),pet_pretrained_path=str(path)))['model']
    ct=build_ct_only_model(cfg(pretrained=True,ct_pretrained_path=str(path)))['model']
    assert isinstance(dual.fusion,AddFusion)
    assert isinstance(dual.enc_ct,SegformerFeatureBackbone) and type(dual.enc_pet) is type(dual.enc_ct)
    assert dual.enc_ct is not dual.enc_pet
    assert {p.data_ptr() for p in dual.enc_ct.parameters()}.isdisjoint({p.data_ptr() for p in dual.enc_pet.parameters()})
    for name,value in source.enc_ct.state_dict().items():
        for encoder in (dual.enc_ct,dual.enc_pet,ct.enc_ct):
            torch.testing.assert_close(value,encoder.state_dict()[name],atol=0,rtol=0)
    assert all(p.requires_grad for p in dual.parameters())
    assert not hasattr(ct,'enc_pet') and not hasattr(ct,'fusion')
    assert ct.enc_ct.feature_info.channels()==[64,128,320,512]
    with pytest.raises(FileNotFoundError):
        build_ct_only_model(cfg(pretrained=True,ct_pretrained_path=str(tmp_path/'absent')))


@pytest.mark.parametrize('mode,start_epoch',[('full',0),('ct_only',3)])
def test_complete_train_save_and_final_evaluation(monkeypatch,tmp_path,mode,start_epoch):
    from configs.seg_mdt import SegMDTConfig
    import run_ct_only_seg as ct_runner
    import run_full_add_baseline as full_runner
    runner=full_runner if mode=='full' else ct_runner
    config_cls=full_runner.FullAddConfig if mode=='full' else SegMDTConfig
    monkeypatch.setattr(sys,'argv',['x','--ct_backbone','mit_b1','--pet_backbone','mit_b1',
        '--pretrained','false','--ct_pretrained_path','must-not-read','--pet_pretrained_path','must-not-read',
        '--checkpoint_root',str(tmp_path),'--hash','smoke','--learning_rate','4e-5',
        '--batch_size','2','--image_size_2d','64','--epochs','1','--cosine_warmup','0',
        '--hd95_mode','surface','--mixed_precision','false','--ema_enabled','true',
        '--ema_start_epoch',str(start_epoch)])
    parsed=config_cls.parse_arguments()
    monkeypatch.setattr(config_cls,'parse_arguments',classmethod(lambda cls:parsed))
    mask=torch.zeros(2,1,64,64);mask[:,:,20:30,20:30]=1
    batch=dict(ct=torch.randn(2,3,64,64),mask=mask,case_id=['1','2'])
    if mode=='full': batch['pet']=torch.randn_like(batch['ct'])
    monkeypatch.setattr(runner,'_loaders',lambda conf:([batch,batch],[batch],[batch]))
    seen=[]
    original=MDTSegTeacher.evaluate
    def audit(task,loader,eval_mode='full',tag='val',model=None):
        seen.append(eval_mode)
        if mode=='ct_only': assert not hasattr(model,'enc_pet')
        return original(task,loader,eval_mode,tag,model)
    monkeypatch.setattr(MDTSegTeacher,'evaluate',audit)
    runner.main()
    assert seen==(['full','full'] if mode=='full' else ['ct','ct'])
    folder=Path(parsed.checkpoint_dir)
    checkpoint=MDTSegTeacher.load_state_dicts(folder/('ckpt.best_full.pth.tar' if mode=='full' else 'ckpt.best_ct.pth.tar'))
    expected_tag='ema' if start_epoch==0 else 'raw'
    assert checkpoint['eval_weights']==expected_tag
    assert checkpoint['config']['ct_backbone']=='mit_b1'
    model=(build_dual_model(parsed) if mode=='full' else build_ct_only_model(parsed))['model'].eval()
    for key in ('model','model_ema'):
        model.load_state_dict(checkpoint[key],strict=True)
    result=json.loads((folder/('final_test_full.json' if mode=='full' else 'final_test_ct.json')).read_text())
    assert result['weights']==expected_tag and 0<=result['dice']<=1


def test_original_ct_backbone_default_and_new_guard():
    from configs.seg_mdt import SegMDTConfig
    from run_ct_only_seg import _assert_ct_only_protocol
    parser=SegMDTConfig.model_parser()
    parsed=parser.parse_args([])
    assert parsed.ct_backbone=='convnextv2_nano'
    for backbone in ('convnextv2_nano','mit_b1','mit-b1'):
        _assert_ct_only_protocol(cfg(ct_backbone=backbone,accumulation_steps=1,optimizer='adamw'))
    with pytest.raises(ValueError):
        _assert_ct_only_protocol(cfg(ct_backbone='wrong',accumulation_steps=1,optimizer='adamw'))
