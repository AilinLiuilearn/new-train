import importlib.util
from pathlib import Path
import pytest
import torch
import json
import numpy as np
from PIL import Image
from torch import nn
from models.components.group_unet_decoder import UNetStyleDecoder
from models.staged_petct import StagedPETCT
import run_staged as runner


torch.set_num_threads(1)


def test_legacy_gn_oracle_exact():
    path=Path(__file__).parent/'oracles/group_unet_decoder_before_bn.py'
    spec=importlib.util.spec_from_file_location('old_gn_decoder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(17);old=module.UNetStyleDecoder().eval()
    torch.manual_seed(17);new=UNetStyleDecoder().eval()
    assert set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[n]) for n,v in old.state_dict().items())
    features=[torch.randn(2,c,s,s) for c,s in zip((64,128,320,512),(16,8,4,2))]
    with torch.no_grad():assert torch.equal(old(features,(64,64))['logits'],new(features,(64,64))['logits'])


def test_bn_only_changes_norm_and_buffers_with_exact_initial_weights():
    torch.manual_seed(23);gn=StagedPETCT(2)
    torch.manual_seed(23);bn=StagedPETCT(2,decoder_norm='BN')
    assert sum(p.numel() for p in gn.parameters())==sum(p.numel() for p in bn.parameters())==35907538
    assert all(torch.equal(v,bn.state_dict()[n]) for n,v in gn.state_dict().items())
    for decoder in (bn.decoder_ct,bn.decoder_full):
        assert sum(isinstance(m,nn.BatchNorm2d) for m in decoder.modules())==10
        assert not any(isinstance(m,nn.GroupNorm) for m in decoder.modules())
    for a,b in zip(bn.decoder_ct.parameters(),bn.decoder_full.parameters()):assert a.data_ptr()!=b.data_ptr()
    before=bn.decoder_ct.proj4[1].running_mean.clone()
    bn.train();out=bn(torch.randn(2,3,64,64),torch.randn(2,3,64,64),mode='full')
    (.5*(out['ct_logits'].square().mean()+out['full_logits'].square().mean())).backward()
    assert not torch.equal(before,bn.decoder_ct.proj4[1].running_mean)
    assert bn.decoder_full.proj4[1].num_batches_tracked.item()==1
    bn.load_state_dict(bn.state_dict(),strict=True)
    with pytest.raises(RuntimeError):gn.load_state_dict(bn.state_dict(),strict=True)


def test_bn_checkpoint_evaluate_resume_and_authoritative_norm(tmp_path):
    root=tmp_path/'data';root.mkdir()
    ids=[]
    for i in range(4):
        case=f'case{i}';folder=root/case;folder.mkdir();image_id=case+'_001';ids.append(image_id)
        image=np.random.RandomState(i).randint(0,256,(64,64),dtype=np.uint8)
        mask=np.zeros((64,64),dtype=np.uint8);mask[20:35,23:34]=255
        for suffix,a in [('CT',image),('PET',image[::-1]),('mask',mask)]:Image.fromarray(a).save(folder/(image_id+'_'+suffix+'.png'))
    for name,items in [('train_original',ids[:2]),('val',ids[2:3]),('test',ids[3:])]:
        (root/(name+'.txt')).write_text('\n'.join(items)+'\n')
    out=tmp_path/'run'
    result=runner.main(['--stage','2','--joint_from_pretrained','--no_pretrained','--decoder_norm','BN',
        '--root',str(root),'--output',str(out),'--image_size_2d','64','--batch_size','2','--num_workers','0',
        '--epochs','1','--cosine_warmup','0','--hd95_mode','surface','--no-mixed_precision',
        '--aug_mode','none','--device','cpu','--save_last_checkpoint'])
    assert runner.load(out/'best.pt')['config']['decoder_norm']=='BN'
    reevaluated=runner.main(['--evaluate',str(out/'best.pt'),'--device','cpu'])
    assert result['metrics']==reevaluated['metrics']
    with pytest.raises(ValueError,match='authoritative'):
        runner.main(['--evaluate',str(out/'best.pt'),'--device','cpu','--decoder_norm','GN'])
    resumed=runner.main(['--resume',str(out/'last.pt'),'--device','cpu'])
    assert result['metrics']==resumed['metrics']
