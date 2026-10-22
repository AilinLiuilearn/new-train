"""Synthetic geometry/cache contracts plus real-model paired training smoke."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import run_staged as runner
from datasets.pclt20k_seg import _records_from_ids,PCLT20KSegDataset
from utils.dual_state_maps import MODEL,atomic_json,sha256,fingerprints
from utils.paired_copy_paste import POLICY,PasteBank,candidates,body_mask

torch.set_num_threads(1)


@pytest.fixture()
def bank_fixture(tmp_path):
    root=tmp_path/'data';root.mkdir();maps=tmp_path/'maps';maps.mkdir()
    ids=[f'p{i}_001' for i in range(6)]
    for i,name in enumerate(ids):
        folder=root/f'p{i}';folder.mkdir()
        ct=np.full((64,64),90+i*2,np.uint8);gt=np.zeros_like(ct)
        start=48 if i==0 else 25
        gt[start:start+8,start:start+8]=255;ct[start:start+8,start:start+8]=140+i*2
        for suffix,array in [('CT',ct),('PET',ct+7),('mask',gt)]:Image.fromarray(array).save(folder/f'{name}_{suffix}.png')
        d=np.zeros_like(ct);d[start:start+2,start:start+8]=1;d[start+6:start+8,start:start+8]=2;d[2,2]=1
        Image.fromarray(d).save(maps/f'{name}.png')
    for split,selected in [('train_original',ids[:4]),('val',ids[4:5]),('test',ids[5:])]:
        (root/f'{split}.txt').write_text('\n'.join(selected)+'\n')
    cfg=runner.parser().parse_args(['--stage','2','--joint_from_pretrained','--root',str(root),
                                  '--image_size_2d','64','--hd95_mode','surface'])
    records=_records_from_ids(str(root),ids[:4])
    manifest={'schema':1,'complete':True,'partial':False,'count':4,'image_ids':ids[:4],
        'model':MODEL,'threshold':'sigmoid_fp32 > 0.5','baseline_config':vars(cfg),
        'train_split_sha256':sha256(root/'train_original.txt'),'content_hash':True,
        'source':{'sha256':'synthetic-test-source'},'samples':{}}
    index={'complete':True,'policy':POLICY,'image_ids':ids[:4],'candidates':[],'body_bboxes':[]}
    for i,r in enumerate(records):
        manifest['samples'][r['image_id']]={'files':fingerprints(r,True),'map':r['image_id']+'.png',
                                           'map_sha256':sha256(maps/(r['image_id']+'.png'))}
        gt=np.array(Image.open(r['mask_path']));d=np.array(Image.open(maps/(r['image_id']+'.png')))
        found=candidates(gt,d)
        assert len(found)==1 and found[0]['flags']==[1,2] and found[0]['area']==64
        for item in found:item['record_idx']=i
        index['candidates'].extend(found)
        index['body_bboxes'].append([0,0,64,64])
    atomic_json(maps/'manifest.json',manifest)
    index['map_manifest_sha256']=sha256(maps/'manifest.json')
    ip=tmp_path/'index.json';atomic_json(ip,index)
    return root,maps,ip,cfg,records


def test_whole_gt_components_and_false_positive_exclusion():
    gt=np.zeros((64,64),np.uint8);gt[20:28,20:28]=255
    d=np.zeros_like(gt);d[2,2]=1
    c=candidates(gt,d)
    assert len(c)==1 and not c[0]['flags'] and not c[0]['hard']
    d[21,21]=1;assert candidates(gt,d)[0]['flags']==[1]
    # Close second target would be cut at the context boundary: reject it.
    gt[29:32,20:28]=255
    assert not candidates(gt,d)


def test_paired_paste_exact_labels_cross_case_fairness_and_bounds(bank_fixture):
    _,maps,index,cfg,records=bank_fixture
    bank=PasteBank(index,maps,vars(cfg));ct,pet,gt=bank.read(0)
    counts=[0,0]
    np.random.seed(41)
    for _ in range(300):
        mixed_ct,mixed_pet,mixed_gt,info,trace=bank.paste(ct,pet,gt,0,'dual_state',1.,trace=True)
        if not info['success']:continue
        counts[trace['category']-1]+=1
        assert not trace['donor_id'].startswith('p0_')
        x0,y0,x1,y1=trace['recipient_bbox']
        assert 0<=x0<x1<=64 and 0<=y0<y1<=64
        assert not gt[y0:y1,x0:x1].any()
        assert np.array_equal(mixed_pet.astype(int)-mixed_ct.astype(int),np.full((64,64),7))
        assert (mixed_gt>0).sum()==128
        assert set(np.unique(mixed_gt))<={0,255}
    assert sum(counts)>20 and .35<counts[0]/sum(counts)<.65
    assert np.array_equal(ct,bank.read(0)[0]) and (gt>0).sum()==64


def test_off_probability_fallback_corruption_identity_and_cache_bound(bank_fixture):
    _,maps,index,cfg,records=bank_fixture
    bank=PasteBank(index,maps,vars(cfg));ct,pet,gt=bank.read(0)
    np.random.seed(17);expected=np.random.get_state()
    result=bank.paste(ct,pet,gt,0,'dual_state',0)
    actual=np.random.get_state()
    assert np.array_equal(expected[1],actual[1]) and expected[2:]==actual[2:]
    assert result[0] is ct and result[3]['attempts']==0
    bank.pools[1]=[];bank.pools[2]=[]
    assert bank.paste(ct,pet,gt,0,'dual_state',1)[3]['fallback_no_donor']==1
    bank=PasteBank(index,maps,vars(cfg))
    full=np.full_like(gt,255)
    out=bank.paste(ct,pet,full,0,'dual_state',1)
    assert not out[3]['success'] and out[3]['fallback_placement']==1 and np.array_equal(out[2],full)
    with pytest.raises(ValueError,match='identity'):PasteBank(index,maps,vars(cfg),expected_identity={})
    path=Path(records[1]['ct_path']);raw=path.read_bytes();path.write_bytes(raw+b'x')
    with pytest.raises(ValueError,match='fingerprint'):bank.read(1)


def test_prespecified_dominance_iteration_changes_only_group_membership(bank_fixture):
    _,maps,index,cfg,_=bank_fixture
    original=json.loads(index.read_text())
    original['candidates'][1]['errors'][1]+=1
    original['candidates'][2]['errors'][2]+=1
    atomic_json(index,original)
    overlap=PasteBank(index,maps,vars(cfg));dominance=PasteBank(index,maps,vars(cfg),selection='dominance')
    assert len(overlap.pools[1])==len(overlap.pools[2])==4
    assert [i['record_idx'] for i in dominance.pools[1]]==[1]
    assert [i['record_idx'] for i in dominance.pools[2]]==[2]
    assert overlap.pools['random']==dominance.pools['random'] and overlap.pools['hard']==dominance.pools['hard']
    assert overlap.identity!=dominance.identity


def test_dataset_off_exact_and_real_model_training_missing_evaluation(bank_fixture,tmp_path):
    root,maps,index,cfg,records=bank_fixture
    a=PCLT20KSegDataset(records,image_size=64,train=True)
    b=PCLT20KSegDataset(records,image_size=64,train=True,copy_paste_prob=0)
    np.random.seed(31);x=a[0];state=np.random.get_state()
    np.random.seed(31);y=b[0];after=np.random.get_state()
    for key in ('ct','pet','mask'):assert torch.equal(x[key],y[key])
    assert np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    bank=PasteBank(index,maps,vars(cfg))
    ds=PCLT20KSegDataset(records,image_size=64,train=True,copy_paste='dual_state',copy_paste_prob=1,paste_bank=bank)
    loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=2)
    assert sum(float(batch['paste_stats']['samples'].sum()) for batch in loader)==4
    output=tmp_path/'run'
    result=runner.main(['--stage','2','--joint_from_pretrained','--no_pretrained','--root',str(root),
        '--output',str(output),'--image_size_2d','64','--hd95_mode','surface','--batch_size','2',
        '--num_workers','0','--epochs','1','--cosine_warmup','0','--device','cpu',
        '--copy_paste','dual_state','--copy_paste_prob','1','--paste_index',str(index),'--guidance_map_dir',str(maps)])
    assert (output/'best.pt').exists() and not (output/'last.pt').exists()
    log=json.loads((output/'copy_paste.jsonl').read_text());assert log['samples']==4 and log['attempts']==4
    checkpoint=runner.load(output/'best.pt')
    assert checkpoint['config']['copy_paste_identity']['index_sha256']==sha256(index)
    assert checkpoint['source']['stage1_checkpoint'] is None
    # Saved weights/evaluation are independent of the donor pool and PET files.
    maps.rename(tmp_path/'maps-removed');index.unlink()
    for p in root.rglob('*_PET.png'):p.unlink()
    missing=runner.main(['--evaluate',str(output/'best.pt'),'--eval_mode','missing','--device','cpu'])
    assert missing['metrics']['missing']==result['metrics']['missing']
