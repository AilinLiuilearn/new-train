"""Real-model tests for best-only saving and changing storage policy on resume."""
from pathlib import Path
import json
import pytest
import run_staged as runner
from test_staged import make_pngs


def test_best_only_and_resume_storage_policy(tmp_path,monkeypatch):
    root=tmp_path/'data';root.mkdir();make_pngs(root)
    common=['--stage','2','--joint_from_pretrained','--no_pretrained','--root',str(root),
            '--image_size_2d','64','--hd95_mode','surface','--batch_size','2','--num_workers','0',
            '--epochs','2','--cosine_warmup','0','--ema_start_epoch','0','--device','cpu','--aug_mode','none']
    output=tmp_path/'best_only'
    result=runner.main([*common,'--output',str(output)])
    assert (output/'best.pt').is_file() and not (output/'last.pt').exists()
    checkpoint=runner.load(output/'best.pt')
    assert checkpoint['config']['save_last_checkpoint'] is False
    assert checkpoint['val']['joint_dice']==.5*(checkpoint['val']['full']['dice']+checkpoint['val']['missing']['dice'])
    assert runner.main(['--evaluate',str(output/'best.pt'),'--device','cpu'])==result
    resumed=tmp_path/'resumed';real_save=runner.atomic_save
    class Interrupted(Exception):pass
    def stop(payload,path):
        real_save(payload,path)
        if Path(path).name=='last.pt' and payload['epoch']==1:raise Interrupted()
    monkeypatch.setattr(runner,'atomic_save',stop)
    with pytest.raises(Interrupted):runner.main([*common,'--save_last_checkpoint','--output',str(resumed)])
    source=resumed/'last.pt';mtime=source.stat().st_mtime_ns
    calls=[]
    def tracked(payload,path):
        calls.append(Path(path).name);real_save(payload,path)
    monkeypatch.setattr(runner,'atomic_save',tracked)
    runner.main(['--resume',str(source),'--device','cpu','--no-save_last_checkpoint'])
    assert 'last.pt' not in calls and source.stat().st_mtime_ns==mtime
    assert json.loads((resumed/'config.json').read_text())['save_last_checkpoint'] is False
    assert [json.loads(line)['epoch'] for line in (resumed/'train.jsonl').read_text().splitlines()]==[1,2]
