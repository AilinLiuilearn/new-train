"""Check that STOP/failure cancels successors and completion runs exactly V/T/S."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import REPO


@pytest.mark.parametrize('mode',['stop','failure','complete'])
def test_queue_lifecycle(tmp_path,monkeypatch,mode):
    spec=importlib.util.spec_from_file_location('test_scgr_supervisor',REPO/'scripts/run_scgr.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'git_identity',lambda:{'dirty':False,'commit':'test-only'})
    monkeypatch.setattr(module,'local_paths',lambda p:{'output_root':str(tmp_path)})
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    # GPU preflight itself is covered separately; don't couple a lifecycle test
    # to a mutable report hash while the report is being regenerated.
    original_sha=module.sha
    checks=json.loads((REPO/'docs/scgr/GPU_PREFLIGHT.json').read_text())
    expected={str(REPO/k):v for k,v in checks['source_sha256'].items()}
    monkeypatch.setattr(module,'sha',lambda p:expected.get(str(p),original_sha(p)))
    launched=[];root=tmp_path/'queue'

    class Child:
        def __init__(self,command,**kwargs):
            self.pid=12345;self.returncode=None;self.n=0
            self.recipe=command[command.index('--recipe')+1];launched.append(self.recipe)
            self.out=Path(command[command.index('--output')+1]);self.out.mkdir()
            if mode=='stop':(root/'STOP').touch()

        def poll(self):
            self.n+=1
            if mode=='stop' and self.n==1:return None
            if mode=='stop':
                assert (self.out/'STOP').exists()
                (self.out/'status.json').write_text(json.dumps({'state':'stopped_by_request'}))
                self.returncode=0;return 0
            if mode=='failure':self.returncode=2;return 2
            i=len(launched)
            result={'complete':True,'metrics':{'dice':.77+i*.001,'hd95':18-i*.1},
                    'best_epoch':4,'weights':'ema','target_788_hd17_met':False,
                    'error_diagnostics':{'counts':{k:10-i for k in ['fn_inner3','fn_deep','fp_outer3','fp_outer3_12','fp_far12','fp_empty_gt','fp_detached']}}}
            (self.out/'RESULTS.json').write_text(json.dumps(result))
            (self.out/'status.json').write_text(json.dumps({'state':'complete'}))
            self.returncode=0;return 0

    monkeypatch.setattr(module.subprocess,'Popen',Child)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    module.main(SimpleNamespace(paths=str(REPO/'paths.local.json'),campaign='queue',gpu='0'))
    status=json.loads((root/'STATUS.json').read_text())
    assert launched==(['V','T','S'] if mode=='complete' else ['V'])
    assert status['state']=={'stop':'stopped_by_request','failure':'needs_analysis','complete':'complete'}[mode]
    if mode=='complete':
        comparison=json.loads((root/'COMPARISON.json').read_text())
        assert set(comparison['comparisons'])=={'T_minus_V','S_minus_T'}
