import pytest
import torch
from models.simmlm2d import Expert2D, DualExpert2D


def test_joint_initial_average_gradients_and_missing_isolation():
    torch.manual_seed(4)
    model=DualExpert2D((4,8,16),8)
    ct,pet=torch.randn(2,1,32,32),torch.randn(2,1,32,32)
    out=model(ct,pet)
    torch.testing.assert_close(out['logits'],.5*(out['ct_logits']+out['pet_logits']))
    torch.testing.assert_close(out['weights'],torch.full((2,2),.5))
    out['logits'].square().mean().backward()
    for module in [model.ct,model.pet,model.gate]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    model.zero_grad(set_to_none=True)
    def forbidden(*args): raise AssertionError('missing called PET or gate')
    h1=model.pet.register_forward_pre_hook(forbidden); h2=model.gate.register_forward_pre_hook(forbidden)
    missing=model(ct,pet*float('nan'),mode='missing')
    missing['logits'].square().mean().backward()
    assert all(p.grad is None for p in model.pet.parameters())
    torch.testing.assert_close(missing['weights'],torch.tensor([[1.,0.],[1.,0.]]))
    assert torch.isfinite(missing['logits']).all()
    h1.remove();h2.remove()


def test_checkpoint_roundtrip_and_shapes():
    model=DualExpert2D((4,8,16),8).eval()
    copy=DualExpert2D((4,8,16),8).eval(); copy.load_state_dict(model.state_dict())
    x=torch.randn(2,1,32,32)
    with torch.no_grad():
        torch.testing.assert_close(model(x,x)['logits'],copy(x,x)['logits'],rtol=0,atol=0)
    with pytest.raises(ValueError): model(x)
    with pytest.raises(ValueError): model(x,x,mode='bad')
    with pytest.raises(ValueError): Expert2D((4,8,16))(torch.randn(1,3,32,32))


def test_dataset_binary_masks_and_ct_only(tmp_path):
    import cv2,numpy as np
    from datasets.pclt_experts import SliceDataset
    x=np.zeros((32,32),np.uint8); x[8:24,8:24]=200
    mask=(x>0).astype(np.uint8)
    cv2.imwrite(str(tmp_path/'ct.png'),x);cv2.imwrite(str(tmp_path/'mask.png'),mask)
    record={'ct_path':str(tmp_path/'ct.png'),'mask_path':str(tmp_path/'mask.png'),'case_id':'a','image_id':'a_1'}
    ds=SliceDataset([record],32,'zscore',paired=False)
    out=ds[0];assert 'pet' not in out
    assert out['mask'].sum()==256
    assert out['ct'].shape==(1,32,32)
    assert torch.isfinite(out['ct']).all()


def test_training_pipeline_resume_and_reevaluation(tmp_path,monkeypatch):
    import cv2,numpy as np,json,sys
    import run_experts as runner
    root=tmp_path/'data';root.mkdir()
    for case in ['a','b','c','d']:
        folder=root/case;folder.mkdir()
        x=np.zeros((32,32),np.uint8);x[8:24,8:24]=170
        for suffix,data in [('CT',x),('PET',x//2),('mask',(x>0).astype(np.uint8)*255)]:
            cv2.imwrite(str(folder/f'{case}_0_{suffix}.png'),data)
    for split,ids in [('train_original.txt','a_0\nb_0\n'),('val.txt','c_0\n'),('test.txt','d_0\n')]:
        (root/split).write_text(ids)
    common=['--root',str(root),'--image_size','32','--batch_size','2','--features','4','8','16',
            '--gate_hidden','8','--num_workers','0','--epochs','2','--warmup_epochs','0',
            '--ema_start_epoch','0','--device','cpu','--hd95_mode','surface','--log_interval','1']
    def run(extra):
        monkeypatch.setattr(sys,'argv',['run_experts.py',*common,*extra]);runner.main()
    for mode in ['ct','pet']:
        run(['--mode',mode,'--output',str(tmp_path/mode)])
    init=['--mode','joint','--ct_checkpoint',str(tmp_path/'ct/best.pt'),
          '--pet_checkpoint',str(tmp_path/'pet/best.pt')]
    run([*init,'--output',str(tmp_path/'reference')])
    expected=runner.load(tmp_path/'reference/last.pt')
    original=runner.forward_loss
    calls=[0]
    def interrupt_second_epoch(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2: raise InterruptedError('simulate interruption after epoch 1')
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'forward_loss',interrupt_second_epoch)
    with pytest.raises(InterruptedError): run([*init,'--output',str(tmp_path/'resumed')])
    monkeypatch.setattr(runner,'forward_loss',original)
    run(['--resume',str(tmp_path/'resumed/last.pt')])
    actual=runner.load(tmp_path/'resumed/last.pt')
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key],actual['model'][key],rtol=0,atol=0)
    assert actual['epoch']==2
    run(['--evaluate',str(tmp_path/'resumed/best.pt')])
    final=json.loads((tmp_path/'resumed/final_test.json').read_text())
    reevaluated=json.loads(next((tmp_path/'resumed').glob('reeval_*.json')).read_text())
    assert final['metrics']==reevaluated['metrics']
