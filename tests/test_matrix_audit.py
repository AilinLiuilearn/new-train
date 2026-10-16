import tempfile
from pathlib import Path
import torch
from torch import nn
from models.ct_full_matrix import MatrixScale
from utils.matrix_audit import audit_scale,audit_findings
from audit_ct_full_matrix import run_audit


def test_scale_identity_and_nonidentity():
    scale=MatrixScale(2,clusters=2,capacity=8,min_candidates=4,min_cluster_samples=2)
    for bank in scale.banks:
        bank.valid.fill_(True);bank.keys.copy_(torch.eye(2));bank.transforms.copy_(torch.eye(2).expand(2,2,2))
    x=torch.ones(1,2,4,4);gt=torch.ones(1,1,4,4)
    variants,rows=audit_scale(scale,x,x,gt,8)
    assert all(torch.allclose(v,x) for v in variants.values())
    assert all(r['ct_error_sum']==0 and r['normal_error_sum']==0 for r in rows)
    for bank in scale.banks:bank.transforms.mul_(2)
    variants,rows=audit_scale(scale,x,2*x,gt,8)
    assert torch.allclose(variants['normal'],1.1*x)
    assert torch.allclose(variants['g100'],2*x)
    assert rows[0]['normal_error_sum']<1e-8 and rows[0]['ct_error_sum']>0
    torch.manual_seed(7)
    for bank in scale.banks:bank.transforms.copy_(torch.randn_like(bank.transforms))
    x=torch.randn_like(x)
    variants,_=audit_scale(scale,x,x,gt,8)
    assert torch.allclose(variants['normal'],scale(x),atol=1e-6)
    assert torch.allclose(variants['shuffle'],scale(x,read_mode='shuffle'),atol=1e-6)


def test_complete_audit(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.matrix=nn.Module();self.matrix.scales=nn.ModuleList([
                MatrixScale(2,clusters=2,capacity=8,min_candidates=4,min_cluster_samples=2) for _ in range(4)])
            self.matrix_read_mode='normal'
            for scale in self.matrix.scales:
                for bank in scale.banks:
                    bank.valid.fill_(True);bank.keys.copy_(torch.eye(2));bank.transforms.copy_((2*torch.eye(2)).expand(2,2,2))
        def _encode_ct(self,x):return [x]*4
        def _encode_pet(self,x):return [x]*4
        def fusion(self,c,p):return [a+b for a,b in zip(c,p)]
        def _decode(self,features,size):return {'logits':sum(features).mean(1,keepdim=True)-5}
        def forward(self,ct,pet=None,forward_mode='missing'):
            c=self._encode_ct(ct)
            f=self.fusion(c,self._encode_pet(pet)) if forward_mode=='full' else [s(x,read_mode=self.matrix_read_mode) for s,x in zip(self.matrix.scales,c)]
            return self._decode(f,ct.shape[-2:])
    model=Model();x=torch.ones(2,2,8,8);gt=torch.zeros(2,1,8,8);gt[:,:,2:4,2:4]=1
    batch=dict(ct=x,pet=x,mask=gt,case_id=['a','b'])
    report=run_audit(model,[batch],'cpu','surface',.5,str(tmp_path),samples=8)
    assert len(report['summary'])==16
    assert len(report['feature_summary'])==8
    assert report['summary']['off']['changed_pixels']==0
    assert report['metadata']['full_cached_path_max_probability_difference']==0
    assert (tmp_path/'report.md').exists()


if __name__=='__main__':
    test_scale_identity_and_nonidentity()
    with tempfile.TemporaryDirectory() as directory:test_complete_audit(Path(directory))
    print('matrix audit tests passed')
