"""Independent low-frequency-context-guided bidirectional wavelet fusion.

Destination: models/context_guided_wavelet_fusion.py. Python>=3.10, torch>=2.1.
Default pyramid forward is AddFusion-compatible: list of four fused features.
forward_features returns (fused_list, ct_enhanced_list, pet_enhanced_list).
No encoder, decoder, library, retrieval, auxiliary losses, or hidden global state.

Provenance / boundaries:
WFANet (AAAI2025), https://github.com/Jie-1203/WFANet/blob/master/net_torch.py
provides Haar analysis/synthesis and frequency-query/structure-key inspiration.
We implement orthonormal Haar with standard torch autograd (coefficients +/-0.5),
not its differently scaled custom backward. This is a task adaptation, not a
reproduction: both modalities decompose; LL exchanges bidirectionally; detail
Q/K receive LL context; V always comes from corresponding source detail band.
AAAI2026 retinal TEWF inspires bidirectional corresponding-band exchange;
no expert routing, classification head, high-frequency-only assumption, or
WaveMamba absolute-maximum selection is included.

Each scale has the SAME structure/settings and independent parameters.
Attention is exact spatial attention WITHIN fixed non-overlapping windows,
not global spatial attention. Padding keys are excluded. All four bands retained.
Odd input dimensions use replicate padding before DWT and crop after IDWT.
Missing returns CT unchanged, never executing DWT/attention. PET enhanced output
in Missing is an absent-feature zero placeholder, NOT generated PET.
Caller must route image rows before PET encoding; this module sees features only.

Final equation F=(C+R_C(Y_C-X_C))+(P+R_P(Y_P-X_P)). R outputs zero initialized.
Do not add C+P a second time. Initially exact addition; internal gradient starts
only after the final output projections have learned nonzero weights.

Run embedded tests: python context_guided_wavelet_fusion.py --test
Real feature sizes: python context_guided_wavelet_fusion.py --smoke --full-size
"""
from __future__ import annotations
import argparse
import math
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['HaarWavelet2D', 'ContextGuidedWaveletFusion', 'ContextGuidedWaveletFusionPyramid']
BANDS = ('LL', 'LH', 'HL', 'HH')

class HaarWavelet2D(nn.Module):
    """Orthonormal separable Haar; band sign/order consistent within both transforms.

    LH here means variation along rows; HL along columns. Naming conventions
    differ across libraries: use this module's matching synthesis, not a foreign IDWT.
    """
    def analysis(self, x: Tensor) -> tuple[tuple[Tensor, ...], tuple[int, int]]:
        if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
            raise ValueError('Haar analysis requires nonempty floating BCHW')
        h, w = x.shape[-2:]
        x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
        a,b,c,d = x[...,0::2,0::2],x[...,0::2,1::2],x[...,1::2,0::2],x[...,1::2,1::2]
        return ((a+b+c+d)*.5, (a+b-c-d)*.5,
                (a-b+c-d)*.5, (a-b-c+d)*.5), (h,w)

    def synthesis(self, bands: Sequence[Tensor], size: tuple[int,int]) -> Tensor:
        if len(bands) != 4 or any(t.shape != bands[0].shape for t in bands):
            raise ValueError('Synthesis requires four identically shaped bands')
        ll,lh,hl,hh = bands
        a,b = (ll+lh+hl+hh)*.5, (ll+lh-hl-hh)*.5
        c,d = (ll-lh+hl-hh)*.5, (ll-lh-hl+hh)*.5
        top = torch.stack((a,b),dim=-1).flatten(-2)
        bottom = torch.stack((c,d),dim=-1).flatten(-2)
        out = torch.stack((top,bottom),dim=-2).flatten(-3,-2)
        h,w = size
        if h not in (out.shape[-2],out.shape[-2]-1) or w not in (out.shape[-1],out.shape[-1]-1):
            raise ValueError('Requested size is inconsistent with bands')
        return out[...,:h,:w]

    def forward(self, x: Tensor):
        return self.analysis(x)

class _TokenProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim,dim)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm(x.permute(0,2,3,1))).permute(0,3,1,2)

class _WindowExchange(nn.Module):
    """One direction, query on target positions; K/V on source positions."""
    def __init__(self, dim: int, heads: int, window: int, context: bool):
        super().__init__()
        self.dim,self.heads,self.window,self.context = dim,heads,window,context
        self.q,self.k,self.v = [_TokenProjection(dim) for _ in range(3)]
        if context:
            # Independent normalization/projection for detail and context terms.
            self.q_context,self.k_context = _TokenProjection(dim),_TokenProjection(dim)
        self.out = nn.Conv2d(dim,dim,1)
        self.relative_bias = nn.Parameter(torch.zeros(heads,(2*window-1)**2))
        coords = torch.stack(torch.meshgrid(torch.arange(window),torch.arange(window),indexing='ij')).flatten(1)
        delta = coords[:,:,None]-coords[:,None,:]
        index = (delta[0]+window-1)*(2*window-1)+delta[1]+window-1
        self.register_buffer('relative_index',index,persistent=False)

    def _pack(self,x: Tensor) -> Tensor:
        b,d,h,w=x.shape;s=self.window
        return x.reshape(b,d,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,d)

    def forward(self, target: Tensor, source: Tensor,
                target_context: Tensor | None=None, source_context: Tensor | None=None) -> Tensor:
        q,k,v=self.q(target),self.k(source),self.v(source)
        if self.context:
            if target_context is None or source_context is None:
                raise ValueError('Detail exchange requires both LL contexts')
            q=q+self.q_context(target_context); k=k+self.k_context(source_context)
        b,d,h,w=q.shape;s=self.window;ph=(-h)%s;pw=(-w)%s
        q,k,v=[F.pad(t,(0,pw,0,ph)) for t in (q,k,v)]
        hp,wp=h+ph,w+pw
        qt,kt,vt=[self._pack(t).reshape(-1,s*s,self.heads,d//self.heads).transpose(1,2) for t in (q,k,v)]
        valid=F.pad(q.new_ones(1,1,h,w),(0,pw,0,ph))
        valid=self._pack(valid).squeeze(-1).bool().repeat(b,1)
        # Complete all-pairs window attention; no top-k, pooling or approximations.
        bias=self.relative_bias[:,self.relative_index].float()[None].expand(qt.shape[0],-1,-1,-1)
        bias=bias.masked_fill(~valid[:,None,None,:],-torch.inf)
        with torch.autocast(device_type=target.device.type,enabled=False):
            attended=F.scaled_dot_product_attention(qt.float(),kt.float(),vt.float(),attn_mask=bias,dropout_p=0.)
        tokens=attended.transpose(1,2).reshape(-1,s*s,d).to(v.dtype)
        spatial=tokens.reshape(b,hp//s,wp//s,s,s,d).permute(0,5,1,3,2,4).reshape(b,d,hp,wp)[...,:h,:w]
        return self.out(spatial)

class ContextGuidedWaveletFusion(nn.Module):
    """Single scale returns (fused, ct_plus, pet_plus)."""
    def __init__(self, channels: int, dim: int=32, heads: int=4,
                 window: int=8, checkpoint_attention: bool=True):
        super().__init__()
        if channels<1 or dim<8 or heads<1 or dim%heads or window<1:
            raise ValueError('Require channels>0, dim>=8, dim%heads==0, window>0')
        self.channels,self.checkpoint_attention=channels,checkpoint_attention
        self.ct_project=nn.Sequential(nn.GroupNorm(math.gcd(channels,8),channels),nn.Conv2d(channels,dim,1))
        self.pet_project=nn.Sequential(nn.GroupNorm(math.gcd(channels,8),channels),nn.Conv2d(channels,dim,1))
        self.wavelet=HaarWavelet2D()
        self.ct_exchange=nn.ModuleDict({b:_WindowExchange(dim,heads,window,b!='LL') for b in BANDS})
        self.pet_exchange=nn.ModuleDict({b:_WindowExchange(dim,heads,window,b!='LL') for b in BANDS})
        self.ct_out,self.pet_out=nn.Conv2d(dim,channels,1),nn.Conv2d(dim,channels,1)
        for output in (self.ct_out,self.pet_out):
            nn.init.zeros_(output.weight);nn.init.zeros_(output.bias)

    def _exchange(self, block, *args):
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            return checkpoint(block,*args,use_reentrant=False)
        return block(*args)

    def forward(self, ct: Tensor, pet: Tensor | None=None, mode: str='full'):
        if mode not in ('full','missing'):raise ValueError('mode must be full or missing')
        if ct.ndim!=4 or ct.shape[1]!=self.channels or min(ct.shape)<1 or not ct.is_floating_point():
            raise ValueError('Invalid CT BCHW/channels/dtype')
        if mode=='missing':return ct,ct,torch.zeros_like(ct)
        if pet is None:raise ValueError('Full requires PET features')
        if pet.shape!=ct.shape or pet.device!=ct.device or pet.dtype!=ct.dtype:
            raise ValueError('CT/PET shape/device/dtype mismatch')
        xc,xp=self.ct_project(ct),self.pet_project(pet)
        cb,size=self.wavelet.analysis(xc);pb,_=self.wavelet.analysis(xp)
        # Both LL updates use original bands; no sequential modality preference.
        lc=cb[0]+self._exchange(self.ct_exchange['LL'],cb[0],pb[0])
        lp=pb[0]+self._exchange(self.pet_exchange['LL'],pb[0],cb[0])
        uc,up=[lc],[lp]
        for i,b in enumerate(BANDS[1:],1):
            uc.append(cb[i]+self._exchange(self.ct_exchange[b],cb[i],pb[i],lc,lp))
            up.append(pb[i]+self._exchange(self.pet_exchange[b],pb[i],cb[i],lp,lc))
        yc,yp=self.wavelet.synthesis(uc,size),self.wavelet.synthesis(up,size)
        # Original C/P appear only once; branch's pre-existing content subtracted.
        dc=self.ct_out(yc-xc).to(ct.dtype);dp=self.pet_out(yp-xp).to(pet.dtype)
        ce,pe=ct+dc,pet+dp
        return ce+pe,ce,pe

class ContextGuidedWaveletFusionPyramid(nn.Module):
    """Fine-to-coarse four scales. Use baseline's upstream Full/Missing routing.

    Selected Full rows can be passed normally, as in the clean baseline auto path.
    This wrapper deliberately does not re-encode PET or implement a missing bank.
    """
    def __init__(self, channels: Sequence[int]=(64,128,320,512),dim: int=32,
                 heads: int=4,window: int=8,checkpoint_attention: bool=True):
        super().__init__()
        self.channels=tuple(channels)
        if len(self.channels)!=4:raise ValueError('Exactly four scales required')
        self.scales=nn.ModuleList([ContextGuidedWaveletFusion(c,dim,heads,window,checkpoint_attention) for c in self.channels])
    def forward_features(self,ct: Sequence[Tensor],pet: Sequence[Tensor] | None=None,mode: str='full'):
        if not isinstance(ct,(list,tuple)) or len(ct)!=4:raise ValueError('CT requires four scales')
        if mode not in ('full','missing'):raise ValueError('mode must be full or missing')
        if mode=='full' and (not isinstance(pet,(list,tuple)) or len(pet)!=4):raise ValueError('Full requires four PET scales')
        if mode == 'full' and any(p.dtype != c.dtype for p, c in zip(pet, ct)):
            # AMP autocast can leave CT (BatchNorm tail, fp32) and PET (fp16)
            # in different dtypes. Align PET to the CT dtype once here; this
            # changes representation only, not the fusion computation (all
            # attention math already runs in fp32 internally).
            pet = [p.to(dtype=c.dtype) if p.dtype != c.dtype else p
                   for p, c in zip(pet, ct)]
        outputs=[[],[],[]]
        for i,block in enumerate(self.scales):
            if i and (ct[i].shape[0]!=ct[0].shape[0] or ct[i].device!=ct[0].device or ct[i].dtype!=ct[0].dtype or any(a>b for a,b in zip(ct[i].shape[-2:],ct[i-1].shape[-2:]))):
                raise ValueError('Scales require shared batch/device/dtype, fine-to-coarse order')
            result=block(ct[i],pet[i] if mode=='full' else None,mode)
            for out,t in zip(outputs,result):out.append(t)
        return tuple(outputs)
    def forward(self,ct,pet=None,mode='full'):
        return self.forward_features(ct,pet,mode)[0]


def _tests():
    import unittest
    import io
    from unittest.mock import patch
    torch.set_num_threads(2)
    class Tests(unittest.TestCase):
        def setUp(self):torch.manual_seed(2026)
        def test_haar_reconstruction_energy_and_gradcheck(self):
            wave=HaarWavelet2D()
            for shape in ((2,3,8,6),(1,2,5,7)):
                x=torch.randn(*shape,dtype=torch.double,requires_grad=True)
                bands,size=wave.analysis(x);y=wave.synthesis(bands,size)
                torch.testing.assert_close(x,y)
                self.assertTrue(torch.autograd.gradcheck(lambda t: wave.synthesis(*wave.analysis(t)),(x,)))
                if shape[-1]%2==0:
                    torch.testing.assert_close(sum(t.square().sum() for t in bands),x.square().sum())
            x=torch.randn(1,1,4,4,dtype=torch.double,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda t: torch.cat(wave.analysis(t)[0],1),(x,)))
        def test_window_against_explicit_reference(self):
            m=_WindowExchange(8,2,4,False)
            with torch.no_grad():m.relative_bias.normal_(0,.1)
            x,y=torch.randn(1,8,5,6),torch.randn(1,8,5,6)
            actual=m(x,y);q,k,v=m.q(x),m.k(y),m.v(y)
            rows=[]
            for yy in range(5):
                row=[]
                for xx in range(6):
                    scores,vals=[],[]
                    for sy in range((yy//4)*4,min((yy//4+1)*4,5)):
                        for sx in range((xx//4)*4,min((xx//4+1)*4,6)):
                            qi=q[0,:,yy,xx].reshape(2,4);ki=k[0,:,sy,sx].reshape(2,4)
                            index=(yy-sy+3)*7+xx-sx+3
                            scores.append((qi*ki).sum(-1)/2+m.relative_bias[:,index])
                            vals.append(v[0,:,sy,sx].reshape(2,4))
                    w=torch.stack(scores,-1).softmax(-1)
                    row.append((w[:,:,None]*torch.stack(vals,1)).sum(1).flatten())
                rows.append(torch.stack(row,-1))
            expected=m.out(torch.stack(rows,-2)[None])
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        def test_zero_init_two_step_all_gradients_and_context(self):
            m=ContextGuidedWaveletFusion(16,dim=8,heads=2,window=4)
            c,p=[torch.randn(2,16,9,7,requires_grad=True) for _ in range(2)]
            f,ce,pe=m(c,p)
            self.assertTrue(torch.equal(f,c+p));self.assertTrue(torch.equal(ce,c));self.assertTrue(torch.equal(pe,p))
            opt=torch.optim.SGD(m.parameters(),lr=.05)
            (f-torch.randn_like(f)).square().mean().backward()
            self.assertGreater(m.ct_out.weight.grad.abs().sum().item(),0)
            opt.step();opt.zero_grad();m(c,p)[0].square().mean().backward()
            for name,t in m.named_parameters():
                self.assertIsNotNone(t.grad,name);self.assertTrue(torch.isfinite(t.grad).all(),name)
                self.assertGreater(t.grad.abs().sum().item(),0,name)
            with patch.object(m.wavelet,'analysis',side_effect=AssertionError('Missing called DWT')):
                self.assertTrue(torch.equal(m(c,None,'missing')[0],c))
            with self.assertRaises(ValueError):m(c,None)
        def test_checkpoint_equivalence_roundtrip_amp(self):
            m=ContextGuidedWaveletFusion(16,8,2,4,True)
            nn.init.normal_(m.ct_out.weight,std=.01);nn.init.normal_(m.pet_out.weight,std=.01)
            n=ContextGuidedWaveletFusion(16,8,2,4,False);n.load_state_dict(m.state_dict())
            c,p=torch.randn(2,16,7,9),torch.randn(2,16,7,9)
            a,b=m(c,p)[0],n(c,p)[0];torch.testing.assert_close(a,b)
            a.square().mean().backward();b.square().mean().backward()
            for x,y in zip(m.parameters(),n.parameters()):torch.testing.assert_close(x.grad,y.grad)
            buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);n.load_state_dict(torch.load(buf,weights_only=True))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=m(c,p)[0];self.assertTrue(torch.isfinite(out).all());out.mean().backward()
        def test_pyramid(self):
            m=ContextGuidedWaveletFusionPyramid(dim=8,heads=2,window=4)
            c=[torch.randn(1,d,n,n) for d,n in zip(m.channels,(16,8,4,2))]
            p=[torch.randn_like(t) for t in c]
            for out,x,y in zip(m(c,p),c,p):self.assertTrue(torch.equal(out,x+y))
            for out,x in zip(m(c,mode='missing'),c):self.assertTrue(torch.equal(out,x))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--test',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--full-size',action='store_true');parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if args.test:_tests()
    if args.smoke:
        torch.set_num_threads(2);torch.manual_seed(2026)
        m=ContextGuidedWaveletFusionPyramid().to(args.device)
        sizes=(128,64,32,16) if args.full_size else (16,8,4,2)
        c=[torch.randn(args.batch_size,d,n,n,device=args.device,requires_grad=True) for d,n in zip(m.channels,sizes)]
        p=[torch.randn_like(t) for t in c]
        f=m(c,p);assert all(torch.equal(t,x+y) for t,x,y in zip(f,c,p))
        sum(t.square().mean() for t in f).backward()
        print('PASS full-size forward/backward, exact initial addition:',[tuple(t.shape) for t in f])
        print('Trainable parameters:',sum(t.numel() for t in m.parameters()))
