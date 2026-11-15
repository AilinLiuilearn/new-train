"""CC-SCMI V1. Paper-inspired routing; custom ST mask and per-dimension matching."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from third_party.ktb_sampling import sample_union


class Router(nn.Module):
    def __init__(self,width):
        super().__init__();self.score=nn.Sequential(nn.Linear(width,width//4),nn.GELU(),nn.Linear(width//4,width))

    def forward(self,x,extra=None,capture=False):
        with torch.autocast(device_type=x.device.type,enabled=False):
            score=self.score(x.float().mean((-2,-1)))
            if extra is not None:score=score+extra.float()
        k=score.shape[1]//2;order=score.argsort(dim=1,descending=True,stable=True)
        hard=torch.zeros_like(score).scatter(1,order[:,:k],1.)
        if self.training:
            sorted_score=score.gather(1,order)
            threshold=((sorted_score[:,k-1]+sorted_score[:,k])/2).detach()
            soft=torch.sigmoid(score-threshold[:,None])
            mask=hard+(soft-soft.detach())
        else:mask=hard
        if capture and score.requires_grad:score.retain_grad()
        return mask,score,order[:,:k]


class Offset(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,grid):
        delta=self.net(F.adaptive_avg_pool2d(x,(8,8))).permute(0,2,3,1).float().tanh()/8
        return grid+delta,delta


class Direction(nn.Module):
    def __init__(self,width,per_dimension=True):
        super().__init__();self.per_dimension=per_dimension
        self.q=nn.Linear(32,32,bias=False);self.k=nn.Linear(32,32,bias=False);self.v=nn.Linear(32,32,bias=False)
        self.position=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.position[-1].weight);nn.init.zeros_(self.position[-1].bias)
        self.beta=nn.Parameter(torch.zeros(()));self.out=nn.Conv2d(32,width,1,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self,receiver,donor,ct_points,pet_points,grid,mask,capture=False):
        # Explicit disabled autocast: the scores, softmax and weighted sum really are FP32.
        with torch.autocast(device_type=receiver.device.type,enabled=False):
            r=F.adaptive_avg_pool2d(receiver.float(),(8,8)).flatten(2).transpose(1,2)
            samples=sample_union(donor.float(),ct_points.float(),pet_points.float())
            q=F.layer_norm(self.q(r),(32,),eps=1e-5);k=F.layer_norm(self.k(samples),(32,),eps=1e-5)
            v=self.v(samples)
            coords=torch.cat([ct_points.flatten(1,2),pet_points.flatten(1,2)],1)
            bias=self.position(grid.flatten(1,2)[:,:,None,:]-coords[:,None,:,:]).squeeze(-1)
            product=q[:,:,None,:]*k[:,None,:,:]
            spatial=product.sum(-1)/math.sqrt(32)+bias
            correction=product if self.per_dimension else product.mean(-1,keepdim=True)*math.sqrt(32)
            scores=spatial[:,:,:,None]+self.beta.tanh()*correction
            attention=scores.softmax(dim=2)
            output=(attention*v[:,None,:,:]).sum(2).transpose(1,2).reshape(-1,32,8,8)
            info={}
            if capture:
                a=attention.detach()
                info={'attention_entropy':-(a*a.clamp_min(1e-12).log()).sum(2).mean(),
                      'attention_channel_std':a.std(-1,unbiased=False).mean(),
                      'score_correction_rms':(self.beta.tanh()*correction.detach()).square().mean().sqrt(),
                      'beta':self.beta.detach().tanh()}
        delta=self.out(F.interpolate(output,size=receiver.shape[-2:],mode='bilinear',align_corners=False))
        return delta*mask.to(delta.dtype)[:,:,None,None],info


class Scale(nn.Module):
    def __init__(self,width,per_dimension):
        super().__init__();self.ct_router=Router(width);self.pet_router=Router(width)
        self.ct_project=nn.Conv2d(width,32,1,bias=False);self.pet_project=nn.Conv2d(width,32,1,bias=False)
        self.ct_offset=Offset();self.pet_offset=Offset()
        self.to_ct=Direction(width,per_dimension);self.to_pet=Direction(width,per_dimension)
        x=(torch.arange(8,dtype=torch.float32)*2+1)/8-1;y,x=torch.meshgrid(x,x,indexing='ij')
        self.register_buffer('grid',torch.stack([x,y],-1)[None])

    def forward(self,c,p,text_scores=None,capture=False):
        ec,ep=(None,None) if text_scores is None else text_scores
        mc,sc,ic=self.ct_router(c,ec,capture);mp,sp,ip=self.pet_router(p,ep,capture)
        lc=self.ct_project(c*mc[:,:,None,None]);lp=self.pet_project(p*mp[:,:,None,None])
        grid=self.grid.expand(c.shape[0],-1,-1,-1)
        pc,oc=self.ct_offset(lc,grid);pp,op=self.pet_offset(lp,grid)
        dc,ac=self.to_ct(lc,lp,pc,pp,grid,mc,capture);dp,ap=self.to_pet(lp,lc,pc,pp,grid,mp,capture)
        info={}
        if capture:info={'ct_score':sc,'pet_score':sp,'ct_indices':ic,'pet_indices':ip,
                        'ct_mask':mc.detach(),'pet_mask':mp.detach(),'ct_offset':oc,'pet_offset':op,
                        'ct_points':pc,'pet_points':pp,'ct_delta':dc,'pet_delta':dp,
                        'ct_attention':ac,'pet_attention':ap}
        return c+dc,p+dp,info


class TextCondition(nn.Module):
    def __init__(self,width):
        super().__init__();self.visual=nn.ModuleList(nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32)) for _ in range(2))
        self.text=nn.ModuleList(nn.Sequential(nn.Linear(512,32),nn.LayerNorm(32)) for _ in range(2))
        self.score=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,width))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,c,p,text):
        return [self.score(torch.cat([self.visual[i](x.mean((-2,-1))),
                  self.text[i](text[i]).expand(x.shape[0],-1)],-1)) for i,x in enumerate([c,p])]


class CCSCMI(nn.Module):
    def __init__(self,text_features,text_enabled=False,per_dimension=True):
        super().__init__();self.text_enabled=text_enabled
        self.register_buffer('text_features',text_features.detach().float().clone())
        self.blocks=nn.ModuleList(Scale(d,per_dimension) for d in [64,128,320,512])
        # Create optional layers only after all common visual parameters, within a separate RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2087)
            self.conditions=nn.ModuleList(TextCondition(d) for d in [64,128,320,512]) if text_enabled else None

    def forward(self,ct,pet,capture=False):
        cs=[];ps=[];details=[]
        for i,(block,c,p) in enumerate(zip(self.blocks,ct,pet)):
            text=self.conditions[i](c,p,self.text_features) if self.text_enabled else None
            a,b,d=block(c,p,text,capture);cs.append(a);ps.append(b);details.append(d)
        return cs,ps,details
