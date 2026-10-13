"""GT-audited paired correction queues for 2D PET-CT segmentation.

Design adaptation of PNDC's uncertainty/region calibration and SAR-Net's
paired dynamic memory, not an exact implementation of either paper.
No missing PET reconstruction, K-means or feature-distillation objective.

Physical keys store detached CT descriptors BEFORE projection. A shared live
projection maps both query and stored descriptors at read time, preventing
stale projection weights from invalidating older projected keys. Source
encoder drift is separately limited by momentum sourcing, TTL and FIFO.
All queue buffers are persistent tensors and must be COPIED, never EMA averaged.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def audit_regions(reference_logits, plus_logits, target, feature_size,
                  margin=.02, min_support=.6, max_negative_fraction=.1):
    """Evaluate at original GT resolution; reduce FG/BG separately per cell.

Low entropy alone never marks a positive region: gain and GT support are
required. Binary entropy is -p log p -(1-p)log(1-p), including empty-GT slices.
Adaptive masked means keep tiny foreground represented in coarse cells.
Returned full-resolution negative mask also covers confident wrong Full
predictions when the CT reference was correct (PNDC-inspired checksum).
"""
    size = target.shape[-2:]
    ref = F.interpolate(reference_logits.float(), size=size, mode='bilinear', align_corners=False)
    plus = F.interpolate(plus_logits.float(), size=size, mode='bilinear', align_corners=False)
    y = (target.float() >= .5).float()
    er = F.binary_cross_entropy_with_logits(ref, y, reduction='none')
    ep = F.binary_cross_entropy_with_logits(plus, y, reduction='none')
    gain = er-ep
    p = plus.sigmoid().clamp(1e-6, 1-1e-6)
    probability_correct = y*p + (1-y)*(1-p)
    correct = p.ge(.5).eq(y.bool())
    reference_correct = ref.ge(0).eq(y.bool())
    entropy = -(p*p.log()+(1-p)*(1-p).log())
    confidence = (1-entropy/math.log(2)).clamp(0,1)
    negative = (gain < -margin) | (reference_correct & ~correct & (confidence > .5))

    pool = lambda x: F.adaptive_avg_pool2d(x, feature_size)
    fg_mass, bg_mass = pool(y), pool(1-y)
    fg_present, bg_present = fg_mass.gt(0), bg_mass.gt(0)
    denom = fg_present.float()+bg_present.float()

    def balanced(x):
        fg = pool(x*y)/fg_mass.clamp_min(1e-12)
        bg = pool(x*(1-y))/bg_mass.clamp_min(1e-12)
        return (fg*fg_present+bg*bg_present)/denom.clamp_min(1)

    cell_gain = balanced(gain)
    support = balanced(probability_correct)
    negative_fraction = balanced(negative.float())
    # A cell containing lesion pixels cannot qualify through background alone.
    fg_gain = pool(gain*y)/fg_mass.clamp_min(1e-12)
    fg_support = pool(probability_correct*y)/fg_mass.clamp_min(1e-12)
    eligible = (cell_gain > margin) & (support >= min_support)
    eligible &= (~fg_present) | ((fg_gain > margin) & (fg_support >= min_support))
    eligible &= negative_fraction <= max_negative_fraction
    reliability = (balanced(confidence)*torch.sigmoid(cell_gain/.1)).clamp(0,1)
    reliability = reliability*eligible
    return {'eligible': eligible, 'reliability': reliability, 'gain': cell_gain,
            'foreground': fg_present, 'negative': negative.detach(),
            'positive': ((gain > margin)&correct).detach()}


class PairedCorrectionMemory(nn.Module):
    def __init__(self, capacity, key_channels, value_channels, max_age=128):
        super().__init__()
        if min(capacity,key_channels,value_channels,max_age) <= 0:
            raise ValueError('capacity/channels/max_age must be positive')
        self.capacity, self.max_age = int(capacity), int(max_age)
        self.register_buffer('raw_keys', torch.zeros(capacity,key_channels))
        self.register_buffer('values', torch.zeros(capacity,value_channels))
        self.register_buffer('reliability', torch.zeros(capacity))
        self.register_buffer('sample_ids', torch.full((capacity,),-1,dtype=torch.long))
        self.register_buffer('group_ids', torch.full((capacity,),-1,dtype=torch.long))
        self.register_buffer('written_at', torch.full((capacity,),-1,dtype=torch.long))
        self.register_buffer('valid', torch.zeros(capacity,dtype=torch.bool))
        self.register_buffer('write_ptr', torch.zeros((),dtype=torch.long))
        # Training-GT-derived metadata ONLY for diagnostics. Never used in
        # matching, admission at query time, or correction weights.
        self.register_buffer('foreground',torch.zeros(capacity,dtype=torch.bool))
        self.register_buffer('source_gain',torch.zeros(capacity))
        self.register_buffer('audit_known',torch.zeros(capacity,dtype=torch.bool))

    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        # Old queues contain no audit metadata: mark it unknown rather than
        # fabricating foreground/gain measurements; original retrieval is unchanged.
        for name in ('foreground','source_gain','audit_known'):
            state_dict.setdefault(prefix+name,torch.zeros_like(getattr(self,name)))
        super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)

    @torch.no_grad()
    def enqueue(self, keys, values, reliability, sample_ids, group_ids, step,
                foreground=None,source_gain=None):
        n=keys.shape[0]
        if not n:
            return
        if keys.shape != (n,self.raw_keys.shape[1]) or values.shape != (n,self.values.shape[1]):
            raise ValueError('queue key/value shape mismatch')
        if any(x.shape != (n,) for x in (reliability,sample_ids,group_ids)):
            raise ValueError('queue metadata shape mismatch')
        if not all(torch.isfinite(x).all() for x in (keys,values,reliability)):
            raise ValueError('non-finite memory candidate')
        known=foreground is not None and source_gain is not None
        if known and (foreground.shape!=(n,) or source_gain.shape!=(n,) or not torch.isfinite(source_gain).all()):
            raise ValueError('invalid diagnostic metadata')
        # Keep the latest capacity candidates, preserving paired indices.
        n=min(n,self.capacity)
        keys,values,reliability,sample_ids,group_ids = [x.detach()[-n:].to(self.raw_keys.device)
                                                      for x in (keys,values,reliability,sample_ids,group_ids)]
        indices=(torch.arange(n,device=self.raw_keys.device)+self.write_ptr)%self.capacity
        self.raw_keys[indices]=keys.float()
        self.values[indices]=values.float()
        self.reliability[indices]=reliability.float().clamp(0,1)
        self.sample_ids[indices]=sample_ids.long()
        self.group_ids[indices]=group_ids.long()
        self.written_at[indices]=int(step)
        self.valid[indices]=True
        self.foreground[indices]=foreground.detach()[-n:].to(self.raw_keys.device).bool() if known else False
        self.source_gain[indices]=source_gain.detach()[-n:].to(self.raw_keys.device).float() if known else 0.
        self.audit_known[indices]=known
        self.write_ptr.copy_((self.write_ptr+n)%self.capacity)

    def allowed(self, sample_id, group_id, step, exclude_group=False):
        ok=self.valid & (self.written_at <= step) & ((step-self.written_at) <= self.max_age)
        if int(sample_id) >= 0:
            ok=ok & self.sample_ids.ne(sample_id)
        if exclude_group and int(group_id) >= 0:
            ok=ok & self.group_ids.ne(group_id)
        return ok


class CorrectionRetriever(nn.Module):
    def __init__(self, ct_channels, decoder_channels, capacity=2048, key_dim=32,
                 topk=4, query_chunk=256, temperature=.1, min_entries=16,
                 max_age=128, exclude_group=False):
        super().__init__()
        if min(key_dim,topk,query_chunk,min_entries) <= 0 or min_entries > capacity or temperature<=0:
            raise ValueError('invalid retrieval configuration')
        self.topk,self.query_chunk,self.temperature = int(topk),int(query_chunk),float(temperature)
        self.min_entries,self.exclude_group=int(min_entries),bool(exclude_group)
        self.memory=PairedCorrectionMemory(capacity,ct_channels,decoder_channels,max_age)
        self.key_projection=nn.Linear(ct_channels,key_dim,bias=False)
        hidden=max(8,decoder_channels//4)
        self.context=nn.Sequential(nn.Conv2d(decoder_channels,hidden,1),nn.GELU(),
                                   nn.Conv2d(hidden,decoder_channels,1),nn.Tanh())
        self.output=nn.Conv2d(decoder_channels,decoder_channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, h, ct_query, sample_ids, group_ids, step,
                return_diagnostics=False,return_maps=False):
        return_diagnostics=bool(return_diagnostics or return_maps)
        if ct_query.shape[-2:] != h.shape[-2:]:
            ct_query=F.interpolate(ct_query,size=h.shape[-2:],mode='bilinear',align_corners=False)
        reads=[]; hits=[]
        diagnostic_maps={name:[] for name in ('top1_cos','source_rel','attention_entropy',
                                             'known_fg_mass','metadata_known_mass')}
        available=[]
        # Float32 similarity/normalization even inside AMP. Block queries, never
        # allocate a B*H*W*capacity dense attention tensor.
        with torch.autocast(device_type=h.device.type,enabled=False):
            for b in range(h.shape[0]):
                allowed=self.memory.allowed(sample_ids[b],group_ids[b],step,self.exclude_group)
                n=int(allowed.sum())
                available.append(n)
                if n < self.min_entries:
                    reads.append(torch.zeros_like(h[b],dtype=torch.float32))
                    hits.append(torch.zeros((1,*h.shape[-2:]),device=h.device,dtype=torch.bool))
                    if return_diagnostics:
                        for name in diagnostic_maps:
                            diagnostic_maps[name].append(torch.zeros((1,*h.shape[-2:]),device=h.device))
                    continue
                keys=F.normalize(self.key_projection(self.memory.raw_keys[allowed].detach().float()),dim=-1,eps=1e-6)
                vals=self.memory.values[allowed].detach().float()
                rel=self.memory.reliability[allowed].detach().float()
                if return_diagnostics:
                    known=self.memory.audit_known[allowed].float()
                    fg=self.memory.foreground[allowed].float()*known
                queries=ct_query[b].flatten(1).T.float()
                pieces=[]
                dp={name:[] for name in diagnostic_maps}
                for qraw in queries.split(self.query_chunk):
                    q=F.normalize(self.key_projection(qraw),dim=-1,eps=1e-6)
                    score,index=(q @ keys.T).topk(min(self.topk,n),dim=1)
                    attention=(score/self.temperature).softmax(dim=1)
                    a=attention*rel[index]
                    pieces.append((a.unsqueeze(-1)*vals[index]).sum(dim=1))
                    if return_diagnostics:
                        with torch.no_grad():
                            ap=attention.detach()
                            dp['top1_cos'].append(score[:,0].detach())
                            dp['source_rel'].append((ap*rel[index]).sum(1))
                            entropy=-(ap*ap.clamp_min(1e-12).log()).sum(1)
                            dp['attention_entropy'].append(entropy/math.log(ap.shape[1]) if ap.shape[1]>1 else entropy*0)
                            dp['known_fg_mass'].append((ap*fg[index]).sum(1))
                            dp['metadata_known_mass'].append((ap*known[index]).sum(1))
                reads.append(torch.cat(pieces).T.reshape_as(h[b]))
                hits.append(torch.ones((1,*h.shape[-2:]),device=h.device,dtype=torch.bool))
                if return_diagnostics:
                    for name in diagnostic_maps:
                        diagnostic_maps[name].append(torch.cat(dp[name]).reshape(1,*h.shape[-2:]))
        r=torch.stack(reads).to(h.dtype)
        hit=torch.stack(hits)
        has_hit=any(n>=self.min_entries for n in available)
        if not has_hit:
            correction=torch.zeros_like(h)
        else:
            correction=self.output(r*(1+self.context(h)))*hit.to(h.dtype)
        # Value-dependent adaptation: no correction at all when R is zero.
        result=h+correction
        if not return_diagnostics:
            return result,hit
        with torch.no_grad():
            maps={name:torch.stack(parts).detach() for name,parts in diagnostic_maps.items()}
            maps['coverage']=hit.detach().float()
            maps['read_rms']=r.detach().float().square().mean(1,keepdim=True).sqrt()
            maps['correction_rms']=correction.detach().float().square().mean(1,keepdim=True).sqrt()
            hrms=h.detach().float().square().mean(1,keepdim=True).sqrt()
            maps['correction_ratio']=maps['correction_rms']/hrms.clamp_min(1e-8)
            count=hit.sum().clamp_min(1)
            info={name:(value*hit).sum()/count for name,value in maps.items()
                  if name not in ('coverage','correction_ratio')}
            if not has_hit:
                for name in ('top1_cos','source_rel','attention_entropy','known_fg_mass','metadata_known_mass'):
                    info[name]=None  # unavailable is not a measured zero match/reliability
            info['coverage']=hit.float().mean()
            info['correction_ratio']=correction.detach().float().square().mean().sqrt()/h.detach().float().square().mean().sqrt().clamp_min(1e-8)
            info['available_mean']=torch.tensor(available,device=h.device,dtype=torch.float32).mean()
            if return_maps:
                info['maps']=maps
        return result,hit,info


@torch.no_grad()
def select_candidates(ct, delta, audit, sample_ids, group_ids, per_sample=16):
    """Bounded paired token selection; half FG quota when available.

Weighted random sampling (seeded by runner) avoids always retaining the same
largest/saturated gain. No negative-region or GT-mask payload enters memory.
"""
    if ct.shape[-2:] != delta.shape[-2:]:
        ct=F.interpolate(ct,size=delta.shape[-2:],mode='bilinear',align_corners=False)
    rows=[]
    for b in range(ct.shape[0]):
        ok=audit['eligible'][b].flatten()
        fg=audit['foreground'][b].flatten()
        weights=audit['reliability'][b].flatten()
        def sample(mask,count):
            ids=mask.nonzero(as_tuple=True)[0]
            if ids.numel()==0 or count==0:
                return ids[:0]
            return ids[torch.multinomial(weights[ids].clamp_min(1e-6),min(count,ids.numel()),replacement=False)]
        first=sample(ok&fg,(per_sample+1)//2)
        remaining=ok.clone(); remaining[first]=False
        chosen=torch.cat([first,sample(remaining,per_sample-first.numel())])
        if chosen.numel():
            n=chosen.numel()
            rows.append((ct[b].flatten(1).T[chosen],delta[b].flatten(1).T[chosen],weights[chosen],
                         sample_ids[b].expand(n),group_ids[b].expand(n),fg[chosen],
                         audit['gain'][b].flatten()[chosen]))
    if not rows:
        return None
    return tuple(torch.cat([r[i] for r in rows]).detach() for i in range(7))
