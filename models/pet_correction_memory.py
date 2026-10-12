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

    @torch.no_grad()
    def enqueue(self, keys, values, reliability, sample_ids, group_ids, step):
        n=keys.shape[0]
        if not n:
            return
        if keys.shape != (n,self.raw_keys.shape[1]) or values.shape != (n,self.values.shape[1]):
            raise ValueError('queue key/value shape mismatch')
        if any(x.shape != (n,) for x in (reliability,sample_ids,group_ids)):
            raise ValueError('queue metadata shape mismatch')
        if not all(torch.isfinite(x).all() for x in (keys,values,reliability)):
            raise ValueError('non-finite memory candidate')
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

    def forward(self, h, ct_query, sample_ids, group_ids, step):
        if ct_query.shape[-2:] != h.shape[-2:]:
            ct_query=F.interpolate(ct_query,size=h.shape[-2:],mode='bilinear',align_corners=False)
        reads=[]; hits=[]
        # Float32 similarity/normalization even inside AMP. Block queries, never
        # allocate a B*H*W*capacity dense attention tensor.
        with torch.autocast(device_type=h.device.type,enabled=False):
            for b in range(h.shape[0]):
                allowed=self.memory.allowed(sample_ids[b],group_ids[b],step,self.exclude_group)
                n=int(allowed.sum())
                if n < self.min_entries:
                    reads.append(torch.zeros_like(h[b],dtype=torch.float32))
                    hits.append(torch.zeros((1,*h.shape[-2:]),device=h.device,dtype=torch.bool))
                    continue
                keys=F.normalize(self.key_projection(self.memory.raw_keys[allowed].detach().float()),dim=-1,eps=1e-6)
                vals=self.memory.values[allowed].detach().float()
                rel=self.memory.reliability[allowed].detach().float()
                queries=ct_query[b].flatten(1).T.float()
                pieces=[]
                for qraw in queries.split(self.query_chunk):
                    q=F.normalize(self.key_projection(qraw),dim=-1,eps=1e-6)
                    score,index=(q @ keys.T).topk(min(self.topk,n),dim=1)
                    a=(score/self.temperature).softmax(dim=1)*rel[index]
                    pieces.append((a.unsqueeze(-1)*vals[index]).sum(dim=1))
                reads.append(torch.cat(pieces).T.reshape_as(h[b]))
                hits.append(torch.ones((1,*h.shape[-2:]),device=h.device,dtype=torch.bool))
        r=torch.stack(reads).to(h.dtype)
        hit=torch.stack(hits)
        if not hit.any():
            return h,hit
        # Value-dependent adaptation: no correction at all when R is zero.
        correction=self.output(r*(1+self.context(h)))
        return h+correction*hit.to(h.dtype),hit


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
                         sample_ids[b].expand(n),group_ids[b].expand(n)))
    if not rows:
        return None
    return tuple(torch.cat([r[i] for r in rows]).detach() for i in range(5))
