"""Independent Full-add / Missing correction-memory model.

Encoder scales ascending S1..S4; decoder channels ascending d1..d4.
decode_stage reuses the original proj/fuse modules verbatim. Controlled
probes share the CT upstream state; they do not subtract two unrelated heads.
Forward NEVER writes a queue. Only the task commits detached candidates after
a successful optimizer step. Eval needs no GT or reference/PET probe passes.
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.pet_correction_memory import CorrectionRetriever, audit_regions, select_candidates


class PETCorrectionModel(DualSharedAddPETCTBaseline):
    def __init__(self, memory_capacity=2048, key_dim=32, topk=4, query_chunk=256,
                 memory_min_entries=16, memory_max_age=128, exclude_group=False,
                 temperature=.1, decoder_channels=(512,256,128,64), **kwargs):
        if kwargs.get('use_deep_supervision',False):
            raise ValueError('Use the correction stage heads, not baseline deep supervision')
        super().__init__(decoder_channels=decoder_channels, **kwargs)
        self.decoder_value_channels=tuple(reversed(decoder_channels))
        self.ct_aux_decoder=copy.deepcopy(self.decoder)
        # S1 uses the real final classifier; only S2..S4 need new readouts.
        # Do not alias seg_head in ModuleList: EMA would update aliased state twice.
        self.stage_heads=nn.ModuleList([nn.Conv2d(c,1,1) for c in self.decoder_value_channels[1:]])
        self.retrievers=nn.ModuleList([
            CorrectionRetriever(c,d,memory_capacity,key_dim,topk,query_chunk,temperature,
                                memory_min_entries,memory_max_age,exclude_group)
            for c,d in zip((64,128,320,512),self.decoder_value_channels)])
        self.register_buffer('memory_step',torch.zeros((),dtype=torch.long))
        self.register_buffer('retrieval_active',torch.zeros((),dtype=torch.bool))

    def project(self, features):
        return [getattr(self.decoder,f'proj{s+1}')(x) for s,x in enumerate(features)]

    def decode_stage(self, s, skip, previous):
        if s==3:
            if previous is not None:
                raise ValueError('deepest stage has no upstream feature')
            return skip
        upstream=F.interpolate(previous,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return getattr(self.decoder,f'fuse{s+1}')(torch.cat([upstream,skip],dim=1))

    def decode_features(self, projected, query=None, sample_ids=None, group_ids=None):
        hs=[None]*4
        for s in (3,2,1,0):
            h=self.decode_stage(s,projected[s],None if s==3 else hs[s+1])
            if query is not None and bool(self.retrieval_active):
                h,_=self.retrievers[s](h,query[s],sample_ids,group_ids,int(self.memory_step))
            hs[s]=h
        return hs

    def logits(self, hs, target_size):
        return F.interpolate(self.decoder.seg_head(hs[0]),size=target_size,
                             mode='bilinear',align_corners=False)

    def stage_readout(self,s,h):
        return self.decoder.seg_head(h) if s==0 else self.stage_heads[s-1](h)

    def controlled_comparison(self, ct_features, fused_features, projected_ct=None):
        pc=self.project(ct_features) if projected_ct is None else projected_ct
        pf=self.project(fused_features)
        reference=self.decode_features(pc)
        plus=[self.decode_stage(s,pf[s],None if s==3 else reference[s+1]) for s in range(4)]
        return {'delta':[a-b for a,b in zip(plus,reference)],
                'reference_logits':[self.stage_readout(s,x) for s,x in enumerate(reference)],
                'plus_logits':[self.stage_readout(s,x) for s,x in enumerate(plus)]}

    @torch.no_grad()
    def make_candidates(self, ct_features, pet_full, full_index, target,
                        sample_ids, group_ids, margin=.02, min_support=.6,
                        max_negative_fraction=.1, tokens_per_full=16):
        """Called in eval mode on the stable source; reads no queue and writes none."""
        if full_index.numel()==0:
            return [None]*4
        cf=[x.index_select(0,full_index) for x in ct_features]
        fused=self.fusion(cf,self._encode_pet(pet_full))
        details=self.controlled_comparison(cf,fused)
        candidates=[]
        for s in range(4):
            audit=audit_regions(details['reference_logits'][s],details['plus_logits'][s],
                                target.index_select(0,full_index),details['delta'][s].shape[-2:],
                                margin,min_support,max_negative_fraction)
            candidates.append(select_candidates(cf[s],details['delta'][s],audit,
                              sample_ids.index_select(0,full_index),group_ids.index_select(0,full_index),
                              tokens_per_full))
        return candidates

    @torch.no_grad()
    def commit_candidates(self, candidates, step):
        if not self.training:
            raise RuntimeError('Memory writes are forbidden in evaluation')
        for r,payload in zip(self.retrievers,candidates):
            if payload is not None:
                r.memory.enqueue(*payload,step=step)
        self.memory_step.fill_(step)

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto',
                sample_ids=None, group_ids=None, query_features=None, training_details=False):
        b=ct.shape[0]; target_size=ct.shape[-2:] if target_size is None else target_size
        if forward_mode=='full':
            state=torch.ones(b,device=ct.device,dtype=torch.long)
        elif forward_mode=='missing':
            state=torch.zeros(b,device=ct.device,dtype=torch.long)
        elif forward_mode=='auto':
            state=_validate_state(torch.ones(b,dtype=torch.long) if pet_available is None
                                  else pet_available,b).to(ct.device)
        else:
            raise ValueError(f'Unsupported forward_mode={forward_mode}')
        if training_details and not self.training:
            raise RuntimeError('GT audit details are training-only')
        def ids(x):
            t=torch.full((b,),-1,device=ct.device,dtype=torch.long) if x is None else torch.as_tensor(x,device=ct.device,dtype=torch.long)
            if t.shape!=(b,):
                raise ValueError('one stable ID is required per sample')
            return t
        sample_ids,group_ids=ids(sample_ids),ids(group_ids)
        if self.training and bool(self.retrieval_active) and (sample_ids < 0).any():
            raise ValueError('Training retrieval requires stable sample IDs')
        fi=state.eq(1).nonzero(as_tuple=True)[0]; mi=state.eq(0).nonzero(as_tuple=True)[0]
        cf=self._encode_ct(ct)
        projected=self.project(cf)
        logits=ct.new_zeros((b,1,*target_size))
        details={}
        if fi.numel():
            if pet is None or pet.shape[0]!=b:
                raise ValueError('Full rows require a batch-aligned real PET tensor')
            cfull=[x.index_select(0,fi) for x in cf]
            ffull=self.fusion(cfull,self._encode_pet(pet.index_select(0,fi)))
            full_h=self.decode_features(self.project(ffull))
            logits=logits.index_copy(0,fi,self.logits(full_h,target_size).to(logits.dtype))
            if training_details:
                details=self.controlled_comparison(cfull,ffull,[p.index_select(0,fi) for p in projected])
                details.update(ct_full=cfull,fused_full=ffull)
        if mi.numel():
            q=cf if query_features is None else query_features
            if len(q)!=4 or any(x.shape[0]!=b for x in q):
                raise ValueError('query feature batch/scale mismatch')
            missing_h=self.decode_features([x.index_select(0,mi) for x in projected],
                      query=[x.index_select(0,mi) for x in q],
                      sample_ids=sample_ids[mi],group_ids=group_ids[mi])
            logits=logits.index_copy(0,mi,self.logits(missing_h,target_size).to(logits.dtype))
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite correction-model logits')
        out={'logits':logits,'pred':logits,'aux':{},'full_index':fi,'missing_index':mi,
             'num_full':fi.numel(),'num_missing':mi.numel(),**details}
        if training_details:
            out['ct_aux_logits']=self.ct_aux_decoder(cf,target_size)['logits']
        return out


def build_correction_model(cfg):
    return {'model':PETCorrectionModel(
        ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path,pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=cfg.pretrained,decoder_channels=tuple(cfg.decoder_channels),
        memory_capacity=cfg.memory_capacity,key_dim=cfg.memory_key_dim,topk=cfg.memory_topk,
        query_chunk=cfg.memory_query_chunk,memory_min_entries=cfg.memory_min_entries,
        memory_max_age=cfg.memory_max_age,exclude_group=cfg.memory_exclude_group,
        temperature=cfg.memory_temperature)}
