import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.c2maot_w1_loss import C2MAOTHierarchicalW1Loss
from models.pg_mtr import PETGroundedMetabolicTokenRetrieval, _valid_group_count


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualDecoderPGMTRC2MAOTOT(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512,256,128,64), use_deep_supervision=False, pg_mtr_stages='all', pg_mtr_num_tokens=8, pg_mtr_temperature=0.07, pg_mtr_detach_bank_missing=True, pg_mtr_ot_alpha=1.5):
        super().__init__(); self.use_deep_supervision=bool(use_deep_supervision); self.pg_mtr_detach_bank_missing=bool(pg_mtr_detach_bank_missing)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels); self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder'); load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels()); pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels); self.fusion = AddFusion(); self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision); self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.pg_mtr = PETGroundedMetabolicTokenRetrieval(pet_channels, num_tokens=pg_mtr_num_tokens, temperature=pg_mtr_temperature, stage_mode=pg_mtr_stages)
        self.retrieval_adapters = nn.ModuleDict();
        for s in self.pg_mtr.active_stage_numbers:
            ch = pet_channels[s-1]; adapter = nn.Sequential(nn.Conv2d(self.pg_mtr.latent_dim, ch, 1, bias=False), nn.GroupNorm(_valid_group_count(ch), ch, affine=False)); nn.init.kaiming_normal_(adapter[0].weight, mode='fan_out', nonlinearity='linear'); self.retrieval_adapters[str(s)] = adapter; self.register_parameter(f'gamma_s{s}', nn.Parameter(torch.tensor(0.01).log()))
        self.stage_queries = nn.ModuleDict({str(s): self.pg_mtr.stage_modules[str(s)] for s in self.pg_mtr.active_stage_numbers})
        self.c2maot_ot_loss = C2MAOTHierarchicalW1Loss(alpha=pg_mtr_ot_alpha)

    @staticmethod
    def _to_3ch(x): return x.repeat(1,3,1,1) if x.shape[1]==1 else x
    def _encode_ct(self, ct): ct=self._to_3ch(ct); feats=self.enc_ct(ct); _check_tensor_list('ct_feats', feats); return self.ct_align(feats)
    def _encode_pet(self, pet): pet=self._to_3ch(pet); feats=self.enc_pet(pet); _check_tensor_list('pet_feats', feats); return feats
    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}
    def _decode_with(self, decoder, features, target_size): out=self._finalize_decoder_output(decoder(features, target_size)); _check_tensor('logits', out['logits']); out['pred']=out['logits']; out['aux']={}; return out
    def _forward_full(self, ct, pet, target_size, mask=None):
        aligned=self._encode_ct(ct); petf=self._encode_pet(pet); out=self._decode_with(self.full_decoder, self.fusion(aligned, petf, pet_available=None), target_size)
        if self.training:
            ot_ct_feats = [feat.detach() for feat in aligned]
            retrieved, _, retrieval_diag = self.pg_mtr(ot_ct_feats, mode='missing', detach_bank=False)
            source_feats = {}
            target_feats = {}
            for s in self.pg_mtr.active_stage_numbers:
                source = self.retrieval_adapters[str(s)](retrieved[s])
                target = petf[s - 1].detach()
                if source.shape != target.shape:
                    raise RuntimeError(f'OT stage {s} shape mismatch: source={tuple(source.shape)} target={tuple(target.shape)}')
                source_feats[s] = source
                target_feats[s] = target
            ot_loss, ot_diag = self.c2maot_ot_loss(source_feats, target_feats, self.pg_mtr.active_stage_numbers)
            diag = dict(retrieval_diag)
            diag.update(ot_diag)
            out['aux_losses'] = {'pg_mtr_ot_loss': ot_loss}
            out['diagnostics'] = diag
        else:
            out['aux_losses'] = {}
            out['diagnostics'] = {}
        return out
    def _rms_tensor(self, x): return x.detach().float().pow(2).mean().add(1e-12).sqrt()
    def _apply_retrieved_memory(self, aligned, retrieved, diag):
        missing=list(aligned); out=dict(diag)
        for s in self.pg_mtr.active_stage_numbers:
            i=s-1; feat=self.retrieval_adapters[str(s)](retrieved[s]); gamma=torch.nn.functional.softplus(getattr(self, f'gamma_s{s}')); delta=gamma*feat; missing[i]=aligned[i]+delta; out[f'pg_mtr_s{s}_gamma']=gamma.detach(); out[f'pg_mtr_s{s}_ct_rms']=self._rms_tensor(aligned[i]); out[f'pg_mtr_s{s}_injection_rms']=self._rms_tensor(delta); out[f'pg_mtr_s{s}_injection_ct_ratio']=out[f'pg_mtr_s{s}_injection_rms']/(out[f'pg_mtr_s{s}_ct_rms']+1e-6)
        return missing, out
    def _forward_missing(self, ct, target_size):
        aligned=self._encode_ct(ct); retrieved,_,diag=self.pg_mtr(aligned, mode='missing', detach_bank=self.pg_mtr_detach_bank_missing); missing,diag=self._apply_retrieved_memory(aligned, retrieved, diag); out=self._decode_with(self.missing_decoder, missing, target_size); out['aux_losses']={}; out['diagnostics']=diag; return out
    def _merge_outputs(self, full_outputs, missing_outputs, full_idx, missing_idx, batch_size):
        merged={}; logits=full_outputs['logits'].new_zeros((batch_size,)+tuple(full_outputs['logits'].shape[1:]));
        if full_idx.numel()>0: logits=logits.index_copy(0, full_idx, full_outputs['logits'])
        if missing_idx.numel()>0: logits=logits.index_copy(0, missing_idx, missing_outputs['logits'])
        merged['logits']=logits; merged['pred']=logits; merged['aux']={}; return merged
    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            pa = pet_available.long().view(-1)
            if torch.all(pa == 1):
                return self._forward_full(ct, pet, target_size, mask=mask)
            if torch.all(pa == 0):
                return self._forward_missing(ct, target_size)
            aligned = self._encode_ct(ct)
            full_idx = torch.nonzero(pa == 1, as_tuple=False).flatten()
            missing_idx = torch.nonzero(pa == 0, as_tuple=False).flatten()
            full_outputs = None
            missing_outputs = None
            if full_idx.numel() > 0:
                full_ct = [f.index_select(0, full_idx) for f in aligned]
                pet_full = pet.index_select(0, full_idx)
                pet_full_feats = self._encode_pet(pet_full)
                full_outputs = self._decode_with(self.full_decoder, self.fusion(full_ct, pet_full_feats, pet_available=None), target_size)
                full_outputs['aux_losses'] = {}
                full_outputs['diagnostics'] = {}
            if missing_idx.numel() > 0:
                miss_ct = [f.index_select(0, missing_idx) for f in aligned]
                retrieved, _, diag = self.pg_mtr(miss_ct, mode='missing', detach_bank=self.pg_mtr_detach_bank_missing)
                miss_feat, diag = self._apply_retrieved_memory(miss_ct, retrieved, diag)
                missing_outputs = self._decode_with(self.missing_decoder, miss_feat, target_size)
                missing_outputs['diagnostics'] = diag
            if full_outputs is None:
                return missing_outputs
            if missing_outputs is None:
                return full_outputs
            return self._merge_outputs(full_outputs, missing_outputs, full_idx, missing_idx, ct.shape[0])
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}. Expected \'full\', \'missing\', or \'auto\'.')
