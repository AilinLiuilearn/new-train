import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
                 asym_fusion_enabled=False, asym_use_text=True, asym_clip_path='pretrained/clip-vit-base-patch32',
                 asym_checkpoint_attention=False, asym_grid_cap=32, asym_pet_dims=(64, 128, 160, 256), asym_heads=4,
                 fusion_text_embeddings=None, decoder_norm='bn'):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        # Keep the original construction order/state_dict identical when the
        # asymmetric fusion is off: AddFusion (no parameters) and the shared
        # decoder are created first, then the fusion is replaced. Swapping the
        # fusion before the decoder would change RNG consumption and silently
        # re-init the decoder.
        self.fusion = AddFusion()
        if decoder_norm not in ('bn', 'group'):
            raise ValueError(f'Unsupported decoder_norm={decoder_norm!r}')
        self.decoder_norm = decoder_norm
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision, norm_type=decoder_norm)
        self.asym_fusion_enabled = bool(asym_fusion_enabled)
        if self.asym_fusion_enabled:
            from models.full_petct_asymmetric_fusion import FullPETCTAsymmetricFusion
            self.fusion = FullPETCTAsymmetricFusion(
                clip_path=asym_clip_path if fusion_text_embeddings is None else None,
                channels=tuple(pet_channels),
                pet_dims=tuple(asym_pet_dims),
                heads=int(asym_heads),
                grid_cap=int(asym_grid_cap),
                use_text=bool(asym_use_text),
                text_embeddings=fusion_text_embeddings,
                checkpoint_attention=bool(asym_checkpoint_attention),
            )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('API-style baseline requires PET input before fusion-time masking')
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _fuse_features(self, ct_feats, pet_feats, state):
        """Unified fusion entry.

        Baseline (asym off): AddFusion ignores the third argument exactly.
        Asymmetric fusion (on): per-row routing. Full rows call the Full-only
        module with ``state='full'``; Missing rows return the aligned-CT
        features without entering the new module. Whole-batch decode once.
        """
        if not self.asym_fusion_enabled:
            return self.fusion(ct_feats, pet_feats, state)
        state = torch.as_tensor(state, device=ct_feats[0].device).view(-1)
        if state.numel() != ct_feats[0].shape[0]:
            raise ValueError('fusion state must contain one value per sample')
        if state.dtype != torch.long:
            if not torch.isfinite(state.float()).all():
                raise ValueError('fusion state must be finite 0/1 values')
            if not torch.all((state == 0) | (state == 1)):
                raise ValueError('fusion state values must be 0 or 1')
            state = state.long()
        elif not torch.all((state == 0) | (state == 1)):
            raise ValueError('fusion state values must be 0 or 1')
        full_idx = state.eq(1).nonzero(as_tuple=True)[0]
        if full_idx.numel() == 0:
            return list(ct_feats)
        batch = ct_feats[0].shape[0]
        dtype = ct_feats[0].dtype
        # AMP: the two heterogeneous backbones may emit different dtypes under
        # autocast (e.g. CT fp16 vs PET fp32). Align PET to the CT dtype with a
        # grad-preserving cast instead of rejecting the batch.
        pet_feats = [p.to(dtype=dtype) if p.dtype != dtype else p for p in pet_feats]
        for c, p in zip(ct_feats, pet_feats):
            if tuple(c.shape) != tuple(p.shape):
                raise ValueError('CT/PET feature shapes must match per scale; no implicit interpolation')
            if c.shape[0] != batch:
                raise ValueError('CT/PET features must share the batch size across scales')
            if p.dtype != c.dtype or c.dtype != dtype:
                raise ValueError('CT/PET features must share dtype across scales')
            if p.device != c.device or c.device != ct_feats[0].device:
                raise ValueError('CT/PET features must share device across scales')
        ct_full = [c.index_select(0, full_idx) for c in ct_feats]
        pet_full = [p.index_select(0, full_idx).to(dtype=c.dtype)
                    for c, p in zip(ct_feats, pet_feats)]
        fused_full = self.fusion(ct_full, pet_full, state='full')
        # AMP: fusion output may be fp16/bf16 while the CT base is fp32.
        return [c.index_copy(0, full_idx, f.to(dtype=c.dtype))
                for c, f in zip(ct_feats, fused_full)]

    def _forward_full(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        state = torch.ones(ct.shape[0], dtype=torch.long, device=ct.device)
        fused_feats = self._fuse_features(ct_feats, pet_feats, state)
        return self._decode(fused_feats, target_size)

    def _forward_missing(self, ct, pet, target_size):
        # Preserve the baseline contract: real PET is still encoded, then the
        # Missing rows are zeroed before fusion (encode-then-zero). With the
        # asymmetric fusion enabled the all-Missing state bypasses the new
        # module and returns the CT identity, so the output equals CT exactly.
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_feats_masked = [torch.zeros_like(feat) for feat in pet_feats_real]
        state = torch.zeros(ct.shape[0], dtype=torch.long, device=ct.device)
        fused_feats = self._fuse_features(ct_feats, pet_feats_masked, state)
        return self._decode(fused_feats, target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        # Strict validation on the RAW state: length-B 0/1 integers or bools.
        # Never .long() first: that would silently truncate 0.5 -> 0.
        raw = torch.as_tensor(pet_available, device=ct.device)
        if raw.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if raw.dtype == torch.bool:
            pet_available = raw.long().view(-1)
        elif raw.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if not torch.all((raw == 0) | (raw == 1)):
                raise ValueError('pet_available values must be 0 or 1')
            pet_available = raw.long().view(-1)
        else:
            raise ValueError('pet_available must be 0/1 integers or bools, no silent float truncation')
        pet_feats_masked = []
        for feat in pet_feats_real:
            availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
            pet_feats_masked.append(feat * availability_mask)
        fused_feats = self._fuse_features(ct_feats, pet_feats_masked, pet_available)
        out = self._decode(fused_feats, target_size)
        out['pet_available'] = pet_available.detach().cpu()
        out['num_full'] = int(pet_available.eq(1).sum())
        out['num_missing'] = int(pet_available.eq(0).sum())
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
