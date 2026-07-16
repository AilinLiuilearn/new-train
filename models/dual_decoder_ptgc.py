import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_decoder_add_baseline import StageChannelAlign
from models.ptgc_module import PatchTaskGainCompensator


def _strip_prefixes(state_dict):
    prefixes = ('module.', 'model.', 'networks.model.')
    stripped = {}
    for key, value in state_dict.items():
        nk = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        stripped[nk] = value
    return stripped


class DualDecoderPTGC(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        ptgc_base_checkpoint=None,
        ptgc_alpha=0.25,
        ptgc_loss_weight=0.2,
        gpnd_rank_weight=1.0,
        gpnd_support_weight=0.05,
        ptgc_delta_active_threshold=1e-4,
        use_gpnd=True,
        **kwargs,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.ptgc_alpha = float(ptgc_alpha)
        self.ptgc_loss_weight = float(ptgc_loss_weight)
        self.gpnd_rank_weight = float(gpnd_rank_weight)
        self.gpnd_support_weight = float(gpnd_support_weight)
        self.ptgc_delta_active_threshold = float(ptgc_delta_active_threshold)
        self.use_gpnd = bool(use_gpnd)

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=False)
        self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.ptgc = PatchTaskGainCompensator(task_channels=pet_channels[-1])
        self._load_base_checkpoint(ptgc_base_checkpoint)
        self.set_ptgc_training_mode()

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def set_ptgc_training_mode(self):
        for module in (self.enc_ct, self.enc_pet, self.ct_align, self.full_decoder, self.missing_decoder):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        for p in self.ptgc.parameters():
            p.requires_grad = True
        self.ptgc.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.enc_ct.eval(); self.enc_pet.eval(); self.ct_align.eval(); self.full_decoder.eval(); self.missing_decoder.eval(); self.ptgc.train(mode)
        return self

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        ct_feats_raw = self.enc_ct(ct)
        _check_tensor_list('ct_feats_raw', ct_feats_raw)
        return self.ct_align(ct_feats_raw)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode_with_states(self, decoder, features, target_size):
        x1, x2, x3, x4 = features
        d4 = decoder.proj4(x4)
        s3 = decoder.proj3(x3)
        d3 = decoder.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = decoder.proj2(x2)
        d2 = decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = decoder.proj1(x1)
        d1 = decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        logits = decoder.seg_head(d1)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': _sanitize(logits)}, [d1, d2, d3, d4]

    def _decode_missing_with_delta(self, ct_feats, d4_base, delta_d4, target_size):
        x1, x2, x3, x4 = ct_feats
        d4_comp = d4_base.detach() + delta_d4
        s3 = self.missing_decoder.proj3(x3.detach())
        d3 = self.missing_decoder.fuse3(torch.cat([F.interpolate(d4_comp, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = self.missing_decoder.proj2(x2.detach())
        d2 = self.missing_decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = self.missing_decoder.proj1(x1.detach())
        d1 = self.missing_decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        comp_logits = self.missing_decoder.seg_head(d1)
        comp_logits = F.interpolate(comp_logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': _sanitize(comp_logits)}, [d1, d2, d3, d4_comp]

    def _forward_full_train(self, ct, pet, target_size):
        with torch.no_grad():
            ct_feats = self._encode_ct(ct)
            pet_feats = self._encode_pet(pet)
            full_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
            full_out, full_states = self._decode_with_states(self.full_decoder, full_feats, target_size)
            ct_out, ct_states = self._decode_with_states(self.missing_decoder, ct_feats, target_size)
        ptgc_out = self.ptgc(ct_states[-1].detach(), ct_out['logits'].detach())
        comp_out, _ = self._decode_missing_with_delta(ct_feats, ct_states[-1], ptgc_out['delta_d4'], target_size)
        return {
            'logits': comp_out['logits'],
            'ptgc_ct_logits': ct_out['logits'].detach(),
            'ptgc_full_logits': full_out['logits'].detach(),
            'ptgc_comp_logits': comp_out['logits'],
            'ptgc_gain_pred': ptgc_out['gain_pred_signed'],
            'ptgc_benefit_pred': ptgc_out['benefit_pred'],
            'ptgc_delta_d4': ptgc_out['delta_d4'],
            'ptgc_entropy_patch': ptgc_out['entropy_patch'],
        }

    def _forward_full_eval(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
        full_out, _ = self._decode_with_states(self.full_decoder, full_feats, target_size)
        return {'logits': full_out['logits']}

    def _forward_missing(self, ct, target_size):
        ct_feats = self._encode_ct(ct)
        ct_out, ct_states = self._decode_with_states(self.missing_decoder, ct_feats, target_size)
        ptgc_out = self.ptgc(ct_states[-1].detach(), ct_out['logits'].detach())
        comp_out, _ = self._decode_missing_with_delta(ct_feats, ct_states[-1], ptgc_out['delta_d4'], target_size)
        return {'logits': comp_out['logits'], 'ptgc_ct_logits': ct_out['logits'].detach(), 'ptgc_gain_pred': ptgc_out['gain_pred_signed'], 'ptgc_benefit_pred': ptgc_out['benefit_pred'], 'ptgc_delta_d4': ptgc_out['delta_d4']}

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pet_available = pet_available.long().view(-1)
        if torch.all(pet_available == 1):
            return self._forward_full_eval(ct, pet, target_size) if not self.training else self._forward_full_train(ct, pet, target_size)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, target_size)
        raise ValueError('dual_decoder_ptgc modular experiment supports homogeneous full or missing batches only.')

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            if self.training:
                if pet is None:
                    raise ValueError('forward_mode="full" requires pet input.')
                return self._forward_full_train(ct, pet, target_size)
            if pet is None:
                raise ValueError('forward_mode="full" requires pet input.')
            return self._forward_full_eval(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long) if pet is not None else torch.zeros(ct.shape[0], device=ct.device, dtype=torch.long)
            if pet is None and torch.any(pet_available == 1):
                raise ValueError('forward_mode="auto" requires pet input when full samples exist.')
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')

    def _load_base_checkpoint(self, ptgc_base_checkpoint):
        if not ptgc_base_checkpoint:
            raise ValueError('ptgc_base_checkpoint must be provided for dual_decoder_ptgc.')
        if not os.path.exists(ptgc_base_checkpoint):
            raise FileNotFoundError(ptgc_base_checkpoint)
        ckpt = torch.load(ptgc_base_checkpoint, map_location='cpu')
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ('model', 'networks.model', 'state_dict'):
                if key in ckpt:
                    state_dict = ckpt[key]
                    break
        if state_dict is None:
            state_dict = ckpt
        state_dict = _strip_prefixes(state_dict)
        model_state = self.state_dict()
        loadable = {}
        allowed_missing = []
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape == value.shape:
                loadable[key] = value
        missing_keys = []
        unexpected_keys = []
        for sub_name in ('enc_ct', 'enc_pet', 'ct_align', 'full_decoder', 'missing_decoder'):
            submodule = getattr(self, sub_name)
            prefix = f'{sub_name}.'
            sub_state = {k[len(prefix):]: v for k, v in loadable.items() if k.startswith(prefix)}
            msg = submodule.load_state_dict(sub_state, strict=False)
            missing_keys.extend([f'{sub_name}.{k}' for k in msg.missing_keys])
            unexpected_keys.extend([f'{sub_name}.{k}' for k in msg.unexpected_keys])
        missing = [k for k in missing_keys if not k.startswith('ptgc.')]
        unexpected = [k for k in unexpected_keys if not k.startswith('ptgc.')]
        if missing:
            raise RuntimeError(f'Base checkpoint missing keys: {missing[:20]}')
        if unexpected:
            raise RuntimeError(f'Base checkpoint unexpected keys: {unexpected[:20]}')
        print(f'PTGC base checkpoint path: {ptgc_base_checkpoint}')
        print(f'loaded base tensors: {len(loadable)}')
        print(f'missing PTGC tensors: {[k for k in missing_keys if k.startswith("ptgc.")] }')
        print(f'unexpected tensors: {unexpected_keys}')
        frozen = sum(p.numel() for m in (self.enc_ct, self.enc_pet, self.ct_align, self.full_decoder, self.missing_decoder) for p in m.parameters())
        trainable = sum(p.numel() for p in self.ptgc.parameters() if p.requires_grad)
        print(f'frozen base parameter count: {frozen}')
        print(f'trainable PTGC parameter count: {trainable}')
