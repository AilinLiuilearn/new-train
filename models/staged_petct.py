"""Fixed two-stage CT-pretrained PET/CT baseline; no retrieval or distillation.

Inputs: normalized Bx3xHxW (Bx1xHxW is repeated for smoke checks).
Four aligned scales: 64,128,320,512. The existing BN aligners and GN decoder
are deliberately unchanged so old clean CT-only state dictionaries load exactly.
"""
import copy
import torch
from torch import nn
from models.components.backbones import create_feature_backbone, load_local_weights_safe
from models.components.channel_align import StageChannelAlign
from models.components.group_unet_decoder import UNetStyleDecoder
from models.components.add_fusion import AddFusion

CHANNELS = (64, 128, 320, 512)
DECODER_CHANNELS = (512, 256, 128, 64)


class StagedPETCT(nn.Module):
    def __init__(self, stage, ct_pretrained_path=None, pet_pretrained_path=None):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        self.stage = stage
        self.enc_ct = create_feature_backbone('convnextv2_nano', in_channels=3)
        self.ct_align = StageChannelAlign(self.enc_ct.feature_info.channels(), CHANNELS)
        decoder = UNetStyleDecoder(CHANNELS, DECODER_CHANNELS, use_deep_supervision=False)
        if stage == 1:
            # Keep the legacy stage-1 key "decoder.*" for exact checkpoint compatibility.
            self.decoder = decoder
        else:
            self.decoder_ct = decoder
            self.decoder_full = copy.deepcopy(decoder)
            self.enc_pet = create_feature_backbone('mit_b1', in_channels=3)
            self.pet_align = StageChannelAlign(self.enc_pet.feature_info.channels(), CHANNELS)
            self.fusion = AddFusion()
        if ct_pretrained_path:
            load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        if stage == 2 and pet_pretrained_path:
            load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')

    @staticmethod
    def _input(x):
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            raise ValueError('expected Bx1xHxW or Bx3xHxW')
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def encode_ct(self, ct):
        return self.ct_align(self.enc_ct(self._input(ct)))

    def forward(self, ct, pet=None, mode='missing', full_indices=None):
        """missing: only CT; full: full+CT on all rows; mixed: PET on selected rows.

        mixed expects PET already SUBSETTED, in full_indices order. CT is encoded
        and decoded once on all rows; only selected rows enter PET/fusion/D_full.
        """
        if mode not in ('missing', 'full', 'mixed'):
            raise ValueError(mode)
        if self.stage == 1 and mode != 'missing':
            raise ValueError('stage 1 has only a CT path')
        c = self.encode_ct(ct)
        decoder = self.decoder if self.stage == 1 else self.decoder_ct
        ct_logits = decoder(c, ct.shape[-2:])['logits']
        result = {'ct_logits': ct_logits}
        if mode == 'missing':
            result['logits'] = ct_logits
            return result  # PET is intentionally not touched, even if supplied.
        if mode == 'full':
            indices = torch.arange(len(ct), device=ct.device)
        else:
            indices = full_indices
            if indices is None or indices.dtype != torch.long or indices.ndim != 1:
                raise ValueError('mixed requires 1D long full_indices')
            if indices.numel() * 2 != len(ct) or indices.unique().numel() != indices.numel():
                raise ValueError('mixed requires exactly half of the batch, without duplicates')
            if indices.min() < 0 or indices.max() >= len(ct):
                raise ValueError('full_indices out of range')
        if pet is None or len(pet) != len(indices) or pet.shape[-2:] != ct.shape[-2:]:
            raise ValueError('PET must contain exactly the selected paired rows, with matching H,W')
        p = self.pet_align(self.enc_pet(self._input(pet)))
        fused = self.fusion([x.index_select(0, indices) for x in c], p)
        full_logits = self.decoder_full(fused, ct.shape[-2:])['logits']
        result.update(full_logits=full_logits, full_indices=indices, logits=full_logits)
        return result

    def import_stage1_state(self, state):
        """No partial loading: verify exact old clean CT-only keys and shapes first."""
        if self.stage != 2:
            raise ValueError('only stage 2 imports stage 1')
        expected = {}
        for name, module in [('enc_ct', self.enc_ct), ('ct_align', self.ct_align),
                             ('decoder', self.decoder_ct)]:
            expected.update({f'{name}.{k}': v for k, v in module.state_dict().items()})
        if set(state) != set(expected):
            raise ValueError(f'Not an exact clean CT-only state: missing={sorted(set(expected)-set(state))[:8]}, '
                             f'unexpected={sorted(set(state)-set(expected))[:8]}')
        for key, value in expected.items():
            if not torch.is_tensor(state[key]) or state[key].shape != value.shape:
                raise ValueError(f'checkpoint shape mismatch: {key}')
            if state[key].is_floating_point() and not torch.isfinite(state[key]).all():
                raise ValueError(f'nonfinite checkpoint tensor: {key}')
        for name, module in [('enc_ct', self.enc_ct), ('ct_align', self.ct_align),
                             ('decoder', self.decoder_ct)]:
            module.load_state_dict({k[len(name)+1:]: v for k, v in state.items()
                                    if k.startswith(name+'.')}, strict=True)
        self.decoder_full.load_state_dict(self.decoder_ct.state_dict(), strict=True)
        if any(a.data_ptr() == b.data_ptr() for a, b in
               zip(self.decoder_ct.parameters(), self.decoder_full.parameters())):
            raise AssertionError('decoders must not share parameters')
        return {'loaded_tensors': len(expected),
                'copied_decoder_tensors': len(self.decoder_ct.state_dict()),
                'all_parameters_trainable': all(p.requires_grad for p in self.parameters())}
