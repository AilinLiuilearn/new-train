import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, SharedDecoder
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(ci, co, kernel_size=1) for ci, co in zip(in_channels_list, out_channels_list)])

    def forward(self, feats):
        return [proj(f) for proj, f in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_ct = create_feature_backbone(config.ct_backbone, in_channels=3)
        self.enc_pet = create_feature_backbone(config.pet_backbone, in_channels=3)
        load_local_weights_safe(self.enc_ct, config.ct_pretrained_path, 'CT_Encoder')
        load_local_weights_safe(self.enc_pet, config.pet_pretrained_path, 'PET_Encoder')
        ct_ch = list(self.enc_ct.feature_info.channels())
        pet_ch = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_ch, pet_ch)
        self.fusion = AddFusion()
        self.decoder = SharedDecoder(pet_ch, out_channels=1)

    def _to3(self, x): return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    def _ct_feats(self, ct): return self.ct_align(self.enc_ct(self._to3(ct)))

    def forward(self, ct, pet=None, forward_mode='missing'):
        ct_feats = self._ct_feats(ct)
        if forward_mode == 'full':
            if pet is None: raise ValueError('pet required for full')
            pet_feats = self.enc_pet(self._to3(pet))
            feats = self.fusion(ct_feats, pet_feats)
        else:
            feats = ct_feats
        return self.decoder(feats, ct.shape[-2:])
