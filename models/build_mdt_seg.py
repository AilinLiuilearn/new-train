import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.emcad_decoder import EMCADDecoder


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return state_dict


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ('module.', 'model.', 'backbone.', 'encoder.', 'visual.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        nk = nk.replace('stages.', 'stages_')
        cleaned[nk] = v
    return cleaned


def load_local_weights_safe(model, path, name='Encoder'):
    if not path or not os.path.exists(path):
        print(f'[-] {name}: Path not found {path}. Training from scratch.')
        return
    if os.path.isdir(path):
        for cand in ('pytorch_model.bin', 'model.safetensors', 'pvt_v2_b2.pth'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                break
    print(f'[+] {name}: Loading local weights from {path}')
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        state_dict = torch.load(path, map_location='cpu')
    state_dict = _sanitize_state_dict(_unwrap_state_dict(state_dict))

    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)
    msg = model.load_state_dict(loadable, strict=False)
    print(f'[+] {name} loaded params: {len(loadable)}, skipped: {len(skipped)}')
    print(f'[+] {name} load status: {msg}')


class DualPVTB2EMCAD(nn.Module):
    """Teacher baseline: dual-stream CT/PET PVTv2-B2 encoder with additive EMCAD decoder."""

    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        self.enc_ct = timm.create_model(
            'pvt_v2_b2', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        self.enc_pet = timm.create_model(
            'pvt_v2_b2', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')

        c1, c2, c3, c4 = self.enc_ct.feature_info.channels()
        self.decoder = EMCADDecoder(
            channels=(c4, c3, c2, c1),
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation,
        )
        self.out_head4 = nn.Conv2d(c4, out_channels, 1)
        self.out_head3 = nn.Conv2d(c3, out_channels, 1)
        self.out_head2 = nn.Conv2d(c2, out_channels, 1)
        self.out_head1 = nn.Conv2d(c1, out_channels, 1)

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)

        x1 = ct_feats[0] + pet_feats[0]
        x2 = ct_feats[1] + pet_feats[1]
        x3 = ct_feats[2] + pet_feats[2]
        x4 = ct_feats[3] + pet_feats[3]

        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])
        if target_size is None:
            target_size = ct.shape[-2:]
        p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
        return [p1, p2, p3, p4]

    def set_epoch(self, epoch):
        return None


def build_mdt_seg_teacher(config):
    model = DualPVTB2EMCAD(
        pretrained_path=getattr(config, 'pretrained_path', None),
        in_channels=3,
        out_channels=1,
    )
    return dict(model=model)
