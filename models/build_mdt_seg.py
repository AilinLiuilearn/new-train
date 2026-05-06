import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:
    from .biomedclip_tcpm_unet import BiomedCLIPTextEncoder, MULTI_shuffle_high_text
except ImportError:
    from biomedclip_tcpm_unet import BiomedCLIPTextEncoder, MULTI_shuffle_high_text


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
        for cand in ('pytorch_model.bin', 'model.safetensors', 'pvt_v2_b1.pth', 'pvt_v2_b2.pth'):
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(192, 128, 64, 32)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.proj4 = nn.Conv2d(c4, d4, 1, bias=False)
        self.proj3 = nn.Conv2d(c3, d3, 1, bias=False)
        self.proj2 = nn.Conv2d(c2, d2, 1, bias=False)
        self.proj1 = nn.Conv2d(c1, d1, 1, bias=False)

        self.up3 = nn.Conv2d(d4, d3, 1, bias=False)
        self.up2 = nn.Conv2d(d3, d2, 1, bias=False)
        self.up1 = nn.Conv2d(d2, d1, 1, bias=False)

        self.dec3 = DepthwiseSeparableConv(d3, d3)
        self.dec2 = DepthwiseSeparableConv(d2, d2)
        self.dec1 = DepthwiseSeparableConv(d1, d1)
        self.out_head = nn.Conv2d(d1, 1, 1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features

        d4 = self.proj4(x4)
        d3 = self.up3(self._upsample_to(d4, x3)) + self.proj3(x3)
        d3 = self.dec3(d3)

        d2 = self.up2(self._upsample_to(d3, x2)) + self.proj2(x2)
        d2 = self.dec2(d2)

        d1 = self.up1(self._upsample_to(d2, x1)) + self.proj1(x1)
        d1 = self.dec1(d1)

        logit = self.out_head(d1)
        return F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)


class DualPVTB1LightUNet(nn.Module):
    """Dual CT/PET PVTv2-B1 teacher with BiomedCLIP-guided TCPM fusion."""

    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1, freeze_text_encoder=True):
        super().__init__()
        self.enc_ct = timm.create_model(
            'pvt_v2_b1', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        self.enc_pet = timm.create_model(
            'pvt_v2_b1', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')

        encoder_channels = self.enc_ct.feature_info.channels()
        self.text_encoder = BiomedCLIPTextEncoder(freeze=freeze_text_encoder)
        self.tcpm_blocks = nn.ModuleList([
            MULTI_shuffle_high_text(ch_dim=channel, num_heads=head, lin_ch=512)
            for channel, head in zip(encoder_channels, (1, 2, 4, 8))
        ])
        self.decoder = LightUNetDecoder(encoder_channels)
        if out_channels != 1:
            self.decoder.out_head = nn.Conv2d(self.decoder.out_head.in_channels, out_channels, 1)

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None, text_code=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)

        if text_code is None:
            text_code = self.text_encoder(batch_size=ct.shape[0], device=ct.device)
        else:
            text_code = F.normalize(text_code.to(device=ct.device, dtype=torch.float32), dim=-1)

        fused_feats = [
            tcpm(pet_feat, ct_feat, text_code)[0]
            for ct_feat, pet_feat, tcpm in zip(ct_feats, pet_feats, self.tcpm_blocks)
        ]

        if target_size is None:
            target_size = ct.shape[-2:]
        return self.decoder(fused_feats, target_size)

    def set_epoch(self, epoch):
        return None


def build_mdt_seg_teacher(config):
    model = DualPVTB1LightUNet(
        pretrained_path=getattr(config, 'pretrained_path', None),
        in_channels=3,
        out_channels=1,
        freeze_text_encoder=getattr(config, 'freeze_text_encoder', True),
    )
    return dict(model=model)
