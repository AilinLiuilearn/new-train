import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

LIGHT_TEACHER_BACKBONES = {
    'convnext_atto': (0, 1, 2, 3),
    'convnext_femto': (0, 1, 2, 3),
    'convnext_pico': (0, 1, 2, 3),
    'convnext_nano': (0, 1, 2, 3),
}

LIGHT_TEACHER_DECODER_PRESETS = {
    'convnext_atto': (256, 128, 64, 32),
    'convnext_femto': (256, 128, 64, 32),
    'convnext_pico': (384, 192, 96, 48),
    'convnext_nano': (512, 256, 128, 64),
}

LIGHT_TEACHER_DECODER_PRESETS_LIGHT = {
    'convnext_atto': (128, 64, 32, 16),
    'convnext_femto': (192, 96, 48, 24),
    'convnext_pico': (256, 128, 64, 32),
    'convnext_nano': (384, 192, 96, 48),
}


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
        nk = nk.replace('stem.', 'stem_')
        cleaned[nk] = v
    return cleaned


def load_local_weights_safe(model, path, name='Encoder'):
    if not path:
        print(f'[-] {name}: No pretrained path provided. Training from scratch.')
        return
    if not os.path.exists(path):
        print(f'[-] {name}: Path not found {path}. Training from scratch.')
        return
    if os.path.isdir(path):
        found = False
        for cand in ('pytorch_model.bin', 'model.safetensors',
                     'pvt_v2_b1.pth', 'pvt_v2_b1.bin', 'pvt_v2_b1.pt',
                     'convnext_nano.pth', 'convnext_nano.bin', 'convnext_nano.pt'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                found = True
                break
        if not found:
            print(f'[-] {name}: No supported weight file found in {path}. Training from scratch.')
            return
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
    if skipped:
        print(f'[+] {name} skipped examples: {skipped[:8]}')
    print(f'[+] {name} load status: {msg}')


def _get_backbone_out_indices(backbone):
    if backbone == 'pvt_v2_b1':
        return (0, 1, 2, 3)
    if backbone in ('convnext_nano', 'convnextv2_nano'):
        return (0, 1, 2, 3)
    if backbone in ('efficientnet_b3', 'efficientnet_b4'):
        return (1, 2, 3, 4)
    raise ValueError(f'Unsupported backbone: {backbone}. Supported: pvt_v2_b1, convnext_nano, convnextv2_nano, efficientnet_b3, efficientnet_b4.')


def create_feature_backbone(backbone, in_channels=3):
    return timm.create_model(
        backbone,
        pretrained=False,
        features_only=True,
        out_indices=_get_backbone_out_indices(backbone),
        in_chans=in_channels,
    )


def create_light_teacher_backbone(backbone, in_channels=3):
    if backbone not in LIGHT_TEACHER_BACKBONES:
        raise ValueError(
            f'Unsupported light teacher backbone: {backbone}. '
            f'Supported: {list(LIGHT_TEACHER_BACKBONES.keys())}'
        )
    return timm.create_model(
        backbone,
        pretrained=False,
        features_only=True,
        out_indices=LIGHT_TEACHER_BACKBONES[backbone],
        in_chans=in_channels,
    )


def resolve_local_pretrained(backbone):
    path = os.path.join('.', 'pretrained', backbone)
    return path if os.path.isdir(path) else None


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        w = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


class LightASPP(nn.Module):
    """Lightweight ASPP for multi-scale context at the bottleneck."""

    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12)):
        super().__init__()
        branch_ch = out_channels // len(dilations)
        self.branches = nn.ModuleList([
            ConvBNAct(in_channels, branch_ch, kernel_size=3 if d > 1 else 1, dilation=d)
            for d in dilations
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(in_channels, branch_ch, kernel_size=1),
        )
        total_ch = branch_ch * (len(dilations) + 1)
        self.fuse = ConvBNAct(total_ch, out_channels, kernel_size=1)

    def forward(self, x):
        outs = [br(x) for br in self.branches]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        outs.append(gap)
        return self.fuse(torch.cat(outs, dim=1))


class SESkip(nn.Module):
    """Squeeze-and-Excitation gate on skip connections to recalibrate channels."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.se(x).view(b, c, 1, 1)


class AttentionUNetDecoder(nn.Module):
    """UNet decoder with CBAM attention, lightweight ASPP bottleneck, and SE-gated skips."""

    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.bottleneck = LightASPP(c4, d4, dilations=(1, 6, 12))

        self.skip3 = nn.Sequential(ConvBNAct(c3, d3, kernel_size=1), SESkip(d3))
        self.skip2 = nn.Sequential(ConvBNAct(c2, d2, kernel_size=1), SESkip(d2))
        self.skip1 = nn.Sequential(ConvBNAct(c1, d1, kernel_size=1), SESkip(d1))

        self.fuse3 = nn.Sequential(ConvBNAct(d4 + d3, d3), ConvBNAct(d3, d3), CBAM(d3))
        self.fuse2 = nn.Sequential(ConvBNAct(d3 + d2, d2), ConvBNAct(d2, d2), CBAM(d2))
        self.fuse1 = nn.Sequential(ConvBNAct(d2 + d1, d1), ConvBNAct(d1, d1), CBAM(d1))

        self.head4 = nn.Conv2d(d4, out_channels, 1)
        self.head3 = nn.Conv2d(d3, out_channels, 1)
        self.head2 = nn.Conv2d(d2, out_channels, 1)
        self.head1 = nn.Conv2d(d1, out_channels, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _up_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features

        d4 = self.bottleneck(x4)

        s3 = self.skip3(x3)
        d3 = self.fuse3(torch.cat([self._up(d4, s3), s3], dim=1))

        s2 = self.skip2(x2)
        d2 = self.fuse2(torch.cat([self._up(d3, s2), s2], dim=1))

        s1 = self.skip1(x1)
        d1 = self.fuse1(torch.cat([self._up(d2, s1), s1], dim=1))

        p1 = self._up_size(self.head1(d1), target_size)
        p2 = self._up_size(self.head2(d2), target_size)
        p3 = self._up_size(self.head3(d3), target_size)
        p4 = self._up_size(self.head4(d4), target_size)
        return {'preds': [p1, p2, p3, p4], 'pred': p1}


class LightConcatUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)

        self.fuse3 = nn.Sequential(
            ConvBNAct(d4 + d3, d3),
            ConvBNAct(d3, d3),
        )
        self.fuse2 = nn.Sequential(
            ConvBNAct(d3 + d2, d2),
            ConvBNAct(d2, d2),
        )
        self.fuse1 = nn.Sequential(
            ConvBNAct(d2 + d1, d1),
            ConvBNAct(d1, d1),
        )

        self.head4 = nn.Conv2d(d4, out_channels, 1)
        self.head3 = nn.Conv2d(d3, out_channels, 1)
        self.head2 = nn.Conv2d(d2, out_channels, 1)
        self.head1 = nn.Conv2d(d1, out_channels, 1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _upsample_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features

        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([self._upsample_to(d4, s3), s3], dim=1))

        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([self._upsample_to(d3, s2), s2], dim=1))

        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([self._upsample_to(d2, s1), s1], dim=1))

        p1 = self._upsample_size(self.head1(d1), target_size)
        p2 = self._upsample_size(self.head2(d2), target_size)
        p3 = self._upsample_size(self.head3(d3), target_size)
        p4 = self._upsample_size(self.head4(d4), target_size)
        return {'preds': [p1, p2, p3, p4], 'pred': p1}



class DualBackboneUNet(nn.Module):
    """Dual CT/PET teacher with configurable timm backbone and traditional UNet decoder."""

    def __init__(self, backbone='pvt_v2_b1', pretrained_path=None,
                 in_channels=3, out_channels=1, use_tcpm=False, disable_text_fusion=False,
                 pet_drop_rate=0.0, use_shaspec_disentangle=False):
        super().__init__()
        self.backbone = backbone
        self.use_tcpm = use_tcpm
        self.pet_drop_rate = pet_drop_rate
        self.use_shaspec_disentangle = use_shaspec_disentangle
        self.enc_ct = create_feature_backbone(backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(backbone, in_channels=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')

        enc_channels = self.enc_ct.feature_info.channels()
        self.decoder = AttentionUNetDecoder(enc_channels, out_channels=out_channels)

        if use_shaspec_disentangle:
            from models.cudm_text_fusion import MultiStageShaSpecDisentangle
            self.disentangle = MultiStageShaSpecDisentangle(enc_channels)
        else:
            self.disentangle = None

        if use_tcpm:
            from models.cudm_text_fusion import MultiStageCUDMTextFusion
            self.fusion = MultiStageCUDMTextFusion(
                enc_channels,
                disable_text=disable_text_fusion,
            )
        else:
            self.fusion = None

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)

        drop_pet = self.training and self.pet_drop_rate > 0 and torch.rand(1).item() < self.pet_drop_rate
        if drop_pet:
            pet_feats = None
        else:
            pet_feats = self.enc_pet(pet)

        if self.disentangle is not None:
            ct_feats, pet_feats, disentangle_aux = self.disentangle(ct_feats, pet_feats)
        else:
            disentangle_aux = None

        if self.fusion is not None:
            fusion_out = self.fusion(ct_feats, pet_feats)
            if isinstance(fusion_out, tuple):
                fused_feats, fusion_aux = fusion_out
            else:
                fused_feats, fusion_aux = fusion_out, None
        else:
            if pet_feats is not None:
                fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
            else:
                fused_feats = ct_feats
            fusion_aux = None

        if target_size is None:
            target_size = ct.shape[-2:]
        outputs = self.decoder(fused_feats, target_size)
        if fusion_aux is not None and isinstance(outputs, dict):
            outputs['fusion_aux'] = fusion_aux
        if disentangle_aux is not None and isinstance(outputs, dict):
            outputs['disentangle_aux'] = disentangle_aux
        return outputs

    def set_epoch(self, epoch):
        return None


def build_mdt_seg_teacher(config):
    model = DualBackboneUNet(
        backbone=getattr(config, 'backbone', 'pvt_v2_b1'),
        pretrained_path=getattr(config, 'pretrained_path', None),
        in_channels=3,
        out_channels=1,
        use_tcpm=getattr(config, 'use_tcpm', False),
        disable_text_fusion=getattr(config, 'disable_text_fusion', False),
        pet_drop_rate=getattr(config, 'pet_drop_rate', 0.0),
        use_shaspec_disentangle=getattr(config, 'use_shaspec_disentangle', False),
    )
    return dict(model=model)


class DualLightBackboneUNet(nn.Module):
    """Dual-modality teacher using student-scale ConvNeXt encoders plus teacher fusion."""

    def __init__(
        self,
        backbone='convnext_atto',
        pretrained_path=None,
        in_channels=3,
        out_channels=1,
        use_tcpm=True,
        disable_text_fusion=False,
        pet_drop_rate=0.0,
        use_shaspec_disentangle=False,
        decoder_type='light',
        decoder_channels=None,
        share_encoder=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_tcpm = use_tcpm
        self.pet_drop_rate = pet_drop_rate
        self.use_shaspec_disentangle = use_shaspec_disentangle
        self.decoder_type = decoder_type
        self.share_encoder = share_encoder
        self.enc_ct = create_light_teacher_backbone(backbone, in_channels=in_channels)
        self.enc_pet = self.enc_ct if share_encoder else create_light_teacher_backbone(backbone, in_channels=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='LightTeacher_CT_Encoder')
            if not share_encoder:
                load_local_weights_safe(self.enc_pet, pretrained_path, name='LightTeacher_PET_Encoder')

        enc_channels = self.enc_ct.feature_info.channels()
        if decoder_channels is None:
            if decoder_type == 'light':
                decoder_channels = LIGHT_TEACHER_DECODER_PRESETS_LIGHT.get(backbone, (128, 64, 32, 16))
            else:
                decoder_channels = LIGHT_TEACHER_DECODER_PRESETS.get(backbone, (256, 128, 64, 32))

        if decoder_type == 'light':
            self.decoder = LightConcatUNetDecoder(enc_channels, out_channels=out_channels, decoder_channels=decoder_channels)
        elif decoder_type == 'attention':
            self.decoder = AttentionUNetDecoder(enc_channels, out_channels=out_channels, decoder_channels=decoder_channels)
        else:
            raise ValueError(f'Unknown decoder_type={decoder_type}. Supported: light, attention')

        if use_shaspec_disentangle:
            from models.cudm_text_fusion import MultiStageShaSpecDisentangle
            self.disentangle = MultiStageShaSpecDisentangle(enc_channels)
        else:
            self.disentangle = None

        if use_tcpm:
            from models.cudm_text_fusion import MultiStageCUDMTextFusion
            self.fusion = MultiStageCUDMTextFusion(
                enc_channels,
                disable_text=disable_text_fusion,
            )
        else:
            self.fusion = None

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)

        drop_pet = self.training and self.pet_drop_rate > 0 and torch.rand(1).item() < self.pet_drop_rate
        if drop_pet:
            pet_feats = None
        else:
            pet_feats = self.enc_pet(pet)

        if self.disentangle is not None:
            ct_feats, pet_feats, disentangle_aux = self.disentangle(ct_feats, pet_feats)
        else:
            disentangle_aux = None

        if self.fusion is not None:
            fusion_out = self.fusion(ct_feats, pet_feats)
            if isinstance(fusion_out, tuple):
                fused_feats, fusion_aux = fusion_out
            else:
                fused_feats, fusion_aux = fusion_out, None
        else:
            if pet_feats is not None:
                fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
            else:
                fused_feats = ct_feats
            fusion_aux = None

        if target_size is None:
            target_size = ct.shape[-2:]
        outputs = self.decoder(fused_feats, target_size)
        if fusion_aux is not None and isinstance(outputs, dict):
            outputs['fusion_aux'] = fusion_aux
        if disentangle_aux is not None and isinstance(outputs, dict):
            outputs['disentangle_aux'] = disentangle_aux
        return outputs

    def set_epoch(self, epoch):
        return None


def build_mdt_seg_light_teacher(config):
    pretrained_path = getattr(config, 'light_pretrained_path', None)
    if not getattr(config, 'light_no_pretrained', False) and not pretrained_path:
        pretrained_path = resolve_local_pretrained(getattr(config, 'light_backbone', 'convnext_atto'))
    if getattr(config, 'light_no_pretrained', False):
        pretrained_path = None
    model = DualLightBackboneUNet(
        backbone=getattr(config, 'light_backbone', 'convnext_atto'),
        pretrained_path=pretrained_path,
        in_channels=3,
        out_channels=1,
        use_tcpm=getattr(config, 'use_tcpm', True),
        disable_text_fusion=getattr(config, 'disable_text_fusion', False),
        pet_drop_rate=getattr(config, 'pet_drop_rate', 0.0),
        use_shaspec_disentangle=getattr(config, 'use_shaspec_disentangle', False),
        decoder_type=getattr(config, 'light_decoder_type', 'light'),
        decoder_channels=getattr(config, 'light_decoder_channels', None),
        share_encoder=getattr(config, 'light_share_encoder', False),
    )
    return dict(model=model)

