"""A1: PET-Prompted CT Decoder (VoxTell-style prompt injection on CT UNet decoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.baseline_petct_unet import _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe


def _prompt_map_stats(pet_map, stage_idx):
    return {
        f'stage{stage_idx}_pet_prompt_map_mean': pet_map.detach().float().mean(),
        f'stage{stage_idx}_pet_prompt_map_std': pet_map.detach().float().std(),
    }


class FrozenDINOv3PETEncoder(nn.Module):
    """Frozen DINOv3 dense feature extractor for PET images."""

    def __init__(
        self,
        model_name='vit_small_patch16_dinov3',
        pretrained_path=None,
    ):
        super().__init__()
        self.model_name = model_name
        # Use the full ViT so local checkpoints (162 tensors) load completely.
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.out_dim = int(getattr(self.model, 'num_features', 384))
        if pretrained_path:
            load_local_weights_safe(self.model, pretrained_path, name='DINOv3_PET_Encoder')
        else:
            print('[-] DINOv3_PET_Encoder: no pretrained path; using random init (frozen)')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def _tokens_to_map(self, tokens, height, width):
        patch_h = height // 16
        patch_w = width // 16
        num_patches = patch_h * patch_w
        if tokens.shape[1] < num_patches:
            raise ValueError(
                f'DINOv3 tokens={tokens.shape[1]} smaller than expected patches={num_patches}'
            )
        patch_tokens = tokens[:, -num_patches:]
        return patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            self.out_dim,
            patch_h,
            patch_w,
        )

    def forward(self, pet):
        pet = self._to_3ch(pet).float()
        with torch.no_grad():
            device_type = 'cuda' if pet.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                tokens = self.model.forward_features(pet)
        if tokens.ndim != 3:
            raise ValueError(f'DINOv3 forward_features must return [B,N,C], got {tuple(tokens.shape)}')
        feat = self._tokens_to_map(tokens, pet.shape[-2], pet.shape[-1])
        _check_tensor('F_pet_dino', feat)
        return feat


class StagePromptMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)


class PETPromptProjector(nn.Module):
    """Project frozen DINOv3 PET features into per-decoder-stage prompt embeddings."""

    def __init__(self, in_channels, base_channels=256, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        d4, d3, d2, d1 = decoder_channels
        num_groups = min(32, base_channels)
        while base_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, base_channels),
            nn.GELU(),
        )
        self.mlp_e4 = StagePromptMLP(base_channels, d4)
        self.mlp_e3 = StagePromptMLP(base_channels, d3)
        self.mlp_e2 = StagePromptMLP(base_channels, d2)
        self.mlp_e1 = StagePromptMLP(base_channels, d1)

    def forward(self, f_pet_dino):
        f_base = self.proj(f_pet_dino)
        pooled = F.adaptive_avg_pool2d(f_base, 1).flatten(1)
        return (
            self.mlp_e4(pooled),
            self.mlp_e3(pooled),
            self.mlp_e2(pooled),
            self.mlp_e1(pooled),
        )


class PETPromptInjection(nn.Module):
    """VoxTell-style einsum prompt response + channel restore."""

    def __init__(self, channels):
        super().__init__()
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
        )

    def forward(self, d_ct, prompt_embed):
        pet_map = torch.einsum('b c h w, b n c -> b n h w', d_ct, prompt_embed)
        fused = torch.cat([d_ct, pet_map], dim=1)
        return self.fuse(fused), pet_map


class PETPromptedCTDecoder(nn.Module):
    """CT UNet decoder with PET prompt injection at every stage."""

    def __init__(
        self,
        encoder_channels=(96, 192, 384, 768),
        decoder_channels=(512, 256, 128, 64),
        out_channels=1,
        use_deep_supervision=True,
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.use_deep_supervision = bool(use_deep_supervision)
        self.decoder_channels = decoder_channels

        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)

        self.fuse3 = nn.Sequential(
            ConvBNAct(d4 + d3, d3, kernel_size=3),
            ConvBNAct(d3, d3, kernel_size=3),
        )
        self.fuse2 = nn.Sequential(
            ConvBNAct(d3 + d2, d2, kernel_size=3),
            ConvBNAct(d2, d2, kernel_size=3),
        )
        self.fuse1 = nn.Sequential(
            ConvBNAct(d2 + d1, d1, kernel_size=3),
            ConvBNAct(d1, d1, kernel_size=3),
        )

        self.inject4 = PETPromptInjection(d4)
        self.inject3 = PETPromptInjection(d3)
        self.inject2 = PETPromptInjection(d2)
        self.inject1 = PETPromptInjection(d1)

        self.seg_head = nn.Conv2d(d1, out_channels, kernel_size=1)
        if self.use_deep_supervision:
            self.aux_head_d2 = nn.Conv2d(d2, out_channels, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(d4, out_channels, kernel_size=1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _upsample_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, ct_features, prompt_embeds, target_size):
        e4, e3, e2, e1 = prompt_embeds
        x1, x2, x3, x4 = ct_features
        prompt_stats = {}

        d4_ct = self.proj4(x4)
        d4, pet_map4 = self.inject4(d4_ct, e4)
        prompt_stats.update(_prompt_map_stats(pet_map4, 4))

        s3 = self.proj3(x3)
        d3_ct = self.fuse3(torch.cat([self._upsample_to(d4, s3), s3], dim=1))
        d3, pet_map3 = self.inject3(d3_ct, e3)
        prompt_stats.update(_prompt_map_stats(pet_map3, 3))

        s2 = self.proj2(x2)
        d2_ct = self.fuse2(torch.cat([self._upsample_to(d3, s2), s2], dim=1))
        d2, pet_map2 = self.inject2(d2_ct, e2)
        prompt_stats.update(_prompt_map_stats(pet_map2, 2))

        s1 = self.proj1(x1)
        d1_ct = self.fuse1(torch.cat([self._upsample_to(d2, s1), s1], dim=1))
        d1, pet_map1 = self.inject1(d1_ct, e1)
        prompt_stats.update(_prompt_map_stats(pet_map1, 1))

        logits = self.seg_head(d1)
        final_logits = self._upsample_size(logits, target_size)

        out = {
            'logits': final_logits,
            'prompt_stats': prompt_stats,
        }
        if not self.use_deep_supervision:
            return out

        out['aux_logits'] = [
            self.aux_head_d2(d2),
            self.aux_head_d3(d3),
            self.aux_head_d4(d4),
        ]
        return out


class PETPromptedCTSegmentation(nn.Module):
    """A1: CT encoder + frozen DINOv3 PET prompts + prompted CT decoder."""

    def __init__(
        self,
        ct_backbone='convnext_tiny',
        ct_pretrained_path=None,
        dinov3_model_name='vit_small_patch16_dinov3',
        dinov3_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        out_channels=1,
        use_deep_supervision=True,
        pet_prompt_base_channels=256,
        **kwargs,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=3)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        ct_channels = self.enc_ct.feature_info.channels()

        self.pet_dino_encoder = FrozenDINOv3PETEncoder(
            model_name=dinov3_model_name,
            pretrained_path=dinov3_pretrained_path,
        )
        self.pet_prompt_projector = PETPromptProjector(
            in_channels=self.pet_dino_encoder.out_dim,
            base_channels=pet_prompt_base_channels,
            decoder_channels=decoder_channels,
        )
        self.decoder = PETPromptedCTDecoder(
            encoder_channels=tuple(ct_channels),
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, pet_available=None, target_size=None, return_aux=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_3ch(ct)

        ct_feats = self.enc_ct(ct)
        _check_tensor_list('ct_feats', ct_feats)

        f_pet_dino = self.pet_dino_encoder(pet)
        prompt_embeds = self.pet_prompt_projector(f_pet_dino)

        dec_out = self.decoder(ct_feats, prompt_embeds, target_size)
        outputs = {
            'logits': _sanitize(dec_out['logits']),
            'pred': _sanitize(dec_out['logits']),
            'aux': {},
        }
        if 'aux_logits' in dec_out:
            outputs['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
        if 'prompt_stats' in dec_out:
            outputs['aux'].update(dec_out['prompt_stats'])
        _check_tensor('logits', outputs['logits'])
        return outputs


PET_PROMPT_LOG_KEYS = [
    'stage1_pet_prompt_map_mean',
    'stage1_pet_prompt_map_std',
    'stage2_pet_prompt_map_mean',
    'stage2_pet_prompt_map_std',
    'stage3_pet_prompt_map_mean',
    'stage3_pet_prompt_map_std',
    'stage4_pet_prompt_map_mean',
    'stage4_pet_prompt_map_std',
]
