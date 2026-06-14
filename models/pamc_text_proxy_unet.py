import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct, LightConcatUNetDecoder, create_feature_backbone
from models.lapa_meddino import LAPA
from models.meddino_wrapper import FrozenMedDINOv3Encoder
from models.pamc_fusion import TextGuidedCorrectionFusion
from models.text_controller import TextModalityController
from models.tppc import TextGuidedPseudoPETCorrection


class PAMCTextProxyUNet(nn.Module):
    def __init__(
        self,
        ct_backbone='convnext_tiny',
        pet_backbone='mit_b0',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_type='light',
        text_embed_dim=256,
        in_channels=3,
        out_channels=1,
        use_meddino=True,
        meddino_ckpt=None,
        use_lapa=True,
        **kwargs,
    ):
        super().__init__()
        self.ct_backbone = ct_backbone
        self.pet_backbone = pet_backbone
        self.text_embed_dim = int(text_embed_dim)
        self.use_meddino = bool(use_meddino)
        self.use_lapa = bool(use_lapa)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        ct_channels = self.enc_ct.feature_info.channels()
        pet_channels = self.enc_pet.feature_info.channels()
        self.text_controller = TextModalityController(ct_channels, embed_dim=self.text_embed_dim)
        self.meddino = FrozenMedDINOv3Encoder(ckpt_path=meddino_ckpt, use_placeholder_if_missing=True, out_channels=ct_channels)
        self.lapa = LAPA(ct_channels, prior_channels=ct_channels) if self.use_lapa else nn.Identity()
        self.pet_proj = nn.ModuleList([ConvBNAct(pin, cout, kernel_size=1) for pin, cout in zip(pet_channels, ct_channels)])
        self.tppc = TextGuidedPseudoPETCorrection(ct_channels, embed_dim=self.text_embed_dim)
        self.fusion = TextGuidedCorrectionFusion()
        self.decoder = LightConcatUNetDecoder(ct_channels, out_channels=out_channels)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(ct_channels[0], max(8, ct_channels[0] // 2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, ct_channels[0] // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, ct_channels[0] // 2), 1, kernel_size=1),
        )

        if ct_pretrained_path:
            print(f'[+] PAMCTextProxyUNet CT encoder pretrained path provided: {ct_pretrained_path}')
        if pet_pretrained_path:
            print(f'[+] PAMCTextProxyUNet PET encoder pretrained path provided: {pet_pretrained_path}')
        if decoder_type != 'light':
            raise ValueError(f'Unsupported decoder_type: {decoder_type}. Only light decoder is supported in PAMCTextProxyUNet.')

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, pet_available=None, target_size=None):
        if pet_available is None:
            pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=ct.dtype)
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        prior_feats = self.meddino(ct)
        controller_out = self.text_controller(pet_available)
        text_embed = controller_out['text_embed']
        prior_gates = controller_out['prior_gates']
        pet_gates = controller_out['pet_gates']
        text_gates = controller_out['text_gates']

        if self.use_lapa:
            enhanced_ct_feats = self.lapa(ct_feats, prior_feats, prior_gates=prior_gates)
        else:
            enhanced_ct_feats = ct_feats

        pet_feats_raw = self.enc_pet(pet)
        pet_feats = []
        for feat, proj, ref in zip(pet_feats_raw, self.pet_proj, enhanced_ct_feats):
            aligned = proj(feat)
            if aligned.shape[-2:] != ref.shape[-2:]:
                aligned = F.interpolate(aligned, size=ref.shape[-2:], mode='bilinear', align_corners=False)
            pet_feats.append(aligned)
        text_feats = self.tppc(enhanced_ct_feats, text_embed)
        fused_feats, fusion_aux = self.fusion(
            enhanced_ct_feats,
            pet_feats,
            text_feats,
            pet_gates,
            text_gates,
            pet_available,
        )
        decoder_out = self.decoder(fused_feats, target_size)
        logits = decoder_out['pred']
        boundary_feat = fused_feats[0]
        boundary_logits = self.boundary_head(boundary_feat)
        if boundary_logits.shape[-2:] != target_size:
            boundary_logits = F.interpolate(boundary_logits, size=target_size, mode='bilinear', align_corners=False)
        prior_gate_mean = torch.stack([g.mean() for g in prior_gates]).mean()
        decoder_out.update({
            'logits': logits,
            'boundary_logits': boundary_logits,
            'aux': {
                'pet_available': pet_available,
                'pet_gate_mean': fusion_aux['pet_gate_mean'],
                'text_gate_mean': fusion_aux['text_gate_mean'],
                'prior_gate_mean': prior_gate_mean,
            },
        })
        return decoder_out
