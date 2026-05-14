#!/usr/bin/env python3
"""One-shot patch: add AdaptivePETGate to MT-CUDM fusion."""
import pathlib

def patch_fusion():
    p = pathlib.Path('models/cudm_text_fusion.py')
    t = p.read_text()

    old1 = (
        'class CUDMTextGate(nn.Module):\n'
        '    """Commonality-Uniqueness Disentanglement with optional text-gated attention."""\n'
        '\n'
        '    def __init__(self, channels, heads=4, text_dim=512, attn_ratio=0.125, disable_text=False):'
    )
    new1 = (
        'class AdaptivePETGate(nn.Module):\n'
        '    """Sample-wise gate controlling PET contribution to both base and enhanced."""\n'
        '\n'
        '    def __init__(self, channels, reduction=8):\n'
        '        super().__init__()\n'
        '        hidden = max(8, channels // reduction)\n'
        '        self.mlp = nn.Sequential(\n'
        '            nn.Linear(channels * 3, hidden, bias=False),\n'
        '            nn.ReLU(inplace=True),\n'
        '            nn.Linear(hidden, 1, bias=True),\n'
        '        )\n'
        '\n'
        '    def forward(self, ct_feature, pet_feature, tumor_feature):\n'
        '        ct_desc = ct_feature.mean(dim=(2, 3))\n'
        '        pet_desc = pet_feature.mean(dim=(2, 3))\n'
        '        tumor_desc = tumor_feature.mean(dim=(2, 3))\n'
        '        return torch.sigmoid(self.mlp(torch.cat([ct_desc, pet_desc, tumor_desc], dim=1))).view(-1, 1, 1, 1)\n'
        '\n'
        '\n'
        'class CUDMTextGate(nn.Module):\n'
        '    """Commonality-Uniqueness Disentanglement with optional text-gated attention."""\n'
        '\n'
        '    def __init__(self, channels, heads=4, text_dim=512, attn_ratio=0.125,\n'
        '                 disable_text=False, adaptive_pet_gate=False):'
    )
    assert old1 in t, 'CUDMTextGate class header not found'
    t = t.replace(old1, new1)

    t = t.replace(
        '        self.disable_text = disable_text\n        self.mutual_gate = MutualGate(channels)',
        '        self.disable_text = disable_text\n        self.adaptive_pet_gate = adaptive_pet_gate\n        self.mutual_gate = MutualGate(channels)',
    )

    old_tgca = (
        '        self.tgca = None if disable_text else TextGuidedChannelAttention(\n'
        '            channels, text_dim=text_dim, reduction=8,\n'
        '        )\n'
        '        self.nq = LayerNorm2d(channels)'
    )
    new_tgca = (
        '        self.tgca = None if disable_text else TextGuidedChannelAttention(\n'
        '            channels, text_dim=text_dim, reduction=8,\n'
        '        )\n'
        '        self.pet_gate = AdaptivePETGate(channels) if adaptive_pet_gate else None\n'
        '        self.nq = LayerNorm2d(channels)'
    )
    assert old_tgca in t, 'tgca block not found'
    t = t.replace(old_tgca, new_tgca)

    old_fwd = (
        '        attn_out = self.attn(self.nq(clean_query), self.nkv(tumor))\n'
        '        enhanced = attn_out + self.ffn(self.nf(attn_out))\n'
        '\n'
        '        base = pet_feature + ct_feature\n'
        '        gate = torch.sigmoid(self.out_gate)\n'
        '        out = base * (1.0 - gate) + (base + enhanced) * gate\n'
        '        return out, {"common": common, "tumor": tumor}'
    )
    new_fwd = (
        '        attn_out = self.attn(self.nq(clean_query), self.nkv(tumor))\n'
        '        enhanced = attn_out + self.ffn(self.nf(attn_out))\n'
        '\n'
        '        pet_gate_value = None\n'
        '        if self.pet_gate is not None:\n'
        '            pet_gate_value = self.pet_gate(ct_feature, pet_feature, tumor)\n'
        '            base = ct_feature + pet_gate_value * pet_feature\n'
        '            enhanced = enhanced * pet_gate_value\n'
        '        else:\n'
        '            base = pet_feature + ct_feature\n'
        '\n'
        '        gate = torch.sigmoid(self.out_gate)\n'
        '        out = base + gate * enhanced\n'
        '        aux = {"common": common, "tumor": tumor}\n'
        '        if pet_gate_value is not None:\n'
        '            aux["pet_gate"] = pet_gate_value\n'
        '        return out, aux'
    )
    assert old_fwd in t, 'forward output block not found'
    t = t.replace(old_fwd, new_fwd)

    t = t.replace(
        'def __init__(self, encoder_channels, text_dim=512, heads_per_stage=(1, 2, 4, 8), disable_text=False):',
        'def __init__(self, encoder_channels, text_dim=512, heads_per_stage=(1, 2, 4, 8), disable_text=False, adaptive_pet_gate=False):',
    )
    t = t.replace(
        'CUDMTextGate(channels=ch, heads=head, text_dim=text_dim, attn_ratio=0.125, disable_text=disable_text)',
        'CUDMTextGate(channels=ch, heads=head, text_dim=text_dim, attn_ratio=0.125, disable_text=disable_text, adaptive_pet_gate=adaptive_pet_gate)',
    )
    p.write_text(t)
    print('[1/3] cudm_text_fusion.py patched')

def patch_build():
    p = pathlib.Path('models/build_mdt_seg.py')
    t = p.read_text()
    t = t.replace(
        'use_tcpm=False, disable_text_fusion=False):',
        'use_tcpm=False, disable_text_fusion=False, adaptive_pet_gate=False):',
    )
    t = t.replace(
        'MultiStageCUDMTextFusion(enc_channels, disable_text=disable_text_fusion)',
        'MultiStageCUDMTextFusion(enc_channels, disable_text=disable_text_fusion, adaptive_pet_gate=adaptive_pet_gate)',
    )
    t = t.replace(
        "disable_text_fusion=False,\n        decoder_type=",
        "disable_text_fusion=False,\n        adaptive_pet_gate=False,\n        decoder_type=",
    )
    t = t.replace(
        "disable_text_fusion=getattr(config, 'disable_text_fusion', False),\n    )\n    return dict(model=model)\n\n\nclass DualLightBackboneUNet",
        "disable_text_fusion=getattr(config, 'disable_text_fusion', False),\n        adaptive_pet_gate=getattr(config, 'adaptive_pet_gate', False),\n    )\n    return dict(model=model)\n\n\nclass DualLightBackboneUNet",
    )
    t = t.replace(
        "disable_text_fusion=getattr(config, 'disable_text_fusion', False),\n        decoder_type=",
        "disable_text_fusion=getattr(config, 'disable_text_fusion', False),\n        adaptive_pet_gate=getattr(config, 'adaptive_pet_gate', False),\n        decoder_type=",
    )
    p.write_text(t)
    print('[2/3] build_mdt_seg.py patched')

def patch_config():
    p = pathlib.Path('configs/seg_mdt.py')
    t = p.read_text()
    old_arg = '        p.add_argument("--disable_text_fusion", type=str2bool, default=False,\n                       help="Disable text-guided query while keeping CUDM visual fusion.")'
    new_arg = (
        '        p.add_argument("--disable_text_fusion", type=str2bool, default=False,\n'
        '                       help="Disable text-guided query while keeping CUDM visual fusion.")\n'
        '        p.add_argument("--adaptive_pet_gate", type=str2bool, default=False,\n'
        '                       help="Enable sample-wise adaptive PET gating in MT-CUDM fusion.")'
    )
    assert old_arg in t, 'disable_text_fusion arg not found in config'
    t = t.replace(old_arg, new_arg)
    p.write_text(t)
    print('[3/3] seg_mdt.py patched')

if __name__ == '__main__':
    patch_fusion()
    patch_build()
    patch_config()
    print('ALL DONE')