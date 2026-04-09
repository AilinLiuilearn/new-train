import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# 兼容性导入：处理不同版本的 timm 位置编码插值工具
try:
    from timm.layers import resample_abs_pos_embed
except ImportError:
    try:
        from timm.models.layers import resample_abs_pos_embed
    except ImportError:
        resample_abs_pos_embed = None
        print("⚠️ Warning: resample_abs_pos_embed not found. Positional embeddings will not be interpolated.")

from models.emcad_decoder import EMCADDecoder

def load_local_weights_safe(model, path, name="Encoder"):
    """
    专家级加载函数：
    1. 自动键名转换 (stages.x -> stages_x) 解决 IncompatibleKeys
    2. 适配 512x512 的位置编码插值
    3. 绕过 PyTorch 2.6+ weights_only 安全限制
    """
    if not path or not os.path.exists(path):
        print(f"[-] {name}: Path not found {path}. Training from scratch.")
        return
    
    if os.path.isdir(path):
        candidates = ['pytorch_model.bin', 'model.safetensors', 'pvt_v2_b2.pth', 'pvt_v2_b0.pth']
        found = False
        for c in candidates:
            if os.path.exists(os.path.join(path, c)):
                path = os.path.join(path, c)
                found = True
                break
        if not found:
            print(f"[-] {name}: No weight file found in {path}.")
            return

    print(f"[+] {name}: Loading local weights from {path}")
    
    # 兼容 PyTorch 2.6+ 加载逻辑
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except:
        state_dict = torch.load(path, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    # --- 核心修正：键名映射转换 (解决 stages.0 -> stages_0) ---
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('stages.', 'stages_')
        
        # 处理位置编码插值 (适配 512x512)
        if 'pos_embed' in new_key and resample_abs_pos_embed is not None:
            print(f"[+] {name}: Interpolating {new_key} for 512x512")
            v = resample_abs_pos_embed(
                v, 
                new_size=(512 // 32, 512 // 32), 
                num_prefix_tokens=0
            )
        new_state_dict[new_key] = v
    
    # 加载权重
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"[+] {name} Load Status: {msg}")


class SinglePVTEMCAD(nn.Module):
    """Student: single-stream CT (3-ch), PVTv2-b0 + EMCAD."""
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        self.encoder = timm.create_model(
            'pvt_v2_b0', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        
        if pretrained_path:
            load_local_weights_safe(self.encoder, pretrained_path, name="Student_Encoder")

        c1, c2, c3, c4 = self.encoder.feature_info.channels()

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

    def forward(self, ct, pet=None, target_size=None):
        if ct.shape[1] == 1: ct = ct.repeat(1, 3, 1, 1)
        x1, x2, x3, x4 = self.encoder(ct)
        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

        if target_size is None: target_size = ct.shape[-2:]
        p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
        return [p1, p2, p3, p4]


class DualPVTB2EMCAD(nn.Module):
    """Teacher: dual-stream CT/PET, two PVTv2-b2 encoders + additive fusion + EMCAD."""
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        self.enc_ct = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
        self.enc_pet = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
        
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name="Teacher_CT_Encoder")
            load_local_weights_safe(self.enc_pet, pretrained_path, name="Teacher_PET_Encoder")

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

    def forward(self, ct, pet, target_size=None):
        if ct.shape[1] == 1: ct = ct.repeat(1, 3, 1, 1)
        if pet.shape[1] == 1: pet = pet.repeat(1, 3, 1, 1)

        feats_ct = self.enc_ct(ct)
        feats_pet = self.enc_pet(pet)
        
        # 纯净 Baseline：最简单的逐元素相加融合
        x1, x2, x3, x4 = [a + b for a, b in zip(feats_ct, feats_pet)]

        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

        if target_size is None: target_size = ct.shape[-2:]
        p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
        return [p1, p2, p3, p4]

def build_mdt_seg_teacher(config):
    p_path = getattr(config, 'pretrained_path', None)
    model = DualPVTB2EMCAD(pretrained_path=p_path, in_channels=3, out_channels=1)
    return dict(model=model)

def build_mdt_seg_student(config):
    s_path = getattr(config, 'student_pretrained_path', None) or getattr(config, 'pretrained_path', None)
    model = SinglePVTEMCAD(pretrained_path=s_path, in_channels=3, out_channels=1)
    return dict(model=model)