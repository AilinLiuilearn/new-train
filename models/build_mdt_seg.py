# # # -*- coding: utf-8 -*-
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import timm

# # from models.emcad_decoder import EMCADDecoder


# # def _create_timm_features(model_name, pretrained, in_chans, out_indices=(0, 1, 2, 3)):
# #     try:
# #         return timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=out_indices, in_chans=in_chans)
# #     except Exception as e:
# #         if pretrained:
# #             print(f"[build_mdt_seg] {model_name} pretrained load failed: {e}. Fallback to pretrained=False")
# #             return timm.create_model(model_name, pretrained=False, features_only=True, out_indices=out_indices, in_chans=in_chans)
# #         raise


# # class SinglePVTEMCAD(nn.Module):
# #     """Student: single-stream CT (3-ch), PVTv2-b0 + EMCAD."""

# #     def __init__(self, pretrained=True, pretrained_path=None, in_channels=3, out_channels=1,
# #                  kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
# #                  lgag_ks=3, activation='relu6'):
# #         super().__init__()
# #         self.encoder = _create_timm_features('pvt_v2_b0', pretrained, in_channels)
# #         if pretrained_path:
# #             state = torch.load(pretrained_path, map_location='cpu')
# #             if isinstance(state, dict) and 'state_dict' in state:
# #                 state = state['state_dict']
# #             self.encoder.load_state_dict(state, strict=False)

# #         channels = self.encoder.feature_info.channels() if hasattr(self.encoder, 'feature_info') else [32, 64, 160, 256]
# #         c1, c2, c3, c4 = channels

# #         self.decoder = EMCADDecoder(
# #             channels=(c4, c3, c2, c1),
# #             kernel_sizes=kernel_sizes,
# #             expansion_factor=expansion_factor,
# #             dw_parallel=dw_parallel,
# #             add=add,
# #             lgag_ks=lgag_ks,
# #             activation=activation,
# #         )

# #         self.out_head4 = nn.Conv2d(c4, out_channels, 1)
# #         self.out_head3 = nn.Conv2d(c3, out_channels, 1)
# #         self.out_head2 = nn.Conv2d(c2, out_channels, 1)
# #         self.out_head1 = nn.Conv2d(c1, out_channels, 1)

# #     def forward(self, ct, pet=None, target_size=None):
# #         x1, x2, x3, x4 = self.encoder(ct)
# #         d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

# #         if target_size is None:
# #             target_size = ct.shape[-2:]
# #         p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
# #         p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
# #         p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
# #         p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
# #         return [p1, p2, p3, p4]


# # class DualPVTB2EMCAD(nn.Module):
# #     """Teacher: dual-stream CT/PET, two PVTv2-b2 encoders + additive fusion + EMCAD."""

# #     def __init__(self, pretrained=True, pretrained_path=None, in_channels=3, out_channels=1,
# #                  kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
# #                  lgag_ks=3, activation='relu6'):
# #         super().__init__()
# #         self.enc_ct = _create_timm_features('pvt_v2_b2', pretrained, in_channels)
# #         self.enc_pet = _create_timm_features('pvt_v2_b2', pretrained, in_channels)
# #         if pretrained_path:
# #             state = torch.load(pretrained_path, map_location='cpu')
# #             if isinstance(state, dict) and 'state_dict' in state:
# #                 state = state['state_dict']
# #             self.enc_ct.load_state_dict(state, strict=False)
# #             self.enc_pet.load_state_dict(state, strict=False)

# #         channels = self.enc_ct.feature_info.channels() if hasattr(self.enc_ct, 'feature_info') else [64, 128, 320, 512]
# #         c1, c2, c3, c4 = channels

# #         self.decoder = EMCADDecoder(
# #             channels=(c4, c3, c2, c1),
# #             kernel_sizes=kernel_sizes,
# #             expansion_factor=expansion_factor,
# #             dw_parallel=dw_parallel,
# #             add=add,
# #             lgag_ks=lgag_ks,
# #             activation=activation,
# #         )

# #         self.out_head4 = nn.Conv2d(c4, out_channels, 1)
# #         self.out_head3 = nn.Conv2d(c3, out_channels, 1)
# #         self.out_head2 = nn.Conv2d(c2, out_channels, 1)
# #         self.out_head1 = nn.Conv2d(c1, out_channels, 1)

# #     def forward(self, ct, pet, target_size=None):
# #         feats_ct = self.enc_ct(ct)
# #         feats_pet = self.enc_pet(pet)
# #         x1, x2, x3, x4 = [a + b for a, b in zip(feats_ct, feats_pet)]

# #         d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

# #         if target_size is None:
# #             target_size = ct.shape[-2:]
# #         p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
# #         p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
# #         p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
# #         p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
# #         return [p1, p2, p3, p4]


# # def build_mdt_seg_teacher(config):
# #     pretrained = getattr(config, 'pretrained_backbone', True)
# #     pretrained_path = getattr(config, 'pretrained_path', None)
# #     model = DualPVTB2EMCAD(
# #         pretrained=pretrained,
# #         pretrained_path=pretrained_path,
# #         in_channels=3,
# #         out_channels=1,
# #     )
# #     return dict(model=model)


# # def build_mdt_seg_student(config):
# #     pretrained = getattr(config, 'pretrained_backbone', True)
# #     student_pretrained_path = getattr(config, 'student_pretrained_path', None)
# #     model = SinglePVTEMCAD(
# #         pretrained=pretrained,
# #         pretrained_path=student_pretrained_path,
# #         in_channels=3,
# #         out_channels=1,
# #     )
# #     return dict(model=model)
# # -*- coding: utf-8 -*-
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# # 兼容性导入：处理不同版本的 timm 位置编码插值工具
# try:
#     from timm.layers import resample_abs_pos_embed
# except ImportError:
#     try:
#         from timm.models.layers import resample_abs_pos_embed
#     except ImportError:
#         resample_abs_pos_embed = None
#         print("⚠️ Warning: resample_abs_pos_embed not found. Positional embeddings will not be interpolated.")

# from models.emcad_decoder import EMCADDecoder

# def load_local_weights_safe(model, path, name="Encoder"):
#     """
#     专家级加载函数：
#     1. 自动键名转换 (stages.x -> stages_x) 解决 IncompatibleKeys
#     2. 适配 512x512 的位置编码插值
#     3. 绕过 PyTorch 2.6+ weights_only 安全限制
#     """
#     if not path or not os.path.exists(path):
#         print(f"[-] {name}: Path not found {path}. Training from scratch.")
#         return
    
#     if os.path.isdir(path):
#         candidates = ['pytorch_model.bin', 'model.safetensors', 'pvt_v2_b2.pth', 'pvt_v2_b0.pth']
#         found = False
#         for c in candidates:
#             if os.path.exists(os.path.join(path, c)):
#                 path = os.path.join(path, c)
#                 found = True
#                 break
#         if not found:
#             print(f"[-] {name}: No weight file found in {path}.")
#             return

#     print(f"[+] {name}: Loading local weights from {path}")
    
#     # 兼容 PyTorch 2.6+ 加载逻辑
#     try:
#         state_dict = torch.load(path, map_location='cpu', weights_only=False)
#     except:
#         state_dict = torch.load(path, map_location='cpu')

#     if 'state_dict' in state_dict:
#         state_dict = state_dict['state_dict']
#     elif 'model' in state_dict:
#         state_dict = state_dict['model']
    
#     # --- 核心修正：键名映射转换 (解决 stages.0 -> stages_0) ---
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # 针对 timm 官方权重命名进行清洗
#         new_key = k.replace('stages.', 'stages_')
        
#         # 处理位置编码插值 (适配 512x512)
#         if 'pos_embed' in new_key and resample_abs_pos_embed is not None:
#             print(f"[+] {name}: Interpolating {new_key} for 512x512")
#             v = resample_abs_pos_embed(
#                 v, 
#                 new_size=(512 // 32, 512 // 32), 
#                 num_prefix_tokens=0
#             )
#         new_state_dict[new_key] = v
    
#     # 加载权重
#     msg = model.load_state_dict(new_state_dict, strict=False)
#     print(f"[+] {name} Load Status: {msg}")


# class SinglePVTEMCAD(nn.Module):
#     """Student: single-stream CT (3-ch), PVTv2-b0 + EMCAD."""
#     def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
#                  kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
#                  lgag_ks=3, activation='relu6'):
#         super().__init__()
#         # 强制断网
#         self.encoder = timm.create_model(
#             'pvt_v2_b0', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
#         )
        
#         if pretrained_path:
#             load_local_weights_safe(self.encoder, pretrained_path, name="Student_Encoder")

#         c1, c2, c3, c4 = self.encoder.feature_info.channels()

#         self.decoder = EMCADDecoder(
#             channels=(c4, c3, c2, c1),
#             kernel_sizes=kernel_sizes,
#             expansion_factor=expansion_factor,
#             dw_parallel=dw_parallel,
#             add=add,
#             lgag_ks=lgag_ks,
#             activation=activation,
#         )

#         self.out_head4 = nn.Conv2d(c4, out_channels, 1)
#         self.out_head3 = nn.Conv2d(c3, out_channels, 1)
#         self.out_head2 = nn.Conv2d(c2, out_channels, 1)
#         self.out_head1 = nn.Conv2d(c1, out_channels, 1)

#     def forward(self, ct, pet=None, target_size=None):
#         if ct.shape[1] == 1: ct = ct.repeat(1, 3, 1, 1)
#         x1, x2, x3, x4 = self.encoder(ct)
#         d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

#         if target_size is None: target_size = ct.shape[-2:]
#         p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
#         p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
#         p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
#         p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
#         return [p1, p2, p3, p4]


# class DualPVTB2EMCAD(nn.Module):
#     """Teacher: dual-stream CT/PET, two PVTv2-b2 encoders + additive fusion + EMCAD."""
#     def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
#                  kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
#                  lgag_ks=3, activation='relu6'):
#         super().__init__()
#         # 实例化两个编码器
#         self.enc_ct = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
#         self.enc_pet = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
        
#         if pretrained_path:
#             load_local_weights_safe(self.enc_ct, pretrained_path, name="Teacher_CT_Encoder")
#             load_local_weights_safe(self.enc_pet, pretrained_path, name="Teacher_PET_Encoder")

#         c1, c2, c3, c4 = self.enc_ct.feature_info.channels()

#         # 核心：确保只有一个 Decoder 实例
#         self.decoder = EMCADDecoder(
#             channels=(c4, c3, c2, c1),
#             kernel_sizes=kernel_sizes,
#             expansion_factor=expansion_factor,
#             dw_parallel=dw_parallel,
#             add=add,
#             lgag_ks=lgag_ks,
#             activation=activation,
#         )

#         self.out_head4 = nn.Conv2d(c4, out_channels, 1)
#         self.out_head3 = nn.Conv2d(c3, out_channels, 1)
#         self.out_head2 = nn.Conv2d(c2, out_channels, 1)
#         self.out_head1 = nn.Conv2d(c1, out_channels, 1)

#     def forward(self, ct, pet, target_size=None):
#         if ct.shape[1] == 1: ct = ct.repeat(1, 3, 1, 1)
#         if pet.shape[1] == 1: pet = pet.repeat(1, 3, 1, 1)

#         feats_ct = self.enc_ct(ct)
#         feats_pet = self.enc_pet(pet)
        
#         # 简单融合 Baseline
#         x = [a + b for a, b in zip(feats_ct, feats_pet)]
#         d4, d3, d2, d1 = self.decoder(x[3], [x[2], x[1], x[0]])

#         if target_size is None: target_size = ct.shape[-2:]
#         p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
#         p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
#         p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
#         p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
#         return [p1, p2, p3, p4]

# def build_mdt_seg_teacher(config):
#     p_path = getattr(config, 'pretrained_path', None)
#     model = DualPVTB2EMCAD(pretrained_path=p_path, in_channels=3, out_channels=1)
#     return dict(model=model)

# def build_mdt_seg_student(config):
#     s_path = getattr(config, 'student_pretrained_path', None) or getattr(config, 'pretrained_path', None)
#     model = SinglePVTEMCAD(pretrained_path=s_path, in_channels=3, out_channels=1)
#     return dict(model=model)


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
        # 针对 timm 官方权重命名进行清洗
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
                 kernel_sizes=(1, 3 , 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        # 强制断网
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


# ==============================================================================
# 核心创新模块：TSCAFBlock (容差感知与语义引导的跨模态自适应滤波模块)
# ==============================================================================
class TSCAFBlock(nn.Module):
    """
    Tolerance-aware Semantic-guided Cross-modality Adaptive Filtering
    用于在网络深层利用 PET 的语义/容差能量场来过滤 CT 的嘈杂边缘
    """
    def __init__(self, channels):
        super().__init__()
        # 1. CT 非对称边缘提取 (使用 groups=channels 保持极致轻量)
        self.ct_1x3 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.ct_3x1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        
        # 2. PET 容差多尺度能量场提取
        self.pet_core = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pet_halo = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        
        # 3. 动态自适应滤波器生成
        # 空间滤波器: 7x7大核捕获平滑容差区域 (输出 1 通道掩码)
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # 通道滤波器: Squeeze-and-Excitation 思想，挑选有价值的特征通道
        self.channel_filter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 4. 特征融合降维 (将过滤后的 CT 与原始 PET 语义重组)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ct_feat, pet_feat):
        # Step 1: 提取 CT 的高频十字形解剖边缘
        ct_edge = self.ct_1x3(ct_feat) + self.ct_3x1(ct_feat)
        
        # Step 2: 构建 PET 多尺度容差能量场
        p_core = self.pet_core(pet_feat)
        p_halo = self.pet_halo(pet_feat)
        p_cat = torch.cat([p_core, p_halo], dim=1)
        
        # Step 3: 动态生成滤波器
        w_s = self.spatial_filter(p_cat) # 空间权重
        w_c = self.channel_filter(p_cat) # 通道权重
        
        # Step 4: 自适应滤波 (仅放行符合 PET 定位和容差特性的 CT 边缘)
        ct_filtered = ct_edge * w_s * w_c
        
        # Step 5: 重组纯净边缘与 PET 语义
        out = self.fusion(torch.cat([ct_filtered, pet_feat], dim=1))
        return out


class DualPVTB2EMCAD(nn.Module):
    """Teacher: dual-stream CT/PET, two PVTv2-b2 encoders + TSCAF Decoupled Fusion + EMCAD."""
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        # 实例化两个编码器
        self.enc_ct = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
        self.enc_pet = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, in_chans=in_channels)
        
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name="Teacher_CT_Encoder")
            load_local_weights_safe(self.enc_pet, pretrained_path, name="Teacher_PET_Encoder")

        c1, c2, c3, c4 = self.enc_ct.feature_info.channels()

        # ---------------------------------------------------------
        # 核心集成：在深层引入 TSCAF 模块 (动态适配通道数)
        # ---------------------------------------------------------
        self.tscaf_x1 = TSCAFBlock(channels=c1)
        self.tscaf_x2 = TSCAFBlock(channels=c2)
        self.tscaf_x3 = TSCAFBlock(channels=c3)
        self.tscaf_x4 = TSCAFBlock(channels=c4)

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
        
        c1, c2, c3, c4 = feats_ct
        p1, p2, p3, p4 = feats_pet

        # =========================================================
        # 架构级创新：深层强引导滤波，浅层纯净边缘解耦
        # =========================================================
        
        # 1. 浅层解耦 (x1, x2): 
        # 坚决切断 PET 的弥散信号注入，让高分辨率的 CT 独挑边缘勾勒的大梁
        # x1 = c1
        # x2 = c2
        x1 = self.tscaf_x1(c1, p1)
        x2 = self.tscaf_x2(c2, p2)
        
        # 2. 深层自适应滤波融合 (x3, x4): 
        # 利用 TSCAF 模块，用 PET 的能量场去过滤 CT 的背景假阳性边缘
        x3 = self.tscaf_x3(c3, p3)
        x4 = self.tscaf_x4(c4, p4)
        
        # =========================================================

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