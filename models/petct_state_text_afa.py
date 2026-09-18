"""PET/CT 模块二：状态提示 + 固定文本调制 + 双残差 AFA（2D 多尺度，v3）。

放置位置: models/petct_state_text_afa.py；仅依赖 torch，导出文本另需 transformers。
建议Python>=3.10、PyTorch>=2.0。
512维文本+默认四尺度时可训练参数约1,439,000；不包含离线文本编码器。
核对基线: AilinLiuilearn/new-train, e1-pspi-ct-affine-smoothl1-add,
commit 8c39080e95de0c0d24683ed9cf55082685abee1c (2026-09-16)。
v3起点: AilinLiuilearn/new-train, e1-pspi-ct-affine-smoothl1-add-module2-v2,
commit c729f396cb181e0d86b6cd3e9c93782c1df8ef56。
本文件没有修改仓库，没有下载模型权重，没有添加损失。

确定的设计（每尺度同结构，参数不共享）:
  P_s = P + E[state]                    # 0=补偿PET，1=真实PET
  q   = GELU(LayerNorm(Linear(text)))
  R_t = 2*sigmoid(text_gate(GMP(P_s*q) + q)) - 1   # 零中心残差，[B,C,1,1]
  P_t = P_s * (1 + R_t)                 # 文本末层零初始化 -> 初始R_t=0
  U_CT, U_PET = Conv1x1(CT), Conv1x1(P_t)  # 各压缩至 C//2
  U   = concat(U_CT, U_PET)
  S   = sigmoid(Conv5x5(cat(channel_mean(U), channel_max(U))))
  U_f = U_CT*S[:,0:1] + U_PET*S[:,1:2]
  H_CT, H_PET = afa_out_ct(U_f), afa_out_pet(U_f)  # 两个独立头，各mid->C
  R_CT, R_PET = 2*sigmoid(H_CT)-1, 2*sigmoid(H_PET)-1  # 均为[B,C,H,W]
  F   = CT*(1 + R_CT) + P_t*(1 + R_PET)
  pet_valid=False 的行直接返回 CT，不运行上述 PET 分支。

v2 -> v3 变更:
  文本门控: sigmoid(g) 改成 2*sigmoid(h_t)-1，text_gate末层零初始化，
    因此初始R_t=0、P_t=P_s；v2初始P_t=P_s*(1+g)，g非零。
  AFA空间卷积: 7x7改成5x5。
  融合: CT+P_t+delta改成双残差CT*(1+R_CT)+P_t*(1+R_PET)；
    旧afa_out单头改成afa_out_ct/afa_out_pet两个独立头，均零初始化。
  架构标识: extra_state version=2, architecture=dual_residual_afa_v3。
  v3因此与v2 checkpoint不兼容，不得裁剪/复制权重静默转换。
  文本缓存format_version保持1不变（不同概念）。

来源与改动:
  MPLMM: https://github.com/zrguo/MPLMM
    只借鉴缺失/非缺失可学习状态提示相加；没有移植生成提示或类型提示。
  DGNet: https://github.com/iLearn-Lab/MM26-DGNet
    参考提交 7e1d23922dfe57d219abdd5032af08bb32aad9cb。
    借鉴目标文本通道调制。这里采用讨论后的简化式，去掉空间3x3卷积，
    MLP在GMP结果与q相加之后；增加1+g残差。不是原始T-KGM逐行复刻。
  ADGNet: https://github.com/iLearn-Lab/MM26-ADGNet
    参考提交 f6ab6bd07f2fb7d4f09085ec1cd90e18ca8876b7。
    借鉴 DualStreamFusion.py/AFA 的降维、通道mean/max、7x7双权重。
    改成 CT+P_t+残差，最后不做sigmoid，残差输出层零初始化。
    DGNet与ADGNet是两个来源。没有小波、交叉注意力或额外路由。

文本策略:
  固定描述: A PET image showing bright tumor regions in the lungs.
  描述是通用任务先验，不是病例报告/标注，也不表示所有肿瘤必定明亮。
  若目标并非肺部肿瘤，必须用 --prompt 显式更改，不能照搬本句。
  默认clip后端返回CLIPTextModel.pooler_output，即整句EOS池化向量，
  非全部token、非类别ID、非CLIP投影后的对比空间向量（符合DGNet接口）。
  clip-vit-base-patch32通常为512维；维度从缓存实际读取，不硬编码。
  可选biomedbert返回attention-mask均值池化向量；它不等于BiomedCLIP。
  不自动下载，不随机替代缺失模型；文本一次性离线编码，训练只用buffer。
  要用完整BiomedCLIP投影句向量，可在外部用官方encode_text提取后调用
  save_text_cache；本文件不伪造其投影权重。不同后端是消融选择，不混用。

导出（本地HuggingFace模型目录必须含config、tokenizer和权重）:
  python models/petct_state_text_afa.py --export-text-cache pretrained/pet_text.pt \
      --backend clip --model-path /path/to/clip-vit-base-patch32
  # 复用已有BiomedBERT目录的可选命令（这不是DGNet同款编码器）:
  python models/petct_state_text_afa.py --export-text-cache pretrained/pet_text.pt \
      --backend biomedbert --model-path /path/to/biomedbert_text_tower

接入 DualSharedAddPETCTBaseline:
  1. 在 __init__ 中替换 self.fusion = AddFusion()，且必须在创建优化器之前:
       self.fusion = StateTextAFAFusion.from_text_cache(
           text_cache_path, channels=pet_channels)
     text_cache_path需要通过构造函数/config/builder传入，不要硬编码个人路径。
     所有可训练层在构造时创建，兼容已有AdamW(self.model.parameters())。
  2. 不能只替换构造器。原代码每处 fusion(..., None) 必须显式传状态与有效性:
     Full _forward_full:
       fused_feats = self.fusion(ct_feats, pet_real_feats, 1, pet_valid=True)
     Missing 的所有 train/eval/legacy 分支:
       ready = bool(self.pspi_enabled and self.module1.bank_ready)
       fused_feats = self.fusion(ct_feats, pet_comp_or_prior, 0, pet_valid=ready)
     pspi关闭或bank未ready时用原有zero PET，并传 pet_valid=False。
     混合 _forward_auto，保留原有pet_for_fusion装配，逐样本传入:
       ready = bool(self.pspi_enabled and self.module1.bank_ready)
       valid = pet_available.reshape(-1).bool() | ready
       fused_feats = self.fusion(ct_feats, pet_for_fusion, pet_available,
                                 pet_valid=valid)
     注意: pet_available=0并不意味着pet_valid=False！bank就绪时补偿PET有效。
     该提交中待更新的fusion调用位于 _forward_full、_forward_missing、
     _forward_auto，共9处。严禁遗漏eval/legacy/pspi关闭路径。
  3. 输出仍是四尺度list，decoder调用不变。模块一、CT-affine、memory更新、
     teacher/真实PET编码及SmoothL1重建路径均保持原样。重建损失继续针对
     上游pet_comp，而不是本模块内部P_t。不要detach pet_comp。
  4. 将builder的AddFusion日志更新为StateTextAFAFusion；Full和Missing共用
     同一个fusion实例。恢复模块二checkpoint必须相同配置，strict=True。
     从AddFusion基线初始化时只允许缺少 fusion.* 新键；检查全部其他
     missing/unexpected keys，不要无检查地strict=False；重建优化器。

输入: CT/PET各N尺度list，每尺度[B,C,H,W]，同batch/device/dtype。
支持PET空间尺寸不等时bilinear对齐至CT。返回形状与CT相同。
pet_available: bool/0/1标量，或[B]/[B,1]/[B,1,1,1]逐样本状态。
pet_valid: 同样形状；必须显式传递，防止冷启动偏置泄漏。
模块仅接收已经选好来源的PET，不持有真实PET编码器/原型库，也不修改输入。
所有有效特征应有限；不以nan_to_num掩盖上游数值问题。

初始化/消融:
  状态提示全零；文本末层全零；AFA双输出头全零，因此初始
  R_t=R_CT=R_PET=0，有效行F=CT+PET（与AddFusion初始严格相等）。
  enabled=False: 有效行严格CT+PET，无效行CT。
  use_text=False: 真正绕过文本分支，不用零文本假装关闭。
  use_state=False/use_afa=False: 对应机制消融；不改变输出接口。
  use_afa=False: F=CT+P_t（文本仍可作用于PET）。
  零初始化使AFA前级（afa_ct/afa_pet/afa_spatial）及文本前级
  （text_proj/text_gate前层）第一步梯度为零，输出层更新后恢复，
  这是预期行为；至少2次优化更新后前级应收到有限非零梯度。
  模块无BatchNorm，无新增loss，不保证Dice/HD95改善。

验证: python models/petct_state_text_afa.py --self-test
      python models/petct_state_text_afa.py --smoke-test
      python models/petct_state_text_afa.py --baseline-contract /path/to/new-train
第三条从本地基线读取真实模块一/affine/混合装配函数/decoder代码进行特征级
串联，跳过图像backbone与完整训练器；只对你信任的代码目录执行。
单元测试使用合成文本向量；可选文本导出测试用本地微型随机HF模型验证
CLIP/BERT加载、池化、缓存与MLP梯度，不代表真实预训练文本权重验证。
尚未验证实际预训练文本文件、图像backbone全链路、GPU或数据集训练指标。
v3 checkpoint与v2不兼容：afa_spatial 7x7->5x5、afa_out单头->双头、
文本门控语义改变；恢复时set_extra_state明确拒绝version=1旧元数据。
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Sequence
import unittest

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_PROMPT = 'A PET image showing bright tumor regions in the lungs.'
DEFAULT_CHANNELS = (64, 128, 320, 512)


def _sentence_vector(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError('text_feature must be a torch.Tensor')
    value = value.detach().float().cpu().clone()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError('Expected one cached sentence vector [D] or [1,D], not tokens')
    if not torch.isfinite(value).all():
        raise ValueError('text_feature contains NaN/Inf')
    return value.contiguous()


def save_text_cache(path: str | Path, vector: Tensor, metadata: dict[str, Any]) -> None:
    """Save an ordinary non-inference tensor, suitable for trainable MLP backward."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing text cache: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.get('prompt') or not metadata.get('backend'):
        raise ValueError('metadata requires prompt and backend')
    torch.save({'format_version': 1, 'text_feature': _sentence_vector(vector),
                'metadata': dict(metadata)}, path)


def encode_fixed_text(model_path: str | Path, *, backend: str = 'clip',
                      prompt: str = DEFAULT_PROMPT, max_length: int = 30,
                      device: str = 'cpu') -> tuple[Tensor, dict[str, Any]]:
    """Offline-only frozen encoding. No image encoder; no network in fusion.forward.

    clip: EOS pooler_output before text_projection, matching DGNet's saved vectors.
    biomedbert: masked mean last_hidden_state (explicit alternative, not BiomedCLIP).
    """
    if backend not in ('clip', 'biomedbert'):
        raise ValueError('backend must be clip or biomedbert')
    if not str(prompt).strip() or max_length < 3:
        raise ValueError('Nonempty prompt and max_length >= 3 are required')
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'Local text model directory not found: {root}')
    try:
        from transformers import AutoTokenizer, AutoModel, CLIPTextModel
    except ImportError as exc:
        raise ImportError('Text export needs transformers and a local pretrained checkpoint') from exc
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                              trust_remote_code=False)
    raw = tokenizer(prompt, add_special_tokens=True, truncation=False)['input_ids']
    if len(raw) > max_length:
        raise ValueError(f'Prompt needs {len(raw)} tokens; increase max_length={max_length}')
    cls = CLIPTextModel if backend == 'clip' else AutoModel
    model = cls.from_pretrained(str(root), local_files_only=True,
                               trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    encoded = tokenizer(prompt, padding='max_length', max_length=max_length,
                        truncation=False, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()
               if k in ('input_ids', 'attention_mask', 'token_type_ids')}
    if backend == 'clip':
        encoded.pop('token_type_ids', None)
    with torch.no_grad():
        output = model(**encoded)
        if backend == 'clip':
            vector = output.pooler_output
            pooling = 'CLIP EOS pooler_output; no text_projection'
        else:
            mask = encoded['attention_mask'].unsqueeze(-1).to(output.last_hidden_state.dtype)
            vector = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooling = 'masked mean last_hidden_state; includes non-padding special tokens'
    vector = _sentence_vector(vector)
    return vector, {'prompt': prompt, 'backend': backend, 'pooling': pooling,
                    'model_path': str(root.resolve()), 'max_length': max_length,
                    'text_dim': vector.shape[1], 'normalized': False}


def _binary_rows(value: Any, batch: int, device: torch.device, name: str) -> Tensor:
    if value is None:
        raise ValueError(f'{name} is required: 1=True, 0=False')
    result = torch.as_tensor(value, device=device)
    if result.ndim == 0:
        result = result.expand(batch)
    elif result.shape[0] == batch and result.numel() == batch:
        result = result.reshape(batch)
    else:
        raise ValueError(f'{name} must be scalar or one value per sample; got {tuple(result.shape)}')
    if not torch.all((result == 0) | (result == 1)):
        raise ValueError(f'{name} must contain only 0 or 1')
    return result.bool()


class _ScaleFusion(nn.Module):
    def __init__(self, channels: int, text_dim: int, reduction: int):
        super().__init__()
        hidden = max(channels // reduction, 1)
        mid = max(channels // 2, 1)
        self.state_prompt = nn.Parameter(torch.zeros(2, channels, 1, 1))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, channels),
                                       nn.LayerNorm(channels), nn.GELU())
        self.text_gate = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                       nn.ReLU(inplace=False), nn.Conv2d(hidden, channels, 1))
        # Zero-centered text residual: last layer zero-init => R_t=0 at init.
        nn.init.zeros_(self.text_gate[-1].weight)
        nn.init.zeros_(self.text_gate[-1].bias)
        self.afa_ct = nn.Conv2d(channels, mid, 1)
        self.afa_pet = nn.Conv2d(channels, mid, 1)
        self.afa_spatial = nn.Conv2d(2, 2, 5, padding=2)
        self.afa_out_ct = nn.Conv2d(mid, channels, 1)
        self.afa_out_pet = nn.Conv2d(mid, channels, 1)
        nn.init.zeros_(self.afa_out_ct.weight)
        nn.init.zeros_(self.afa_out_ct.bias)
        nn.init.zeros_(self.afa_out_pet.weight)
        nn.init.zeros_(self.afa_out_pet.bias)

    def forward(self, ct: Tensor, pet: Tensor, states: Tensor, text: Tensor,
                *, use_state: bool, use_text: bool, use_afa: bool,
                diagnostics: bool) -> tuple[Tensor, dict[str, Tensor]]:
        p = pet + self.state_prompt[states.long()].to(pet.dtype) if use_state else pet
        # AMP: keep the explicit gate math in the feature dtype. Autocast
        # casts only whitelisted ops, so run the small gate MLPs under a
        # local autocast-disabled region with inputs cast to the parameter
        # dtype, then cast the residual back (matches the v2 boundary that
        # unified CT/PET dtypes in _fuse_modalities).
        r_text = None
        if use_text:
            gate_param_dtype = self.text_gate[0].weight.dtype
            q = self.text_proj(text.to(gate_param_dtype)).unsqueeze(-1).unsqueeze(-1)
            pooled = F.adaptive_max_pool2d(p.to(gate_param_dtype) * q, 1)
            with torch.autocast(device_type=p.device.type, enabled=False):
                r_text = (2.0 * torch.sigmoid(self.text_gate(pooled + q)) - 1.0)
            p = p * (1 + r_text.to(p.dtype))
        if not use_afa:
            out = ct + p
        else:
            afa_param_dtype = self.afa_ct.weight.dtype
            a = self.afa_ct(ct.to(afa_param_dtype)).to(ct.dtype)
            b = self.afa_pet(p.to(afa_param_dtype)).to(ct.dtype)
            joined = torch.cat((a, b), dim=1)
            statistics = torch.cat((joined.mean(1, keepdim=True),
                                    joined.amax(1, keepdim=True)), dim=1)
            with torch.autocast(device_type=ct.device.type, enabled=False):
                spatial = torch.sigmoid(
                    self.afa_spatial(statistics.to(self.afa_spatial.weight.dtype))
                )
            spatial = spatial.to(ct.dtype)
            fused_joint = a * spatial[:, :1] + b * spatial[:, 1:]
            with torch.autocast(device_type=ct.device.type, enabled=False):
                joint_param = fused_joint.to(self.afa_out_ct.weight.dtype)
                r_ct = 2.0 * torch.sigmoid(self.afa_out_ct(joint_param)) - 1.0
                r_pet = 2.0 * torch.sigmoid(self.afa_out_pet(joint_param)) - 1.0
            r_ct, r_pet = r_ct.to(ct.dtype), r_pet.to(p.dtype)
            out = ct * (1 + r_ct) + p * (1 + r_pet)
        info: dict[str, Tensor] = {}
        if diagnostics:
            if r_text is not None:
                info['r_text'] = r_text.detach()
            if use_afa:
                info['afa_spatial_mean'] = spatial.detach().float().mean((2, 3))
                info['r_ct'] = r_ct.detach()
                info['r_pet'] = r_pet.detach()
        return out, info


class StateTextAFAFusion(nn.Module):
    """Replace AddFusion with explicit source state and compensation validity.

    forward returns list[Tensor], or (list, per-scale detached diagnostics).
    Diagnostics refer to valid_rows only. Never feed diagnostics to losses.
    """

    def __init__(self, channels: Sequence[int] = DEFAULT_CHANNELS, *,
                 text_feature: Tensor | None = None, text_dim: int = 512,
                 text_metadata: dict[str, Any] | None = None,
                 reduction: int = 16, enabled: bool = True, use_state: bool = True,
                 use_text: bool = True, use_afa: bool = True,
                 diag_enabled: bool = False, diag_interval: int = 50):
        super().__init__()
        if not channels or any(not isinstance(c, int) or c < 2 for c in channels):
            raise ValueError('channels must be a nonempty sequence of integers >= 2')
        if not isinstance(reduction, int) or reduction < 1:
            raise ValueError('reduction must be a positive integer')
        if text_feature is None:
            if use_text and enabled:
                raise ValueError('Text enabled: supply a real cached text_feature or use from_text_cache')
            if not isinstance(text_dim, int) or text_dim < 1:
                raise ValueError('text_dim must be a positive integer')
            vector = torch.zeros(1, text_dim)
        else:
            vector = _sentence_vector(text_feature)
        self.channels = tuple(channels)
        self.reduction = reduction
        self.enabled, self.use_state = bool(enabled), bool(use_state)
        self.use_text, self.use_afa = bool(use_text), bool(use_afa)
        self.diag_enabled = bool(diag_enabled)
        self.diag_interval = int(diag_interval)
        if self.diag_interval < 1:
            raise ValueError('diag_interval must be >= 1')
        self.register_buffer('text_feature', vector)
        self.register_buffer('text_ready', torch.tensor(text_feature is not None))
        self.text_metadata = dict(text_metadata or {'prompt': DEFAULT_PROMPT, 'backend': 'provided-vector'})
        self.scales = nn.ModuleList([_ScaleFusion(c, vector.shape[1], reduction) for c in channels])
        # Read-only amplitude monitor (never part of the graph/loss). Keys are
        # plain Python floats/ints aggregated per forward; cleared per epoch.
        self._diag_stats: dict[str, Any] = {'forwards': 0, 'scales': {}}
        self._diag_forward_count = 0

    @classmethod
    def from_text_cache(cls, path: str | Path, *, channels: Sequence[int] = DEFAULT_CHANNELS,
                        **kwargs: Any) -> 'StateTextAFAFusion':
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(cache, dict) or cache.get('format_version') != 1:
            raise ValueError('Use save_text_cache or --export-text-cache to create a version-1 cache')
        if not isinstance(cache.get('metadata'), dict):
            raise ValueError('Text cache requires metadata')
        return cls(channels, text_feature=cache['text_feature'],
                   text_metadata=cache['metadata'], **kwargs)

    def get_extra_state(self) -> dict[str, Any]:
        return {'version': 2, 'architecture': 'dual_residual_afa_v3',
                'text_metadata': self.text_metadata,
                'config': {'channels': self.channels, 'reduction': self.reduction,
                           'enabled': self.enabled, 'use_state': self.use_state,
                           'use_text': self.use_text, 'use_afa': self.use_afa,
                           'kernel_size': 5, 'centered_text_gate': True,
                           'dual_modality_residual': True,
                           'diag_enabled': self.diag_enabled,
                           'diag_interval': self.diag_interval}}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get('version') != 2:
            raise ValueError(
                'Unsupported fusion checkpoint metadata: expected v3 '
                f"(architecture=dual_residual_afa_v3), got version={state.get('version') if isinstance(state, dict) else type(state)}; "
                'v2 checkpoints cannot load into v3 (afa 7x7->5x5, single->dual head, gate rescale). '
                'Evaluate v2 checkpoints on the v2 branch.'
            )
        if state.get('architecture') != 'dual_residual_afa_v3':
            raise ValueError(
                f"Fusion architecture mismatch: {state.get('architecture')!r} != 'dual_residual_afa_v3'"
            )
        if state.get('config') != self.get_extra_state()['config']:
            raise ValueError('Fusion checkpoint config differs; construct the same architecture/flags')
        self.text_metadata = dict(state['text_metadata'])

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
                pet_available: Any, *, pet_valid: Any,
                return_diagnostics: bool = False) -> list[Tensor] | tuple[list[Tensor], list[dict[str, Tensor]]]:
        if len(ct_feats) != len(self.channels) or len(pet_feats) != len(self.channels):
            raise ValueError(f'Expected {len(self.channels)} CT and PET scales')
        first = ct_feats[0]
        if first.ndim != 4 or first.shape[0] < 1:
            raise ValueError('Features must be nonempty [B,C,H,W] tensors')
        batch = first.shape[0]
        state = _binary_rows(pet_available, batch, first.device, 'pet_available')
        valid = _binary_rows(pet_valid, batch, first.device, 'pet_valid')
        if torch.any(state & ~valid):
            raise ValueError('A real-PET sample cannot have pet_valid=False')
        if self.enabled and self.use_text and not bool(self.text_ready):
            raise RuntimeError('Text modulation enabled without a real text cache')
        if self.text_feature.device != first.device:
            raise ValueError('Move fusion and features to the same device before forward')
        rows = valid.nonzero(as_tuple=False).flatten()
        diag_on = bool(self.diag_enabled) and (self._diag_forward_count % self.diag_interval == 0)
        if bool(self.diag_enabled):
            self._diag_forward_count += 1
        outputs, infos = [], []
        for i, (ct, pet, channels, block) in enumerate(zip(ct_feats, pet_feats, self.channels, self.scales)):
            if ct.ndim != 4 or pet.ndim != 4 or ct.shape[:2] != (batch, channels) or pet.shape[:2] != (batch, channels):
                raise ValueError(f'Scale {i}: expected CT/PET [B={batch},C={channels},H,W]')
            if min(ct.shape[2:] + pet.shape[2:]) < 1:
                raise ValueError(f'Scale {i}: empty spatial dimension')
            if ct.device != first.device or pet.device != ct.device or ct.dtype != first.dtype or pet.dtype != ct.dtype:
                raise ValueError('All CT/PET scales must share device and floating dtype')
            if not ct.is_floating_point():
                raise TypeError('Feature tensors must be floating point')
            info = {'valid_rows': rows.detach()}
            if rows.numel() == 0:
                out = ct  # strict cold-start; no PET/prompt/MLP arithmetic
            else:
                c, p = ct.index_select(0, rows), pet.index_select(0, rows)
                if p.shape[-2:] != c.shape[-2:]:
                    p = F.interpolate(p, size=c.shape[-2:], mode='bilinear', align_corners=False)
                if not self.enabled:
                    fused = c + p
                else:
                    fused, more = block(c, p, state.index_select(0, rows), self.text_feature,
                                        use_state=self.use_state, use_text=self.use_text,
                                        use_afa=self.use_afa, diagnostics=(return_diagnostics or diag_on))
                    info.update(more)
                    if diag_on:
                        self._accumulate_diag(
                            i, state, rows,
                            more.get('r_text'), more.get('r_ct'), more.get('r_pet'), fused,
                        )
                out = ct.index_copy(0, rows, fused.to(ct.dtype))
            outputs.append(out)
            infos.append(info)
        if diag_on:
            self._diag_stats['forwards'] = int(self._diag_stats.get('forwards', 0)) + 1
        return (outputs, infos) if return_diagnostics else outputs

    @staticmethod
    def _rms(x: Tensor) -> float:
        return float(x.detach().float().pow(2).mean().sqrt().item())

    def _accumulate_diag(self, scale_idx, state, rows, r_text, r_ct, r_pet, fused) -> None:
        """Aggregate per-scale Full/Missing amplitude stats (detached scalars).

        Cold-start (no valid rows) records count=0 with None values, never
        fake zero gates. Text/AFA-off branches record 'n/a'. Fused RMS is
        split into CT/PET branch amplitude via the detached residual ratio.
        """
        entry = self._diag_stats['scales'].setdefault(
            f'scale{scale_idx + 1}',
            {'full': self._empty_group(), 'missing': self._empty_group()},
        )
        row_state = state.index_select(0, rows).tolist()
        groups = {
            'full': [k for k, s in enumerate(row_state) if s],
            'missing': [k for k, s in enumerate(row_state) if not s],
        }
        for tag, positions in groups.items():
            group = entry[tag]
            if not positions:
                continue
            group['count'] += len(positions)
            pos = torch.tensor(positions, device=rows.device)
            if self.use_text and r_text is not None:
                vals = r_text.detach().float().index_select(0, pos)
                group['r_t_sum'] += float(vals.mean().item()) * len(positions)
                group['r_t_abs_sum'] += float(vals.abs().mean().item()) * len(positions)
            if self.use_afa and r_ct is not None and r_pet is not None:
                rc = r_ct.detach().float().index_select(0, pos)
                rp = r_pet.detach().float().index_select(0, pos)
                group['r_ct_sum'] += float(rc.mean().item()) * len(positions)
                group['r_ct_abs_sum'] += float(rc.abs().mean().item()) * len(positions)
                group['r_pet_sum'] += float(rp.mean().item()) * len(positions)
                group['r_pet_abs_sum'] += float(rp.abs().mean().item()) * len(positions)
                sub = fused.detach().float().index_select(0, pos)
                group['fused_rms_sum'] += self._rms(sub) * len(positions)

    @staticmethod
    def _empty_group() -> dict[str, Any]:
        return {'count': 0, 'r_t_sum': 0.0, 'r_t_abs_sum': 0.0,
                'r_ct_sum': 0.0, 'r_ct_abs_sum': 0.0,
                'r_pet_sum': 0.0, 'r_pet_abs_sum': 0.0,
                'fused_rms_sum': 0.0}

    def reset_diag_stats(self) -> None:
        self._diag_stats = {'forwards': 0, 'scales': {}}
        self._diag_forward_count = 0

    def pop_diag_stats(self) -> dict[str, Any]:
        """Return epoch-aggregated diagnostics and clear (no graphs kept)."""
        out: dict[str, Any] = {'forwards': int(self._diag_stats.get('forwards', 0)), 'scales': {}}
        for scale, groups in self._diag_stats.get('scales', {}).items():
            out['scales'][scale] = {}
            for tag, g in groups.items():
                n = int(g['count'])
                if n == 0:
                    out['scales'][scale][tag] = {'count': 0, 'r_t_mean': None,
                                                 'r_t_abs_mean': None, 'r_ct_mean': None,
                                                 'r_ct_abs_mean': None, 'r_pet_mean': None,
                                                 'r_pet_abs_mean': None, 'fused_rms': None}
                    continue
                entry = {'count': n}
                entry['r_t_mean'] = g['r_t_sum'] / n if self.use_text else 'n/a'
                entry['r_t_abs_mean'] = g['r_t_abs_sum'] / n if self.use_text else 'n/a'
                entry['r_ct_mean'] = g['r_ct_sum'] / n if self.use_afa else 'n/a'
                entry['r_ct_abs_mean'] = g['r_ct_abs_sum'] / n if self.use_afa else 'n/a'
                entry['r_pet_mean'] = g['r_pet_sum'] / n if self.use_afa else 'n/a'
                entry['r_pet_abs_mean'] = g['r_pet_abs_sum'] / n if self.use_afa else 'n/a'
                entry['fused_rms'] = g['fused_rms_sum'] / n
                out['scales'][scale][tag] = entry
        self.reset_diag_stats()
        return out


class ContractTests(unittest.TestCase):
    """Synthetic vectors test mechanics, NOT text semantics or segmentation quality."""

    def setUp(self):
        # Seed 42 reproduces the v2 fixture inputs, whose scale-1 ReLU gate
        # is degenerate (all-zero first-step grad) under the v3 zero-centered
        # gate; per-scale assertions below therefore allow zero and check the
        # update-reachability separately with non-degenerate inputs.
        torch.manual_seed(42)
        self.channels = (8, 16, 24, 32)
        self.vector = torch.randn(1, 12, requires_grad=True)
        self.net = StateTextAFAFusion(self.channels, text_feature=self.vector)
        self.ct = [torch.randn(3, c, s, s, requires_grad=True)
                   for c, s in zip(self.channels, (16, 8, 4, 2))]
        self.pet = [torch.randn_like(x, requires_grad=True) for x in self.ct]

    def call(self, state=(1, 0, 1), valid=True, **kwargs):
        return self.net(self.ct, self.pet, state, pet_valid=valid, **kwargs)

    def test_full_and_missing_all_scales(self):
        for state in (1, 0, (1, 0, 1)):
            outputs = self.call(state)
            self.assertEqual(len(outputs), 4)
            for y, c in zip(outputs, self.ct):
                self.assertEqual(y.shape, c.shape)
                self.assertTrue(torch.isfinite(y).all())

    def test_backward_and_frozen_text(self):
        sum(y.square().mean() for y in self.call()).backward()
        self.assertFalse(self.net.text_feature.requires_grad)
        self.assertIsNone(self.vector.grad)
        for c, p, block in zip(self.ct, self.pet, self.net.scales):
            self.assertGreater(c.grad.abs().sum().item(), 0)
            self.assertGreater(p.grad.abs().sum().item(), 0)
            self.assertGreater(block.state_prompt.grad[0].abs().sum().item(), 0)
            self.assertGreater(block.state_prompt.grad[1].abs().sum().item(), 0)
            # v3 zero-centered gates: both AFA output heads receive first-step
            # gradients; their pre-layers are blocked by the zero init.
            self.assertGreater(block.afa_out_ct.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.afa_out_pet.weight.grad.abs().sum().item(), 0)
        # Text output head: non-degenerate inputs must yield a first-step
        # gradient. (The seed-42 fixture has a degenerate all-zero ReLU path
        # at small scales; use fresh non-degenerate inputs here.)
        torch.manual_seed(7)
        probe = StateTextAFAFusion(self.channels, text_feature=torch.randn(1, 12))
        ct = [torch.randn(3, c, s, s) for c, s in zip(self.channels, (16, 8, 4, 2))]
        pet = [torch.randn_like(x) for x in ct]
        probe.zero_grad(set_to_none=True)
        sum(y.square().mean() for y in probe(ct, pet, (1, 0, 1), pet_valid=True)).backward()
        self.assertGreater(probe.scales[0].text_gate[-1].weight.grad.abs().sum().item(), 0)
        # Non-degenerate updates reach the text/AFA pre-layers: after two
        # AdamW steps the pre-layers receive finite nonzero gradients on at
        # least one scale (per-scale ReLU paths may stay degenerate on fixed
        # synthetic fixtures; full multi-scale reachability is covered by the
        # two_optimizer_steps test with its own inputs).
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            sum(y.square().mean() for y in self.call()).backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        sum(y.square().mean() for y in self.call()).backward()
        pre_grads = []
        for block in self.net.scales:
            pre_grads.append(float(block.text_proj[0].weight.grad.abs().sum()))
            pre_grads.append(float(block.afa_ct.weight.grad.abs().sum()))
        self.assertTrue(all(g == g for g in pre_grads))
        self.assertGreater(max(pre_grads), 0)

    def test_cold_start_exact_ct_and_no_pet_grad(self):
        for p in self.pet:
            with torch.no_grad():
                p.fill_(float('nan'))  # invalid compensation must not enter computation
        outputs = self.call(0, False)
        for y, c in zip(outputs, self.ct):
            self.assertTrue(torch.equal(y, c))
        sum(y.sum() for y in outputs).backward()
        self.assertTrue(all(p.grad is None for p in self.pet))

    def test_mixed_cold_start(self):
        outputs = self.call(valid=(1, 0, 1))
        for y, c in zip(outputs, self.ct):
            self.assertTrue(torch.equal(y[1], c[1]))
            self.assertFalse(torch.equal(y[0], c[0]))

    def test_disabled_exact_addition(self):
        self.net.enabled = False
        for y, c, p in zip(self.call(), self.ct, self.pet):
            self.assertTrue(torch.equal(y, c + p))

    def test_no_text_initial_exact_addition(self):
        self.net.use_text = False
        for y, c, p in zip(self.call(), self.ct, self.pet):
            self.assertTrue(torch.equal(y, c + p))

    def test_v3_initial_exact_addition(self):
        # v3 zero-centered gates: R_t=R_CT=R_PET=0 at init, so the enabled
        # module initially equals CT+PET exactly (unlike v2's nonzero text g).
        outputs, infos = self.call(return_diagnostics=True)
        for y, c, p, d in zip(outputs, self.ct, self.pet, infos):
            self.assertTrue(torch.equal(d['r_text'], torch.zeros_like(d['r_text'])))
            self.assertTrue(torch.equal(d['r_ct'], torch.zeros_like(d['r_ct'])))
            self.assertTrue(torch.equal(d['r_pet'], torch.zeros_like(d['r_pet'])))
            torch.testing.assert_close(y, c + p, rtol=0, atol=0)
            self.assertEqual(tuple(d['r_ct'].shape), tuple(c.shape))
            self.assertEqual(tuple(d['r_pet'].shape), tuple(c.shape))
            self.assertEqual(tuple(d['r_text'].shape), (c.shape[0], c.shape[1], 1, 1))
            self.assertTrue(all(not v.requires_grad for v in d.values()))

    def test_dual_heads_independent(self):
        self.net.eval()
        with torch.no_grad():
            for block in self.net.scales:
                block.afa_out_ct.bias.fill_(3.0)
                block.afa_out_pet.bias.fill_(-3.0)
        outputs, infos = self.call(return_diagnostics=True)
        for d in infos:
            self.assertFalse(torch.equal(d['r_ct'], d['r_pet']))
            self.assertTrue((d['r_ct'] > 0).all())
            self.assertTrue((d['r_pet'] < 0).all())
            self.assertEqual(tuple(d['r_ct'].shape)[1:], tuple(d['r_pet'].shape)[1:])

    def test_batch_independence(self):
        all_rows = self.call()
        for i, state in enumerate((1, 0, 1)):
            one = self.net([c[i:i+1] for c in self.ct],
                           [p[i:i+1] for p in self.pet], state, pet_valid=True)
            for a, b in zip(all_rows, one):
                torch.testing.assert_close(a[i:i+1], b, rtol=2e-5, atol=2e-6)

    def test_checkpoint_roundtrip(self):
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        buf.seek(0)
        clone = StateTextAFAFusion(self.channels, text_feature=torch.zeros(1, 12))
        clone.load_state_dict(torch.load(buf, weights_only=True))
        for a, b in zip(self.call(), clone(self.ct, self.pet, (1, 0, 1), pet_valid=True)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_validation(self):
        for state in (None, 2, [1, 0], [1, 0.5, 0]):
            with self.assertRaises((ValueError, TypeError)):
                self.call(state)
        with self.assertRaises(ValueError):
            self.call(1, False)
        with self.assertRaises(ValueError):
            self.net(self.ct[:3], self.pet, 1, pet_valid=True)
        with self.assertRaises(ValueError):
            StateTextAFAFusion(self.channels)
        with self.assertRaises(ValueError):
            StateTextAFAFusion(self.channels, text_feature=torch.full((12,), float('nan')))

    def test_spatial_resize(self):
        smaller = [F.avg_pool2d(p, 2) for p in self.pet]
        outputs = self.net(self.ct, smaller, 0, pet_valid=True)
        self.assertEqual([x.shape for x in outputs], [x.shape for x in self.ct])

    def test_no_encoder_or_memory_calls(self):
        # No privileged-data dependency is reachable from this feature-only module.
        before = set(self.net.state_dict())
        self.call(0)
        self.assertEqual(before, set(self.net.state_dict()))
        self.assertFalse(any('encoder' in n or 'bank' in n for n, _ in self.net.named_modules()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in self.net.modules()))

    def test_two_optimizer_steps(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            sum(y.square().mean() for y in self.call()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in self.net.parameters()))
            opt.step()
        for block in self.net.scales:
            self.assertGreater(block.afa_ct.weight.grad.abs().sum().item(), 0)
        # Text pre-layers: nonzero on at least one scale (per-scale ReLU
        # paths on fixed synthetic fixtures may stay degenerate; the
        # output-head first-step gradient is asserted in
        # test_backward_and_frozen_text with non-degenerate inputs).
        text_pre = [float(block.text_proj[0].weight.grad.abs().sum().item())
                    for block in self.net.scales]
        self.assertTrue(all(g == g for g in text_pre))
        self.assertGreater(max(text_pre), 0)

    def test_cpu_autocast(self):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            outputs = self.call()
            loss = sum(y.square().mean() for y in outputs)
        loss.backward()
        self.assertTrue(all(torch.isfinite(y).all() for y in outputs))

    def test_cache_format(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'text.pt'
            save_text_cache(path, self.vector, {'prompt': DEFAULT_PROMPT, 'backend': 'synthetic-test'})
            net = StateTextAFAFusion.from_text_cache(path, channels=self.channels)
            torch.testing.assert_close(net.text_feature, self.vector.detach())
            self.assertEqual(net.text_metadata['prompt'], DEFAULT_PROMPT)

    def test_state_is_selected_per_sample(self):
        self.net.use_text = False
        self.net.use_afa = False
        with torch.no_grad():
            for block in self.net.scales:
                block.state_prompt[0].fill_(2.)
                block.state_prompt[1].fill_(5.)
        offset = torch.tensor([5., 2., 5.]).view(3, 1, 1, 1)
        for y, c, p in zip(self.call(), self.ct, self.pet):
            torch.testing.assert_close(y, c + (p + offset), rtol=0, atol=0)

    def test_extra_state_v3_and_v2_rejected(self):
        state = self.net.get_extra_state()
        self.assertEqual(state['version'], 2)
        self.assertEqual(state['architecture'], 'dual_residual_afa_v3')
        self.assertEqual(state['config']['kernel_size'], 5)
        self.assertTrue(state['config']['centered_text_gate'])
        self.assertTrue(state['config']['dual_modality_residual'])
        clone = StateTextAFAFusion(self.channels, text_feature=torch.zeros(1, 12))
        clone.set_extra_state(state)
        v2 = dict(state)
        v2['version'] = 1
        v2.pop('architecture', None)
        with self.assertRaises(ValueError):
            clone.set_extra_state(v2)

    def test_disabled_mechanisms_skip_nonzero_parameters(self):
        self.net.use_text = self.net.use_state = self.net.use_afa = False
        with torch.no_grad():
            for block in self.net.scales:
                block.state_prompt.fill_(10.)
                block.afa_out_ct.bias.fill_(10.)
                block.afa_out_pet.bias.fill_(10.)
                block.text_gate[-1].bias.fill_(10.)
        for y, c, p in zip(self.call(), self.ct, self.pet):
            self.assertTrue(torch.equal(y, c + p))

    def test_uninitialized_text_cannot_be_enabled_later(self):
        model = StateTextAFAFusion(self.channels, use_text=False)
        model.use_text = True
        with self.assertRaises(RuntimeError):
            model(self.ct, self.pet, 1, pet_valid=True)

    def test_optional_local_text_export(self):
        """Real HF loading/tokenization APIs, tiny RANDOM fixtures, no downloaded weights."""
        import importlib.util
        import tempfile
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('optional transformers not installed; pretrained export not exercised')
        from transformers import (PreTrainedTokenizerFast, CLIPTextConfig,
                                  CLIPTextModel, BertConfig, BertModel)
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        vocab = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
                 'A': 4, 'PET': 5, 'image': 6, 'showing': 7, 'bright': 8,
                 'tumor': 9, 'regions': 10, 'in': 11, 'the': 12, 'lungs': 13, '.': 14}
        base = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        base.pre_tokenizer = pre_tokenizers.Whitespace()
        base.post_processor = processors.TemplateProcessing(
            single='[BOS] $A [EOS]', special_tokens=[('[BOS]', 2), ('[EOS]', 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token='[UNK]',
                                            pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
        for backend in ('clip', 'biomedbert'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                tokenizer.save_pretrained(folder)
                if backend == 'clip':
                    config = CLIPTextConfig(vocab_size=len(vocab), hidden_size=12,
                                            intermediate_size=24, num_hidden_layers=1,
                                            num_attention_heads=2, max_position_embeddings=32,
                                            bos_token_id=2, eos_token_id=3, pad_token_id=0)
                    model = CLIPTextModel(config)
                else:
                    config = BertConfig(vocab_size=len(vocab), hidden_size=12,
                                        intermediate_size=24, num_hidden_layers=1,
                                        num_attention_heads=2, max_position_embeddings=32,
                                        pad_token_id=0)
                    model = BertModel(config)
                model.eval().save_pretrained(folder)
                vector, metadata = encode_fixed_text(folder, backend=backend)
                self.assertEqual(vector.shape, (1, 12))
                self.assertFalse(vector.requires_grad)
                self.assertTrue(torch.isfinite(vector).all())
                inputs = tokenizer(DEFAULT_PROMPT, padding='max_length', max_length=30, return_tensors='pt')
                if backend == 'clip':
                    inputs.pop('token_type_ids', None)
                with torch.no_grad():
                    output = model(**inputs)
                    if backend == 'clip':
                        expected = output.pooler_output
                    else:
                        mask = inputs['attention_mask'].unsqueeze(-1)
                        expected = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
                torch.testing.assert_close(vector, expected)
                cache = Path(folder) / 'vector.pt'
                save_text_cache(cache, vector, metadata)
                net = StateTextAFAFusion.from_text_cache(cache, channels=self.channels)
                sum(x.sum() for x in net(self.ct, self.pet, 0, pet_valid=True)).backward()
                self.assertTrue(all(b.text_proj[0].weight.grad is not None for b in net.scales))
                with self.assertRaises(ValueError):
                    encode_fixed_text(folder, backend=backend, max_length=3)


def smoke_test() -> None:
    """Default channel widths and 512x512-input feature sizes; synthetic embedding."""
    torch.manual_seed(7)
    net = StateTextAFAFusion(text_feature=torch.randn(1, 512))
    ct = [torch.randn(1, c, size, size, requires_grad=True)
          for c, size in zip(DEFAULT_CHANNELS, (128, 64, 32, 16))]
    pet = [torch.randn_like(x, requires_grad=True) for x in ct]
    out = net(ct, pet, 0, pet_valid=True)
    sum(x.square().mean() for x in out).backward()
    assert all(torch.isfinite(x).all() for x in out)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in pet)
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'CPU forward/backward PASS; trainable parameters={count:,}')
    print('Output shapes:', [tuple(x.shape) for x in out])
    print('Text: synthetic 512-D fixture, NOT pretrained checkpoint validation.')


def baseline_contract_test(repo_root: str | Path) -> None:
    """Run original feature-path code; no backbone mocks presented as full training.

    Extract the original decoder classes/assembly method with AST to avoid importing
    timm and unrelated image backbones. No baseline source is modified.
    """
    import ast
    import importlib.util
    import sys
    from unittest.mock import patch

    root = Path(repo_root).resolve() / 'models'

    def load_file(filename: str):
        name = '_fusion_contract_' + Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses needs its defining module
        spec.loader.exec_module(module)
        return module

    namespace = {'torch': torch, 'nn': nn, 'F': F}
    for filename, class_name in (('build_mdt_seg.py', 'ConvBNAct'),
                                 ('baseline_blocks.py', 'UNetStyleDecoder')):
        source = ast.parse((root / filename).read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(root / filename), 'exec'), namespace)
    source = ast.parse((root / 'dual_shared_add_baseline.py').read_text(encoding='utf-8'))
    baseline = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DualSharedAddPETCTBaseline')
    method = next(n for n in baseline.body if isinstance(n, ast.FunctionDef) and n.name == '_assemble_mixed_pet_fusion')
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<original-mixed-assembly>', 'exec'), namespace)
    assemble = namespace['_assemble_mixed_pet_fusion']
    Module1 = load_file('paired_semantic_prototype_imputation.py').PairedSemanticPrototypeImputation
    Affine = load_file('ct_conditioned_pet_affine.py').CTConditionedPETAffine
    torch.manual_seed(19)
    channels, sizes = (8, 12, 16, 20), (16, 8, 4, 2)
    bank = Module1(channels, num_clusters=2, cluster_max_iter=3)
    affine = Affine(channels)
    fusion = StateTextAFAFusion(channels, text_feature=torch.randn(1, 12))
    decoder = namespace['UNetStyleDecoder'](channels, (32, 24, 16, 8), use_deep_supervision=True)
    bank.train()
    for _ in range(3):
        c = [torch.randn(4, ch, s, s) for ch, s in zip(channels, sizes)]
        p = [torch.randn_like(x) for x in c]
        mask = torch.zeros(4, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1
        bank.collect_candidates(c, p, mask)
    bank.finalize_epoch(epoch=1)
    assert bank.bank_ready
    before = {name: value.clone() for name, value in bank.named_buffers()}
    ct = [torch.randn(2, ch, s, s, requires_grad=True) for ch, s in zip(channels, sizes)]
    real = [torch.randn_like(x, requires_grad=True) for x in ct]
    state = torch.tensor([1, 0])
    missing = state.eq(0)
    # No memory update or privileged PET reference is allowed during retrieval/fusion.
    with patch.object(bank, 'collect_candidates', side_effect=AssertionError('unexpected collect')), \
         patch.object(bank, 'finalize_epoch', side_effect=AssertionError('unexpected finalize')):
        prior, _ = bank.retrieve_pet_prior([x[missing] for x in ct])
        comp, _, _ = affine([x[missing] for x in ct], prior)
        scattered = [torch.zeros_like(x).index_copy(0, torch.tensor([1]), y) for x, y in zip(ct, comp)]
        pet = assemble(None, real, scattered, missing, ct[0])
        features = fusion(ct, pet, state, pet_valid=True)
        changed_real = [torch.cat((x[:1], x[1:] + 999), dim=0) for x in real]
        unchanged_pet = assemble(None, changed_real, scattered, missing, ct[0])
        check = fusion(ct, unchanged_pet, state, pet_valid=True)
        for a, b in zip(features, check):
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        result = decoder(features, target_size=(64, 64))
        assert result['logits'].shape == (2, 1, 64, 64)
        assert len(result['aux_logits']) == 3
        loss = result['logits'].square().mean()
        loss = loss + sum(x.square().mean() for x in result['aux_logits'])
        loss.backward()
    assert all(torch.equal(before[n], b) for n, b in bank.named_buffers())
    assert all(not b.requires_grad and b.grad is None for b in bank.buffers())
    assert all(x.grad is not None and x.grad[0].abs().sum() > 0 and x.grad[1].abs().sum() == 0 for x in real)
    for part in (bank, affine, fusion):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in part.parameters())
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in ct)
    fresh = Module1(channels, num_clusters=2)
    zeros, _ = fresh.retrieve_pet_prior(ct)
    cold = fusion(ct, zeros, 0, pet_valid=False)
    assert all(torch.equal(a, b) for a, b in zip(cold, ct))
    print('PASS: original PSPI -> CT affine -> original mixed assembly -> module2 -> original decoder')
    print('PASS: upstream gradients; Missing real-PET isolation; frozen/unmodified bank; cold-start CT')
    print('Feature-path test only: synthetic features/text; no image backbones or full training runner.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--self-test', action='store_true')
    action.add_argument('--smoke-test', action='store_true')
    action.add_argument('--baseline-contract', type=Path)
    action.add_argument('--export-text-cache', type=Path)
    parser.add_argument('--backend', choices=('clip', 'biomedbert'), default='clip')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--max-length', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.self_test or args.smoke_test or args.baseline_contract:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    elif args.smoke_test:
        smoke_test()
    elif args.baseline_contract:
        baseline_contract_test(args.baseline_contract)
    else:
        if args.model_path is None:
            parser.error('--export-text-cache requires --model-path')
        vector, metadata = encode_fixed_text(args.model_path, backend=args.backend,
                                             prompt=args.prompt, max_length=args.max_length,
                                             device=args.device)
        save_text_cache(args.export_text_cache, vector, metadata)
        print(f'Saved {tuple(vector.shape)} sentence embedding to {args.export_text_cache}')
        print(metadata)


if __name__ == '__main__':
    main()
