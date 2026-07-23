"""
APSF: Asymmetric PET-Surrogate Fusion
======================================

面向 2D PET-CT 肿瘤分割的独立四尺度特征融合模块。

设计边界
--------
1. Full 路径只接收 CT 特征与真实 PET 特征。
2. Missing 路径只接收 CT 特征与 MPPC 已生成的补偿特征。
3. APSF 不读取 MPPC 的 memory、置信度、检索结果或其他内部状态。
4. 不需要分割标签，不引入辅助损失，也没有需要搜索的标量超参数。
5. Full/Missing 使用各自的前端整形，之后进入同一个共享融合核。
6. 三个输出投影采用零初始化，因此初始化时严格退化为原始 SUM：

       Full:    fused = CT + PET
       Missing: fused = CT + PET_proxy

推荐接法
--------
    apsf = APSF(channels=(64, 128, 320, 512))

    if pet_available:
        pet_feats = pet_encoder(pet)
        pet_feats = align_pet_channels(pet_feats)
        fused_feats = apsf.forward_full(ct_feats, pet_feats)
    else:
        proxy_feats = mppc(ct_feats)
        fused_feats = apsf.forward_missing(ct_feats, proxy_feats)

    prediction = shared_decoder(fused_feats)

输入的每个尺度均须为 [B, C_s, H_s, W_s]，且同尺度 CT 与辅助特征
必须已经完成通道和空间对齐。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FeaturePyramid = Sequence[Tensor]
DebugInfo = Dict[str, Tensor]


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """
    Sparsemax 激活。

    与 Softmax 不同，Sparsemax 能产生精确的零，从而得到无需 Top-K、
    比例或阈值超参数的稀疏支持集。

    为提升 AMP 下的数值稳定性，排序与阈值计算始终在 float32 中完成，
    最终结果再转换回输入 dtype。
    """
    if dim < 0:
        dim += logits.ndim
    if dim != logits.ndim - 1:
        logits = logits.transpose(dim, -1)
        restore_dim = dim
    else:
        restore_dim = None

    original_dtype = logits.dtype
    z = logits.float()
    z = z - z.max(dim=-1, keepdim=True).values

    z_sorted = torch.sort(z, dim=-1, descending=True).values
    z_cumsum = z_sorted.cumsum(dim=-1)

    rank_shape = [1] * z.ndim
    rank_shape[-1] = z.shape[-1]
    ranks = torch.arange(
        1,
        z.shape[-1] + 1,
        device=z.device,
        dtype=z.dtype,
    ).view(rank_shape)

    support = 1.0 + ranks * z_sorted > z_cumsum
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)

    tau_sum = z_cumsum.gather(
        dim=-1,
        index=support_size - 1,
    )
    tau = (tau_sum - 1.0) / support_size.to(z.dtype)
    probabilities = torch.clamp(z - tau, min=0.0)
    probabilities = probabilities.to(original_dtype)

    if restore_dim is not None:
        probabilities = probabilities.transpose(restore_dim, -1)
    return probabilities


def unit_sparsemax(logits: Tensor) -> Tensor:
    """
    将 Sparsemax 输出按每个样本的最大值归一化至 [0, 1]。

    Sparsemax 的零仍保持为精确零；正支持集中的最大响应变为 1。
    eps 仅用于数值稳定，不是实验超参数。
    """
    probabilities = sparsemax(logits, dim=-1)
    eps = torch.finfo(probabilities.dtype).eps
    maximum = probabilities.amax(dim=-1, keepdim=True)
    return probabilities / maximum.clamp_min(eps)


def standardize_scores(scores: Tensor) -> Tensor:
    """按样本在空间 token 维度做无可调参数的标准化。"""
    scores_fp32 = scores.float()
    mean = scores_fp32.mean(dim=-1, keepdim=True)
    variance = (scores_fp32 - mean).square().mean(dim=-1, keepdim=True)
    eps = torch.finfo(scores_fp32.dtype).eps
    normalized = (scores_fp32 - mean) * torch.rsqrt(variance + eps)
    return normalized.to(scores.dtype)


def map_to_tokens(feature: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, H*W, C]."""
    return feature.flatten(2).transpose(1, 2).contiguous()


def tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
    """[B, H*W, C] -> [B, C, H, W]."""
    batch, token_count, channels = tokens.shape
    expected = height * width
    if token_count != expected:
        raise ValueError(
            f"Token count mismatch: got {token_count}, expected {expected}."
        )
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def zero_init_linear(layer: nn.Linear) -> None:
    """将线性层初始化为严格的零映射。"""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class APSFStage(nn.Module):
    """
    单尺度 APSF。

    前端：
      - Full: PET 自身显著性与 CT 任务相关性取稀疏并集。
      - Missing: CT 任务相关性与 CT-proxy 局部一致性形成稀疏共识。

    后端：
      - 对 Sparsemax 非零位置执行真正的 gather。
      - 使用轻量交叉通道调制。
      - CT 作为 Query，辅助特征作为 Key/Value。
      - 将交互残差 scatter 回原空间并叠加到 SUM baseline。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.channels = int(channels)
        self.attention_scale = self.channels**-0.5

        # CT 条件任务查询。每个尺度仅一个向量。
        self.task_token = nn.Parameter(torch.empty(1, self.channels))

        # 三类输入的独立归一化。
        self.ct_norm = nn.LayerNorm(self.channels)
        self.real_pet_norm = nn.LayerNorm(self.channels)
        self.proxy_norm = nn.LayerNorm(self.channels)
        self.query_norm = nn.LayerNorm(self.channels)

        # 状态专用前端。二者不共享参数，且均为零初始化残差。
        self.real_refine = nn.Linear(self.channels, self.channels)
        self.proxy_refine = nn.Linear(self.channels, self.channels)

        # Cross-SE 式跨源通道调制。只作用于新交互残差。
        self.ct_from_aux_scale = nn.Parameter(torch.ones(self.channels))
        self.ct_from_aux_bias = nn.Parameter(torch.zeros(self.channels))
        self.aux_from_ct_scale = nn.Parameter(torch.ones(self.channels))
        self.aux_from_ct_bias = nn.Parameter(torch.zeros(self.channels))

        # 单头、单方向 CT -> auxiliary 交互。
        self.q_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.k_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.v_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.task_token, std=0.02)

        for norm in (
            self.ct_norm,
            self.real_pet_norm,
            self.proxy_norm,
            self.query_norm,
        ):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(projection.weight)

        # 保证 APSF 初始状态严格等价于原始逐尺度 SUM。
        zero_init_linear(self.real_refine)
        zero_init_linear(self.proxy_refine)
        zero_init_linear(self.out_proj)

    @staticmethod
    def _validate_pair(ct: Tensor, auxiliary: Tensor) -> None:
        if ct.ndim != 4 or auxiliary.ndim != 4:
            raise ValueError(
                "APSFStage expects 4D feature maps [B, C, H, W], "
                f"got CT {tuple(ct.shape)} and auxiliary {tuple(auxiliary.shape)}."
            )
        if ct.shape != auxiliary.shape:
            raise ValueError(
                "CT and auxiliary features must be aligned before APSF. "
                f"Got CT {tuple(ct.shape)} and auxiliary {tuple(auxiliary.shape)}."
            )

    def _ct_conditioned_query(
        self,
        ct_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        使用任务 token 从 CT 中读取当前病例的结构语义。

        Returns:
            query: [B, C]
            normalized_ct: [B, N, C]
        """
        normalized_ct = self.ct_norm(ct_tokens)
        batch = normalized_ct.shape[0]
        task = self.task_token.expand(batch, -1)

        attention_logits = torch.einsum(
            "bd,bnd->bn",
            task,
            normalized_ct,
        ) * self.attention_scale
        attention = attention_logits.softmax(dim=-1)
        ct_context = torch.einsum(
            "bn,bnd->bd",
            attention,
            normalized_ct,
        )
        query = self.query_norm(task + ct_context)
        return query, normalized_ct

    def _full_frontend(
        self,
        ct_tokens: Tensor,
        pet_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        真实 PET 前端：保留优先的稀疏并集。

        PET 自身认为重要，或 CT 条件任务查询认为重要的位置，均可进入
        后续共享融合核。
        """
        query, _ = self._ct_conditioned_query(ct_tokens)
        normalized_pet = self.real_pet_norm(pet_tokens)

        pet_global = normalized_pet.mean(dim=1)
        self_scores = torch.einsum(
            "bd,bnd->bn",
            pet_global,
            normalized_pet,
        ) * self.attention_scale
        task_scores = torch.einsum(
            "bd,bnd->bn",
            query,
            normalized_pet,
        ) * self.attention_scale

        selection = torch.maximum(
            unit_sparsemax(self_scores),
            unit_sparsemax(task_scores),
        )

        signed_selection = 2.0 * selection.unsqueeze(-1) - 1.0
        refined_pet = pet_tokens + self.real_refine(
            signed_selection * normalized_pet
        )
        return refined_pet, selection

    def _missing_frontend(
        self,
        ct_tokens: Tensor,
        proxy_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        MPPC 补偿前端：可靠优先的 CT 共识。

        补偿特征必须同时受到 CT 任务位置和 CT-proxy 局部一致性的约束，
        不使用 MPPC 的任何内部检索信息。
        """
        query, normalized_ct = self._ct_conditioned_query(ct_tokens)
        normalized_proxy = self.proxy_norm(proxy_tokens)

        ct_task_scores = torch.einsum(
            "bd,bnd->bn",
            query,
            normalized_ct,
        ) * self.attention_scale
        local_agreement_scores = (
            normalized_ct * normalized_proxy
        ).sum(dim=-1) * self.attention_scale

        consensus_scores = 0.5 * (
            standardize_scores(ct_task_scores)
            + standardize_scores(local_agreement_scores)
        )
        selection = unit_sparsemax(consensus_scores)

        signed_selection = 2.0 * selection.unsqueeze(-1) - 1.0
        refined_proxy = proxy_tokens + self.proxy_refine(
            signed_selection * normalized_proxy
        )
        return refined_proxy, selection

    @staticmethod
    def _gather_sparse_support(
        tokens: Tensor,
        selection: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        真正 gather Sparsemax 的非零支持集。

        不同样本的支持集长度可能不同，因此输出会补齐到当前 batch 的
        最大长度，并额外返回 valid mask。

        Returns:
            selected_tokens: [B, K_max, C]
            indices: [B, K_max]
            valid: [B, K_max]
        """
        batch, token_count, channels = tokens.shape
        support = selection > 0
        support_count = support.sum(dim=-1)

        if bool((support_count == 0).any()):
            # 理论上 Sparsemax 至少保留一个元素；该检查用于防御异常 dtype。
            fallback = selection.argmax(dim=-1, keepdim=True)
            support = support.scatter(dim=-1, index=fallback, value=True)
            support_count = support.sum(dim=-1)

        max_selected = int(support_count.max().item())
        flat_indices = torch.arange(
            token_count,
            device=tokens.device,
        ).unsqueeze(0).expand(batch, -1)

        # 未选位置被放到末尾；已选位置保持原始空间顺序。
        ordered_indices = torch.where(
            support,
            flat_indices,
            torch.full_like(flat_indices, token_count),
        ).sort(dim=-1).values[:, :max_selected]

        valid = ordered_indices < token_count
        safe_indices = ordered_indices.clamp(max=token_count - 1)
        gather_index = safe_indices.unsqueeze(-1).expand(
            -1,
            -1,
            channels,
        )
        selected_tokens = tokens.gather(dim=1, index=gather_index)
        return selected_tokens, safe_indices, valid

    @staticmethod
    def _masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
        weights = valid.unsqueeze(-1).to(tokens.dtype)
        numerator = (tokens * weights).sum(dim=1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def _cross_channel_modulation(
        self,
        ct_selected: Tensor,
        aux_selected: Tensor,
        valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """由另一来源的全局描述调制当前来源的通道。"""
        ct_global = self._masked_mean(ct_selected, valid)
        aux_global = self._masked_mean(aux_selected, valid)

        ct_gate = torch.sigmoid(
            aux_global * self.ct_from_aux_scale + self.ct_from_aux_bias
        )
        aux_gate = torch.sigmoid(
            ct_global * self.aux_from_ct_scale + self.aux_from_ct_bias
        )

        ct_modulated = ct_selected * ct_gate.unsqueeze(1)
        aux_modulated = aux_selected * aux_gate.unsqueeze(1)
        return ct_modulated, aux_modulated

    @staticmethod
    def _scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """
        单头注意力，优先调用 PyTorch 的内存优化实现。

        valid 仅屏蔽 padding key；padding query 的输出会在调用后归零。
        """
        key_mask = valid[:, None, None, :]

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query.unsqueeze(1),
                key.unsqueeze(1),
                value.unsqueeze(1),
                attn_mask=key_mask,
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(1)
        else:
            scale = query.shape[-1] ** -0.5
            logits = torch.matmul(query, key.transpose(-2, -1)) * scale
            logits = logits.masked_fill(
                ~valid[:, None, :],
                torch.finfo(logits.dtype).min,
            )
            attention = logits.softmax(dim=-1)
            attended = torch.matmul(attention, value)

        return attended * valid.unsqueeze(-1).to(attended.dtype)

    @staticmethod
    def _scatter_selected(
        selected_tokens: Tensor,
        indices: Tensor,
        valid: Tensor,
        token_count: int,
    ) -> Tensor:
        """将变长选择结果 scatter 回 [B, N, C]。"""
        batch, _, channels = selected_tokens.shape
        source = selected_tokens * valid.unsqueeze(-1).to(selected_tokens.dtype)
        scatter_index = indices.unsqueeze(-1).expand(-1, -1, channels)
        output = selected_tokens.new_zeros(batch, token_count, channels)
        return output.scatter_add(dim=1, index=scatter_index, src=source)

    def _shared_fusion(
        self,
        ct_tokens: Tensor,
        auxiliary_tokens: Tensor,
        selection: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full/Missing 共用的 CT 锚定选择性交互核。

        原始 CT 与辅助特征不参与可靠性竞争；选择与通道调制只控制
        新产生的交互残差。
        """
        ct_selected, indices, valid = self._gather_sparse_support(
            ct_tokens,
            selection,
        )
        aux_selected, _, _ = self._gather_sparse_support(
            auxiliary_tokens,
            selection,
        )

        ct_modulated, aux_modulated = self._cross_channel_modulation(
            ct_selected,
            aux_selected,
            valid,
        )

        query = self.q_proj(ct_modulated)
        key = self.k_proj(aux_modulated)
        value = self.v_proj(aux_modulated)

        attended = self._scaled_dot_product_attention(
            query,
            key,
            value,
            valid,
        )
        selected_residual = self.out_proj(attended)
        residual = self._scatter_selected(
            selected_residual,
            indices,
            valid,
            token_count=ct_tokens.shape[1],
        )

        fused_tokens = ct_tokens + auxiliary_tokens + residual
        support_count = valid.sum(dim=-1)
        return fused_tokens, support_count

    def forward_full(
        self,
        ct: Tensor,
        pet: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, DebugInfo]]:
        """融合 CT 与真实 PET 特征。"""
        self._validate_pair(ct, pet)
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Stage was built for {self.channels} channels, "
                f"but received {ct.shape[1]}."
            )

        _, _, height, width = ct.shape
        ct_tokens = map_to_tokens(ct)
        pet_tokens = map_to_tokens(pet)

        refined_pet, selection = self._full_frontend(ct_tokens, pet_tokens)
        fused_tokens, support_count = self._shared_fusion(
            ct_tokens,
            refined_pet,
            selection,
        )
        fused = tokens_to_map(fused_tokens, height, width)

        if not return_debug:
            return fused
        debug = {
            "selection_map": selection.reshape(
                selection.shape[0],
                1,
                height,
                width,
            ),
            "support_count": support_count,
        }
        return fused, debug

    def forward_missing(
        self,
        ct: Tensor,
        proxy: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, DebugInfo]]:
        """融合 CT 与 MPPC 已输出的 PET 补偿特征。"""
        self._validate_pair(ct, proxy)
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Stage was built for {self.channels} channels, "
                f"but received {ct.shape[1]}."
            )

        _, _, height, width = ct.shape
        ct_tokens = map_to_tokens(ct)
        proxy_tokens = map_to_tokens(proxy)

        refined_proxy, selection = self._missing_frontend(
            ct_tokens,
            proxy_tokens,
        )
        fused_tokens, support_count = self._shared_fusion(
            ct_tokens,
            refined_proxy,
            selection,
        )
        fused = tokens_to_map(fused_tokens, height, width)

        if not return_debug:
            return fused
        debug = {
            "selection_map": selection.reshape(
                selection.shape[0],
                1,
                height,
                width,
            ),
            "support_count": support_count,
        }
        return fused, debug

    def extra_repr(self) -> str:
        return f"channels={self.channels}, selection=sparsemax, heads=1"


class APSF(nn.Module):
    """
    四尺度 APSF 封装。

    默认通道与当前 PET-CT baseline 的对齐后通道一致：
        [64, 128, 320, 512]

    channels 只是输入接口的结构描述，不是实验中需要搜索的标量超参数。
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one stage.")

        self.channels = tuple(int(value) for value in channels)
        self.stages = nn.ModuleList(
            [APSFStage(value) for value in self.channels]
        )

    def _validate_pyramids(
        self,
        ct_features: FeaturePyramid,
        auxiliary_features: FeaturePyramid,
    ) -> None:
        if len(ct_features) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} CT stages, "
                f"got {len(ct_features)}."
            )
        if len(auxiliary_features) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} auxiliary stages, "
                f"got {len(auxiliary_features)}."
            )

    def forward_full(
        self,
        ct_features: FeaturePyramid,
        pet_features: FeaturePyramid,
        return_debug: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[DebugInfo]]]:
        """
        Full 路径。

        Args:
            ct_features: 四尺度、已对齐的 CT 特征。
            pet_features: 四尺度、已对齐的真实 PET 特征。
            return_debug: 是否额外返回选择图和每样本支持集大小。
        """
        self._validate_pyramids(ct_features, pet_features)

        fused_features: List[Tensor] = []
        debug_info: List[DebugInfo] = []
        for stage, ct, pet in zip(self.stages, ct_features, pet_features):
            result = stage.forward_full(ct, pet, return_debug=return_debug)
            if return_debug:
                fused, debug = result
                fused_features.append(fused)
                debug_info.append(debug)
            else:
                fused_features.append(result)

        if return_debug:
            return fused_features, debug_info
        return fused_features

    def forward_missing(
        self,
        ct_features: FeaturePyramid,
        proxy_features: FeaturePyramid,
        return_debug: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[DebugInfo]]]:
        """
        Missing 路径。

        proxy_features 必须由 APSF 外部的 MPPC 产生；APSF 不持有或调用
        MPPC，也不读取其 memory、检索置信度或其他内部变量。
        """
        self._validate_pyramids(ct_features, proxy_features)

        fused_features: List[Tensor] = []
        debug_info: List[DebugInfo] = []
        for stage, ct, proxy in zip(
            self.stages,
            ct_features,
            proxy_features,
        ):
            result = stage.forward_missing(
                ct,
                proxy,
                return_debug=return_debug,
            )
            if return_debug:
                fused, debug = result
                fused_features.append(fused)
                debug_info.append(debug)
            else:
                fused_features.append(result)

        if return_debug:
            return fused_features, debug_info
        return fused_features

    def forward(
        self,
        ct_features: FeaturePyramid,
        auxiliary_features: FeaturePyramid,
        mode: str,
        return_debug: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[DebugInfo]]]:
        """
        可选的统一调用入口。

        推荐模型代码直接调用 forward_full / forward_missing，以使两条
        数据路径在代码层面保持明确。mode 仅用于函数分派，不会被编码为
        availability embedding，也不会进入特征计算。
        """
        normalized_mode = mode.lower().strip()
        if normalized_mode == "full":
            return self.forward_full(
                ct_features,
                auxiliary_features,
                return_debug=return_debug,
            )
        if normalized_mode == "missing":
            return self.forward_missing(
                ct_features,
                auxiliary_features,
                return_debug=return_debug,
            )
        raise ValueError(
            f"mode must be 'full' or 'missing', got {mode!r}."
        )

    def trainable_parameter_count(self) -> int:
        """返回 APSF 的可训练参数量。"""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def _build_random_pyramid(
    batch_size: int,
    channels: Sequence[int],
    spatial_sizes: Sequence[Tuple[int, int]],
    requires_grad: bool,
) -> List[Tensor]:
    return [
        torch.randn(
            batch_size,
            channel,
            height,
            width,
            requires_grad=requires_grad,
        )
        for channel, (height, width) in zip(channels, spatial_sizes)
    ]


def self_test() -> None:
    """
    最小独立自测：
      1. Full/Missing 四尺度形状正确；
      2. 零初始化时与逐尺度 SUM 严格等价；
      3. debug 选择图尺寸正确且支持集非空；
      4. 反向传播能够运行并更新零初始化输出层。
    """
    torch.manual_seed(7)

    test_channels = (16, 32, 64, 128)
    spatial_sizes = ((16, 16), (8, 8), (4, 4), (2, 2))
    batch_size = 2

    module = APSF(channels=test_channels)
    ct_features = _build_random_pyramid(
        batch_size,
        test_channels,
        spatial_sizes,
        requires_grad=True,
    )
    pet_features = _build_random_pyramid(
        batch_size,
        test_channels,
        spatial_sizes,
        requires_grad=True,
    )
    proxy_features = _build_random_pyramid(
        batch_size,
        test_channels,
        spatial_sizes,
        requires_grad=True,
    )

    full_features, full_debug = module.forward_full(
        ct_features,
        pet_features,
        return_debug=True,
    )
    missing_features, missing_debug = module.forward_missing(
        ct_features,
        proxy_features,
        return_debug=True,
    )

    for stage_index, (full, missing) in enumerate(
        zip(full_features, missing_features)
    ):
        torch.testing.assert_close(
            full,
            ct_features[stage_index] + pet_features[stage_index],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            missing,
            ct_features[stage_index] + proxy_features[stage_index],
            rtol=0.0,
            atol=0.0,
        )

        height, width = spatial_sizes[stage_index]
        expected_selection_shape = (batch_size, 1, height, width)
        if tuple(full_debug[stage_index]["selection_map"].shape) != (
            expected_selection_shape
        ):
            raise AssertionError("Unexpected Full selection-map shape.")
        if tuple(missing_debug[stage_index]["selection_map"].shape) != (
            expected_selection_shape
        ):
            raise AssertionError("Unexpected Missing selection-map shape.")
        if not bool((full_debug[stage_index]["support_count"] > 0).all()):
            raise AssertionError("Full sparse support must be non-empty.")
        if not bool((missing_debug[stage_index]["support_count"] > 0).all()):
            raise AssertionError("Missing sparse support must be non-empty.")

    loss = sum(
        feature.square().mean()
        for feature in full_features + missing_features
    )
    loss.backward()

    for stage in module.stages:
        if stage.real_refine.weight.grad is None:
            raise AssertionError("real_refine did not receive gradients.")
        if stage.proxy_refine.weight.grad is None:
            raise AssertionError("proxy_refine did not receive gradients.")
        if stage.out_proj.weight.grad is None:
            raise AssertionError("out_proj did not receive gradients.")

    production_module = APSF()
    print("APSF self-test passed.")
    print(
        "Default trainable parameters: "
        f"{production_module.trainable_parameter_count():,}"
    )


if __name__ == "__main__":
    self_test()