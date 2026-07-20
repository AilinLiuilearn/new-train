"""MTPI: Multi-scale Task-oriented PET Prototype Imputation.

用于 2D PET-CT 肺部肿瘤分割的四尺度缺失 PET 特征补偿模块。

设计边界
--------
1. 本模块应放在 CT/PET 编码特征完成原有通道对齐之后、逐尺度 SUM 融合之前。
2. Full batch：返回真实 PET 特征，并在训练时用标注无梯度更新 CT/PET 语义原型。
3. Missing batch：``pet_features`` 必须为 ``None``，只用 CT 特征生成 PET 补偿特征。
4. 本模块不包含 PET Encoder、融合模块、Decoder、教师模型或 PET 重建分支。
5. 参考原型是 buffer，不参加反向传播；可训练参数只在 Missing 路径中使用。

默认四尺度通道数 ``(64, 128, 320, 512)`` 对应现有通道对齐后的 MiT-B1
PET 特征维度。若 baseline 的对齐层输出不同，只需修改 ``channels``。

典型接入方式
------------
::

    # 一个 batch 只能选择一个状态；推荐训练顺序从 full batch 开始。
    ct_features = align_ct(ct_encoder(ct_image))

    if modality_state == "full":
        pet_features = align_pet(pet_encoder(pet_image))
        pet_for_fusion = mtpi(
            ct_features,
            pet_features=pet_features,
            modality_state="full",
            target=mask,              # 训练时用于无梯度更新原型
        )
    else:
        # 不加载 PET，不创建零 PET，不运行 PET Encoder。
        pet_for_fusion = mtpi(
            ct_features,
            pet_features=None,
            modality_state="missing",
        )

    fused = [ct + pet for ct, pet in zip(ct_features, pet_for_fusion)]
    logits = shared_decoder(fused)
    seg_loss = segmentation_loss(logits, mask)

    if modality_state == "missing":
        loss, loss_items = mtpi.training_loss(
            seg_loss,
            pet_for_fusion,
            mask,
            reference_weight=0.1,
        )
    else:
        loss = seg_loss

论文思想边界（概念适配，不复制原论文代码）
--------------------------------------
- API：类别语义原型、中心过滤聚合、原型检索与实例自适应仿射调制。
- CalMRL：只依据已观测模态完成表征级补偿。
- TMDC：由分割任务驱动，只补偿任务相关信息而非完整恢复 PET。

新增研究超参数只有两个：
- ``filter_ratio=0.05``：Full batch 原型更新时丢弃的离群病例比例；
- ``reference_weight=0.1``：Missing batch 语义参考辅助损失权重。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["MTPICompensation", "MTPI"]


FeatureSequence = Sequence[Tensor]
ModalityState = Union[str, int, bool]


def _canonical_state(state: ModalityState) -> str:
    """将 batch 级状态统一成 ``full`` 或 ``missing``。"""
    if isinstance(state, str):
        value = state.strip().lower()
        if value in {"full", "complete", "present", "1"}:
            return "full"
        if value in {"missing", "absent", "0"}:
            return "missing"
    elif isinstance(state, bool):
        return "full" if state else "missing"
    elif isinstance(state, int) and state in (0, 1):
        return "full" if state == 1 else "missing"
    raise ValueError(
        "modality_state 必须是 'full'/'missing'，或对应的 batch 级标识 1/0。"
    )


def _prepare_binary_target(target: Tensor, batch_size: int) -> Tensor:
    """将二值标签统一成 ``[B, H, W]`` 的 long tensor。"""
    if target.ndim == 4:
        if target.shape[1] == 1:
            target = target[:, 0]
        elif target.shape[1] == 2:
            target = target.argmax(dim=1)
        else:
            raise ValueError(
                "四维 target 只能是 [B,1,H,W] 标签或 [B,2,H,W] 二类 one-hot/logit。"
            )
    if target.ndim != 3:
        raise ValueError(f"target 应为 [B,H,W]，实际形状为 {tuple(target.shape)}。")
    if target.shape[0] != batch_size:
        raise ValueError(
            f"target batch={target.shape[0]}，但特征 batch={batch_size}。"
        )

    target = target.long()
    unique_values = torch.unique(target.detach())
    if not bool(torch.all((unique_values == 0) | (unique_values == 1))):
        raise ValueError(
            "MTPI 当前只支持背景/肿瘤二值标签（0/1）；请先处理 ignore label。"
        )
    return target


def _resize_target(target: Tensor, spatial_size: Tuple[int, int]) -> Tensor:
    """使用最近邻插值把标签缩放到某个编码尺度。"""
    return F.interpolate(
        target.unsqueeze(1).float(), size=spatial_size, mode="nearest"
    ).squeeze(1).long()


@torch.no_grad()
def _distributed_concat_rows(rows: Tensor, enabled: bool) -> Tensor:
    """在 DDP 各进程间收集数量不等的二维向量，保证原型 buffer 一致。"""
    if not enabled or not dist.is_available() or not dist.is_initialized():
        return rows

    if rows.ndim != 2:
        raise ValueError("DDP 原型同步输入必须为 [N,C]。")

    world_size = dist.get_world_size()
    local_size = torch.tensor([rows.shape[0]], device=rows.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if max_size == 0:
        return rows.new_empty((0, rows.shape[1]))

    padded = rows.new_zeros((max_size, rows.shape[1]))
    if rows.shape[0] > 0:
        padded[: rows.shape[0]].copy_(rows)

    gathered_rows = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_rows, padded)
    valid_rows = [part[:size] for part, size in zip(gathered_rows, sizes) if size > 0]
    return torch.cat(valid_rows, dim=0)


class _ScaleReferenceBank(nn.Module):
    """单尺度背景/肿瘤 CT-PET 配对原型；仅含 buffers，无可训练参数。"""

    NUM_CLASSES = 2

    def __init__(
        self,
        channels: int,
        filter_ratio: float,
        sync_distributed: bool,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正整数。")
        if not 0.0 <= filter_ratio < 1.0:
            raise ValueError("filter_ratio 必须满足 0 <= filter_ratio < 1。")

        self.channels = int(channels)
        self.filter_ratio = float(filter_ratio)
        self.sync_distributed = bool(sync_distributed)

        self.register_buffer(
            "ct_prototypes", torch.zeros(self.NUM_CLASSES, self.channels)
        )
        self.register_buffer(
            "pet_prototypes", torch.zeros(self.NUM_CLASSES, self.channels)
        )
        # count 统计被中心过滤后接纳的病例级向量数，用累计均值避免 EMA 动量超参数。
        self.register_buffer("counts", torch.zeros(self.NUM_CLASSES, dtype=torch.float64))

    @property
    def ready_mask(self) -> Tensor:
        return self.counts > 0

    @property
    def fully_initialized(self) -> bool:
        return bool(torch.all(self.ready_mask).item())

    @torch.no_grad()
    def reset(self) -> None:
        self.ct_prototypes.zero_()
        self.pet_prototypes.zero_()
        self.counts.zero_()

    @staticmethod
    def _case_level_vectors(
        feature: Tensor,
        scaled_target: Tensor,
        class_index: int,
    ) -> Tensor:
        """对每个含该类别的病例分别做 mask pooling，避免按像素量加权。"""
        vectors: List[Tensor] = []
        for batch_index in range(feature.shape[0]):
            class_mask = scaled_target[batch_index] == class_index
            if bool(class_mask.any()):
                # feature[b, :, H, W] -> [C, N] -> [C]
                vectors.append(feature[batch_index, :, class_mask].mean(dim=1))
        if not vectors:
            return feature.new_empty((0, feature.shape[1]))
        return torch.stack(vectors, dim=0)

    def _centroid_filter(
        self,
        ct_vectors: Tensor,
        pet_vectors: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """按配对 CT-PET 向量到联合中心的余弦偏离过滤离群病例。"""
        if ct_vectors.shape != pet_vectors.shape:
            raise RuntimeError("CT/PET 病例级原型必须一一配对且维度相同。")
        num_cases = ct_vectors.shape[0]
        if num_cases <= 1 or self.filter_ratio == 0.0:
            return ct_vectors, pet_vectors

        joint = torch.cat(
            [
                F.normalize(ct_vectors.float(), dim=1),
                F.normalize(pet_vectors.float(), dim=1),
            ],
            dim=1,
        )
        centroid = F.normalize(joint.mean(dim=0, keepdim=True), dim=1)
        distance = 1.0 - F.cosine_similarity(joint, centroid.expand_as(joint), dim=1)
        keep_count = max(1, int(math.ceil(num_cases * (1.0 - self.filter_ratio))))
        keep_indices = torch.topk(distance, k=keep_count, largest=False).indices
        return ct_vectors[keep_indices], pet_vectors[keep_indices]

    @torch.no_grad()
    def update(self, ct_feature: Tensor, pet_feature: Tensor, target: Tensor) -> None:
        """从一次 Full forward 的配对特征更新本尺度原型。"""
        if ct_feature.shape != pet_feature.shape:
            raise ValueError(
                "原型更新要求通道对齐后的 CT/PET 特征形状一致；"
                f"得到 {tuple(ct_feature.shape)} 与 {tuple(pet_feature.shape)}。"
            )
        if ct_feature.ndim != 4 or ct_feature.shape[1] != self.channels:
            raise ValueError("原型更新特征必须是与本尺度 channels 一致的 [B,C,H,W]。")

        scaled_target = _resize_target(target, ct_feature.shape[-2:])
        for class_index in range(self.NUM_CLASSES):
            ct_vectors = self._case_level_vectors(
                ct_feature.detach(), scaled_target, class_index
            )
            pet_vectors = self._case_level_vectors(
                pet_feature.detach(), scaled_target, class_index
            )

            # 两个集合在本地由同一标签产生，行数和顺序严格一致；DDP 分别同步后仍一致。
            ct_vectors = _distributed_concat_rows(ct_vectors, self.sync_distributed)
            pet_vectors = _distributed_concat_rows(pet_vectors, self.sync_distributed)
            if ct_vectors.shape[0] == 0:
                continue

            ct_vectors, pet_vectors = self._centroid_filter(ct_vectors, pet_vectors)
            new_count = float(ct_vectors.shape[0])
            old_count = float(self.counts[class_index].item())
            total_count = old_count + new_count

            ct_mean = ct_vectors.float().mean(dim=0).to(self.ct_prototypes.dtype)
            pet_mean = pet_vectors.float().mean(dim=0).to(self.pet_prototypes.dtype)
            if old_count == 0.0:
                self.ct_prototypes[class_index].copy_(ct_mean)
                self.pet_prototypes[class_index].copy_(pet_mean)
            else:
                self.ct_prototypes[class_index].mul_(old_count / total_count)
                self.ct_prototypes[class_index].add_(ct_mean, alpha=new_count / total_count)
                self.pet_prototypes[class_index].mul_(old_count / total_count)
                self.pet_prototypes[class_index].add_(pet_mean, alpha=new_count / total_count)
            self.counts[class_index].fill_(total_count)


class _ScalePrototypeImputer(nn.Module):
    """CT 原型查询 + PET 原型检索 + 实例自适应仿射调制。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)

        # 对应 API 中的轻量 DSP 与 APG；不引入隐藏宽度这一额外超参数。
        self.descriptor_projection = nn.Linear(channels, channels)
        self.gamma_projection = nn.Linear(channels, channels)
        self.beta_projection = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 方阵 DSP 以恒等映射开始；仿射头使初始输出等于检索到的 PET 原型特征。
        nn.init.eye_(self.descriptor_projection.weight)
        nn.init.zeros_(self.descriptor_projection.bias)
        nn.init.zeros_(self.gamma_projection.weight)
        nn.init.zeros_(self.gamma_projection.bias)
        nn.init.zeros_(self.beta_projection.weight)
        nn.init.zeros_(self.beta_projection.bias)

    def forward(
        self,
        ct_feature: Tensor,
        ct_prototypes: Tensor,
        pet_prototypes: Tensor,
        ready_mask: Tensor,
    ) -> Tensor:
        if ct_feature.ndim != 4 or ct_feature.shape[1] != self.channels:
            raise ValueError(
                f"补偿器期望 [B,{self.channels},H,W]，实际为 {tuple(ct_feature.shape)}。"
            )
        ready_indices = torch.nonzero(ready_mask, as_tuple=False).flatten()
        if ready_indices.numel() == 0:
            raise RuntimeError(
                "PET 参考原型尚未初始化。请确保训练交替从 Full batch 开始，"
                "并将包含 target 的 Full forward 保存进 checkpoint 后再做 Missing 推理。"
            )

        ct_reference = ct_prototypes.index_select(0, ready_indices).detach()
        pet_reference = pet_prototypes.index_select(0, ready_indices).detach()

        # 每个空间位置依据 CT 与 CT 类别原型的相似度，软检索对应 PET 类别原型。
        normalized_ct = F.normalize(ct_feature.float(), dim=1)
        normalized_ct_reference = F.normalize(ct_reference.float(), dim=1)
        similarity = torch.einsum(
            "bchw,kc->bkhw", normalized_ct, normalized_ct_reference
        )
        assignment = F.softmax(similarity, dim=1).to(pet_reference.dtype)
        retrieved_pet = torch.einsum(
            "bkhw,kc->bchw", assignment, pet_reference
        ).to(ct_feature.dtype)

        # CT 全局描述对检索结果做逐通道仿射调制；gamma 初始为 1，beta 初始为 0。
        descriptor = F.adaptive_avg_pool2d(ct_feature, output_size=1).flatten(1)
        descriptor = F.gelu(self.descriptor_projection(descriptor))
        gamma = 2.0 * torch.sigmoid(self.gamma_projection(descriptor))
        beta = self.beta_projection(descriptor)
        return gamma[:, :, None, None] * retrieved_pet + beta[:, :, None, None]


class MTPICompensation(nn.Module):
    """四尺度任务导向 PET 原型补偿模块。

    Parameters
    ----------
    channels:
        四个尺度在现有对齐层之后的共同 CT/PET 通道数。
    filter_ratio:
        Full 原型更新时过滤的离群病例比例，默认 0.05。
    sync_distributed:
        DDP 训练时是否跨进程同步病例级原型统计。它是执行选项而非研究超参数。
    """

    NUM_SCALES = 4

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        filter_ratio: float = 0.05,
        sync_distributed: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != self.NUM_SCALES:
            raise ValueError(f"MTPI 必须接收 {self.NUM_SCALES} 个尺度的 channels。")
        if any(int(channel) <= 0 for channel in channels):
            raise ValueError("所有尺度通道数必须为正整数。")

        self.channels: Tuple[int, ...] = tuple(int(channel) for channel in channels)
        self.filter_ratio = float(filter_ratio)
        self.reference_banks = nn.ModuleList(
            [
                _ScaleReferenceBank(channel, filter_ratio, sync_distributed)
                for channel in self.channels
            ]
        )
        self.imputers = nn.ModuleList(
            [_ScalePrototypeImputer(channel) for channel in self.channels]
        )

    def _validate_features(
        self,
        features: FeatureSequence,
        name: str,
    ) -> List[Tensor]:
        if not isinstance(features, (list, tuple)):
            raise TypeError(f"{name} 必须是包含四个 [B,C,H,W] tensor 的 list/tuple。")
        if len(features) != self.NUM_SCALES:
            raise ValueError(f"{name} 必须包含四个尺度，实际为 {len(features)}。")

        checked = list(features)
        batch_size: Optional[int] = None
        for scale_index, (feature, channels) in enumerate(
            zip(checked, self.channels), start=1
        ):
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise TypeError(f"{name}[{scale_index - 1}] 必须是 [B,C,H,W] tensor。")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"{name} 第 {scale_index} 尺度期望 C={channels}，"
                    f"实际 C={feature.shape[1]}。请确认模块位于原有通道对齐层之后。"
                )
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError(f"{name} 四个尺度的 batch size 必须一致。")
        return checked

    @torch.no_grad()
    def reset_reference_banks(self) -> None:
        """清空所有原型。仅在明确重新训练时调用；恢复训练不要调用。"""
        for bank in self.reference_banks:
            bank.reset()

    def reference_status(self) -> Dict[str, object]:
        """返回各尺度/类别的原型样本计数，便于日志与 checkpoint 检查。"""
        per_scale_counts = [bank.counts.detach().cpu().tolist() for bank in self.reference_banks]
        return {
            "classes": ("background", "tumor"),
            "counts_per_scale": per_scale_counts,
            "fully_initialized": all(
                bank.fully_initialized for bank in self.reference_banks
            ),
        }

    def extra_parameter_count(self) -> int:
        """返回 MTPI 新增可训练参数量（原型 buffers 不计入）。"""
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        ct_features: FeatureSequence,
        pet_features: Optional[FeatureSequence] = None,
        modality_state: ModalityState = "full",
        target: Optional[Tensor] = None,
        update_references: Optional[bool] = None,
    ) -> List[Tensor]:
        """执行单一 batch 状态的前向。

        Full：要求 ``pet_features``，原样返回它；训练且提供 target 时默认更新原型。
        Missing：要求 ``pet_features is None``，仅由 CT 生成四尺度 PET 补偿特征。

        ``update_references=None`` 表示仅在 ``self.training`` 且提供 target 的 Full
        batch 中更新。验证/测试时即使误传 target 也不会默认更新。
        """
        state = _canonical_state(modality_state)
        ct_features_checked = self._validate_features(ct_features, "ct_features")
        batch_size = ct_features_checked[0].shape[0]

        if state == "full":
            if pet_features is None:
                raise ValueError("Full batch 必须提供真实 pet_features。")
            pet_features_checked = self._validate_features(pet_features, "pet_features")
            for scale_index, (ct_feature, pet_feature) in enumerate(
                zip(ct_features_checked, pet_features_checked), start=1
            ):
                if ct_feature.shape != pet_feature.shape:
                    raise ValueError(
                        f"第 {scale_index} 尺度 SUM 前 CT/PET 形状必须一致；"
                        f"得到 {tuple(ct_feature.shape)} 与 {tuple(pet_feature.shape)}。"
                    )

            should_update = (
                self.training and target is not None
                if update_references is None
                else bool(update_references)
            )
            if should_update:
                if target is None:
                    raise ValueError("update_references=True 时 Full batch 必须提供 target。")
                target_checked = _prepare_binary_target(target, batch_size).to(
                    device=ct_features_checked[0].device, non_blocking=True
                )
                for bank, ct_feature, pet_feature in zip(
                    self.reference_banks, ct_features_checked, pet_features_checked
                ):
                    bank.update(ct_feature, pet_feature, target_checked)

            # Full 路径不校准、不重建，保持 baseline 的真实 PET 接口与数值不变。
            return pet_features_checked

        # Missing 路径：严禁调用方传入 PET 特征，从接口层防止隐藏的双路径/置零实现。
        if pet_features is not None:
            raise ValueError(
                "Missing batch 的 pet_features 必须为 None；不要加载/置零 PET，"
                "也不要运行 PET Encoder。"
            )
        if update_references is True:
            raise ValueError("Missing batch 不允许更新 Full CT-PET 参考原型。")

        return [
            imputer(
                ct_feature,
                bank.ct_prototypes,
                bank.pet_prototypes,
                bank.ready_mask,
            )
            for imputer, bank, ct_feature in zip(
                self.imputers, self.reference_banks, ct_features_checked
            )
        ]

    def reference_loss(
        self,
        compensated_pet_features: FeatureSequence,
        target: Tensor,
    ) -> Tensor:
        """唯一辅助损失：补偿特征的病例级类别语义应接近对应 PET 原型。

        只对当前已经建立参考原型且该病例真实存在的类别计算余弦分类损失。
        它不访问当前病例真实 PET，也不做逐元素 PET 特征回归。
        """
        features = self._validate_features(
            compensated_pet_features, "compensated_pet_features"
        )
        target_checked = _prepare_binary_target(target, features[0].shape[0]).to(
            device=features[0].device, non_blocking=True
        )
        losses: List[Tensor] = []

        for feature, bank in zip(features, self.reference_banks):
            ready_indices = torch.nonzero(bank.ready_mask, as_tuple=False).flatten()
            # 只有一个可用类别时分类交叉熵恒为零，等待两个类别原型齐备即可。
            if ready_indices.numel() < 2:
                continue

            scaled_target = _resize_target(target_checked, feature.shape[-2:])
            reference = F.normalize(
                bank.pet_prototypes.index_select(0, ready_indices).detach().float(),
                dim=1,
            )
            class_to_logit = {
                int(class_id.item()): logit_index
                for logit_index, class_id in enumerate(ready_indices)
            }

            pooled_vectors: List[Tensor] = []
            pooled_labels: List[int] = []
            for class_index, logit_index in class_to_logit.items():
                class_vectors = _ScaleReferenceBank._case_level_vectors(
                    feature, scaled_target, class_index
                )
                if class_vectors.shape[0] > 0:
                    pooled_vectors.append(class_vectors)
                    pooled_labels.extend([logit_index] * class_vectors.shape[0])

            if pooled_vectors:
                vectors = F.normalize(torch.cat(pooled_vectors, dim=0).float(), dim=1)
                logits = vectors @ reference.transpose(0, 1)
                labels = torch.tensor(
                    pooled_labels, device=logits.device, dtype=torch.long
                )
                losses.append(F.cross_entropy(logits, labels))

        if not losses:
            # 保持 device/dtype/计算图兼容；原型未就绪时唯一监督仍是分割损失。
            return sum(feature.sum() for feature in features) * 0.0
        return torch.stack(losses).mean()

    def training_loss(
        self,
        segmentation_loss: Tensor,
        compensated_pet_features: FeatureSequence,
        target: Tensor,
        reference_weight: float = 0.1,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """组合 Missing batch 总损失：L = L_seg + lambda_ref * L_ref。"""
        if segmentation_loss.ndim != 0:
            raise ValueError("segmentation_loss 必须是已经归约为标量的 tensor。")
        if reference_weight < 0.0:
            raise ValueError("reference_weight 必须非负。")
        ref_loss = self.reference_loss(compensated_pet_features, target)
        total_loss = segmentation_loss + float(reference_weight) * ref_loss
        return total_loss, {
            "segmentation_loss": segmentation_loss.detach(),
            "reference_loss": ref_loss.detach(),
            "total_loss": total_loss.detach(),
        }


# 简短别名，便于配置文件中使用。
MTPI = MTPICompensation


def _smoke_test() -> None:
    """直接运行本文件时执行 CPU 最小前向/反向检查。"""
    torch.manual_seed(7)
    channels = (8, 16, 24, 32)
    sizes = ((32, 32), (16, 16), (8, 8), (4, 4))
    batch_size = 2

    module = MTPICompensation(channels=channels, sync_distributed=False).train()
    ct_full = [
        torch.randn(batch_size, channel, height, width)
        for channel, (height, width) in zip(channels, sizes)
    ]
    pet_full = [torch.randn_like(feature) for feature in ct_full]
    target = torch.zeros(batch_size, 64, 64, dtype=torch.long)
    target[:, 20:44, 22:46] = 1

    full_output = module(ct_full, pet_full, "full", target=target)
    assert all(output is source for output, source in zip(full_output, pet_full))
    assert module.reference_status()["fully_initialized"]

    ct_missing = [feature.detach().clone().requires_grad_(True) for feature in ct_full]
    compensated = module(ct_missing, pet_features=None, modality_state="missing")
    assert [tuple(value.shape) for value in compensated] == [
        tuple(value.shape) for value in ct_missing
    ]

    fake_seg_loss = sum(value.square().mean() for value in compensated)
    total_loss, _ = module.training_loss(
        fake_seg_loss, compensated, target, reference_weight=0.1
    )
    total_loss.backward()
    assert all(value.grad is not None for value in ct_missing)
    assert any(parameter.grad is not None for parameter in module.imputers.parameters())

    expected_parameters = sum(3 * (channel * channel + channel) for channel in channels)
    assert module.extra_parameter_count() == expected_parameters
    print(
        "MTPI smoke test passed | params=",
        module.extra_parameter_count(),
        "| status=",
        module.reference_status(),
    )


if __name__ == "__main__":
    _smoke_test()