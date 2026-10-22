# PET/CT GN baseline 与局部／区域残差融合

本分支仅维护固定 GN 相加 baseline、Full 融合模块，以及训练、评估、恢复和必要验证。
旧的引导裁剪、双状态增强、Copy-Paste、离线地图和自动实验队列已移除。
基线原有的随机配对增强保持原样，不改变数据增强协议。

## 固定框架

- CT ConvNeXtV2-Nano，PET MiT-B1，四尺度通道64/128/320/512，BN对齐。
- 两个独立GN解码器。每个训练样本同时监督Full和Missing，CT只编码一次，损失各0.5。
- Missing提前返回，不调用PET编码器、融合模块或Full解码器。
- `add`：四尺度C+P；`local_region_residual`：独立CT局部建模、PET区域建模，分别残差后相加。
- 通用编码器预训练初始化；512、batch16、lr8e-5、AdamW、AMP、EMA、60轮调度。
- Joint严格提升选同一个最佳checkpoint，5轮无提升早停，默认仅保存best.pt。
- 原CIPA随机配对增强和ImageNet归一化保持。验证／测试复用test.txt属于开发实验，非独立盲测。

固定配置：`configs/frozen_gn_training.json`。BN decoder兼容代码与Stage1导入接口仅用于旧checkpoint兼容；正式对照固定GN和joint_from_pretrained。

## 环境与数据

使用Python3.10/3.11及匹配的PyTorch/torchvision（融合使用SDPA，PyTorch>=2.0）。
优先迁移原运行环境；`requirements.txt`提供依赖约束，不是原环境的完整锁文件。
已记录的训练环境：torch2.1.2+cu118、torchvision0.16.2+cu118、timm1.0.27、
transformers4.38.2、numpy1.26.4。更换GPU或依赖后不保证逐位复现。

数据、编码器预训练文件和实验best.pt均不随Git上传，需要迁移到新实例。
数据布局：

```text
PCLT20K/
  train_original.txt
  test.txt
  CASE_ID/
    CASE_ID_SLICE_CT.png
    CASE_ID_SLICE_PET.png
    CASE_ID_SLICE_mask.png
```

split每行是图像ID（不含_CT等后缀），case目录取ID第一个下划线之前的部分。
CT/PET预训练路径可分别指向包含`pytorch_model.bin`的目录或加载器支持的权重文件。

## 两种模型的训练

从仓库根目录运行；每次使用全新输出目录。同一GPU上顺序运行两种模型。
以下路径按实例修改：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA_ROOT=/root/autodl-tmp/data/PCLT20K
CT_WEIGHTS=/root/autodl-tmp/pretrained/convnextv2_nano
PET_WEIGHTS=/root/autodl-tmp/pretrained/mit-b1
RUN_ROOT=/root/autodl-tmp/experiments/petct-r1
SEED=2023

COMMON=(--root "$DATA_ROOT" --ct_pretrained_path "$CT_WEIGHTS"
        --pet_pretrained_path "$PET_WEIGHTS" --random_state "$SEED")

# 固定GN相加baseline
bash scripts/train_local_region.sh "$RUN_ROOT/add-s$SEED" \
  --fusion_type add "${COMMON[@]}"

# 完整高斯融合
bash scripts/train_local_region.sh "$RUN_ROOT/fusion-s$SEED" \
  --fusion_type local_region_residual "${COMMON[@]}"

# 可选的去高斯对照；不是默认运行队列
bash scripts/train_local_region.sh "$RUN_ROOT/no-gaussian-s$SEED" \
  --fusion_type local_region_residual --no-fusion_gaussian "${COMMON[@]}"

# 只展开实际CLI，检查配置，不启动训练
bash scripts/train_local_region.sh UNUSED --fusion_type add "${COMMON[@]}" --print_command
```

统一启动器读取冻结配置，仅允许切换融合、高斯开关、实例路径和种子。
种子2023用于复现；2024/2025用于配对稳定性比较。每个种子两种模型使用相同数据、权重和环境。
原`bash scripts/train_joint.sh NEW_OUTPUT [run_staged参数]`仍保留，但冻结协议推荐使用上面的统一入口。

## 评估与恢复

```bash
python run_staged.py --evaluate /PATH/TO/RUN/best.pt --root "$DATA_ROOT" --device cuda
python run_staged.py --evaluate /PATH/TO/RUN/best.pt --root "$DATA_ROOT" --eval_mode missing --device cuda
# 仅适用于创建时显式启用--save_last_checkpoint的run
python run_staged.py --resume /PATH/TO/RUN/last.pt --device cuda
```

结构从checkpoint保存配置重建并strict load；旧add权重缺少fusion配置时默认add。
旧baseline／融合checkpoint中已禁用的增强字段会被剥离；包含已移除增强实验的配置会明确拒绝，需使用历史分支。
正式冻结协议不保存last.pt，不能把best.pt当成最后一轮恢复点。

## 验证与目录

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
# GPU空闲时进行目标配置预检
python scripts/check_staged.py --joint_from_pretrained --fusion_type add \
  --ct_pretrained_path "$CT_WEIGHTS" --pet_pretrained_path "$PET_WEIGHTS" --image_size 512 --batch_size 16
python scripts/check_staged.py --joint_from_pretrained --fusion_type local_region_residual \
  --ct_pretrained_path "$CT_WEIGHTS" --pet_pretrained_path "$PET_WEIGHTS" --image_size 512 --batch_size 16
```

- `models/`：两种模型共用骨架、对齐、解码器及融合模块。
- `datasets/`、`utils/`：数据、基线随机增强、loss、指标、optimizer调度、AMP和EMA。
- `scripts/`：统一训练入口、原joint入口、预检和性能测量。
- `tests/`：模型、数据、strict加载、恢复、保存策略和融合验证。
- `docs/local_region_validation/`：融合设计、性能报告、许可及已完成实验结果。
- `docs/cleanup_verification.json`：本次清理的等价性和验证记录。

最佳权重、日志和数据继续保存在服务器原路径；删除源码中的旧实验实现不删除这些实验记录。
