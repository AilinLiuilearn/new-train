# 双状态差异引导的配对上下文增强 v1

实现于 `exp/ct-pretrain-dual-decoder-add-lr8e5`，审查 HEAD
`3c1c5d491a762552835ecd869524d9249329967c`。未切换、新建或推送分支。
本轮开始时没有适用的 AGENTS.md，已跟踪文件无改动；原有未跟踪的
`diagnostics/`、`scripts/.ipynb_checkpoints/` 保留。原始状态/diff及验证日志保存于
`checkpoints_new/AUGMENTATION/review-v1/`（测试生成的临时权重不作为交付）。

## 固定的模型与初始化

模型仍为 CT ConvNeXtV2-Nano / PET MiT-B1，BN 四尺度对齐到
64/128/320/512，直接相加，两个独立 GN U-Net 解码器；35,907,538 参数。
每个配对样本同时走 Full/Missing，CT 编码一次，损失精确为两路 BCE+Dice
的等权平均。`models/`、损失、优化器、调度、AMP/EMA实现和指标定义未修改。
Missing 模型推理无需 PET、GT、cache，禁止调用 PET/融合/Full 解码器。

已现场核对源 `best.pt`：format=`ct_pretrain_dual_decoder_v2_paired`，stage=2，
protocol=`paired_full_missing`，epoch=6，eval_weights=`ema`；SHA256：
`c2c5891d4fb0b32b4f39914916d10a7f7a97b1461564e740aa248c26781ab467`。
配置：512、batch16、lr8e-5、epochs60、seed2023、ImageNet归一化、CIPA增强、
AdamW wd1e-4、warmup3、flat0.3、min_lr1e-6、clip5、AMP、EMA0.999且epoch>3启用。
train_original/test/test，显式允许 val=test，原警告保留。
历史日志实际有24轮，最佳第6轮 Full=.78091493、Missing=.76488509、
Joint=.77290001，HD95分别17.58248/22.59715（像素）。历史 run 没有 final_test.json，
不能称为完成了60轮；也不能直接称为与新早停 run 同预算的前瞻对照。

新训练只从源配置中指定的通用编码器预训练文件初始化。两个解码器为相同随机
初值的独立副本。建图 best.pt 不用于 resume、任务初始化、optimizer、scaler、
scheduler 或训练 EMA。launcher 只读取 manifest 中的配置，校验通用预训练 hash，
不读取源任务 checkpoint。正式 run 的 initialization.json 继续记录初始化来源。

用户后续明确授权直接训练，并纠正“10轮”为 **early stopping patience=10**。
因此新 launcher 默认连续10轮 Joint 无严格提升才停止；提升后重新计数，60轮上限及
原60轮学习率调度不变。原入口默认 patience=0，保留旧固定预算行为。
新 B0/A1/A2/A3 都应采用相同上限及 patience，而非各跑10轮。

## 方法及边界

参考来源与本轮改造分开理解：

- [FuseMix, CVPR 2024](https://arxiv.org/abs/2312.10144)，
  [官方代码](https://github.com/layer6ai-labs/fusemix)：原方法在预训练编码器潜空间进行多模态增强，
  验证重点是图文/音文检索。本轮只借鉴保持配对，未引入其混合训练、适配器或对比目标。
- [TumorCP, MICCAI 2021](https://arxiv.org/abs/2107.09843)，
  [官方代码](https://github.com/YaoZhang93/TumorCP)：原方法为目标级 Copy-Paste。
  本轮仅以目标组织上下文视图；同图真实裁剪，无肿瘤复制、跨患者粘贴或 nnU-Net。
- [PNDC, TMI 2025](https://doi.org/10.1109/TMI.2025.3526818)，
  [官方代码](https://github.com/yansheng-qiu/PNDC)：借鉴增加模态不保证局部获益的动机；
  本轮没有采用其正负校准损失。

假设是：在随机覆盖保留的情况下，针对两路互补错误提供配对上下文视图可能有益。
Full/Missing解码器不同，分歧也可能来自参数/优化；地图只表示经验分支错误关系，
不是 PET 因果贡献。类别定义依赖训练 GT，预测分歧本身不等于正确性。
训练期可使用 PET/GT 安排窗口，推理期没有额外信息；裁剪不能恢复 CT 不可辨识的 PET 信息。
普通配对增强、目标裁剪、困难样本挖掘均有先例；组合/命名不构成已验证新颖性。

## 离线建图与完整性

只构造 train Dataset：train=False、aug=none、shuffle=False、drop_last=False。
FP32 eval/inference_mode、冻结参数；一次 Full 前向返回两路 logits，严格 sigmoid>0.5。
D=0两路对，1为Missing错Full对，2相反，3两路错。按安全且唯一 image_id 保存uint8 PNG。
不读取val/test图像或标签，不保存浮点logits，不创建训练optimizer/EMA。

manifest记录源路径/hash/epoch/raw或EMA/commit、当前commit、固定模型结构、完整源配置、
train split hash、所有image_id、CT/PET/mask大小及mtime_ns、可选内容hash、地图hash、
像素/8连通区域/样本覆盖、时长、GPU峰值。开启 `--content_hash` 可更严格检查源图像。
普通指纹不能发现恶意保持大小和mtime的内容修改，这是默认快速模式的限制。
源数据在建图中变化立即失败。PNG/manifest原子写，最后才标记complete。
`--limit` 总是标记partial，即使limit>=总数也不允许正式训练。非空目录拒绝覆盖；
不支持隐式续建，失败后保留目录并显式使用新目录重建。地图构建批量OOM不会自动降尺寸、
降精度或跳样本；使用新目录显式减小batch重试，batch记入manifest。

启用模式在train loader初始化时校验完整性、身份、尺寸、归一化、split及所有源文件指纹和
地图hash，每次按需解码再校验hash/shape/类别。worker不常驻全量地图。
resume验证同一manifest/source hash/上下文规则；保存配置权威，不允许CLI换策略/cache。
evaluation完全不加载地图，旧checkpoint缺少新参数时按none处理。

## 裁剪规则与 off 等价

复用原随机仿射、翻转、裁剪的顺序。原 GT 线性仿射/三次裁剪插值和最后阈值化保留。
只扩展辅助图传参：D用相同矩阵/flip，最近邻、边界0，原调用返回数量不变。
原裁剪边长[0.7,0.9]先抽一次，触发概率0.6；未触发不裁剪。
触发后先抽原随机窗口，作为随机退路及审查的同尺寸反事实；引导概率条件于此次触发。

- none：原代码路径，不新增随机数，不读地图。
- foreground：GT前景8连通区域均匀抽取。
- hard：D!=0的总体错误区域均匀抽取，不区分哪一路正确。
- dual_state：若1/2都有则各半，只有一类则用该类；区域内均匀抽连通区域。3不并入。

目标区域若碰到GT前景，则联合所有相交的2D前景连通区域；这不是独立3D病灶。
上下文固定 `max(4, round(0.1*max(bbox_width,bbox_height)))`，在图像边界裁定合法上下文。
在包含目标框及上下文的合法左上角范围均匀抽样，不固定中心，不重抽边长。
候选不存在、变换后消失、目标过大分别计数并用原随机窗口；损坏cache立即报错。
CT/PET/GT同窗裁剪，原插值与归一化恢复输入尺寸，GT仍为二值真实标签。

tests/oracles 内保存了开始修改前真实HEAD的dataset/transform实现，用于独立oracle。
24个固定seed逐样本CT/PET/GT及NumPy RNG状态exact比较，none和guided_prob=0均等价。
核心模型不mock；额外覆盖共享矩阵、最近邻/0边界、连通区域公平性、非中心抖动、空候选、
过大/边界窗口、文件/配置身份、尾批、raw/EMA、BN不变、实际CUDA AMP溢出与有效更新、
多worker计数、epoch边界恢复、无PET/cache的Missing评价。
多worker没有persistent_workers；按现有epoch边界保存Python/NumPy/Torch/CUDA/loader生成器。
CPU num_workers=0 的增强训练恢复已做逐权重比较；不承诺跨worker数/库版本/GPU位级复现。

## 现场建图和审查结果

RTX3090 24GB，myenv Python3.10.20、torch2.1.2+cu118。完整17,416张建图耗时405.43秒；
FP32 batch4，峰值allocated 501,881,856字节（约478.63MiB），无buffer变化。
类别像素数0/1/2/3：4,561,058,606 / 644,483 / 503,226 / 3,293,589。
对应错误类区域数1/2/3：117,803 / 117,948 / 239,497。
45张无差异候选，16,492张两类都有（94.69%）；差异像素约0.0251%，
说明区域很多但非常细碎，不能把“覆盖很多样本”误读成强监督信号。

固定seed2023，1024张：随机序列前补可用类别/背景误报/无候选层，非显著性样本。
审查输出每样本window、随机对照window、seed、大小、选区面积/前景像素、
覆盖、中心及JSONL；拼图包含原图、同步几何后图、D类别颜色、选择/随机框及两种裁剪。

| 策略 | 引导成功/尝试 | 成功引导与随机窗IoU | 像素MAD/255 | 所选区域平均像素 | 区域前景比例 |
|---|---:|---:|---:|---:|---:|
| A1 foreground | 303/303 | .7312 | .1181 | 202.74 | 1.0000 |
| A2 hard | 303/303 | .7337 | .1157 | 37.09 | .2084 |
| A3 dual_state | 302/303 | .7405 | .1148 | 8.84 | .1190 |

A3裁剪638次、引导尝试303次、成功302次（全样本29.49%），随机退路336次；
1次无候选回退。两类都有的285次选择中，1类137次、2类148次（48.07%/51.93%）。
成功引导所选区域覆盖100%。保留随机窗的整体平均IoU=.8772，成功引导子集=.7405，
两者分母不同。大窗口仍高度重叠，真实视图确有变化但增量有限。
真实样例覆盖引导、随机、无裁剪及回退；另有合成边界/过大/消失测试。
人工检查0009_007及0098_001等拼图，CT/PET/GT空间对应；后者仅共同错误，合法回退。

三策略裁剪后的平均前景比例接近（约.00357/.00359/.00357），但选中区域面积和前景构成
差异明显。A3-A2收益即使出现，也不能完全归因于双状态分组；后续最小验证应只匹配
区域面积/前景构成后比较，不同时修改裁剪范围、loss、lr、融合。v1保持原规则不混入v2。

真实编码器CUDA预检512²/batch16：一次有效AdamW+EMA更新，loss=1.4636501，
两路各16个样本、CT/PET各编码一次；非零梯度覆盖全部6个训练模块，Missing隔离通过。
峰值allocated=12.5825GiB、reserved=13.4902GiB；这是合成输入预检，非真实训练吞吐保证。
正式训练/对照结果查看suite status、各run train.jsonl/augmentation.jsonl及comparison.json；
在训练完成前不声称Dice提升。

## 可复制命令

```bash
cd /root/autodl-tmp/mkd-main/new-train
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE_CKPT="$PWD/checkpoints_new/STAGED/dual-add-joint-nocttask-s2023/best.pt"
MAP_DIR="$PWD/checkpoints_new/AUGMENTATION/dual-state-maps-s2023-v1"
AUDIT_DIR="$PWD/checkpoints_new/AUGMENTATION/audit-dual-state-s2023-v1"
RUN_DIR="$PWD/checkpoints_new/STAGED/dual-add-joint-dualstatecrop-s2023-v1"

# 已现场执行完成；目录非空时拒绝重跑，请为新构建指定新目录。
python -u scripts/build_dual_state_maps.py --checkpoint "$BASE_CKPT" \
  --output "$MAP_DIR" --device cuda --batch_size 4 --num_workers 4
python -u scripts/audit_dual_state_augmentation.py --map_dir "$MAP_DIR" \
  --output "$AUDIT_DIR" --num_samples 32 --seed 2023
python -m pytest -q tests/test_dual_state_augmentation.py tests/test_staged.py tests/test_preflight_amp.py
python scripts/check_staged.py --device cuda --stage 2 --batch_size 16 --joint_from_pretrained

# 检查精确展开的配置（不启动训练、不读取任务权重）。
bash scripts/train_joint_aug.sh "$MAP_DIR" "$RUN_DIR" --dry_run

# 四个同协议对照；已由suite启动时，不要手动重复运行。新输出目录不会覆盖旧实验。
bash scripts/train_joint_aug.sh "$MAP_DIR" checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1 --crop_guidance none
bash scripts/train_joint_aug.sh "$MAP_DIR" checkpoints_new/STAGED/dual-add-joint-foregroundcrop-s2023-v1 --crop_guidance foreground
bash scripts/train_joint_aug.sh "$MAP_DIR" checkpoints_new/STAGED/dual-add-joint-hardcrop-s2023-v1 --crop_guidance hard
bash scripts/train_joint_aug.sh "$MAP_DIR" "$RUN_DIR" --crop_guidance dual_state --guided_crop_prob 0.5

# 原baseline入口仍可用；其默认值应与source config核对，下面显式启用用户要求的早停。
bash scripts/train_joint.sh checkpoints_new/STAGED/dual-add-joint-baseline-new --early_stopping_patience 10

# 有界顺序队列，四策略均为预先固定的v1，没有多种子或超参搜索。
python -u scripts/run_crop_pilots.py --map_dir "$MAP_DIR" \
  --output checkpoints_new/AUGMENTATION/pilots-s2023-v1 \
  --strategies dual_state hard foreground none

python -u run_staged.py --resume "$RUN_DIR/last.pt" --device cuda
python -u run_staged.py --evaluate "$RUN_DIR/best.pt" --eval_mode missing --device cuda
```

队列优先启动用户目标A3，其后A2/A1/B0，全为预注册的同配置对照，不依据test表现调参数。
任一子进程失败即停止队列，保留日志，不吞异常、不自动换精度/权重。逐run可用last.pt恢复；
队列本身不猜测如何恢复非空目录，必要时明确使用原run的resume，再为未执行项建新队列目录。
不删除原权重/数据；回退使用新run并设置crop_guidance=none即可走原数值路径。
所有新脚本均支持仓库根目录直接执行及 --help。

## 迭代记录及结论约束

实现修复：旧测试假设torch.amp.GradScaler必存在，按2.1.2真实API修正测试准备及警告预期；
未升级环境/改变AMP代码。增强断点故意在last.pt之后中断时发现日志漏首行，已从checkpoint
恢复augmentation_row；恢复权重与不中断run逐值相同。以上为实现修复，不算性能迭代。

当前只证明完整建图、执行契约、有效视图变化和有效参数更新。训练结果尚待实际输出。
结果必须是同一个Joint选出的epoch/权重，报告Full/Missing/Joint、HD95、漏检和误报相关指标，
不拼接两路各自峰值。validation复用test是开发协议；不能称独立最终测试或根据它自动搜索参数。
至少比较A3相对A1/A2，并注意区域构成混杂；单种子不确定性未量化，无SOTA或因果声明。
若A3不优于A2，优先检查窗口相近、碎小误差与采样构成；只提出一个可证伪的后续对照，
不通过叠加融合/补偿/在线teacher掩盖结果。
