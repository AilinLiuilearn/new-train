# v1结果分析与v2单变量迭代

上一轮只设置了预设实验队列，没有在A3完成后接上分析和代码迭代。这是执行遗漏。
本轮现场读取原始日志、比较同一个Joint选中epoch，补做训练地图诊断并实现v2。

## 已有结果（开发协议，val=test）

| 实验 | 训练结束轮 | 选中轮 | Full Dice | Missing Dice | Joint Dice |
|---|---:|---:|---:|---:|---:|
| A3 双状态 v1 | 16 | 6 | .774132 | .760780 | .767456 |
| A2 总体错误 v1 | 15 | 5 | .773780 | .761597 | .767688 |
| A1 前景 v1 | 19 | 9 | .770686 | .759548 | .765117 |
| 新B0随机裁剪（仍在运行） | 尚未结束 | 当前5 | .778667 | .768756 | .773711 |

A3比A2低0.000232，比新B0当前最佳低0.006255；不支持双状态机制已有效的结论。
A3相对新B0的Full/Missing敏感度均下降，精度略升；这是预测更保守的现象，
不是已证实由某个模块造成的因果解释。A1同样未改善，不能把问题简单归为背景选区。
训练loss持续下降而开发Dice早早平台/下降，也提示泛化限制；不通过训练更久规避早停。

## 训练来源诊断和唯一改动

v1训练增强日志中，A3平均选区6.825像素，59.29%完全不接触GT前景。
A2平均37.649像素、纯背景33.04%；A1平均223.852像素。
固定seed2023随机抽1024张训练地图，共14,022个D=1/2连通区域：
中位面积1像素，60.60%单像素，76.53%不超过2像素，p90为6、p99为71像素。
分析只读取train图像/GT/cache；没有重建教师、调参搜索或额外模型训练。

v2新增 `--guidance_region_sampling sqrt_area`：**只改变选定类别内的连通区域抽样概率，
从均匀改为与面积平方根成正比**。指数固定0.5，不根据test结果搜索。
两类候选仍各半，保留小区域的非零机会，同时对大区域采用次线性权重。
没有区域过滤、没有新增最小面积阈值、没有改上下文、裁剪范围、引导概率或模型。
默认uniform保留v1代码路径和RNG，旧checkpoint缺省uniform。

原始训练地图上按“先类别各半、再区域抽样”计算，≤2像素选区概率从77.82%降到66.57%，
预期面积从7.03升到12.66。纯背景概率从59.90%变为60.86%，**没有减少背景选区**。
因此v2只检验减轻碎小区域过度采样的假设，不宣称修复召回或已证明v1下降原因。

相同seed/1024张真实增强审查，v2平均选区12.828像素，v1为8.845；引导成功均302次。
窗口与同尺寸随机窗IoU分别.73788/.74054；v2改变了区域分布，但大窗口重叠问题仍存在。
同样A2-v2采用sqrt_area，对照A3-v2，避免只给A3更换采样规则。

## 测试、初始化、恢复

本轮22项测试通过（69.57秒），包含新增平方根概率分布、类别各半、窗口合法、
冻结v1默认路径/RNG exact、guided_prob=0 exact、真实编码器有效更新及两路等权监督。
增强完整保存/恢复/evaluate回归采用sqrt_area，验证参数写入checkpoint、逐权重恢复相同、
禁止resume通过CLI换回uniform，Missing评价无需PET/cache。
已有模型结构、损失、AMP/EMA、优化器、调度、数据划分和阈值均未更改。

新run仍从原通用编码器预训练初始化，使用同一份v1离线地图；不会加载任何v1任务权重。
lr8e-5、batch16、512²、seed2023、60轮上限、原60轮调度、patience10均保留。

## 执行状态与输出

用户随后明确要求暂停B0并立即启动v2。已停止B0进程并释放GPU，保留其best.pt和
第10轮last.pt（最佳第5轮Joint=.77371129）；第11轮未完成的进度不计入checkpoint。
v1状态标记paused，未伪标完成；原v2等待记录归档，新的v2队列已直接启动A3-v2，随后执行A2-v2。
新增每个run完成后的 `interpretation.json`：报告对v1同策略的增减、A3对匹配A2/B0的结果。
这是确定性的定量判读，不是能在后台自行编写新代码的代理；v3未预先混入。

数据盘剩约2GB，两个新run的best/last及原子保存临时文件需要更多空间。
只清理本会话生成的/tmp/pytest-of-root临时测试权重，释放系统盘约8.5GB；
用户原始权重和v1实验均未删除/移动。新v2权重实际存放：

`/root/dual-state-experiments/STAGED/`

仓库入口 `checkpoints_new/STAGED-v2` 为该目录的符号链接。只改变输出存储位置。
该系统盘不属于AutoDL数据盘，实例环境变更前应保留该目录；不自动移动用户数据。

日志/状态：`checkpoints_new/AUGMENTATION/pilots-s2023-v2/`。
分析：`checkpoints_new/AUGMENTATION/analysis-v1-to-v2/{analysis.md,analysis.json,curves.png}`。
审查：`checkpoints_new/AUGMENTATION/audit-dual-state-s2023-v2/`。
新源码快照/diff/测试记录：`checkpoints_new/AUGMENTATION/review-v2/`。

```bash
cd /root/autodl-tmp/mkd-main/new-train
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 已启动；不要重复启动同名队列/实验。
python -u scripts/run_crop_pilots.py \
  --map_dir checkpoints_new/AUGMENTATION/dual-state-maps-s2023-v1 \
  --output checkpoints_new/AUGMENTATION/pilots-s2023-v2 \
  --run_root checkpoints_new/STAGED-v2 --tag s2023-v2 \
  --strategies dual_state hard --guidance_region_sampling sqrt_area \
  --reference_suite checkpoints_new/AUGMENTATION/pilots-s2023-v1/status.json

cat checkpoints_new/AUGMENTATION/pilots-s2023-v2/status.json
tail -f checkpoints_new/AUGMENTATION/pilots-s2023-v2.console.log

python -m pytest -q tests/test_region_sampling_v2.py tests/test_dual_state_augmentation.py tests/test_staged.py

# 如果某个训练进程中断，只恢复它的last.pt，不能换采样规则继续同一实验。
python -u run_staged.py --resume \
  checkpoints_new/STAGED-v2/dual-add-joint-dualstatecrop-s2023-v2/last.pt --device cuda
```

性能结论等待v2实际结果，不能把采样分布改变称为Dice提升。
val/test复用仍是原开发协议；本轮是明确说明的算法候选验证，不是独立最终测试。
若A3-v2仍不超过A2-v2，则不支持分组机制；若超过A2但不超过B0，也不能称为baseline改进。
后续候选需先审查实际结果，一次只改变一个因素，不自动堆叠模块或根据test扫描超参。
