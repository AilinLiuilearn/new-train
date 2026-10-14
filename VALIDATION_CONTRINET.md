# ConTriNet MFM 验证记录

## 固定来源

- baseline：`663f9786122365d8b2774e4497df92bc06b715d5` (`exp/aligned-add-baseline`)
- official ConTriNet：`9af315d4a93d897cf0b9abb11e194d1ed1072ff9`
- 官方完整文件 SHA256：`8bae888de081c8560de84565797ab816a602c2c06f369884c2389132b5b40478`
- 完整归档与通过 GitHub connector 取得的固定版本原文逐字符完全一致。
- MFM_0 / MFM 类定义逐字符和 AST 均与官方归档一致；包括原 BatchNorm 与非 GAP 通道瓶颈门控。

## 实际执行结果

CPU，Python 3.12，PyTorch 2.5.1+cpu，torchvision 0.20.1+cpu，timm 1.0.30，transformers 4.57.6。
合成图像及标签，不加载预训练权重。

| 检查 | 结果 |
|---|---|
| 新增模块单测 | 13 passed |
| 新增真实模型/训练/checkpoint 集成测试 | 17 passed |
| 现有 baseline、Full runner、EMA、mixed-batch 回归 | 53 passed |
| 83 项合计 | 全部通过 |
| 512×512 完整真实编码器模型前向 | `(1,1,512,512)`，全部有限 |
| 新增融合参数（默认 residual） | 1,303,144 |
| Python compileall / Bash syntax | 通过 |
| CLI `run_full_contrinet_fusion.py --help` | 通过，Full-only 与 MFM 参数可见 |

首次全量测试为 82 passed + 1 failed；失败为测试环境未安装 cv2，发生于原 `test_ct_only_dataset_has_no_pet` 导入阶段。
补齐测试环境的 opencv-python-headless 后单独重跑该项，1 passed。没有为此修改仓库的数据集或增强代码。
修改源码归档末尾的保存换行后，额外重跑官方源码一致性测试，1 passed。

主要集成断言：

- `add/local/cross/replace` 各运行一次真实编码器训练、验证、保存 best_full、按同开关重建后的最终 Full 评估。
- disabled：融合无参数，state_dict 与原 Add 相同，严格加载及 logits 完全相等。
- enabled residual：共享模块初始权重与 baseline 一致，初始 logits 与 Add 完全相等。
- 第一优化步输出投影有梯度；第二步原 MFM 有非零梯度且参数更新。
- 深层输出对浅层输入的依赖及反向梯度只在启用跨尺度路径时存在。
- 融合所有参数均属于 AdamW；CT、PET、两组对齐及解码器均有有限非零梯度。
- raw/EMA 权重、MFM 开关和严格 checkpoint 重建正确；EMA 尚未启用时不因 payload 存在而误选 EMA。
- Missing/缺失 PET 在进入编码器前被拒绝；原 Full/Missing baseline 的隔离性测试仍通过。
- CPU bfloat16 autocast 前后向、标准四尺度及奇数分辨率通过。

## 验证边界

未进行真实 PCLT20K 训练、GPU FP16、batch 16 显存测量或精度对比。
代码接入通过不代表 MFM 已提高 Full Dice，或已经适配 Missing 补偿路径。
原编码器、ct_align、pet_align、GroupNorm 解码器、criterion、EMA、数据集与指标文件均未改动。

上游源码保留其原始空白格式。自有修改通过 `git diff --check`（排除两份原样源码），不对官方类运行格式化。
