# CT/PET 融合协同实验

当前协同分支：`research/fusion-collab-20261115`。这里只同步可执行代码、配方、划分和固定文本向量。影像、预训练模型、实验权重和输出独立存储。

## 目录

```text
src/                  共享编码器、M1、GN解码器、数据管线、指标与工具
variants/tsbci/       78.4970%版本的通道模块、模型及训练/推理入口
variants/a2/          最近A2版本的模块、模型及训练/推理入口
configs/              两个完整训练配方
resources/            固定BioMedCLIP角色向量、资源校验值和来源
splits/               原训练/开发集划分，跨机器使用同一份
scripts/              myenv入口、资源检查、评估和资源打包
tests/                工程约束检查
paths.local.json      每台机器自己填写，不提交Git
```

历史模型和文献分析索引见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。这是原实验的工程迁移，不是新的性能实验。没有启动正式训练。

最新授权的三组patience=3重训方案及命令见 [docs/RERUN_P3.md](docs/RERUN_P3.md)：原M1、仅CT预对齐的拼接基线、原78.4970%架构。它们使用独立配置，不覆盖旧运行或旧最佳权重。

三组自然完成后的V3.1完整候选试跑见 [docs/V31_PILOT.md](docs/V31_PILOT.md)。按“专家联合训练→冻结基座纠错训练”执行，先自动做GPU预检；上游停止或失败则取消后续，不修改正在运行的三组代码。

## 首次准备

```bash
git clone --depth 1 --branch research/fusion-collab-20261115 https://github.com/AilinLiuilearn/new-train.git fusion
cd fusion
conda activate myenv
cp paths.example.json paths.local.json
```

编辑 `paths.local.json`，填写该机器的三个绝对路径：

- `data_root`：PNG数据集根目录，结构为 `0001/0001_001_CT.png`、对应 `_PET.png`、`_mask.png`。
- `pretrained_root`：包含 `convnextv2_nano/` 和 `mit-b1/` 的目录，各自有 `config.json` 和 `pytorch_model.bin`。
- `output_root`：该卡独立保存运行结果的目录。

已有 `myenv` 不需要重建。新机器可用 `conda env create -f environment.yml`，或在其 `myenv` 中运行 `python -m pip install -r requirements-myenv.txt`。此配方记录了本机验证的Python 3.10、PyTorch 2.1.2/CUDA 11.8环境；驱动和GPU条件也需满足。当前模型不依赖natten。

```bash
bash scripts/myenv.sh scripts/check_environment.py
bash scripts/myenv.sh -m pytest -q
```

固定文本向量已经保存为小型JSON，值与原BioMedCLIP缓存逐元素一致；训练无需启动文本编码器，也无需下载完整BioMedCLIP。修改文本时需要另行编码和校验，不能直接改字符串冒充新向量。

## 在不同卡运行

代码修改完成后先提交并推送新分支。另一台机器在没有未提交修改时执行 `git pull --ff-only`，再运行。每个运行使用新名称，不覆盖历史输出。

```bash
# 卡A：TSBCI
bash scripts/myenv.sh -m variants.tsbci.train --run tsbci_cardA_s2023 --gpu 0

# 卡B：A2；不同机器都可以使用各自的GPU 0
bash scripts/myenv.sh -m variants.a2.train --run a2_cardB_s2023 --gpu 0

# 同一台机器的另一个GPU，显式选择GPU 1
bash scripts/myenv.sh -m variants.a2.train --run a2_gpu1_s2023 --gpu 1
```

上述命令在前台运行；需要断开终端继续训练时放进 `tmux`。每个进程独占一张卡；按物理GPU UUID加锁，不再用同一个全局锁阻塞所有卡。两台机器须使用独立输出目录；共享文件系统时也应使用不同run名称。

默认拒绝未提交代码。若有意验证临时修改，可显式加 `--allow-dirty`，这次运行会记录脏状态和完整源码指纹，不能只用commit号声称版本一致。

每组保存 `run_identity.json`（Git提交、分支、配置、依赖版本、GPU、资源和源代码校验）、代码快照、逐轮指标和诊断。仅保存验证最佳的 `best.pt`，不保存 `last.pt`、优化器、调度器、Scaler或RNG断点。EMA仍用于训练及选模，文件里保存实际选中的权重。中断后须用新run从头训练。

保持种子2023、512输入、batch16、学习率8e-5及原BCE+Dice。只加载编码器通用预训练，任务模块从头学习；每轮仅验证一次完整融合路径。默认不附加G-on/G-off验证，不自动续接其他实验。停止某组可在其输出目录放置 `STOP` 文件。

## 单独验证已有权重

权重在两台机器间自行复制，不经过Git。新代码可以严格加载原TSBCI/A2的推理权重。

```bash
bash scripts/myenv.sh scripts/evaluate.py --variant tsbci --checkpoint /path/to/best.pt --output /path/to/new_evaluation.json
```

## 首次传输数据与预训练资源

同机多卡直接在两个本地路径配置中指向同一份只读资源。不同机器可以用 `rsync` 直接传输；没有现成同步渠道时可生成私人传输包：

```bash
bash scripts/myenv.sh scripts/export_resources.py --include-data --output /path/to/ct_pet_resources.tar.gz
```

将此包传到另一台机器后解压到独立资源目录，并配置 `data_root=资源目录/data/PCLT20K`、`pretrained_root=资源目录/pretrained`。包内包含逐文件SHA256清单；启动前会校验预训练权重、划分及样本文件存在性。数据包和实验权重不要推送到公开仓库。

## 代码同步约定

每个候选使用独立配置或分支；实验前记录明确假设和单因素差异。代码经提交推送后另一张卡再拉取运行。训练期间不切换该工作目录的代码版本，另一候选使用另一份clone或worktree。完整结果自然结束后分析，LLM每5轮检查一次即可，异常或用户询问除外。研究目标仍需实际实验验收，不因工程迁移而宣称性能提升。
