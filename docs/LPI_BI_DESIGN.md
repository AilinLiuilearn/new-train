# LPI-BI V1：最终候选模型设计（无文本）

## Problem Anchor

- Bottom-line problem: 在保留异构双编码器、现有通道对齐、原M1和GN单解码器的前提下，设计一个前置模块，检验监督学习的模态特定定位能否改善跨模态信息选择及最终融合分割。
- Must-solve bottleneck: 过去前置交互未取得可靠增量；目前没有证据说明更多通道交互自然有益，也没有证据说明M1必然缺少定位。
- Non-goals: 不更换M1、解码器或数据；不研究缺失模态；不把固定文本常量包装为语义对齐；设计讨论本身不启动训练；后续用户已明确授权实现与完整模型一组实验。
- Constraints: CT ConvNeXtV2-Nano / PET MiT-B1，原双路对齐，四尺度64/128/320/512；512、batch16、seed2023、AdamW 8e-5/1e-4、原调度/EMA/评估，myenv，best-only。只加载编码器通用预训练，任务模块从头。
- Success condition: 同协议完整模型Dice/HD95有增量，且同定位监督下空间条件交互优于普通交互；目标Dice至少78.8%、争取78.9%以上，同一最佳权重HD95<17。目标不是已取得结果；单种子开发集筛选不能证明稳定性。

## Technical Gap

固定文本投影的单Query与线性Key点积可折叠为1×1前景头，因此本版直接采用定位头。原M1有CT GaussianScaleContrast/local block及PET region block，再叠加MAC可能功能重复。本版不再做独立空间增强，而用监督定位作为检索条件。定位准不等于交换有益，必须保留辅助监督和普通交互对照。

原M1接口证据：code/fusion/src/models/components/local_region_residual_fusion.py 的 enhance 接受同形CT/PET，两路增强后相加。新增模块必须返回两路同形特征。

## Method Thesis / Contribution Focus

主张（待验证）：用受监督的软定位同时构造空间候选偏置与通道匹配条件，可以改善双向局部跨模态检索；原始特征和原M1保持直通。不是已经证实的互补性估计或错误纠正器。

一种定位头、一种局部交互单元（内含通道描述）；无固定文本主干、无Top-K/offset/专家解码器/额外纠错阶段。文本仅是后续独立研究，本版不依赖BioMedCLIP。

选择：相较于固定文本Query的重参数化路线，采用可监督的1×1定位头更直接、易辨识。相较于每像素加η的归一化，使用峰值响应控制的GAP混合，避免固定η产生mean(S)/(mean(S)+η)的病灶面积衰减；峰值不代表经过校准的存在概率，假阳性峰值是风险。

## System Graph

CT/PET → 各自编码器 → 原对齐 → 四尺度LPI-BI → 原四尺度M1 → 原GN单解码器 → 分割logit。

各尺度参数独立；两个方向不共享Q/K/V/output。无跨尺度递归、无共享Query。所有视觉参数端到端更新，架构固定不等于冻结参数。

## 固定配置

|尺度|输入宽度|512输入下特征大小|局部候选|总隐宽/头数/每头宽度|通道MLP隐宽|
|---|---|---|---|---|---|
|S1|64|128×128|有效3×3，内部9点|32/4/8|16|
|S2|128|64×64|有效3×3，内部9点|32/4/8|32|
|S3|320|32×32|有效3×3，内部9点|32/4/8|80|
|S4|512|16×16|有效3×3，内部9点|32/4/8|128|

所有新增Conv/Linear都带bias；新增参数量按此公式配置推导为654,608（约0.655M），不含原模型。这是解析计数，尚不是实例化模型统计或运行基准。

## 1. 定位

每尺度每模态独立Conv1×1(D,1,bias=True)，常规初始化，得到logit z 和 S=sigmoid(z)。S形状B×1×H×W，是下采样前景占比响应，不是PET SUV、临床代谢热点或可靠性概率。GT仅进入loss。

## 2. 空间条件通道描述

对每个样本、每路、每尺度计算：

$$
v_G=\frac1N\sum_iX_i,\qquad
v_S=\frac{\sum_iS_iX_i+\epsilon v_G}{\sum_iS_i+\epsilon},\qquad a=\max_i S_i
$$

$$
v=(1-a)v_G+a v_S,\qquad
g=2\sigma(W_2\operatorname{GELU}(W_1v))
$$

FP32池化，epsilon=1e-6是总质量稳定项，不是每像素的η。S全零或均匀时v=vG；单个高响应不因前景面积小而自动被GAP淹没；低峰值时回退GAP。a是可微（几乎处处）的峰值响应，不称为病灶存在概率。

MLP宽度D→max(16,D/4)→D，W2及其bias置零，g初值1。g只用于Q/K输入，原始跳连与V不被此门控缩放。没有第二次文本拼接。

## 3. 局部双向检索

每尺度潜在总宽度32，4头，每头8；Q/K使用各自X*g经独立Conv1×1投影，V使用源路原始X独立投影。暂不添加额外归一化层、FFN、dropout或空间采样器，保持首版机制可辨识。注意力缩放按每头sqrt(8)，不是sqrt(32)。

对接收位置i，源路有效3×3邻域N(i)；边界外候选必须mask，不能将zero/replicate padding当作有效候选。所有局部统计仅在有效候选上计算。

$$
\ell_j=\log(S_j+\epsilon),\quad
a_i^{\mathrm{local}}=\max_{j\in\mathcal N(i)}S_j,\quad
B_{ij}=a_i^{\mathrm{local}}\tanh\left(\ell_j-\frac1{|\mathcal N(i)|}\sum_{k\in\mathcal N(i)}\ell_k\right)
$$

绝对响应因子抑制全低响应邻域中相对噪声的影响；tanh有界且不除以标准差。B每个方向使用源路S，不从GT生成。

$$
e_{ij}^{h}=\frac{Q_i^h\cdot K_j^h}{\sqrt8}+\beta B_{ij},\qquad
\beta=\sigma(b),\quad b_0=\log(0.1/0.9)
$$

每尺度每方向一个beta，四头共用；0<beta<1，初始0.1。beta非负，使更高源响应的偏好含义一致。不额外增加alpha残差系数、reject gate或Null。该Softmax不能主动选择空消息，因此不宣称逐位置拒绝所有无益信息。

$$
A_{ij}^{h}=\operatorname{softmax}_{j\in\mathcal N(i)}e_{ij}^{h},\qquad
M_i=\operatorname{Concat}_h\sum_j A_{ij}^{h}V_j^h
$$

$$
C'=C+W_o^C M_{C\leftarrow P},\qquad
P'=P+W_o^P M_{P\leftarrow C},\qquad
F_s=\operatorname{M1}_s(C',P')
$$

仅Wo权重与bias置零。原C/P直通，零初始化时完整模型等价于同权重M1；定位loss仍改变编码器学习，故不保证训练轨迹或性能不变。

## 固定实现约定与零初始化时序

- 邻域逐像素滑动而非互不重叠的window partition；坐标按行优先，九候选按左上、上、右上、左、中、右、左下、下、右下，中心候选index4。使用dilation1、stride1、padding1的unfold等价实现；还原顺序必须用坐标哨兵校验。
- Q形状B×4×8×N，K/V邻域形状B×4×8×9×N；点积沿每头8维求和，Softmax只沿九候选轴。S邻域B×1×9×N。有效候选掩码为B×1×9×N，与点积后的B×4×9×N分数广播；对K/V等五维张量操作时再加通道轴为B×1×1×9×N，不能将五维掩码直接用于四维分数。不能在合法位置重复计入边缘值。边角4候选、边缘6候选、内部9候选，中心总有效。
- Conv/Linear允许原训练AMP策略。池化、sigmoid用于辅助监督、log、max/mean、点积、Softmax、V聚合与辅助loss在显式禁用autocast的FP32区执行；输出消息再转回主路径dtype进入Wo。任务头预测logit只resize一次以复用原评估流程，不改变阈值。
- 非空判定按该训练样本空间增强后的原始GT中是否有正前景决定，不用soft GT阈值或预测图判定。所有GT只在训练loss出现；验证与推理不传GT给模型。
- 第一个反向：Wo=0意味着分割loss通常可更新Wo与原直通网络，但不能经消息分支更新Q/K/V、beta、通道MLP或定位头。定位辅助loss可以从第一步更新定位头和编码器。这是串联零出口的正常时序，不是首步梯度故障。
- Wo更新为非零后，任务loss可更新检索路径；W2=0使MLP早层W1的任务梯度再延后。必须用多次真实更新核验最终任务梯度到各个预期路径，不能用第一步所有模块非零梯度作为通过条件，也不能因为只看到辅助梯度便声称交互学到了任务。
- Wo置零确保初始完整模型等价于同权重的原M1。只检查这个初始前向结论，不把它解释成训练收益保护。普通交互对照保持同样零出口初始化。

## Training Plan / Gradient Contract

单次端到端训练；仅编码器通用预训练，所有对齐、定位、交互、M1和GN解码器从头。不是加载78.49/78.81分割权重后微调。每轮完整模型验证一次、60轮上限、连续3轮Dice无提升早停、best-only、myenv。无文本编码器、专家或参考解码器。

原分割loss完全复用。辅助GT在全部空间增强结束后AreaResize到该尺度，不阈值化。

$$
L_{\mathrm{loc}}=\frac1{8B}\sum_{m,s,n}\left[\operatorname{BCEWithLogits}(z,Y_s)+I_n\left(1-\frac{2\sum_i p_iY_{s,i}+\epsilon}{\sum_i p_i^2+\sum_iY_{s,i}^2+\epsilon}\right)\right]
$$

BCE先每图平均像素，I_n=1仅在该图GT含正前景；空GT辅助Dice置0，仍计入固定8B分母。epsilon=1e-6，p=sigmoid(z)，辅助loss与统计FP32。

$$
L=L_{\mathrm{seg,original}}+0.1L_{\mathrm{loc}}
$$

默认不detach S或X：分割loss可经Q/K、池化、S、定位头回到编码器；定位loss更新定位头与编码器；V与跳连保持正常梯度。各组E1/E2/E3/E4/E5使用相同定位头、loss及默认梯度策略。

若未来做仅隔离定位loss对编码器的影响，辅助loss用H(X.detach())，交互用H(X)，共享H参数，或精确选择性梯度累计；不能直接把供交互的整张S也来自detached X并称为只切断辅助loss。

## Failure Modes and Diagnostics

- 定位准确但无增益：比较E3/E2、最终FP/FN/边界，而非只看S。
- PET伪候选/无病灶峰值：记录峰值a与响应质量；a不是校准概率；不能将FP称为已证明生理性摄取。
- 小病灶弱响应：原跳连/V不删位置，统计按病灶大小的定位覆盖与最终漏检；这不等于保证小病灶保留。
- 特征增量过大：每尺度原特征/增量RMS、beta、注意力熵、梯度范数；快照不代替验证贡献。
- 辅助梯度冲突：记录主/辅助loss及同参数梯度范数；可选一次离线方向检查，不能以梯度夹角代替任务增益。
- FP32注意力与AMP安全：log、边界mask、Softmax、pool、lossFP32；只有有效候选参与；unfold显存须batch16真实预检，不作未测显存承诺。

## Claim-Driven Validation (three blocks)

1. 辅助监督：B0原M1 vs E1只加8定位头和相同loss，空间图不进入主路径。判断loss对编码器的单独作用。
2. 空间条件：E2=E1+普通双向local attention；E3=E2+源空间偏置B，无通道MLP。E3优于E2才支持空间条件额外价值；E3只优于B0不够。
3. 通道描述：E4=E3+GAP的同结构通道MLP；E5=E4仅将GAP换为上面的空间混合v（完整模型）。判断响应条件化池化相对普通通道调制的作用。

同组普通 attention 与条件 attention 共用可训练参数布局与初始化，未启用的beta固定非训练或显式忽略；记录1个额外标量差异，不拿少量参数差异冒充机制验证。所有公共参数初始化与数据RNG匹配。首轮2023用于开发筛选，不是多种子稳定性结论；不能仅凭一轮或小数点优势认定成功。

## Literature Grounding / Novelty Boundaries

- Jiang等，ZePT，CVPR2024：https://arxiv.org/html/2312.04964v2 。借鉴显式定位/自提示的组织思想；不复现其器官预训练、advanced query或query–knowledge alignment，也不声称本模型是文本语义对齐。
- Zhang等，TMFormer，AAAI2024：https://ojs.aaai.org/index.php/AAAI/article/view/28572 （DOI10.1609/aaai.v38i7.28572）。借鉴肿瘤区域优先而不删除全部非候选的原则；不复现token merge或缺失模态机制。已核验官方检索摘要和方法摘录，直接PDF读取遇访问限制。
- 本版具体池化、局部相对偏置、source raw V与M1残差接口是任务适配，尚未完成穷尽查新，不称首次或顶会创新已经成立。若空间条件不能超过同监督普通交互，核心主张不成立。

## Compute / Execution Boundary

设计阶段已完成确定性数学检查。后续用户已授权实现并运行完整模型一组；实现与GPU预检见LPI_BI_IMPLEMENTATION.md及LPI_BI_GPU_PREFLIGHT.json。每个实验独立目录、配置和提交，沿用开发集评估边界。无新增数据标注成本。

## 定稿状态

本文件固定了V1完整模型（E5）的接口、公式和首版超参，不表示顶会可投、性能保证或查新完成。核心空间条件先用E1/E2/E3检验；如果E3不优于E2，不靠加入文本或通道MLP把核心结论改写为成立；若E5不优于E4，删除空间条件池化的主张与组件。保持原始M1对照和全部不利结果。

CPU方程检查已通过22项：均匀/零响应GAP回退、单峰在多种网格不按面积稀释、池化梯度、padding候选数、先验有界/均匀为零、无效候选权重为零、softmax归一化、软标签Dice最小点、空GT BCE梯度、零出口首步和后续梯度时序。它们不是完整模型、AMP、显存或最终性能验证。详见管理目录同名设计审查的MATH_CHECKS.json。
