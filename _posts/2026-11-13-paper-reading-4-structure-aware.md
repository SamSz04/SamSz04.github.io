---
layout: "post"
title: "论文阅读(4)——Structure-Aware：结构感知融合与动态物化"
date: "2026-11-13 00:00:00 -0400"
permalink: "/2026/11/13/paper-reading-4-structure-aware/"
description: "理解 Transformer 专用融合内核、物化边界搜索及其性能取舍。"
tags: ["Compiler", "GPU", "Operator Fusion", "Transformer"]
categories: ["AI"]
related_posts: true
toc:
  sidebar: left
_styles: |
  #markdown-content { overflow-wrap: anywhere; }
  #markdown-content figure { margin: 1.8rem 0; }
  #markdown-content figure img { display: block; width: 100%; height: auto; border-radius: 6px; }
  #markdown-content figcaption { font-size: .9rem; line-height: 1.75; margin-top: .7rem; opacity: .85; }
  #markdown-content table { display: block; max-width: 100%; overflow-x: auto; }
  #markdown-content pre { max-width: 100%; overflow-x: auto; }
  #markdown-content mjx-container[display="true"] { display: block; max-width: 100%; overflow-x: auto; overflow-y: hidden; padding: .5rem 0; }
  #markdown-content h2, #markdown-content h3 { scroll-margin-top: 6rem; }
---

**这篇论文采用“识别 Transformer 模式 → 使用专用融合内核 → 搜索残差边界的物化位置”的路线。** 最值得理解的是两个层次怎样配合：内核内部减少中间结果落显存，计算图层面决定哪些边界值得保留。它与 Neptune、RedFuser 的自动归约修正路线形成了有用对照。

论文：_Structure-Aware Operator Fusion and Dynamic Materialization for Efficient Transformer Inference_；Ruiting Sun、Honglu He、Guanwen Zhang、Wei Zhou；IJCNN 2026，DOI：10.1109/IJCNN69854.2026.11729434。本文依据 7 页论文 PDF，含全部正文、算法、图表与参考文献；该文件没有附录。

论文报告：RTX 3090 上，ViT-Base 延迟为 15.89 ms，相对 PyTorch 的 24.22 ms 为 **1.52×**；相对另一组 TVM+Ansor 基线的 17.38 ms 为 **1.09×**。这两个数字使用不同基线，不能混用。

同系列阅读：[Neptune]({{ '/2026/11/13/paper-reading-2-neptune/' | relative_url }}) · [RedFuser]({{ '/2026/11/13/paper-reading-3-redfuser/' | relative_url }})。

<h2 id="problem">1. 从一个 Transformer block 看问题</h2>

普通 Transformer 包含两类大的计算链。Attention 先算 score，再缩放、加 mask、softmax，最后乘 value；FFN 先把隐藏维度扩展到原来的四倍，经过 GeLU，再投影回来。若每一步作为独立内核执行，就需要反复启动内核，并把中间结果写入、读出全局内存。

大算子本身计算量高，不代表整条链已高效。一边是矩阵乘的高吞吐，另一边是中间张量流转和较小的逐元素操作；后一类开销可能使整条链达不到矩阵乘的理想效率。作者因此选择直接识别这两类稳定结构，提供专用实现。

然而，内核内融合完成后，图上还有另一种困难。以教学用残差结构为例：

$$
h=\operatorname{FFN}(x),\qquad y=x+h,\qquad z=\operatorname{LayerNorm}(y).
$$

$$x$$ 同时供 FFN 和残差加法使用。将相邻操作合并时，必须保证两条路径都能获得正确的数据，同时考虑片上存储、寄存器和同步成本。保留所有边界容易损失复用，合并所有边界又可能令资源超限。这就是作者提出搜索物化位置的动机。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/overview.png' | relative_url }}" alt="原文图 1，PDF 第 2 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文图 1，PDF 第 2 页，裁剪。按从左到右阅读：识别 Attention/FFN 子图，替换为定制内核，再在残差边界枚举物化配置，最后生成代码。注意“专用内核”和“边界搜索”是两个不同层次。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/overview.png' | relative_url }}">查看原图</a></figcaption>
</figure>

有三条路线值得区分。通用融合规则覆盖面较广，但不一定能生成高效 attention 算法；专用内核能利用领域结构，却需要显式识别和实现模式；图划分搜索能改善边界，却通常依赖已有候选内核。本文把后两者组合起来，并未展示从任意算子表达式自动推导新在线算法的过程。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/graph.png' | relative_url }}" alt="原文图 2，PDF 第 2 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文图 2，PDF 第 2 页，裁剪。左侧的两个矩阵乘夹着 scale/mask/softmax，右侧被标为 Fused_Attention；FFN 的矩阵乘—GeLU—矩阵乘也被合并。残差加法和 LayerNorm 仍然构成需要处理的边界。图中的 GEMM 不全被归入一个大内核。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/graph.png' | relative_url }}">查看原图</a></figcaption>
</figure>

<h2 id="attention">2. Fused_Attention：保留状态，逐块消费 score</h2>

### 2.1 为什么不能简单逐块做 softmax？

设一个 query 的 score 为 $$s=[0,1,-1,2]$$，对应 value 为 $$v=[1,2,3,4]$$。完整 attention 输出是：

$$
o=\frac{\sum_j e^{s_j-m}v_j}{\sum_j e^{s_j-m}},\qquad m=\max_js_j.
$$

如果分别对前两项、后两项做 softmax，再简单平均两个输出，就丢掉了两个分块在全局分母中的相对权重。必须保留足以统一尺度的统计量。

本文算法 1 使用三项状态：当前最大值 $$m$$、指数和 $$\ell$$、未归一化的加权和 $$U$$。原文把这个累计量也记作 $$O$$；为避免与最终归一化输出混淆，这里在推导中改记为 $$U$$。收到下一块 score $$S$$ 和 value $$V$$ 后：

$$
m'=\max(m,\operatorname{rowmax}(S)),\qquad \alpha=e^{m-m'},
$$

$$
P=e^{S-m'},\qquad \ell'=\alpha\ell+\operatorname{rowsum}(P),
$$

$$
U'=\alpha U+PV.
$$

全部块处理完，再算 $$o=U/\ell$$。对每一行，乘 $$\alpha$$ 恰好将历史结果转换到新的最大值参照下，因此无需回读所有历史 score。初始 $$m=-\infty,\ell=0,U=0$$；这里默认至少有一个有效有限 score。全 mask 行等特殊情况需要明确语义，论文算法未展开。

| 教学步骤                                        | 最大值 $$m$$ | 指数和 $$\ell$$ | 未归一化加权和 $$U$$ | 输出 $$U/\ell$$ |
| ----------------------------------------------- | -----------: | --------------: | -------------------: | --------------: |
| 读入 [0,1]，value [1,2]                         |            1 |        1.367879 |             2.367879 |        1.731059 |
| 读入 [−1,2]，value [3,4]；将旧状态乘 $$e^{-1}$$ |            2 |        1.553002 |             5.020455 |        3.232743 |
| 一次性计算全部四项                              |            2 |        1.553002 |             5.020455 |        3.232743 |

这是本文构造并独立重算的例子，不是作者性能测试。它与前两篇精读使用同一组 score，便于直接看出数学基础的重合。

### 2.2 状态怎样映射到 GPU？

Q 沿 query 行分成 $$B_r$$ 行一块。每个 Q 块遍历 K/V 的 token 分块，每块含 $$B_c$$ 个 key/value，生成一个 $$B_r\times B_c$$ 的 score tile。K 转置后表现为列分块，V 本身沿 token 行分块；“K/V 按列分块”的口头描述需要结合矩阵方向理解。

score tile 在寄存器或片上存储中参与最大值、指数和与矩阵乘，随后即可释放；累计输出和统计量跨循环保留。最终只写回输出块。这样消除的是完整 $$N\times N$$ score/probability 矩阵的显存物化。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/attention.png' | relative_url }}" alt="原文图 3、4，PDF 第 3 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文图 3、4，PDF 第 3 页，裁剪。左图表示分块结果逐步累计，右图标出 Q 的行块与 score 的二维 tile。重点是每次只消费一个 score tile，统计量和输出状态跨块保存。左图省略了稳定化的最大值修正，准确更新以算法 1 与正文式 (1)–(2) 为准。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/attention.png' | relative_url }}">查看原图</a></figcaption>
</figure>

算法 1 展示的是熟悉的在线 softmax attention 路径，与 FlashAttention 的基本原理相通。论文明确引用 Online Softmax 与 FlashAttention；此处的贡献定位应是把这一专用模式纳入其优化框架，而不是重新发现在线更新公式。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/algorithm1.png' | relative_url }}" alt="原文算法 1，PDF 第 3 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文算法 1，PDF 第 3 页，裁剪。沿内层循环看四个动作：读 K/V、算 score、修正旧状态、加入当前块；除以指数和发生在所有块处理完之后。算法省略了标题描述的 scale/mask 细节，完整 attention 实现还需在求最大值前把它们应用到 score。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/algorithm1.png' | relative_url }}">查看原图</a></figcaption>
</figure>

原文图 3 还存在一个局部记号问题：第二块的指数和写成再次累加 $$S^{(1)}$$，按图意应使用 $$S^{(2)}$$；它展示的归一化输出表示也与算法 1 的未归一化累计量不同。正文式 (1)–(2) 和算法 1 给出了带最大值修正的清晰主路径，本文按这一路径解释。

### 2.3 “线性内存”与“线性总流量”要分开

论文第 4 页称全局访存仅为读取 Q/K/V、写出 O，从而为 $$O(Nd)$$。**按它实际列出的双层循环，这个总流量结论缺少必要条件。** 每处理一个 Q 行块，都会再遍历全部 K/V。忽略缓存复用，假定整除、每个元素按一次访问计数，算法层面的全局加载/存储量约为：

$$
2Nd+2\frac{N}{B_r}Nd=2Nd+\frac{2N^2d}{B_r}.
$$

第一项是 Q 与 O，第二项是多轮 K/V 加载。缓存、跨块复用和具体调度会影响实际显存流量，这个式子也不是硬件实测值；但它足以说明，消除 $$N^2$$ 中间矩阵不等于 K/V 各只读一次。

例如教学配置 $$N=2048,d=64,B_r=64$$，Q 有 32 块；上述模型中 K/V 的加载总量是单次读入 K/V 的 32 倍。论文没有提供能证明这些读取全部被缓存消除的流量测量。更准确的成果描述是：**避免二次规模中间张量物化，通过分块降低数据搬运；稠密 attention 的计算量仍为 $$\Theta(N^2d)$$。** 图 6 的有限区间延迟趋势不能据此改写算法复杂度。

<h2 id="ffn">3. Fused_FFN：用片上容量换掉扩维中间张量</h2>

令 $$M=BN$$，将 token 展成二维输入。采用与算法 2 的权重维度一致的行向量约定：

$$
H=\operatorname{GeLU}(XW_1+b_1),\qquad Y=HW_2+b_2,
$$

$$
X\in\mathbb{R}^{M\times d},\quad W_1\in\mathbb{R}^{d\times4d},\quad H\in\mathbb{R}^{M\times4d},\quad W_2\in\mathbb{R}^{4d\times d}.
$$

$$H$$ 的元素数是输入的四倍。算法 2 沿 batch 和 sequence 选择小块，将第一层投影和激活结果保存在 `Hlocal`，同步线程后直接供第二层投影使用。与仅把 GeLU 并入第一层 GEMM 尾部相比，本文希望连第二层 GEMM 也一起融合，从而完全免掉 $$H$$ 的全局写回与读入。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/ffn.png' | relative_url }}" alt="原文算法 2 与图 5，PDF 第 4 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文算法 2 与图 5，PDF 第 4 页，裁剪。左侧先生成 Hlocal、同步，再执行第二层；右侧红框覆盖两个线性层及中间激活。关键是激活张量跨两个 GEMM 在片上存活。算法是高层描述，未展开矩阵乘的线程协作与 Tensor Core 映射。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/ffn.png' | relative_url }}">查看原图</a></figcaption>
</figure>

### 3.1 节约的字节怎样算？

如果参考实现已融合第一层 bias/GeLU，但仍把 $$H$$ 写回再读入，移除这一次物化可节约：

$$
\Delta B_H=2\cdot M\cdot4d\cdot b=8Mdb\quad\text{bytes},
$$

$$b$$ 是每元素字节数。这是明确的中间张量流量，不包括权重、其他激活临时量、缓存和重复加载。

原文说“全局内存流量减半”，但没有给出支持固定比例的完整账本。若作一个理想化模型，X/Y 各访问一次、两组权重各读一次，则融合前后分别是：

$$
B_{\mathrm{before}}\approx(10Md+8d^2)b,\qquad B_{\mathrm{after}}\approx(2Md+8d^2)b.
$$

节约比例随 token 数和权重读取而变，并不恒为一半。真实实现中权重重读、缓存命中及分块不同，还会进一步改变比例。这个模型只用于说明归因，不能替代 profiler。

### 3.2 真正的难点是容量与计算效率一起满足

设一个 tile 有 $$t=T_bT_n$$ 个 token，仅扩维状态就占：

$$
B_H=4tdb.
$$

以教学配置 $$d=768$$、FP32 为例，$$t=4$$ 时 Hlocal 为 48 KiB；$$t=8$$ 时达到 96 KiB，尚未计入输入、权重 tile、输出累计量及流水线缓冲。将部分状态放寄存器会增加寄存器压力，并可能降低占用率。

把 token tile 缩小容易满足容量，却可能让 GEMM 太窄，损失权重复用与矩阵乘吞吐。这是这类融合的关键取舍。原文强调分块适配共享内存，但没有报告具体 $$T_b,T_n$$、共享内存使用、寄存器数、occupancy 或 Tensor Core 调度。因而它交代了设计原理和测得的收益，却还不足以直接复现高性能内核。

此外，第二层依赖完整 GeLU 结果；不能因为都是线性投影就随意交换归约与激活。本文采用片上保留中间值来满足依赖，并未提供类似 Neptune 的代数重写求解器。

<h2 id="materialization">4. Dynamic Materialization 到底搜索什么？</h2>

物化指把中间结果存成可供后续使用的张量；本文关注全局内存中的物化边界。物化允许不同内核独立调度、便于多个消费者共享；融合则有机会让数据保持在寄存器/共享内存，并减少 launch。

作者对竞争子图的入口、出口各放一个布尔变量 $$l,r$$，枚举四种配置：

| $$(l,r)$$ | 入口           | 出口           | 候选方案的含义                                   |
| --------- | -------------- | -------------- | ------------------------------------------------ |
| (真,真)   | 物化           | 物化           | 保留两侧边界，给内核与消费者更多独立性           |
| (假,真)   | 尝试消除边界   | 物化           | 允许与前驱融合，仍写出子图结果                   |
| (真,假)   | 物化           | 尝试消除边界   | 允许与后继融合，如残差加法/归一化                |
| (假,假)   | 两侧均尝试消除 | 两侧均尝试消除 | 最大化跨边界融合机会，同时施加更强资源与依赖约束 |

这里“假”只是允许构造相应候选，能否融合仍由合法性和实际规则决定；论文没有给出每个残差子图在四种配置下的完整展开图。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/search.png' | relative_url }}" alt="原文算法 3，PDF 第 5 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文算法 3，PDF 第 5 页，裁剪。两个布尔循环最多产生四种局部配置；先通过 IsCompatible，再应用融合规则、估计延迟并选择。按这个伪代码，它逐个更新竞争子图，而非同时穷举整图所有边界的组合。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/search.png' | relative_url }}">查看原图</a></figcaption>
</figure>

算法流程是：识别竞争子图与前后节点，枚举配置，检查依赖和资源限制，构造候选图，应用融合规则，按成本模型选择预测延迟最小的有效配置。原文称模型考虑全局内存流量与 kernel launch 开销；没有给出拟合公式、校准数据或预测误差。

### 4.1 为什么菱形图会让融合变复杂？

残差图本身是有向无环图；多消费者不会自动制造环。问题来自不合适的分组。例如有 $$x\to G\to A$$ 和 $$x\to A$$ 两条路径，若把非相邻的 $$x$$ 生产者与 $$A$$ 合成一组，却把 $$G$$ 放在组外，组间可能形成“融合组需要 G，G 又需要融合组输出”的循环。其他候选则可能保持无环，却因同时保留太多值导致 spill。

因此合法性检查和收益判断必须分开。论文的 `IsCompatible` 是接口级描述；它没有展开边界识别、图重写、寄存器溢出估计及重计算生成的具体实现。作者提到重算小段可能比读显存便宜，但算法 3 没有显式的重计算决策变量或代码生成步骤，不能据此把它解释成完整的自动重计算优化器。

### 4.2 “动态”应怎样理解？

从论文展示的流程看，搜索发生在优化图并生成代码的阶段，依赖模型结构和硬件约束。“动态”主要是相对于固定启发式边界，按候选配置进行选择。**文中没有展示请求执行期间根据输入数据、在线负载或运行时形状重新决策的机制。** 因此适合称为“基于候选搜索的物化选择”，不宜直接归类为运行时自适应调度。

若有 $$k$$ 个竞争子图，算法表面最多评估约 $$4k$$ 个局部候选，每次评估本身还有图检查和成本估计开销。完整联合空间可达 $$4^k$$；顺序更新不保证找到全局最优，且可能受子图处理顺序影响。作者声称的最低延迟，准确地说是在所枚举的有效局部候选中、按其成本模型做出的选择。

<h2 id="experiments">5. 实验：哪些结果最有说服力？</h2>

实验硬件是 RTX 3090 与 Intel Xeon Silver 4214R；软件基于 TVM v0.19.dev0、ONNX 1.17。TVM 比较基线启用 Ansor，论文称每个算子 10000 次调优迭代，并保留默认融合启发式。模型来自 timm，包括 ViT、DeiT、Swin。

微基准设 batch 为 32/64，序列长度为 128、256、512、768、1024、2048，精度为 FP32/INT8，head dimension 为 64，FFN 扩维倍数为 4。未明确给出完整 head 数与 FFN 输入隐藏维度、端到端 batch/分辨率/精度、GPU 计时重复数、误差条以及全部比较框架的配置。这些缺项限制结果的可复现性，尤其影响小幅差异的解释。

### 5.1 ViT-Base 跨框架比较

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/cross-compiler.png' | relative_url }}" alt="原文表 II，PDF 第 5 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文表 II，PDF 第 5 页，裁剪。时间越小越好；Speedup 统一以 PyTorch 耗时除以各实现耗时计算。15.89 ms 与 Triton 的 16.13 ms 很接近，需要结合计时波动和配置理解。Size 为作者定义的 compiled artifact size，未给出组成拆分。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/cross-compiler.png' | relative_url }}">查看原图</a></figcaption>
</figure>

| 系统      | 延迟（ms） | 相对 PyTorch 加速 | 报告产物大小（MB） |
| --------- | ---------: | ----------------: | -----------------: |
| PyTorch   |      24.22 |             1.00× |                344 |
| ONNX-MLIR |      19.62 |             1.23× |                292 |
| Triton    |      16.13 |             1.50× |                239 |
| 本文方法  |      15.89 |             1.52× |                231 |

重算 $$24.22/15.89=1.5242$$，支持 1.52×。相对 Triton 的延迟下降是 $$(16.13-15.89)/16.13=1.49\%$$；产物大小相对 PyTorch 减少 32.85%。

较小产物是论文报告的现象，但“由图裁剪或代码生成更高效导致”的因果解释没有组成数据支撑。需要知道大小是否包含权重、运行时、缓存与依赖。原文也未说明这里的 Triton 端到端系统怎样构建，因此这个对照不能等同于已知某个标准 Triton Transformer 实现。

### 5.2 单个内核的收益

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/attention-perf.png' | relative_url }}" alt="原文图 6 与表 III，PDF 第 6 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文图 6 与表 III，PDF 第 6 页，裁剪。上方按 batch 和精度分组，蓝柱为原 Self-Attention，红柱为融合实现。最长序列处差距最大；下表给出选定长度的耗时比。曲线是有限测试点的实测，不能用于证明稠密 attention 已成为线性复杂度。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/attention-perf.png' | relative_url }}">查看原图</a></figcaption>
</figure>

Attention 在 $$N=2048$$、batch 32、FP32 时从 2.519 ms 到 0.916 ms，为 2.75×；batch 64、INT8 时从 1.473 ms 到 0.396 ms，为 3.72×。趋势符合减少大中间张量搬运的预期。不过没有与 FlashAttention/FlashInfer 的同条件对照，也没有流量、带宽或 Tensor Core 利用率测量，因而证明的是相对该基线的效果，不能据此判断专用 attention 内核的领先程度。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/ffn-perf.png' | relative_url }}" alt="原文图 7 与表 IV，PDF 第 6 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文图 7 与表 IV，PDF 第 6 页，裁剪。四组测试中的 FFN 融合普遍更快，典型提升约 1.4–1.6×；但 batch 32、INT8、N=128 只有约 1.24×。读图应同时看形状和精度，而非把统一倍数应用于全部配置。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/ffn-perf.png' | relative_url }}">查看原图</a></figcaption>
</figure>

FFN 在 batch 64、INT8、$$N=2048$$ 时从 2.151 ms 到 1.353 ms，约 1.59×。这支持“片上保存扩维状态的实现能带来收益”。另有一处图表不一致：图 7 的 batch 32、INT8、N=2048 标为 0.848 ms 与 0.512 ms，重算为 1.656×，而表 IV 写成 1.58×；现有材料不足以确定哪一处标注有误。原文进一步称其由 memory-bound 转为 compute-bound；仅凭耗时尚不足以确认瓶颈迁移，需要 roofline 或硬件计数器证据。

原文还报告最大相对误差在 $$10^{-5}$$ 内，但未交代参考实现、测试分布、近零分母处理、逐精度统计和 INT8 量化尺度。INT8 配置不意味着 softmax 与 GeLU 全程在整数域中执行；量化/反量化和累计精度需要另外说明。该数值也不能替代 ImageNet 分类准确率的完整对照，文中没有给出相应准确率表。

### 5.3 消融究竟证明了什么？

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-4-structure-aware/ablation.png' | relative_url }}" alt="原文表 V，PDF 第 7 页，裁剪" loading="lazy" data-zoomable>
  <figcaption>原文表 V，PDF 第 7 页，裁剪。+Attn 与 +FFN 分别只启用一种内核；+Full 同时启用两种内核和动态物化。缺少“两种内核都启用，但不启用动态物化”这一行，因此无法单独隔离物化搜索收益。 <a href="{{ '/assets/img/posts/paper-reading-4-structure-aware/ablation.png' | relative_url }}">查看原图</a></figcaption>
</figure>

| 模型       | TVM 基线 | +Attn |  +FFN | +Full | Full 加速比（重算） |
| ---------- | -------: | ----: | ----: | ----: | ------------------: |
| ViT-Small  |     8.49 |  7.84 |  8.26 |  7.69 |              1.104× |
| ViT-Base   |    17.38 | 16.47 | 17.24 | 15.89 |              1.094× |
| DeiT-Small |     7.86 |  7.68 |  7.74 |  7.53 |              1.044× |
| DeiT-Base  |    15.43 | 15.13 | 15.34 | 14.92 |              1.034× |
| Swin-Tiny  |    10.37 |  9.39 | 10.13 |  9.17 |              1.131× |
| Swin-Small |    18.41 | 16.82 | 17.93 | 16.65 |              1.106× |

这张表清楚支持：Full 在六个模型上均优于该 TVM 基线，也优于各自只启用一种融合内核的版本。但 **Full 相对 +Attn 的差值同时包含 FFN 融合、物化搜索及其相互作用**，不能直接当作 Dynamic Materialization 的收益。相对 Triton 的 1.5% 差异也没有通过控制变量证明由动态物化单独造成。

作者称完整系统具有“超出简单相加”的协同。用表中延迟做一个有限的核查，定义：

$$
\Delta_{\mathrm{extra}}=(T_0-T_{\mathrm{full}})-[(T_0-T_{\mathrm{attn}})+(T_0-T_{\mathrm{ffn}})].
$$

ViT-Base 得到 +0.44 ms，但 ViT-Small、Swin-Tiny、Swin-Small 分别为 −0.08、−0.02、−0.31 ms。因此“所有模型均超加性”并不成立。这种差值本身也不能隔离搜索收益，因为单项优化的影响未必可加。

要验证独立贡献，最直接的补充是加入 `Attn+FFN、固定边界`，再与 `Attn+FFN、搜索边界` 比较，同时列出各模型选择的 $$(l,r)$$、实测与预测延迟、搜索耗时和额外编译成本。它们是当前实验解释所缺的证据。

<h2 id="comparison">6. 与 Neptune、RedFuser 放在一起看</h2>

| 问题             | 本文                                | Neptune                             | RedFuser                             |
| ---------------- | ----------------------------------- | ----------------------------------- | ------------------------------------ |
| 主要优化对象     | 已识别 Attention/FFN 模式及其图边界 | 生产者—消费者归约链与循环调度       | 级联归约树、层级合并与增量执行       |
| 在线公式从哪里来 | 专用 attention 算法直接采用         | 符号构造并验证修正函数              | 分解表达式并生成修正/合并            |
| 自动化重点       | 图模式识别、四配置物化选择          | 用户 schedule 中的可组合变换        | 归约分析、在线表达式与 tile 程序生成 |
| 资源问题         | 中间张量是否落显存，片上容量是否够  | 状态、分段、并行与后端映射          | 存储容量与增量/非增量方案的可行域    |
| 主要实验证据     | 单 RTX 3090，视觉 Transformer 推理  | 多设备 attention 推理与部分其他归约 | attention、routing、量化等推理子图   |

这篇更适合放在你原先框架的**“专用结构融合 + 物化/图划分搜索”**线上。发表时间在 2026 年，不意味着它就属于“自动代数重写、SM 任务化或跨 GPU”这几类贡献。它说明规则、专用内核、图搜索和算法重写在同一时期仍会并存。

本文相关工作把 Neptune 描述成 unified space mappings；结合前次精读，更准确的定位应是归约修正和 loop-to-tile 变换。这也提醒我们，不能只根据一篇论文对前作的一句话来替代前作本身的机制分析。

三篇的连接可以这样理解：Neptune/RedFuser 扩展“哪些算法形式可以成为候选”，本文的物化问题关注“候选内核和相邻子图怎样连接”。二者有潜在组合价值，但本文没有实现或评估与这两个系统的集成。

<h2 id="research">7. 从这篇论文走向更大的研究问题</h2>

这篇推动的是领域结构与图边界决策的结合。更远的目标是：编译器在给定模型、形状和硬件下，同时选择算法、片上状态和跨内核数据流，使端到端成本最小。下面是本材料提出的研究方向，不是原文已验证的成果，也不据此宣称领域内无人研究。

### 7.1 将算法形式与物化位置联合搜索

同一个 attention 子图可以有传统分阶段、在线单段、并行分段等形式；改变算法形式后，产生的临时状态、并行任务和出口张量也会改变。若先固定内核、再选边界，就可能提前排除全局更好的组合。

研究问题是如何把“候选算法状态”纳入图级决策。第一步可选一个 attention—残差—归一化小图，准备少量经验证的在线/分段候选，联合枚举物化和分块，比较分阶段决策与联合决策的差距。核心难点在于候选规模、跨子图资源耦合，以及如何给算法变换保留可检查的语义条件。

### 7.2 让物化成本体现容量、并行度与共享复用

消除一条显存边，可能增加寄存器存活时间、缩小 tile、导致 spill，也可能重复加载权重；保留边界则可能换来更高效 GEMM。领域目标应是选择最低总执行成本，而非最少物化点。

第一步可在不同隐藏维度、token 数和分支数下，系统测量融合与分离实现的转折点，记录共享内存、寄存器、cache 命中、kernel launch 与矩阵乘吞吐。研究难点是将这些相互作用压缩成可迁移、可校准的成本模型，同时能识别“预测不确定，应实测”的候选。

### 7.3 从局部边界选择走向带正确性约束的全图优化

局部 $$(l,r)$$ 决策容易实现，但重叠子图会共享张量与资源，局部最优可能互相冲突。多消费者场景还牵涉复用与重计算，不能只用一个“融合/不融合”标志覆盖全部语义。

可将物化、复制、重计算与融合分组分开表示，建立无环和资源约束；在小图上用穷举获得参考最优解，再比较局部搜索、束搜索或约束求解的质量与编译时间。重要产出不仅是更低延迟，还包括可解释的失败条件和正确性验证接口。

### 7.4 面向实际服务负载的自适应执行

本文搜索尚未展示运行时自适应。更大的机会是根据请求 batch、序列长度、并发和显存压力选择不同预编译方案，避免一个静态方案覆盖所有状态。

第一步可以离线产生少量已验证配置，再在可控形状变化下测量切换策略对吞吐与尾延迟的影响；明确计入选择开销、缓存暖机和额外代码体积。困难是硬件竞争与缓存状态会让离线最优失效。这条研究路线与论文的编译阶段边界枚举相关，但需要新的运行时机制与服务级实验。

<h2 id="assessment">8. 阅读后的总体判断</h2>

**可以肯定的部分**是，本文把在线 attention、双线性层 FFN 融合和残差边界选择组织成一个清楚的优化框架，并报告了单内核及视觉 Transformer 端到端收益。对于理解“减少中间结果物化”这一条优化主线，它提供了直观的实例。

**证据较弱的部分**是，最具区别性的 Dynamic Materialization 只给出高层枚举算法，没有完整成本模型、边界策略实例和独立消融；内核实现细节与数值/计时设置也不充分。关于线性总访存、流量减半和瓶颈迁移的表述需要收窄。适合引用其框架和报告结果，引用时应保留基线、硬件和证据范围。

用一句话概括：**这是“专用内核与图边界选择协同”的系统方案，贡献重心与自动生成归约算法的 Neptune、RedFuser 不同；理解这一区别，比按年份把它们归为同一技术阶段更有帮助。**

<h2 id="sources">9. 来源、覆盖与核查记录</h2>

本文依据同名论文 PDF，7 页，IEEE IJCNN 2026；[DOI 链接](https://doi.org/10.1109/IJCNN69854.2026.11729434)。已完整读取 §I–V、参考文献 [1]–[20]、算法 1–3、图 1–7、表 I–V。文件没有附录；未取得额外补充材料。参考文献列表阅读不等于将其中所有前作再次全文阅读。

| 原文内容                           | 本材料位置                        |
| ---------------------------------- | --------------------------------- |
| §I–II、图 1–2                      | §1 动机与框架；§6 定位            |
| §III-A、图 3–4、算法 1、式 (1)–(2) | §2 状态推导、教学例子和访存核对   |
| §III-B、图 5、算法 2、式 (3)       | §3 FFN 容量、流量与实现取舍       |
| §III-C、算法 3                     | §4 搜索空间、合法性与“动态”的含义 |
| §IV-A、表 I                        | §5 硬件与比较口径                 |
| §IV-B、图 6–7、表 II–IV            | §5 跨框架和微基准结果             |
| §IV-C、表 V                        | §5 独立消融与协同证据             |
| §V 与全文讨论                      | §7–8 研究空间及评价               |

文中未提供作者代码仓库、实现文件或 artifact 链接；本次未取得与论文对应的作者实现。因此关于实现的解释以正文和伪代码为限，没有声称核查实际 CUDA/TVM 源码。与 Neptune、RedFuser 的对照基于两篇论文的全文及固定版本源码核查。

图片为该 PDF 的局部裁剪，版权归原作者/IEEE，保留原图数据与图例，中文图注由本材料补充；引用用于本文的研究讨论与方法讲解，不附原 PDF，也不将其标为开放许可。图 3 的记号不一致在正文指出，没有修改原图。

教学 attention 的状态、表 II/V 的速度比、协同差值及容量/流量示例已用独立计算核对，本文按对应公式独立重算了这些数值。本材料未运行 GPU 内核、未复现论文基准，也未验证作者所称的 INT8 数值误差。

本文基于论文全文、原图与已注明的代码证据整理，使用 AI 辅助梳理和数值核查。性能结果来自论文；补充推导和教学例子不代表 GPU 实测。

---

[下载本篇离线阅读包（HTML、Markdown 与图片）]({{ '/assets/downloads/paper-reading-4-structure-aware.zip' | relative_url }})。离线包的公式预渲染为 MathML。
