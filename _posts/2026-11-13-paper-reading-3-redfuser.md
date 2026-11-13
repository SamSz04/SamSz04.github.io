---
layout: "post"
title: "论文阅读(3)——RedFuser：级联归约的层级合并与增量执行"
date: "2026-11-13 00:00:00 -0400"
permalink: "/2026/11/13/paper-reading-3-redfuser/"
description: "从归约树、表达式分解和增量状态理解 RedFuser 的自动融合方法。"
tags: ["Compiler", "GPU", "Operator Fusion"]
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

**核心判断：RedFuser 用归约树与可分解表达式统一局部计算、结果修正和增量更新，并将推导结果交给 TileLang 生成 GPU 内核。** 其主实验报告 MHA 平均达到 FlashAttention2 的 1.09×、MLA 达到 FlashMLA 的 1.02×；更大的倍数主要来自与通用编译器的子图比较。

同系列阅读：[Neptune]({{ '/2026/11/13/paper-reading-2-neptune/' | relative_url }}) · [Structure-Aware]({{ '/2026/11/13/paper-reading-4-structure-aware/' | relative_url }})。

<h2 id="problem">1. 为什么普通分块仍受级联依赖限制？</h2>

单个归约容易并行：把输入分段，各段算局部和或局部最大值，再合并。难点在于第二个归约使用第一个归约的最终结果。attention 中，指数和依赖最终最大值，加权求和又依赖最大值与指数和；量化中，矩阵乘的输入缩放依赖输入的绝对最大值。

如果每个归约独立完成整棵树，再开始下一棵树，就可能重复读取输入、保存大量中间结果，并在全局完成点上等待。传统融合能把一些计算放到一起，却不自动消除这种“等前一棵树收敛”的依赖。手工在线算法可以绕开等待，但缺少统一的自动生成路径。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig1.png' | relative_url }}" alt="原文图 1（PDF 第 2 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 1（PDF 第 2 页，裁剪）。上图给出有依赖的多个归约，下图展开成归约树链。关注后一棵树对前一棵树根结果的依赖：这正是作者希望改写的执行约束。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig1.png' | relative_url }}">查看原图</a></figcaption>
</figure>

RedFuser 的洞察是：局部块可以先采用局部统计量，等合并到更高层时，再把局部输出修正到统一统计量下。若这种修正只需要小状态，就有机会把多棵依赖的归约树组织成一棵携带联合状态的树。

<h2 id="decomposition">2. 将依赖表达成可撤销、可重设的参数</h2>

论文 §3 将第 $$i$$ 个归约表示为：

$$
d_i=\operatorname{Reduce}_{\oplus_i}F_i(x_j,D_i),
$$

其中 $$D_i$$ 是它依赖的前序归约结果。融合的关键条件之一是：

$$
F_i(x,D)=G_i(x)\otimes_i H_i(D).
$$

这里 $$G_i$$ 只依赖输入，$$H_i$$ 只依赖前序统计量。再要求合适的结合、交换、单位元及分配性质，使同一修正可以作用于整个局部归约。推导还需要相关 $$H_i$$ 值可逆，或者提供具有正确语义的专门处理。

对指数和：

$$
F(x,m)=e^{x-m}=e^x\cdot e^{-m},
$$

所以 $$G(x)=e^x$$，$$H(m)=e^{-m}$$。这是符号层的分解；最终实现可以直接使用差值的指数，保持数值稳定。

假设块 $$b$$ 用自己的局部统计量 $$D_b$$ 算出 $$d_b$$，将它转换到统一参数 $$D$$ 下：

$$
\widetilde d_b=d_b\otimes H(D_b)^{-1}\otimes H(D).
$$

其含义是“撤销旧参数影响，再施加新参数影响”。把转换后的局部结果用原归约操作合并，就得到使用共同参数的结果。这是 §3.2 式 (11) 的核心。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig3.png' | relative_url }}" alt="原文图 3（PDF 第 4 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 3（PDF 第 4 页，裁剪）。上图每个归约阶段分别读取输入并等待前序结果；下图把同层局部状态连接起来。读图重点是依赖从前序归约的最终根结果转移到局部/同层状态及合并修正，而不是所有依赖都凭空消失。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig3.png' | relative_url }}">查看原图</a></figcaption>
</figure>

沿用输入 $$[0,1,-1,2]$$，分成两块 $$[0,1]$$、$$[-1,2]$$：

| 块  | 局部最大值 |   局部指数和 | 转到全局最大值 2 的系数 |
| --- | ---------: | -----------: | ----------------------: |
| A   |          1 | $$1+e^{-1}$$ |              $$e^{-1}$$ |
| B   |          2 | $$1+e^{-3}$$ |                       1 |

两块合并为 $$e^{-1}(1+e^{-1})+1+e^{-3}=1.553002$$。这是同一条修正规律在“分段并行”中的应用。归约树可以递归合并更大的分段，而不要求每个局部计算事先知道最终最大值。

<h2 id="incremental">3. 从树形合并到增量形式：为何多做计算反而打开了新方案？</h2>

树形合并仍可能要求保存上一层的全部输出。RedFuser §3.3 进一步把合并改写为逐段更新。为简化符号，以下展示乘法修正和求和归约的情形：

$$
d_{\mathrm{new}}=d_{\mathrm{old}}\frac{H(D_{\mathrm{new}})}{H(D_{\mathrm{old}})}+d_b\frac{H(D_{\mathrm{new}})}{H(D_b)}.
$$

需要保存旧累计状态、当前块状态，以及新旧参数。这使相关中间状态相对归约长度保持有界；论文所说的 $$O(1)$$ 是相对于分段数量或归约长度，不是整个程序的总内存常数。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig4.png' | relative_url }}" alt="原文图 4（PDF 第 5 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 4（PDF 第 5 页，裁剪）。上图等待全部局部结果并保存它们，底部缓存随输入增长；下图每来一个块就更新累计结果，保留的状态不随归约长度增长。由此省下的是中间状态容量，代价是反复执行修正。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig4.png' | relative_url }}">查看原图</a></figcaption>
</figure>

这解释了两条实现路线：

- **Single-Segment**：一个 CTA 对其负责的归约任务逐块更新。减少跨 CTA 的合并与同步，适合独立任务已足够多的情形。
- **Multi-Segment**：多个 CTA 各自处理一段，然后合并局部结果。增加任务数量，适合 decode 等并发不足的情形。

数学层的“归约树节点”、逻辑层的“输入分段”、程序层的“tile”、执行层的“CTA”并不是同一个对象。实际映射由策略与参数决定；CTA 也不是永久固定的一颗 SM。分段数增大能增加并行任务，同时也增加局部结果保存与合并成本。

论文 §5.4 最重要的结论是：**在相同并行度下，非增量计算可能更快；增量计算的价值在于允许更长分段、更小存储和新的并行度配置。** 它扩展了可行优化空间，而不是保证每个配置都变快。

<h2 id="compiler">4. 编译器如何自动生成公式和内核？</h2>

### 可分解性检查

对一个归约，先根据其类型选取候选结合操作。例如求和使用乘法修正，极值类可以利用共同平移等兼容操作。然后选一个使参考值可逆的固定点 $$(x_0,D_0)$$，检查符号恒等式：

$$
F(x,D)\otimes F(x_0,D_0)=F(x,D_0)\otimes F(x_0,D).
$$

成立时可构造：

$$
G(x)=F(x,D_0),\qquad H(D)=F(x_0,D)\otimes F(x_0,D_0)^{-1}.
$$

“固定点”用来构造分解，不是用一次数值采样代替证明。该方法把任意函数搜索限制到常见归约结构中，实际支持能力取决于符号化、候选运算和化简器。

### 表达式成为程序

论文描述的路径是：模型进入 TVM Relax，识别归约子图，再降低到 TIR；提取归约轴、表达式与依赖，生成融合/增量表达式；构造单分段和多分段的标量循环；最后 tensorization 为 tile 运算，通过 TileLang 生成设备代码。

| 阶段            | 输入 → 输出                 | 关键作用                               |
| --------------- | --------------------------- | -------------------------------------- |
| 归约分析        | TIR → 归约表达式与依赖      | 知道哪些统计量属于生产者，沿哪一轴归约 |
| 符号分解        | 表达式 → $$G,H$$ 与修正公式 | 决定是否能用已有状态完成修正           |
| 在线程序生成    | 公式 → 带旧/新状态的循环    | 安排保存、修正和归约的顺序             |
| 分块及数据放置  | 标量循环 → TileOp           | 控制寄存器/共享内存、拷贝与矩阵乘      |
| TileLang 与调优 | tile 程序 → GPU 内核        | 选择硬件指令、流水深度、线程数及分段数 |

对于 attention，附录 A.4 明确先识别四个归约，重点处理共享 KV 归约轴的后三个；$$QK^T$$ 的内积沿 head dimension 进行，作为每块 score 的生成者嵌入执行。因此“把所有归约看成同一条轴”会丢失重要区别。

已核对的公开入口 `example/flash_attention.py` 从 TE 定义出发，显式设置四个维度的 tile 大小为 64，再串联在线表达式生成、分块、缓冲区变换和 TileLang 生成。这支撑了 TIR 及其后续路径；该示例本身不展示任意 PyTorch 模型自动完成全部 Relax 图划分。`generate_online_expr_pass.py` 按连续且归约长度相同的 reduction block 分组；split 分支要求恰好一个级联组，并注明尚不处理该结构之外的其他算子。因此论文的总体架构、示例入口与当前 pass 的支持范围需要分别理解。

<h2 id="applications">5. 三类应用分别体现什么？</h2>

### Attention：统一局部统计量

局部状态可以写成 $$(m,\ell,u)$$：最大 score、指数和、未归一化的 value 加权和。共同乘以 $$e^{m_b-m}$$ 后合并。RedFuser 的部分公式维护归一化输出 $$O=u/\ell$$，所以还出现新旧 $$\ell$$ 的比例。两种状态表示都可行，但实现时必须明确 $$O$$ 是否已经除过分母。

### MoE routing：利用排序不变性

softmax 对同一行保持顺序，因此 top-k 专家索引可以从原始 score 选择，同时维护最大值与指数和。若还需要完整 softmax 对应的权重，选出的 score $$s_k$$ 应转换为：

$$
w_k=\frac{e^{s_k-m}}{\ell}.
$$

这体现了另一种收益：并非每个下游操作都必须等待归一化之后才执行。论文附录 A.2.2 将最后一步写成 $$s/\ell$$，与其前面把 $$s$$ 定义为原始 top-k score 的符号不一致；阅读时应以实际输出语义区分 logits、指数值和归一化权重。

### FP8 Quant + GEMM：代数尺度与量化语义需一起看

§3.4 用绝对最大值 $$m$$ 产生缩放，推导形如 $$\sum_j (M_{\max}A_j/m)W_j$$ 的修正。实数尺度可以在局部矩阵乘结果上变换，但真实 FP8 包含离散舍入：

$$
Q_{\mathrm{FP8}}(a/s_1)\,s_1\quad\text{与}\quad Q_{\mathrm{FP8}}(a/s_2)\,s_2
$$

一般不相等。局部尺度量化再修正，不自动保留全局尺度量化的结果。因而应把“实数表达式等价”和“量化实现的数值误差可接受”分别验证。这决定了该路径在部署时能否满足给定模型的精度要求。

公开的 `generated_redfuser_ptpc_quant_gemm.py` 确实随块更新输入最大值、按当前尺度转成 FP8，并用新旧最大值比例修正旧累积结果。配套测试将它与全局 per-token 尺度的量化结果比较，输入为 $$4096\times4096$$ 的 `0.01*randn`，输出断言采用 `rtol=1e-2, atol=1e-1`。这提供了一种带容差的测试方式；本次只阅读代码、没有运行该 GPU 测试，且此测试不能替代分布变化与模型精度验证。

<h2 id="experiments">6. 性能结果与取舍证据</h2>

主实验在 A10 24GB 和 H800 80GB 上进行，CUDA 12.8；比较 PyTorch 2.7 Eager、torch.compile/Inductor、TVM 0.21 以及可用的手写库。TVM 使用默认优化管线，明确关闭 CUTLASS 和 FlashInfer 后端。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig5.png' | relative_url }}" alt="原文图 5（PDF 第 10 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 5（PDF 第 10 页，裁剪）。四张图分别是 MHA、MLA、MoE routing、FP8 Quant+GEMM；纵轴均以 PyTorch Eager 归一化。前两张有专家库对照，后两张主要与通用编译器对照，因此不同子图的柱高不能直接当成同一种竞争强度。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig5.png' | relative_url }}">查看原图</a></figcaption>
</figure>

| 子图与设备            | 文中报告结果                                     | 比较口径                                         |
| --------------------- | ------------------------------------------------ | ------------------------------------------------ |
| MHA / A10             | 平均为 FlashAttention2 的 1.09×                  | 手写库对照；LLaMA 配置对 Dynamo/TVM 为 2.8×/2.6× |
| MLA / H800            | 为 FlashMLA 的 1.02×；对 Dynamo/TVM 为 2.4×/8.7× | query 长度固定为 1                               |
| MoE routing / A10     | 对 Dynamo/TVM 为 1.7×/6.6×                       | 未提供同范围专家 routing 内核比较                |
| FP8 Quant+GEMM / H800 | 对 Dynamo/TVM 为 3.4×/12.1×                      | 数值语义需与量化路径一起核对                     |

以上均为论文报告的子图结果，不是整模型端到端加速。摘要“2× 到 5×”是概括，具体结果应引用对应子图、基线和硬件。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-configs.png' | relative_url }}" alt="原文表 2（PDF 第 18 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文表 2（PDF 第 18 页，裁剪）。左上 MHA 多数 batch 为 32，右上 MLA 的 batch 为 32、16、1；左下 routing 的 token 数为 2048，右下量化矩阵的 M 为 4096。配置本身说明该工作不应统一标为 batch 1。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-configs.png' | relative_url }}">查看原图</a></figcaption>
</figure>

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig6.png' | relative_url }}" alt="原文图 6（PDF 第 10 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 6（PDF 第 10 页，裁剪）。左图比较融合层级，纵轴是归一化性能、越高越好；右图同时展示增量/非增量性能和分段长度。最值得看的是非增量曲线的可行范围，以及增量形式能到达的更优配置，而不是简单比较任意两个点。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig6.png' | relative_url }}">查看原图</a></figcaption>
</figure>

融合层级决定修正次数与复用范围：过细的修正有较高计算成本，过晚合并保留更多等待。图 6(a) 的 safe softmax 实验中 intra-block 最好，这是一组特定输入和硬件上的观察，不是普适层级定理。

图 6(b) 中，同并行度的非增量方案通常较快，但容量限制使它无法到达某些较长分段。增量方案在该实验的 Waves per SM=3 附近达到更优配置。由此可见，代数变换会改变“哪些调度可实现”，性能优化必须在新的可行域中重新搜索。

附录 A.6–A.7 补充方差、转动惯量与 NVIDIA A100 / AMD MI308X。跨平台数据扩展了证据范围，但没有代替针对 TPU 等其他加速器的验证。标题中的 AI Accelerators 应结合实际后端覆盖理解。

<h2 id="verification">7. 复核公式：哪些可直接使用，哪些需要修正？</h2>

核心的局部尺度转换和增量更新，在可分解、兼容且可逆的条件下可以逐步推导。复核发现的问题主要集中在边界处理、附录示例和实现语义的衔接。它们应与核心机制的价值分开评估。

**Attention 漏项。** 附录 A.2.1 的式 (30)、(32) 在输出更新项中漏掉对应的 $$V$$ 因子；式 (29) 的定义包含它。直接照抄漏项公式会失去加权求和含义。

**图 13 的重复修正。** 合并阶段先把局部指数和改写为 $$\ell'_b=e^{m_b-m}\ell_b$$，随后又用 $$e^{m_b-m}\ell'_b/\ell$$ 乘局部归一化输出，重复计入指数系数。若使用已修正的 $$\ell'_b$$，正确权重应为 $$\ell'_b/\ell$$。本文用两个单元素分段、score 为 $$[0,1]$$、value 全为 1 的例子重算：正确输出为 1，按图中字面顺序得到 0.8299965984。**但已核对的公开生成代码没有重复这一修正。** `generated_redfuser_flash_decoding.py` 第一阶段保存未归一化的分子和指数和；第二阶段分别将二者缩放一次，最后计算 `all_y1/all_y2`。因此本项结论是论文图 13 的字面伪代码有误，不能据此断言公开实现同样错误。该示例的 query 长度为 512，也不能仅凭文件名将它等同于论文 query 长度为 1 的 decode 性能实验。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig13.png' | relative_url }}" alt="原文图 13（PDF 第 22 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 13（PDF 第 22 页，裁剪）。阅读 combine 段：先观察 psum_part/psum_frag 的原地缩放，再看后续 o_part/o_frag 的系数。问题在于同一个尺度是否已计入分母状态，不在于分段合并思路本身。 <a href="{{ '/assets/img/posts/paper-reading-3-redfuser/redfuser-fig13.png' | relative_url }}">查看原图</a></figcaption>
</figure>

**不可逆情形的证明不足。** 附录 A.1 把不可逆 $$H$$ 替换为单位元，能让表达式可计算，但不足以自动保证原语义。例如 $$F(y,r)=yr$$，$$r$$ 为输入的最大值；两段的 $$r$$ 均为 0、$$y$$ 分别为 2 和 3，原输出为 0。若全部改用 $$H'(0)=1$$ 则得到 5；若局部仍保留原来的零输出，则另取局部 $$r=(0,1)$$，全局为 1 时已丢失第一段的 $$y$$，同样无法仅凭零累计值恢复。需要明确保留额外状态或特殊分支，不能只替换单位元。

**方差的自动化路径未交代充分。** 附录报告方差实验，但 $$(x-m)^2$$ 一般不能直接分解成单一 $$G(x)H(m)$$。可以考虑展开成多个矩，或维护中心矩联合状态；这些是额外变换。文中未充分说明如何由 ACRF 自动走到该实现，因此不能把实验结果直接解释为自动发现了 Welford。公开 `decompose.py` 还将 `sum(A+B)` 拆成两个求和列为 TODO，这进一步说明“展开多个矩后自动分拆”的支持不能从论文结果直接推定。

这些复核限定的是文中具体公式与论证。没有 GPU 执行和逐个测试实现前，不把它们升级为对整个系统正确性的结论。

<h2 id="future-work">8. 作者明确提出的后续问题</h2>

RedFuser 的结论提出三条后续方向。首先，修正计算与寄存器压力可能抵消访存收益，需要系统化成本模型指导是否融合。其次，现有方法只适用于满足分解与代数兼容条件的表达式，需要放宽条件。最后，作者希望将框架拓展到 GPU 之外的 AI 加速器。这三点分别对应收益判断、算法表达能力和硬件映射。

进一步的研究设想是让编译器自动选择并维护联合状态。例如方差可以维护计数、均值与二阶中心矩，而不是直接要求平方差分解成单一乘积。但 Welford 及其分块合并本身是已有算法；新的贡献需要落在状态发现、证明和高效代码生成的方法上，不能仅凭加入一个手写规则宣称通用综合。

<h2 id="sources">9. 来源与核查范围</h2>

依据固定版本 **2603.10026v1.pdf，22 页**。按 arXiv 编号属于 2026 年 3 月条目，PDF 首页面侧标注 2026-02-24；本材料固定使用文件版本，不据此推断优先权。已阅读正文 §1–7、附录 A.1–A.7，复核主要数学推导、应用公式、图 5–6、表 2–3 与生成示例图 11–13。论文标注 CC BY 4.0；本文图像是保留原内容的 PDF 裁剪。

- [论文版本](https://arxiv.org/abs/2603.10026v1)
- [作者仓库](https://github.com/alibaba/redfuser)

教学反例与合并公式已独立数值核对；未执行 GPU 性能复现实验。下列源码链接给出固定核查提交及实际阅读范围。

公开代码核查日期为 2026-11-08，固定提交为 `56bf0b28c569fe160fd0a53a87a9d7f113852940`；这一提交不自动等同于论文性能实验所用提交。实际核对的核心文件如下：

- [python/tvm/redfuser/transform/decompose.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/transform/decompose.py)
- [python/tvm/redfuser/transform/generate_online_expr_pass.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/transform/generate_online_expr_pass.py)
- [python/tvm/redfuser/example/flash_attention.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/example/flash_attention.py)
- [python/tvm/redfuser/example/generated/generated_redfuser_flash_decoding.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/example/generated/generated_redfuser_flash_decoding.py)
- [python/tvm/redfuser/example/generated/generated_redfuser_ptpc_quant_gemm.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/example/generated/generated_redfuser_ptpc_quant_gemm.py)
- [python/tvm/redfuser/example/generated/test_ptpc_quant_gemm.py](https://github.com/alibaba/redfuser/blob/56bf0b28c569fe160fd0a53a87a9d7f113852940/python/tvm/redfuser/example/generated/test_ptpc_quant_gemm.py)

本文基于论文全文、原图与已注明的代码证据整理，使用 AI 辅助梳理和数值核查。性能结果来自论文；补充推导和教学例子不代表 GPU 实测。
