---
layout: "post"
title: "论文阅读(2)——Neptune：把在线算法变成可组合的编译器变换"
date: "2026-11-13 00:00:00 -0400"
permalink: "/2026/11/13/paper-reading-2-neptune/"
description: "从归约修正函数理解 Rolling Update、Split-K 与自动生成融合内核。"
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

**核心判断：Neptune 的贡献是自动构造归约修正，并把这种能力做成调度原语；它让用户用普通计算定义和高层调度，生成 FlashAttention / FlashDecoding 式执行。** 主实验中，相对最佳编译器基线的几何平均加速为 1.35×；相对最佳手工库基线为 1.07×，两者对应不同实验集合。

同系列阅读：[RedFuser]({{ '/2026/11/13/paper-reading-3-redfuser/' | relative_url }}) · [Structure-Aware]({{ '/2026/11/13/paper-reading-4-structure-aware/' | relative_url }})。

<h2 id="problem">1. 从全局依赖到可更新状态</h2>

attention 的困难不只是算子多。对一个 query 而言，score 的最大值必须遍历整段 key 才知道；指数和依赖最大值；加权求和又依赖这些归一化量。按原始顺序实现，很容易把大型中间矩阵写回显存。手写 FlashAttention 利用在线 softmax 与分块解决了这一问题，但为每个 attention 变体重新实现和验证内核，代价很高。

两条已有路线各有所长：调度编译器能表达数学计算和循环变换，却会因归约依赖拒绝某些融合；tile 语言善于利用 Tensor Core 和片上存储，却要求程序员先写出正确的分块算法。Neptune 希望把高层算法变换放进编译器，再把结果交给 tile 编译器。为此，必须同时解决两个问题：融合后怎样保持语义，以及变换后的循环怎样自动成为高性能 tile 程序。

先看一个贯穿本文的教学输入 $$x=[0,1,-1,2]$$。目标是：

$$
m=\max_j x_j,\qquad s=\sum_j e^{x_j-m}.
$$

最终 $$m=2$$，所以 $$s=e^{-2}+e^{-1}+e^{-3}+1\approx1.553002$$。如果边更新最大值、边直接累加指数，就得到 $$3+e^{-2}\approx3.135335$$。原因在于旧项分别使用了不同的最大值。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig1.png' | relative_url }}" alt="原文图 1（PDF 第 4 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 1（PDF 第 4 页，裁剪）。先看左列原程序，再看中列直接融合后的橙色错误值，最后看右列绿色修正项。修正作用于已经累计的历史结果，使每一步状态使用当前最大值；中间状态可以改变，最终输出恢复一致。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig1.png' | relative_url }}">查看原图</a></figcaption>
</figure>

新的设计要求随之出现：历史输入已经离开片上存储，不能逐项重算；只能用一个小状态完成整体修正。对 softmax，这恰好可行：

$$
m'=\max(m,x),\qquad s'=s\,e^{m-m'}+e^{x-m'}.
$$

| 已读输入    | 当前最大值 | 正确在线状态 $$s$$ | 未修正累计值 |
| ----------- | ---------: | -----------------: | -----------: |
| 0           |          0 |           1.000000 |     1.000000 |
| 0, 1        |          1 |           1.367879 |     2.000000 |
| 0, 1, −1    |          1 |           1.503215 |     2.135335 |
| 0, 1, −1, 2 |          2 |           1.553002 |     3.135335 |

这是本文重算的教学例子，不是论文性能测量。它揭示了关键洞察：**暂时使用局部信息并不必然损失正确性，只要旧状态可以低成本转换到新的参照参数下。**

<h2 id="repair">2. 什么样的修正可以作用于整个归约？</h2>

Neptune 将消费端计算写成一个归约：

$$
T=\operatorname{Reduce}_f g(r,c_j).
$$

这里 $$r$$ 是前序归约结果，$$c_j$$ 是当前输入，$$g$$ 生成每个待归约项，$$f$$ 将这些项组合起来。编译器寻找 $$h(t,r,r')$$，把旧参数 $$r$$ 下的结果 $$t$$ 修成新参数 $$r'$$ 下的结果。

首先必须修对一个元素：

$$
h(g(r,c),r,r')=g(r',c).
$$

但这还不够。历史元素已经被折叠成一个状态，因此还要满足：

$$
h(f(a,b),r,r')=f(h(a,r,r'),h(b,r,r')).
$$

后一条说明修正与归约兼容：先合并再修正，等于先修正每项再合并。于是一次作用于累计值的修正，具有“替换所有历史项中旧参数”的效果。论文称这种性质为 tag-updating；tag 标记读取的是生产者哪一次迭代的值，不是新的数据维度。

对于 softmax，$$f(a,b)=a+b$$，$$g(r,c)=e^{c-r}$$。乘同一个系数可以分配到加法的各项，所以：

$$
h(t,r,r')=t e^{r-r'}.
$$

Neptune §4.1 的构造法是：

$$
h(t,r,r')=g\bigl(r',g_c^{-1}(r,t)\bigr).
$$

先用旧参数反解，再换新参数重新代入。softmax 中 $$g_c^{-1}(r,t)=r+\log t$$，化简后得到上述乘法。这是编译期符号计算，运行时不需要执行这次“求逆—重新代入”。编译器还必须验证修正与 $$f$$ 的兼容性；仅求出逆函数，并不能直接宣告融合成功。

当 $$g$$ 对原始输入不可逆，某些表达式仍能通过换元处理，例如把 $$\operatorname{ReLU}^2(c)$$ 视为一个整体变量。多生产者情形则让 $$r$$ 成为参数向量。它们拓宽了范围，但求解器仍有边界：论文明确承认不能自动得到 Welford 方差更新，因为缺少对均值、方差、样本之间关联的利用。

<h2 id="routes">3. 一个修正函数，导出两条执行路线</h2>

### Rolling Update：把历史状态逐次修到当前状态

用户指定要融合的消费端归约和目标循环。编译器沿数据流找到生产者归约，内联中间逐元素计算，分析 $$f,g$$，求解并验证 $$h$$，然后保留生产者的旧值与新值，把消费端更新改成：

$$
T_j=h(T_{j-1},r_{j-1},r_j)\;f\;g(r_j,c_j).
$$

这使读入一个块后就能更新累计状态，避免物化完整的中间结果。代价是每步修正、多保存少量状态，以及产生跨迭代递推依赖。失败时变换返回原程序，论文的正确性结论针对成功完成分析和重写的情形。

附录 D 的证明核心是归纳不变量：处理到第 $$j$$ 项时，累计状态恰好等于“前 $$j$$ 项全部用当前 $$r_j$$ 重新表达后再归约”的结果。最后一次迭代中，当前参数就是最终参数，因此最终输出与原程序相同。证明建立在其代数假设之上；浮点数误差另由实验考察。

### Split-K Update：并行算局部结果，再统一参照

如果任务本来就少，顺序扫描全部 KV 会限制并行度。Split-K 将归约维度切分，让各段使用自己的局部生产者结果计算，最后合并时再修正：

$$
m=\max_b m_b,\qquad s=\sum_b e^{m_b-m}s_b.
$$

同样取 $$[0,1,-1,2]$$，分为 $$[0,1]$$ 和 $$[-1,2]$$。两段分别产生 $$(m_A,s_A)=(1,1+e^{-1})$$ 和 $$(m_B,s_B)=(2,1+e^{-3})$$。统一到最大值 2 后，结果为 $$e^{-1}s_A+s_B=1.553002$$。各段可以独立执行，额外成本是保存局部统计量和合并阶段。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig3-4.png' | relative_url }}" alt="原文图 3、4（PDF 第 12 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 3、4（PDF 第 12 页，裁剪）。左图先生成 max_l、sum_l，再计算 max_g 和修正后的 sum_g；右图中不同颜色表示不同局部分段。观察修正项出现在全局合并中，局部任务因此能够保持独立。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig3-4.png' | relative_url }}">查看原图</a></figcaption>
</figure>

attention 同时保留最大值 $$m$$、指数和 $$\ell$$、未归一化的 value 累积向量 $$u$$。合并两段时：

$$
m=\max(m_A,m_B),\qquad \alpha=e^{m_A-m},\quad\beta=e^{m_B-m},
$$

$$
\ell=\alpha\ell_A+\beta\ell_B,\qquad u=\alpha u_A+\beta u_B,\qquad o=u/\ell.
$$

这里采用未归一化状态解释机制。它说明 FlashAttention 式流式执行和 FlashDecoding 式分段执行可以共享同一数学基础。Prefill 有较多 query 行，倾向于流式局部复用；decode 的 query 长度为 1，分段可提供更多并行任务。最终选择仍取决于形状和硬件，kernel 数量本身不是优化目标。

<h2 id="implementation">4. 从正确循环到高性能 GPU 内核</h2>

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig2.png' | relative_url }}" alt="原文图 2（PDF 第 5 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 2（PDF 第 5 页，裁剪）。顶部有两个输入：计算定义和高层调度模板。沿左侧向下读是编译路径，右侧 autotuner 结合性能测量搜索方案。该图明确区分用户指导的调度与编译器自动完成的代数修正。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig2.png' | relative_url }}">查看原图</a></figcaption>
</figure>

Neptune 建立在 TVM 与 Triton 上：TVM 风格的 tensor expression 和 schedule 进入 loop-scalar IR；高层变换完成后，转换为 HTile IR；再生成 Triton 程序，由后端处理线程映射、数据布局和设备代码。

| 表示或对象     | 保存的信息                         | 产生者与消费者               | 为什么需要                     |
| -------------- | ---------------------------------- | ---------------------------- | ------------------------------ |
| 数学计算定义   | 张量、归约和逐元素公式             | 用户 → 编译器                | 保留原始计算语义               |
| 高层调度模板   | 分块、融合位置、数据放置等选择     | 用户 → 模板优化器/调优器     | 缩小搜索范围并提供结构指导     |
| loop-scalar IR | 显式循环、下标、标量表达式         | 前端/调度器 → 修正分析       | 便于识别依赖并改写累计状态     |
| HTile IR       | tile 切片、广播、置换、dot、reduce | loop-to-tile 转换器 → Triton | 将变换结果接到高效设备后端     |
| 运行时累计状态 | 当前/上次归约值、输出累积          | 生成的内核内部               | 支撑在线修正，无需保存全部历史 |

loop-to-tile 转换不是机械替换下标。比如 $$Y[i,j]=A[i]B[j,i]$$，必须给 $$A$$ 补广播维度、给 $$B$$ 做置换。Neptune 的 dimension order reconciliation 通过 squeeze、permute、unsqueeze 对齐循环维度与张量维度。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig5.png' | relative_url }}" alt="原文图 5（PDF 第 14 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 5（PDF 第 14 页，裁剪）。左侧三个循环分别变成右侧 reduce、dot 和广播/置换表达式。重点看第三个例子：索引访问表达了维度关系，转换器必须保留这些关系才能正确生成 tile 程序。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig5.png' | relative_url }}">查看原图</a></figcaption>
</figure>

附录 A 给出的普通 attention 定义为 38 行、配套调度为 28 行，且示例定义不包含 GQA。调度显式调用 rolling_update，说明自动化的对象主要是变换内部推导与实现。MetaSchedule 的成本模型每轮预测候选，并选取部分方案实测；实验每个设置进行 128 次实测，编译调优总计约 1.5–10 分钟。Triton 阶段占 50%–90%，高级融合分析占约 5%–10%（附录 E.1）。

公开实现也体现了这一分工：`attn_plain.py` 由手写 scheduler 显式调用 `rolling_update` / `split_k_update`；`build.py` 把这个 scheduler 交给 MetaSchedule 的进化搜索与 XGBoost 成本模型。源码默认调优次数为 256，论文实验设置为 128，两者不能混用。符号工具 `sympy_tools.py` 的变量分离目前处理展开后的加法或乘法，遇到混合项会失败；逆函数求解出现零个或多个解时也会拒绝。这些具体实现解释了为什么“具有数学修正”不等于“当前求解器一定能找到修正”。

转换器也有后端限制：例如同一个内层变量重复出现在多个访问维度、或以非线性方式参与索引，可能不被支持。算法变换成功、tile 表示可表达、后端生成高效代码，是三个相连但不同的条件。

<h2 id="experiments">5. 实验怎样支持设计？</h2>

主实验使用 FP16 推理，四类 GPU 为 RTX 6000 Ada、RTX A5000、A100、MI300。10 个 attention 测试类别包括 prefill 与 decode，默认 batch 1。正文方法段列出长度从 $$2^7$$ 开始，但 320 个设置的统计与主要曲线的 8 个长度对应 $$2^8$$ 到 $$2^{15}$$；阅读汇总数据时按其明确的 320 个设置理解，保留这一文字不一致。

| 证据                 | 报告结果                                              | 支持的判断                                        |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| 表 2：最佳编译器基线 | 总体几何平均 1.35×；284/320 更快或持平                | 自动变换结合调优，对多种变体具有竞争力            |
| 表 4：最佳手写库基线 | 总体 1.07×；101/256 更快或持平                        | 在部分形状/变体上超过专家库，成熟常见模式仍有差距 |
| 图 10：逐组件消融    | tile 优化之上，融合带来主要收益；调优对 Global 更明显 | 代数融合确有贡献，底层优化与调优共同决定最终效果  |
| 图 11：非 attention  | 稳定 L2 norm、RMSNorm 配合缩放统计、Performer 获益    | 修正抽象超出 softmax attention 模板               |
| 附录 F：FP64 对照    | prefill 的 RMS 误差低于或等于所比较实现               | 实测数据上数值表现有竞争力                        |

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-tables3-4.png' | relative_url }}" alt="原文表 3、4（PDF 第 19 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文表 3、4（PDF 第 19 页，裁剪）。主要看右表：数值是 Neptune 相对最佳手写库的加速比，大于 1 才更快。A100 的平均为 0.95×，MI300 为 1.36×；Window PF 多处落后。因此总体均值应与逐场景结果一起阅读。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-tables3-4.png' | relative_url }}">查看原图</a></figcaption>
</figure>

1.35× 的比较组包括 Triton 实现和 FlexAttention 等编译器路线；1.07× 的比较组包括 cuDNN、FlashAttention 的 CUTLASS 实现等库，且后者剔除了当时基线未覆盖的 SoftCap 两个类别。不能用一组的总体数值描述另一组。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig9-10.png' | relative_url }}" alt="原文图 9、10（PDF 第 20 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 9、10（PDF 第 20 页，裁剪）。左图是固定序列长度下的 batch 扩展，右图是逐步加入 tile 优化、融合、调优后的表现。右图支撑“融合是主要收益来源”；左图只描述该设备和负载下的扩展，不应解释为所有 GPU 的普遍吞吐规律。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-fig9-10.png' | relative_url }}">查看原图</a></figcaption>
</figure>

Neptune 在 GQA decode 中还利用 mask 补齐矩阵行数，使原本行数太少的 dot 能使用 Tensor Core。这说明最终性能收益并非全来自修正公式。Window PF 表现较差，作者将其归因于 TVM 循环分析未识别合适的划分条件。两者都强调了代数层和工程层必须同时成功。

附录 F 使用 Qwen2.5 在 WikiText 的 100 条、长度 2048 的文本上采集的输入，对照 FP64 未融合 attention。PF GQA 的 RMS 误差分别为 Neptune 4.3e−5、Tri-Dao 4.5e−5、FlexAttention 4.4e−5；DC GQA 则为 3.4e−5、4.5e−5、2.6e−5。decode 中 Neptune 并非所有指标都更好。

<figure>
  <img src="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-accuracy.png' | relative_url }}" alt="原文图 20、表 5（PDF 第 37 页，裁剪）" loading="lazy" data-zoomable>
  <figcaption>原文图 20、表 5（PDF 第 37 页，裁剪）。上方分布展示逐样本误差，下方表列出各变体的绝对误差。先比较同一行的 RMS，再看分位数；这为浮点行为提供经验性证据，与实数代数正确性证明承担不同任务。 <a href="{{ '/assets/img/posts/paper-reading-2-neptune/neptune-accuracy.png' | relative_url }}">查看原图</a></figcaption>
</figure>

<h2 id="discussion">6. 贡献边界与值得继续的问题</h2>

Neptune 最有价值的贡献是把算法级修正提升为可组合的编译器原语。它既能与已有调度操作协作，也能连接成熟 tile 后端。证明说明成功变换在假设下保留语义，实验说明这种抽象没有必然牺牲性能。

实际范围仍由符号求解能力、调度模板、索引表达能力和后端决定。主实验是推理，没有自动生成反向与完整训练的验证。论文 §10 将更自动的调度和新架构优化交给后续 Nautilus 工作；该引用提供路线线索，本材料没有对 Nautilus 做独立精读。

复核发现 §4.4 和附录 C 的部分 attention/Performer 示例把指数符号写成 $$e^{r'-r}$$，与 §4.1、附录 B 及其定义不符；对 $$g(r,c)=e^{c-r}$$ 应使用 $$e^{r-r'}$$。这属于文中局部公式不一致，核心构造公式能导出正确方向。正文讲解均采用按定义重推的方向。

更远的研究问题是：能否让编译器连“保留什么状态”也自动发现？Welford 的失败提示，现有状态空间之外可能需要计数、均值、二阶中心矩等联合状态及其不变量。若能把状态综合、证明与成本选择结合，就能从自动应用已知在线结构迈向自动构造新的执行算法。这需要联合考虑状态不变量、合并规律与硬件代价。

<h2 id="sources">7. 来源与核查范围</h2>

全文依据固定版本 **2510.08726v2.pdf，37 页，2026-04-20 版本，PLDI 2026，DOI 10.1145/3808298**。已阅读正文、附录 A–F，重点回查 §4 公式、附录 D 证明、§7–8/附录 E 实验口径，以及附录 F 数值验证。图像从此版本 PDF 裁剪，保留原内容；中文图注为本材料撰写。论文标注 CC BY 4.0。

- [论文版本](https://arxiv.org/abs/2510.08726v2)
- [作者仓库](https://github.com/uiuc-arc/neptune)

本材料的教学数值由独立小计算核对；未运行 GPU 性能复现实验。下列源码链接固定于核查提交，以区分论文附录与公开实现。

公开代码核查日期为 2026-11-08，固定提交为 `f79750113db136e96516c4bb86a980fb1249559c`；这一提交不自动等同于论文性能实验所用提交。实际核对的核心文件如下：

- [scripts/neptune_bench/ours/attn_plain.py](https://github.com/uiuc-arc/neptune/blob/f79750113db136e96516c4bb86a980fb1249559c/scripts/neptune_bench/ours/attn_plain.py)
- [python/tvm/arith/sympy_tools.py](https://github.com/uiuc-arc/neptune/blob/f79750113db136e96516c4bb86a980fb1249559c/python/tvm/arith/sympy_tools.py)
- [python/tvm/neptune/build.py](https://github.com/uiuc-arc/neptune/blob/f79750113db136e96516c4bb86a980fb1249559c/python/tvm/neptune/build.py)

本文基于论文全文、原图与已注明的代码证据整理，使用 AI 辅助梳理和数值核查。性能结果来自论文；补充推导和教学例子不代表 GPU 实测。

---

[下载本篇离线阅读包（HTML、Markdown 与图片）]({{ '/assets/downloads/paper-reading-2-neptune.zip' | relative_url }})。离线包的公式预渲染为 MathML。
