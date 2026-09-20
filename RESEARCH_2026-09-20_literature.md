# 文献支撑：这类策略能不能被验证，以及别人做到哪一步（2026-09-20）

> 目的：`FALSIFICATION_2026-09-19.md` §8 给出了"下次的三条硬规则"，但那是从本项目
> 自己的失败里归纳的。本文把它们放到文献与同行实践里核对——**结论是三条规则都能被
> 独立证据支持，而且还漏了一条更根本的**（§1）。
>
> 同时记录一个直接可比的独立平行项目（§3），它的三个核心发现与本项目逐条对应。
>
> ⚠️ 本文是文献综述与外部对照，**不产生任何新的策略判据**，不进 `trials.json`。

---

## 0. 先放最干净的那个数

v13 回测头条 **+32.6% / Sharpe 0.8056**，窗口 13,105 根小时 bar = **1.50 年**。
用最基本的关系 `t ≈ SR_ann × √年数`：

**v13 回测自己的 t 统计量 = 0.8056 × √1.50 = 0.985。**

| 门槛 | 需要 SR_ann | v13 实际 | 差距 |
|---|---:|---:|---|
| 单次检验显著（t>2） | 1.64 | **0.81** | 差一半 |
| DSR @ n_trials=77（越过选择运气上限） | 1.91 | **0.81** | 差 2.4× |
| Harvey–Liu 多重检验下"新因子"门槛（t>3） | 2.45 | **0.81** | 差 3× |

**这个回测从来没到过任何一条线，连最宽松的单次检验都差一半。**
后面三个月的 live 不是在检验它——是在确认一件回测阶段就已判定的事。
（这与 §4.4 的 DSR 结论同源：SR 0.008607 / E[max]@77 0.020429 = 0.42 倍。
t 统计量是同一件事的更朴素说法，不需要任何 López de Prado 的机器。）

---

## 1. 验证的算术：§8 漏掉的那条根本规则

Two Sigma 的 Sharpe 技术报告与 Harvey–Liu 的 backtesting 框架给的是同一个关系：

```
t ≈ SR_ann × √(年数)        等价于      SE(SR_ann) ≈ 1/√(年数)
```

**关键性质：它只依赖日历时间，不依赖采样频率。** 把日频换成小时频不会让你更快地
验证同一个策略——n 变大了，但单期 SR 同比例变小。这条否定了一个很自然的错误直觉
（"多取样本就能更快验证"）。

反推本项目的处境：

| 想验证的真实 SR_ann | t=2 所需时间 |
|---:|---:|
| 2.00 | 1.0 年 |
| 1.41 | 2.0 年 |
| 1.00 | 4.0 年 |
| 0.53 | 14.2 年 |
| 0.138 | **210 年** |

注册窗口是 **60 天和 90 天**。

> **§8 应补的第四条规则：先问"这个效应量在我能等的时间里可验证吗"，再决定要不要做。**
> 如果答案是"需要 6 年"，那么这个课题对单人研究者**在结构上不可完成**，
> 与模型好坏、代码质量、数据干净程度全都无关。这不是执行问题，是算术。

MinTRL（minimum track record length）就是这个量的正式名字；
`DaruFinance/quant-research-framework` 把它和 MinBTL、PBO/CSCV、DSR 一起做成了
标准诊断套件——**说明这是个已被工程化的常规检查，而本项目做了 13 个版本都没做过它。**

---

## 2. 这个资产类别上，实际能拿到多高的 Sharpe

发表值（注意：发表值本身已经被选择偏差抬高过一轮）：

| 来源 | 结果 |
|---|---|
| 加密动量横截面 vs 时序对比研究（2020-01~2025-10） | 横截面动量 **14.59%/年**，时序动量 31.96%/年 |
| 同上，参数化最优 | **Sharpe 1.51**（28 天回看 / 5 天持有，假设 **15bps** 成本），市场组合 0.84 |
| 加密横截面交互项 OOS 多空 | **Sharpe 略高于 1**；作者明确提示低流动性抬高成本会侵蚀它 |

**再叠加 McLean & Pontiff (2016) 的衰减**：异象收益 OOS 平均低 **26%**，
发表后低 **58%**（月度对冲组合毛收益 0.58% → 0.40% → 0.26%）。

于是一条完整的推理链：

```
发表最优 Sharpe 1.51
  → 按发表后衰减 58% 折算 ≈ 0.63
  → 验证 SR 0.63 至 t=2 需要 (2/0.63)² ≈ 10 年 live
```

**即使你完美复现了这个领域已发表的最好结果，你也需要十年才能证明它不是运气。**
而 v13 实际拿到的是 0.81（回测、未扣多重检验），live 是负的。

⚠️ 诚实边界：McLean–Pontiff 的样本是**美股异象**，不是加密因子；检索到的综述明确
指出加密市场的同类系统性研究仍很少。把 26%/58% 直接套到加密上是**类比而非证据**，
本文按类比使用，不作为判据。

---

## 3. 一个独立的平行项目：结论逐条对应

[`Jareedd/qr-alpha-lab`](https://github.com/Jareedd/qr-alpha-lab) ——
另一位研究者、另一套代码库，同样的方法论骨架：预注册、CI 里跑 DSR 门槛、
walk-forward + embargo（21 天）、成本感知回测（默认 10bps/边）、**控制臂**
（每个 live cycle 影子跑一个 12-1 动量基准）、paper trading 且预测先提交后下单。

**总成绩：13 个登记 trial，跨三个资产类，零个毕业。**

与本项目的对应关系：

| 他们的发现 | 本项目的对应发现 | 判断 |
|---|---|---|
| **#8 crypto-perp funding carry：净 Sharpe 0.87，t_NW −3.54，仍未过预注册 DSR 门槛（0.865 < 0.95）**；并记录该交易"已从 Sharpe 2.3 衰减到 ~0.4 as the trade institutionalized"，且崩盘偏度 −1.87 | carry 回测 Sharpe 0.53；live 全口径 −1.69；funding 收入全程为正但价格腿吃掉它 | **独立确认**：carry 机制真实但过不了多重检验门槛，且正在被机构化磨平 |
| **#10 carry 尾部版："clean IC signal, negative P&L"**，作者自己标注为整个项目里 IC≠P&L 最干净的展品 | w·y24 = −0.033%/天：IC 每年为正但美元空间为负（`RESEARCH_2026-07-13_extended_window.md`） | **独立复现本项目的核心诊断** |
| 生存偏差对照：同一个 ridge 模型，静态宇宙（今天的成分股）净 Sharpe **+0.82** / IC +0.033；时点真实宇宙 **−0.01** / IC +0.005（t_NW 0.54）。作者结论："The entire edge was hindsight in the universe selection." | CTRL 实验：同窗口同配方，仅去掉 4 个晚上市山寨 + funding 因子，**+32.6% → −13.6%** | **同一失效类的两个实例** |
| #11 CEF 折价回归：原始 SR 1.11、DSR 0.999（看起来是大发现），被一个**入场延迟诊断**推翻（1.11 → 0.10），判定为微结构假象 | 本项目的同类角色 = ENGINE_CROSSCHECK 的成本模型敏感性（−13.1pp） | 提醒：**DSR 过关也不等于真**，执行侧诊断能独立推翻它 |
| #12 fundamental quality：净 SR 0.58 → HML 中性化后 **−0.18**，"quality edge was the value factor in disguise" | — | 本项目缺这一步：**从未做过因子中性化后的增量检验** |
| 合成验证：植入信号 → IC +0.0629 / DSR 0.99 被正确召回；纯噪声 20 trials → IC −0.02 / DSR 0.0004 被正确拒绝 | **本项目此前没有做过** → 已于 2026-09-20 补上（`tools/pipeline_calibration.py`） | 见 §5 |

**这张表的意义**：本项目那三个核心结论——IC 不等于 PnL、宇宙选择主导收益、
carry 即使 funding 为正也过不了 DSR——被一个完全独立的代码库在不同资产上复现了。
它们因此从"我这个项目的特异失败"升级为**这个问题本身的性质**。

同时也暴露了本项目缺的两件事：**控制臂**（他们每个 cycle 影子跑基准，本项目只有
combo 虚拟账本，没有独立基准臂）和**因子中性化增量检验**。

---

## 4. 多重检验：门槛在哪

- **Harvey & Liu**：单次检验用 t>2 在多重检验下不成立；他们给出按试验次数计算的
  haircut，并指出 haircut 是**非线性**的——最高的 Sharpe 只被温和惩罚，
  **边缘 Sharpe 被重罚**。v13 的 0.81 正落在被重罚的那一段。
- **HLZ (2015)** 统计到股票横截面上至少 **316 个**被检验过的因子——"factor zoo"。
  本项目 `trials.json` 记 **77**，量级上远小，但 DSR 已经判否，说明**问题不是试得多，
  是效应量太小**。
- **Bailey & López de Prado 的 DSR** 正是把"E[max SR under n trials]"当基准的 PSR。
  本项目已经在用，只是**用成了事后报告项而不是准入门槛**（§8 已记录这条要改）。

---

## 5. 对三条前路的文献裁决

| 方向 | 文献怎么说 | 裁决 |
|---|---|---|
| **继续日频横截面因子** | 发表最优 Sharpe ~1.5，衰减后 ~0.63，需 10 年验证 | ❌ **结构上不可验证**，停 |
| **机制型：delta-hedged funding capture** | BIS WP 1087 确认 carry 真实存在；但 CEX/DEX 套利研究显示 17% 观测有 ≥20bps 价差，而**最好的机会里只有 40% 在扣成本与价差反转后仍为正**；qr-alpha-lab 记录 Sharpe 2.3→0.4 的机构化衰减 | ⚠️ **算术上可验证（若 Sharpe 真高则窗口短），但赛道正在被磨平**；要做必须新预注册 |
| **高频 / LOB 执行侧** | 报告 Sharpe 1.38–0.86；费率敏感性极端（0.2%→0.6% 费率下成交次数 24.54→7.85）；多数论文成交假设乐观 | ⚠️ 方向对（高 Sharpe 才能快速验证）但**是换数据基建量级的决定**，不是换模型 |
| **合成信号验证管线** | qr-alpha-lab 的标准做法，且是本项目 §6"未确立"栏的唯一可填项 | ✅ **已执行**，见 `tools/pipeline_calibration.py` 与 §6 |

---

## 6. 本次据此补上的工作

`tools/pipeline_calibration.py`（2026-09-20）——把已知强度的合成信号喂进**真实的
v13 回测路径**（config C：banded top-3、enter<3/exit≥6、TWAP 成本），测这套构造
把"排名准确度"转成"美元"的传递函数。三个臂：NULL（纯噪声，必须不赚钱）、
UNIFORM（准确度横截面均匀）、SMALLMOVE（准确度集中在小波动名字上 = 合成复现
v13 被诊断出的失效模式）。

这填的是 `FALSIFICATION_2026-09-19.md` §6 的空白：在此之前，本项目所有负结果都
**无法区分**"市场没有信号"与"有信号但管线转不成钱"。

⚠️ 该工具的分数按设计使用未来信息（由已实现 y24 构造），**不是策略、不可交易**，
任何从中流出的数字若被当作业绩宣称即为造假。

### 6.1 结果（2026-09-20，20 币钉死宇宙，28,453 小时样本 = 3.25 年，5 seed）

| 臂 | 实测 rank IC | 中位 Sharpe | 判读 |
|---|---:|---:|---|
| **NULL**（零信息） | +0.0019 | **−0.58** | ✅ **机器不会凭空造钱**——噪声进去，成本出来 |
| UNIFORM | +0.0210 | −0.46 | 平衡线以下 |
| UNIFORM | +0.0357 | +1.19 | 平衡线以上 |
| UNIFORM | +0.0696 | +3.56 | |
| SMALLMOVE | +0.0244 | +0.15 | |
| SMALLMOVE | +0.0476 | +1.45 | |

**盈亏平衡 rank IC ≈ 0.024**（UNIFORM 0.0244 / SMALLMOVE 0.0238）。

### 6.2 三个结论

**① 管线**不是**瓶颈——§6 的空白填上了，而且是往有利方向填的。**
喂给这套构造一个 IC ≥ 0.024 的信号，它就能赚钱；IC 0.07 时中位 Sharpe 达 3.56。
所以"有信号但管线转不成钱"这个可能性**被排除**。此前所有负结果的二义性消解了：
问题在信号，不在构造。

**② 整个故事可以压缩成一句话：IC 从 0.064 掉到 0.0195，途中穿过了 0.024 这条线。**

| | rank IC | 相对平衡线 | 结果 |
|---|---:|---:|---|
| v13 回测 | 0.064 | **2.6×** | +32.6%，好看 |
| v13 live（96 marks） | 0.0195 | **0.80×** | −16.21%，亏 |

**平衡线正好夹在两者之间。** 这解释了为什么回测漂亮而 live 亏损，
且不需要任何"regime 变了""因子失效了"的额外假设——
**只需要"回测 IC 被选择偏差抬高了约 70%"**，而这正是 DSR 早就指出的
（SR = 运气上限的 0.42 倍）。衰减幅度与 McLean–Pontiff 记录的
OOS −26% / 发表后 −58% 同一量级（⚠️ 类比：MP 是美股异象、且是发表后衰减，
本处是样本内→live 的过拟合衰减，机制相关但不同）。

**③ ⚠️ 我为复现"大波动放错边"而设计的 SMALLMOVE 臂，没能复现它——假设未获支持。**
在**匹配实测 rank IC 之后**，两个臂的平衡点几乎相同（0.0244 vs 0.0238），
即：**这套构造的变现能力对"准确度落在横截面何处"不敏感，只对总体 rank IC 敏感。**
所以 `RESEARCH_2026-07-13` 里 w·y24 < 0 的现象**不能**用"准确度集中在小波动名字上"
来解释——至少不能用这个参数化解释。
可能原因：那个诊断用的是**分数比例权重**（score-proportional），
而本处是 **banded top-3**（只用排名、只碰两端），两种构造对信号结构的敏感度本就不同。
**这条留作未解**，不要在任何地方把它讲成已解释。

### 6.3 三条必须同时说出口的限制

1. **0.024 是下界，不是真实门槛。** 合成信号的质量在整个窗口上是**恒定**的；
   真实信号的 IC 随时间起伏（会有整段为负的时期）。同样均值 IC 下，
   时变信号的变现能力更差 ⇒ **真实所需 IC 高于 0.024。**
2. **窗口与宇宙不同于 v13。** 本标定跑在钉死的 live 20 币宇宙、3.25 年窗口；
   v13 回测是 1.5 年、部分不同的宇宙。平衡线在别的窗口上会移动。
3. **高 IC 端的总收益无意义。** 那是无容量约束的复利（IC 0.40 时 +25 亿%），
   只有符号与平衡区间可读——故上表只列 Sharpe。

另：该工具过程中发现一个**新的静默漂移隐患**——
`run_v13_final.build_from_parquet` 的宇宙是 `sorted(lake_symbols)[:max_assets]`，
**lake 增长后它会静默返回不同的 20 个币**（现会吃进 FET/FIL/PEPE/RENDER，
丢掉 SOL/SUI/UNI/XRP）。这与 §3 表中 qr-alpha-lab 的生存偏差案例是同一失效类。
校准工具已显式钉死宇宙；该隐患本身记录在此，供任何未来复跑者注意。

---

## 参考

1. [Jareedd/qr-alpha-lab](https://github.com/Jareedd/qr-alpha-lab) — falsification-first 量化研究，13 trial 零毕业
2. [DaruFinance/quant-research-framework](https://github.com/DaruFinance/quant-research-framework) — DSR/PSR/MinTRL/MinBTL/PBO-CSCV 诊断套件
3. [Bailey & López de Prado — The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
4. [Harvey & Liu — Backtesting（多重检验 haircut）](https://people.duke.edu/~charvey/Media/2016/Practical_applications_backtesting.pdf)
5. [Two Sigma — Sharpe Ratio: Estimation, Confidence Intervals, and Hypothesis Testing](https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf)
6. [Momentum Trading in Cryptocurrencies: Time-Series vs Cross-Sectional](https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419)
7. [Cross-sectional interactions in cryptocurrency returns](https://www.sciencedirect.com/science/article/abs/pii/S1057521924007415)
8. [McLean & Pontiff — Does Academic Research Destroy Stock Return Predictability?](https://www.signaltrace.wiki/markov-model/Papers/McLean-and-Pontiff-2016)
9. [BIS Working Paper 1087 — Crypto Carry](https://www.bis.org/publ/work1087.pdf)
10. [Exploring risk and return profiles of funding rate arbitrage on CEX and DEX](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
11. [Combining Deep Learning on Order Books with Reinforcement Learning](https://arxiv.org/pdf/2311.02088)
