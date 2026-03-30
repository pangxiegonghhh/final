# SQuAD/CC‑News 基线阶段的固定重写提示词设计与分阶段落地建议

## 执行摘要

我建议在“固定重写提示词 + 缓存”这一步就启动开题里“提示词设计与实验评估”的**最小可行版本**：先定一个带长度约束的 P0 跑通单次重写 baseline；随后再扩展到 3–8 条提示词池做敏感性与对比评估，最后才进入集成与路由。这样既对齐开题，又不把工程复杂度一次性拉满。fileciteturn2file0 citeturn4view0turn3view0

## 这一步的提示词到底要设计到什么程度

你问“是不是在这一步就要完成开题报告(2)？”——我的结论是：**是从这一步开始进入(2)，但不需要在这一刻把(2)全部做完**。fileciteturn2file0

原因很直接：

- 你现在这一步的产物是“重写文本 + 编辑距离/重合度特征”，而开题(2)里写的内容（多种重写指令、长度约束、低成本 API 重写器、计算不变性/等变性/不确定性、在开发集上阈值或轻量分类器评估）本质上就是围绕“重写文本”展开的实验系统。也就是说，**P0 的设计与缓存策略就是(2)的地基**。fileciteturn2file0 citeturn3view0turn4view0  
- 同时，Raidar 原文已经明确指出：不同重写提示词会显著影响最终检测性能，且不存在跨数据源的单一最优提示词。这意味着如果你不从 P0 开始就把“提示词版本化 + 可复现”做好，后面做敏感性/集成/路由会非常痛苦。citeturn4view0

因此我推荐一个“**两阶段交付**”：

- **阶段 A（现在）**：只完成“P0（固定提示词）+ 长度约束 + 单次重写 + 缓存 + 基线评估（LR/SVM）”。这就足够你说“我已启动并跑通开题(2)的 baseline 版本”。fileciteturn2file0  
- **阶段 B（baseline 稳后）**：扩展到 3–8 条提示词池（P1…P8），系统做提示词敏感性评估（含有/无长度约束），再进入你开题里写的集成与路由。fileciteturn2file0 citeturn4view0

## P0 固定重写提示词怎么设计

P0 的设计目标不是“写得花”，而是**最大化语义保持、最小化无关变形、并用长度约束避免输出过长/过短导致的语义漂移**。这一点和 DART 的经验完全一致：他们明确写到如果不做约束，重写文本可能过长或过短，从而扭曲核心语义，并因此采用“按指定词数重写”的 prompt 控制长度。citeturn3view0turn3view2

### P0 的四条硬约束

你可以把 P0 的约束写成“审稿人友好”的原则（并在论文里直接引用）：

1) **语义保持**：不新增事实、不删关键事实，不改实体名、数字、专有名词。citeturn3view2turn4view0  
2) **禁止摘要化**：不要总结/缩写信息量（否则编辑距离信号会混入“压缩”因素）。citeturn4view0  
3) **长度约束**：把输出长度锁到与原文相近（DART 用 “in {n} words” 的形式）。citeturn3view0turn3view2  
4) **只输出正文**：禁止前缀解释（如“Sure,”、“Here is…”），否则会污染编辑距离与 n‑gram overlap。citeturn3view0turn4view0  

### P0 的推荐模板（英文更稳）

因为 SQuAD 与 CC‑News 均为英文文本，我建议 P0 用英文写，从而避免中英文指令混杂造成风格漂移。citeturn3view0turn4view0

（模板 1：遵循 DART 风格的“指定词数”）

```text
System:
You are a careful editor.

User:
Rewrite the following paragraph in {n_words} words.
- Preserve the original meaning and all key details.
- Do NOT add new facts, names, numbers, or claims.
- Do NOT remove important information.
- Do NOT summarize.
- Keep named entities and numeric values unchanged whenever possible.
Output ONLY the rewritten paragraph.

Paragraph:
{input_text}
```

（模板 2：更宽松的“±区间”版本，减少模型为凑词数而扭曲语义）

```text
System:
You are a careful editor.

User:
Rewrite the following paragraph to improve fluency while preserving meaning.
Target length: between {n_min} and {n_max} words (close to the original length).
- Do NOT add new facts, names, numbers, or claims.
- Do NOT delete important information.
- Do NOT summarize.
Output ONLY the rewritten paragraph.

Paragraph:
{input_text}
```

什么时候用哪个？

- **你想最大化可控性**（更像 DART）：用“指定词数（exact n）”，并在代码里做“词数不达标 → 自动重试一次”，把漂移压到最小。citeturn3view0turn3view2  
- **你更担心语义被凑词数破坏**：用“区间约束（±5% 或 ±10%）”，并记录实际词数以便后续分析“长度偏离是否影响特征”。citeturn3view2turn4view0  

### {n_words} 应该怎么定

我建议对 SQuAD 和 CC‑News 统一用同一套规则，并写进配置文件（保证可复现）：

- 先算原文词数 `n0 = word_count(input_text)`。  
- 设定 `n_words = clamp(round_to_nearest_5(n0), n_min_global, n_max_global)`。  
  - 例如 `n_min_global=120`、`n_max_global=260`（你可以按段落长度分布再微调）。  
- 如果模型输出词数偏离太多（例如 |n_out − n_words| > 8），重试一次；第二次仍失败就保留并在缓存里标记 `length_violation=true`，后续统计分析时剔除或单独报告。citeturn3view2turn4view0

这样做的理由可以直接引用 DART：他们之所以做词数约束，是为了避免“过长/过短导致语义扭曲，从而引入无关差异”。citeturn3view2turn3view0

## 如何分阶段完成你开题(2)提到的“多提示词 + 实验评估”

你开题(2)明确写了“设计多种重写指令 + 结合 DART 的长度约束 + 用低成本重写器批量生成重写版本 + 计算编辑距离、不变性、等变性、不确定性 + 在开发集用阈值或轻量分类器评估”。fileciteturn2file0

我建议把它拆成 3 个“可交付”的里程碑（每个都能写进周报/中期）：

### 里程碑一：P0 单提示词基线

交付物：
- P0（固定、带长度约束）
- 重写缓存（至少包含：text_id、dataset、split、prompt_id、n_words、model、temperature、rewritten_text、word_count_out）
- 基线模型：逻辑回归（LR）或 SVM（任选其一先跑通）
- 指标：AUROC、F1、TPR@FPR（按照你开题描述）fileciteturn2file0

这就是你现在要做的范围。

### 里程碑二：提示词敏感性（多提示词池）

为什么一定要做？因为 Raidar 已经实证：**不同提示词会显著影响检测性能，且没有单一最优提示词**。citeturn4view0

做法：
- 设计 3–8 条不同风格的重写指令（例如：严格同义改写/句法重排/正式文体/更口语/更简洁/更展开）。
- 每条都套同样的长度约束策略（DART 风格）。citeturn3view0turn3view2
- 在开发集比较：
  - 单 prompt 的性能
  - prompt 之间的方差（敏感性）
  - 以及简单的“prompt 集成”（例如对多个重写输出取特征平均、或把多 prompt 特征拼接给 LR/SVM）citeturn4view0turn3view2

你不需要在“P0 还没跑通”时就把这一步做完，但你应该在现在就把 prompt_id 与 prompt_pool 的数据结构设计好，避免返工。fileciteturn2file0

### 里程碑三：把“等变性/不确定性”接到同一套缓存接口上

这一步才更接近你开题里写的“特征向量更完整”（不变性/等变性/不确定性）。fileciteturn2file0

一个很实用的工程建议是：  
**先把缓存 schema 设计成“允许多次重写/多种变换”**，哪怕你现在只跑一次。这样未来你加入等变性（例如同一输入在不同 prompt 下重写输出的一致性/分歧）与不确定性（例如多次采样温度>0 的输出分布）时，不需要改动数据落盘结构。fileciteturn2file0

## 缓存键与实验记录建议

因为你明确要“调用 DeepSeek 生成重写文本并缓存”，我建议你把“缓存键”设计成能完全决定输出分布的最小集合：

- `dataset`（squad / cc_news）
- `split`（train/dev/test）
- `text_id`（稳定的 hash 或原始 id）
- `prompt_id`（P0/P1/…）
- `rewriter_model`（如 deepseek-chat）
- `decoding`（temperature, top_p, max_tokens）
- `length_policy`（exact_n / range；n_words 或 [n_min,n_max]）
- `timestamp` + `system_fingerprint`（如果你用的 API 支持类似字段则记录；如果不支持则至少记录日期与 SDK 版本）citeturn3view2turn4view0

这样你后面写论文“实验设置与可复现性声明”会非常顺：你可以明确告诉审稿人“重写输出可复跑、参数可追溯、提示词可版本化”。citeturn4view0turn3view2

## 参考来源清单

- 你的开题(2)原文（提示词设计、DART 长度约束、低成本重写器、轻量分类器评估）。fileciteturn2file0  
- DART：明确给出 rephraser prompt “Please rewrite … in {n} words”，并解释长度约束用于避免过长/过短导致语义扭曲，同时报告 rephrase 前后平均词数变化较小。citeturn3view0turn3view2  
- Raidar：明确指出“不同重写提示词会显著影响检测性能、无单一最优提示词”，并说明单一 prompt 在某些数据源上也可达到很高 F1；同时其动机与特征体系就是你的 baseline 依据。citeturn4view0turn1search6