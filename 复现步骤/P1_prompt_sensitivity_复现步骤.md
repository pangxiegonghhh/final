# P1 多提示词重写 + 提示词敏感性分析 复现步骤

> 数据集：SQuAD / CC-News
> 任务：多提示词重写 → 提示词敏感性分析（H1/H2/H3 假设验证）
> 前置条件：P0 已完成（human_corpus.json / ai_corpus.json 已存在）
> 时间：2026-03-08

---

## 一、环境准备

```powershell
conda activate raidar
cd "E:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect"
pip install datasets scikit-learn xgboost python-Levenshtein numpy pandas matplotlib openai scipy
```

---

## 二、项目目录结构

```
RaidarLLMDetect\
├── configs\
│   ├── p0_config.json              ← 已有，提供 api_key
│   └── p1_prompt_pool.json         ← 新增，7 个 prompt 定义
├── scripts\
│   ├── p0_train_eval.py            ← 已有，Step 3 复用
│   ├── p1_rewrite_multi_prompt.py  ← 新增，Step 1：多 prompt 重写
│   ├── p1_extract_features_multi.py← 新增，Step 2：逐 prompt 特征提取
│   └── p1_sensitivity_analysis.py  ← 新增，Step 4：敏感性分析 + 出图
├── SQuAD\
│   ├── human_corpus.json           ← P0 已有
│   ├── ai_corpus.json              ← P0 已有
│   └── rewrites_p1\                ← 新增目录
│       ├── human_P0_baseline.json  ← 可从 P0 复制
│       ├── ai_P0_baseline.json     ← 可从 P0 复制
│       ├── human_P1_fluency.json
│       ├── ai_P1_fluency.json
│       └── ...（共 14 个文件，7 prompts × human/ai）
├── CCNews\
│   ├── human_corpus.json           ← P0 已有
│   ├── ai_corpus.json              ← P0 已有
│   └── rewrites_p1\                ← 新增目录（同 SQuAD 结构）
└── outputs\
    └── p1_sensitivity\             ← 新增目录
        ├── features\               ← Step 2 产物（.npz × 14）
        ├── metrics\                ← Step 3 产物（.json × 14）
        ├── figures\                ← Step 4 产物（.png/.svg × 2）
        └── tables\                 ← Step 4 产物（.csv/.md + anova.json）
```

---

## 三、配置文件说明

`configs\p1_prompt_pool.json` 包含 7 个提示词，**无需改动**，api_key 继续从 `configs\p0_config.json` 读取：

| Prompt ID | 重写策略 | 长度约束 | 备注 |
|---|---|---|---|
| `P0_baseline` | 流畅度（含详细约束） | ±10% | P0 原始 prompt |
| `P1_fluency` | 流畅度（简洁版） | ±10% | - |
| `P2_synonym` | 同义改写 | ±10% | - |
| `P3_restructure` | 句式重构 | ±10% | - |
| `P4_formal` | 学术正式化 | ±10% | - |
| `P5_concise` | 精简化 | ±10% | - |
| `P6_no_length` | 流畅度（**无**长度约束） | 无 | H3 对照组 |

> **注意**：`P0_baseline` 的重写结果在 P0 阶段已产生，可直接复制跳过 API 调用（见下方 Step 0）。

---

## 四、完整执行命令

所有命令均在 `RaidarLLMDetect\` 根目录下执行。

### Step 0：复用 P0 重写缓存（节省 API 调用，约 8000 次）

> 将 P0 已有的重写结果直接复用为 `P0_baseline` 的缓存，跳过 `P0_baseline` 的 API 调用。

```powershell
# SQuAD
Copy-Item SQuAD\rewrite_squad_human.json SQuAD\rewrites_p1\human_P0_baseline.json
Copy-Item SQuAD\rewrite_squad_ai.json    SQuAD\rewrites_p1\ai_P0_baseline.json

# CCNews
Copy-Item CCNews\rewrite_ccnews_human.json CCNews\rewrites_p1\human_P0_baseline.json
Copy-Item CCNews\rewrite_ccnews_ai.json    CCNews\rewrites_p1\ai_P0_baseline.json
```

> **注意**：P0 的重写 JSON 字段须含 `text_id`、`input`、`rewrite` 字段，否则需先确认格式兼容性。

---

### Step 1：多 Prompt 重写（~8 h，需 API）

> 对 6 个新 prompt（P1~P6）× 2 域 × human/ai = 24 批次，每批 2000 条。
> 支持断点续跑，中断后重新执行同一命令即可。

```powershell
$PROMPTS = @("P1_fluency","P2_synonym","P3_restructure","P4_formal","P5_concise","P6_no_length")

foreach ($promptId in $PROMPTS) {
    # SQuAD - human
    python scripts\p1_rewrite_multi_prompt.py `
      --input_json SQuAD\human_corpus.json `
      --prompt_id $promptId `
      --out_json "SQuAD\rewrites_p1\human_${promptId}.json"

    # SQuAD - AI
    python scripts\p1_rewrite_multi_prompt.py `
      --input_json SQuAD\ai_corpus.json `
      --prompt_id $promptId `
      --out_json "SQuAD\rewrites_p1\ai_${promptId}.json"

    # CCNews - human
    python scripts\p1_rewrite_multi_prompt.py `
      --input_json CCNews\human_corpus.json `
      --prompt_id $promptId `
      --out_json "CCNews\rewrites_p1\human_${promptId}.json"

    # CCNews - AI
    python scripts\p1_rewrite_multi_prompt.py `
      --input_json CCNews\ai_corpus.json `
      --prompt_id $promptId `
      --out_json "CCNews\rewrites_p1\ai_${promptId}.json"
}
```

**耗时估计**：6 新 prompt × 4 数据源 × 2000 条 × 0.5s ≈ **6~8 h**（可分批、分天执行）

> Step 0 + Step 1 完成后，`rewrites_p1\` 中共应有 **28 个 JSON 文件**（7 prompts × 2 域 × 2 类型）。

---

### Step 2：逐 Prompt 特征提取（~10 min）

```powershell
$PROMPTS = @("P0_baseline","P1_fluency","P2_synonym","P3_restructure","P4_formal","P5_concise","P6_no_length")

foreach ($promptId in $PROMPTS) {
    # SQuAD
    python scripts\p1_extract_features_multi.py `
      --human_json "SQuAD\rewrites_p1\human_${promptId}.json" `
      --ai_json    "SQuAD\rewrites_p1\ai_${promptId}.json" `
      --out_npz    "outputs\p1_sensitivity\features\squad_${promptId}.npz"

    # CCNews
    python scripts\p1_extract_features_multi.py `
      --human_json "CCNews\rewrites_p1\human_${promptId}.json" `
      --ai_json    "CCNews\rewrites_p1\ai_${promptId}.json" `
      --out_npz    "outputs\p1_sensitivity\features\ccnews_${promptId}.npz"
}
```

产物：`outputs\p1_sensitivity\features\` 下共 **14 个 .npz 文件**（7 prompts × 2 域）

**特征说明（5 维，与 P0 完全一致）：**

| 维度 | 特征 | 计算方式 |
|------|------|---------|
| 1 | Levenshtein ratio | `1 - lev(input, rewrite) / max(len(input), len(rewrite))` |
| 2 | 1-gram overlap | `共有 unigram 数 / input unigram 总数` |
| 3 | 2-gram overlap | `共有 bigram 数 / input bigram 总数` |
| 4 | 3-gram overlap | `共有 trigram 数 / input trigram 总数` |
| 5 | 4-gram overlap | `共有 4-gram 数 / input 4-gram 总数` |

---

### Step 3：逐 Prompt 10-seed 评估（~5 min）

> 复用 P0 的 `p0_train_eval.py`，P1 只跑 LR（主分类器），节省时间。

```powershell
$PROMPTS = @("P0_baseline","P1_fluency","P2_synonym","P3_restructure","P4_formal","P5_concise","P6_no_length")

foreach ($promptId in $PROMPTS) {
    # SQuAD
    python scripts\p0_train_eval.py `
      --npz          "outputs\p1_sensitivity\features\squad_${promptId}.npz" `
      --out_dir      "outputs\p1_sensitivity\metrics" `
      --dataset_name "squad_${promptId}" `
      --classifiers  "LR"

    # CCNews
    python scripts\p0_train_eval.py `
      --npz          "outputs\p1_sensitivity\features\ccnews_${promptId}.npz" `
      --out_dir      "outputs\p1_sensitivity\metrics" `
      --dataset_name "ccnews_${promptId}" `
      --classifiers  "LR"
}
```

产物：`outputs\p1_sensitivity\metrics\` 下共 **14 个 JSON 文件**，文件名格式如 `squad_P1_fluency_LR.json`

每个 JSON 包含字段：`accuracy_mean`、`accuracy_std`、`f1_mean`、`f1_std`、`auroc_mean`、`auroc_std`、`f1_list`、`accuracy_list`

---

### Step 4：敏感性分析 + 可视化（~1 min）

```powershell
python scripts\p1_sensitivity_analysis.py `
  --metrics_dir outputs\p1_sensitivity\metrics `
  --out_dir     outputs\p1_sensitivity
```

产物：

| 文件 | 路径 | 用途 |
|------|------|------|
| 敏感性总表（CSV） | `outputs\p1_sensitivity\tables\p1_prompt_sensitivity.csv` | 论文 Table |
| 敏感性总表（MD） | `outputs\p1_sensitivity\tables\p1_prompt_sensitivity.md` | 论文 Table（Markdown） |
| 敏感性柱状图 | `outputs\p1_sensitivity\figures\p1_sensitivity_squad.png/.svg` | 论文 Figure |
| 敏感性柱状图 | `outputs\p1_sensitivity\figures\p1_sensitivity_ccnews.png/.svg` | 论文 Figure |
| ANOVA 检验结果 | `outputs\p1_sensitivity\tables\p1_anova.json` | 统计显著性（验证 H1） |

**控制台输出示例（验证假设 H1）：**

```
=== ANOVA (F1 across prompts) ===
  squad:  F=12.34, p=0.000123
  ccnews: F=8.76,  p=0.001456
```

> p < 0.05 即可认为不同 prompt 的 F1 差异显著（H1 成立）。

---

### Step 5：复现留痕

```powershell
# 冻结依赖版本
python -m pip freeze > outputs\repro\requirements_p1.txt

# 保存配置快照
Copy-Item configs\p1_prompt_pool.json outputs\repro\p1_prompt_pool_snapshot.json

# 生成所有 P1 产物的 SHA256
Get-ChildItem -Recurse outputs\p1_sensitivity -File | ForEach-Object {
    "{0}`t{1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.FullName
} | Set-Content -Encoding UTF8 outputs\repro\p1_manifest_sha256.tsv
```

---

## 五、执行时间速查

| 步骤 | 脚本 | 预计耗时 |
|------|------|---------|
| Step 0 | 复制 P0 缓存 | < 1 min |
| Step 1 | `p1_rewrite_multi_prompt.py` × 24 批 | **6~8 h（API 调用）** |
| Step 2 | `p1_extract_features_multi.py` × 14 | ~10 min |
| Step 3 | `p0_train_eval.py` × 14（仅 LR） | ~5 min |
| Step 4 | `p1_sensitivity_analysis.py` | ~1 min |
| Step 5 | 复现留痕命令 | ~1 min |

---

## 六、核心假设验证说明

| 假设 | 验证方法 | 判定标准 |
|------|---------|---------|
| **H1（敏感性）** | Step 4 ANOVA，检验 7 个 prompt 的 F1 跨 prompt 方差 vs 跨 seed 方差 | p < 0.05 且 F_prompt > F_seed |
| **H2（无单一最优）** | 对比 SQuAD 和 CCNews 各自的最优 prompt | 两域最优 prompt 不同 |
| **H3（长度约束交互）** | 对比 `P0_baseline`（有长度约束） vs `P6_no_length`（无约束）的排名变化 | 两者 F1 差异可见，且排名不稳定 |

---

## 七、关键设计说明

| 决策 | 选择 | 理由 |
|------|------|------|
| 评估分类器 | 仅 LR | P0 已验证三分类器，P1 聚焦敏感性分析，只需主分类器 |
| prompt 数量 | 7 个（P0~P6） | P0~P5 覆盖 6 种重写风格，P6 作为去约束对照组 |
| P6 设计 | 去掉长度约束 | 单一变量控制，验证 H3（长度约束对检测性能的影响） |
| 断点续跑 | 逐 50 条保存缓存 | 防止 API 中断导致全量重跑 |
| 评估 seed | 10 个（42~51） | 与 P0 完全对齐，保证结果可比性 |
