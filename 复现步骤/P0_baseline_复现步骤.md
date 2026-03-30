# P0 Baseline 复现步骤

> 数据集：SQuAD / CC-News  
> 任务：AI 生成文本检测（Invariance 基线）  
> 时间：2026-03-05

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
│   └── p0_config.json              ← 统一配置文件（API Key 在此填写）
├── scripts\
│   ├── p0_sample_corpus.py         ← Step 1：下载 & 抽样人类文本
│   ├── p0_generate_ai_text.py      ← Step 2：DeepSeek 生成 AI 对照文本
│   ├── p0_rewrite.py               ← Step 3：P0 固定提示词重写（±10% 区间约束）
│   ├── p0_extract_features.py      ← Step 4：提取 5 维特征（Levenshtein + n-gram）
│   ├── p0_train_eval.py            ← Step 5：训练 LR/XGBoost/MLP，评估 AUROC/F1/TPR@FPR
│   └── p0_export_results.py        ← Step 6：导出 CSV/MD 总表 + 柱状图
├── SQuAD\
│   ├── raw\                        ← HuggingFace 原始数据（自动生成）
│   ├── human_corpus.json           ← Step 1 产物
│   ├── ai_corpus.json              ← Step 2 产物
│   ├── rewrite_squad_human.json    ← Step 3 产物
│   └── rewrite_squad_ai.json       ← Step 3 产物
├── CCNews\                         ← 同 SQuAD 结构
│   ├── raw\
│   ├── human_corpus.json
│   ├── ai_corpus.json
│   ├── rewrite_ccnews_human.json
│   └── rewrite_ccnews_ai.json
└── outputs\
    ├── p0_baseline\
    │   ├── features\               ← Step 4 产物（.npz）
    │   ├── models\
    │   ├── metrics\                ← Step 5 产物（JSON）
    │   └── figures\                ← Step 6 产物（PNG/SVG）
    └── repro\                      ← 复现留痕
```

---

## 三、配置文件说明

编辑 `configs\p0_config.json`，**只需确认 `api_key` 字段正确**，其他参数无需改动：

```json
{
  "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
  "rewriter": {
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "length_tolerance": 0.10
  },
  "eval": {
    "seed_start": 42,
    "num_seeds": 10,
    "fpr_thresholds": [0.01, 0.05]
  }
}
```

> **注意**：脚本优先从配置文件读取 `api_key`，不需要设置环境变量。

---

## 四、完整执行命令

所有命令均在 `RaidarLLMDetect\` 根目录下执行。

### Step 1：抽样人类文本（~5 min）

```powershell
# SQuAD
python scripts\p0_sample_corpus.py --dataset squad --sample_size 2000 --out_json SQuAD\human_corpus.json

# CC-News
python scripts\p0_sample_corpus.py --dataset cc_news --sample_size 2000 --out_json CCNews\human_corpus.json
```

### Step 2：生成 AI 对照文本（~2~4 h，需 API）

```powershell
# SQuAD
python scripts\p0_generate_ai_text.py --human_json SQuAD\human_corpus.json --out_json SQuAD\ai_corpus.json

# CC-News
python scripts\p0_generate_ai_text.py --human_json CCNews\human_corpus.json --out_json CCNews\ai_corpus.json
```

> 支持断点续传：中断后重新运行同一命令即可，已完成的条目自动跳过。

### Step 3：DeepSeek 重写（~4~8 h，需 API）

```powershell
# SQuAD - human
python scripts\p0_rewrite.py --input_json SQuAD\human_corpus.json --out_json SQuAD\rewrite_squad_human.json

# SQuAD - AI
python scripts\p0_rewrite.py --input_json SQuAD\ai_corpus.json --out_json SQuAD\rewrite_squad_ai.json

# CC-News - human
python scripts\p0_rewrite.py --input_json CCNews\human_corpus.json --out_json CCNews\rewrite_ccnews_human.json

# CC-News - AI
python scripts\p0_rewrite.py --input_json CCNews\ai_corpus.json --out_json CCNews\rewrite_ccnews_ai.json
```

> 同样支持断点续传。

### Step 4：提取特征（~5 min）

```powershell
# SQuAD
python scripts\p0_extract_features.py ^
  --human_rewrite_json SQuAD\rewrite_squad_human.json ^
  --ai_rewrite_json SQuAD\rewrite_squad_ai.json ^
  --out_npz outputs\p0_baseline\features\squad_features.npz

# CC-News
python scripts\p0_extract_features.py ^
  --human_rewrite_json CCNews\rewrite_ccnews_human.json ^
  --ai_rewrite_json CCNews\rewrite_ccnews_ai.json ^
  --out_npz outputs\p0_baseline\features\ccnews_features.npz
```

**特征说明（5 维，对齐 RAIDAR 定义）：**

| 维度 | 特征 | 计算方式 |
|------|------|---------|
| 1 | Levenshtein ratio | `1 - lev(input, rewrite) / max(len(input), len(rewrite))` |
| 2 | 1-gram overlap | `共有 unigram 数 / input unigram 总数` |
| 3 | 2-gram overlap | `共有 bigram 数 / input bigram 总数` |
| 4 | 3-gram overlap | `共有 trigram 数 / input trigram 总数` |
| 5 | 4-gram overlap | `共有 4-gram 数 / input 4-gram 总数` |

### Step 5：训练分类器 + 评估（~2 min）

```powershell
# SQuAD
python scripts\p0_train_eval.py ^
  --npz outputs\p0_baseline\features\squad_features.npz ^
  --out_dir outputs\p0_baseline\metrics ^
  --dataset_name squad

# CC-News
python scripts\p0_train_eval.py ^
  --npz outputs\p0_baseline\features\ccnews_features.npz ^
  --out_dir outputs\p0_baseline\metrics ^
  --dataset_name ccnews
```

**分类器与评估指标：**

| 分类器 | 角色 |
|--------|------|
| LR | 主分类器（对齐 RAIDAR 原论文 Table 15） |
| XGBoost | 辅助，验证非线性边界 |
| MLP | 对照，对齐原仓库实现 |

| 指标 | 说明 |
|------|------|
| Accuracy | 分类准确率 |
| F1 | 二分类 F1 |
| AUROC | ROC 曲线下面积 |
| TPR@FPR=1% | FPR ≤ 1% 时的最大 TPR |
| TPR@FPR=5% | FPR ≤ 5% 时的最大 TPR |

每个分类器跑 10 个 seed（42~51），输出均值 ± 标准差。

### Step 6：导出结果总表 + 图（~1 min）

```powershell
python scripts\p0_export_results.py --metrics_dir outputs\p0_baseline\metrics
```

产物：
- `outputs\p0_baseline\metrics\p0_baseline_results.csv`
- `outputs\p0_baseline\metrics\p0_baseline_results.md`
- `outputs\p0_baseline\figures\p0_f1.png / .svg`
- `outputs\p0_baseline\figures\p0_auroc.png / .svg`
- `outputs\p0_baseline\figures\p0_accuracy.png / .svg`

### Step 7：复现留痕

```powershell
# 冻结依赖版本
python -m pip freeze > outputs\repro\requirements.txt

# 保存配置快照
Copy-Item configs\p0_config.json outputs\repro\p0_config_snapshot.json

# 生成所有产物的 SHA256
Get-ChildItem -Recurse outputs\p0_baseline -File | ForEach-Object {
    "{0}`t{1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.FullName
} | Set-Content -Encoding UTF8 outputs\repro\p0_manifest_sha256.tsv
```

---

## 五、执行时间速查

| 步骤 | 脚本 | 预计耗时 |
|------|------|---------|
| Step 1 | `p0_sample_corpus.py` ×2 | ~5 min |
| Step 2 | `p0_generate_ai_text.py` ×2 | **2~4 h（API 调用）** |
| Step 3 | `p0_rewrite.py` ×4 | **4~8 h（API 调用）** |
| Step 4 | `p0_extract_features.py` ×2 | ~5 min |
| Step 5 | `p0_train_eval.py` ×2 | ~2 min |
| Step 6 | `p0_export_results.py` | ~1 min |
| Step 7 | 复现留痕命令 | ~1 min |

---

## 六、关键设计说明

| 决策 | 选择 | 理由 |
|------|------|------|
| 重写提示词 | 区间约束 ±10% | 避免模型为凑精确词数扭曲语义，更稳健 |
| AI 文本来源 | 元信息驱动生成（title → 段落） | 与人类文本同域同长度，对比更公平 |
| 主分类器 | LR | 对齐 RAIDAR 原论文 Table 15 |
| 特征维度 | 5 维 | 对齐 RAIDAR 定义，不引入额外归纳偏置 |
| 评估 seed | 10 个（42~51） | 与仓库其他实验保持一致 |
