# B阶段：提示词敏感性（单 prompt 消融）
## 1.make_prompt_subset_json.py:生成“单 prompt 子集 JSON
## 2.collect_prompt_sensitivity.py:收集单 prompt 结果并画图（mean±std）
## 3.make_prompt_sensitivity_evidence.py：生成单 prompt 消融的证据 将6个单prompt的csv汇总起来计算：计算并输出你论文里最关键的 4 类证据：
### prompt 间分散度（range/std/IQR）
### Top-1 不稳定（top1_change_rate / winner 分布）
### Oracle gap（存在性上界）（oracle_mean - best_single_mean）
### 统计显著性（Friedman test：prompt choice 是否显著影响 F1/Acc）

# 阶段 C：提示词集成（Ensemble）
C1：全 prompts 特征（主线已经具备）
C2：Top-3 prompts（按阶段 B 的平均 F1 选）

## 1.select_topk_prompts.py ：脚本：从阶段 B CSV 自动写出 Top-3 prompt 的 md5 列表
## 2make_prompt_topk_json.py. ：需要一个脚本把 3 个 prompt 合并进同一份 JSON（而不是单 prompt）
## 3.make_stageC_summary.py：C1（全 prompts / baseline） VS C2（Top-3）
## 4.make_stageC_summary_k.py：C1（全 prompts / baseline） VS C2（Top-3） VS C3（Top-5）