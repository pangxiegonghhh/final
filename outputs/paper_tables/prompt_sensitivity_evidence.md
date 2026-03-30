# Stage-B Evidence: Prompt Sensitivity ⇒ Need Ensemble (C) / Routing (D)

解释：
- **range/std across prompts** 大 ⇒ 同域下 prompt 选择敏感（单 prompt 不可靠）
- **top1 change rate** 高 ⇒ 不同 seed 下最优 prompt 频繁切换（需要集成/路由来稳定）
- **oracle gap** > 0 ⇒ 存在可提升上界（集成/路由有客观空间）
- **Friedman p** 小 ⇒ prompt 差异显著（选择 prompt 不是噪声）

| Domain | Rewriter | #Prompts | F1 range | F1 std | Top1 change | Oracle gap | Friedman p |
|---|---|---:|---:|---:|---:|---:|---:|
| Arxiv | deepseek | 7 | 0.0445 | 0.0172 | 0.70 | 0.0176 | 0.2213 |
| Arxiv | repo | 7 | 0.0748 | 0.0261 | 0.40 | 0.0153 | 0.0005693 |
| Code | deepseek | 5 | 0.0776 | 0.0309 | 0.30 | 0.0157 | 0.02754 |
| Code | repo | 5 | 0.0565 | 0.0215 | 0.40 | 0.0049 | 2.171e-05 |
| Yelp | deepseek | 7 | 0.0989 | 0.0335 | 0.20 | 0.0021 | 4.105e-06 |
| Yelp | repo | 7 | 0.0601 | 0.0215 | 0.40 | 0.0135 | 3.545e-05 |