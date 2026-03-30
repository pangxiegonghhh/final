# Stage C Paired Tests: TOP5/TOP3 minus ALL (10 seeds paired)

说明：差值定义为 K - ALL；paired t-test + Wilcoxon；CI 为 bootstrap(BCa) 95%。

| Group | K | Metric | mean(diff) | t p | wilcoxon p | dz | CI95 low | CI95 high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arxiv/repo | TOP5 | acc | 0.0043 | 0.5724 | 0.6094 | 0.185 | -0.0100 | 0.0171 |
| Arxiv/repo | TOP5 | f1 | 0.0027 | 0.7127 | 0.7695 | 0.120 | -0.0116 | 0.0150 |
| Arxiv/repo | TOP3 | acc | -0.0264 | 0.007006 | 0.01562 | -1.099 | -0.0407 | -0.0129 |
| Arxiv/repo | TOP3 | f1 | -0.0268 | 0.009776 | 0.01953 | -1.032 | -0.0420 | -0.0118 |
| Arxiv/deepseek | TOP5 | acc | -0.0371 | 0.03715 | 0.05664 | -0.773 | -0.0629 | -0.0057 |
| Arxiv/deepseek | TOP5 | f1 | -0.0359 | 0.05337 | 0.06445 | -0.703 | -0.0591 | 0.0046 |
| Arxiv/deepseek | TOP3 | acc | -0.0329 | 0.0813 | 0.1055 | -0.621 | -0.0650 | -0.0036 |
| Arxiv/deepseek | TOP3 | f1 | -0.0378 | 0.1172 | 0.1309 | -0.548 | -0.0761 | 0.0043 |
| Code/repo | TOP5 | acc | 0.0000 | nan | 1 | nan | nan | nan |
| Code/repo | TOP5 | f1 | 0.0000 | nan | 1 | nan | nan | nan |
| Code/repo | TOP3 | acc | 0.0030 | 0.3434 | 0.5 | 0.316 | -0.0030 | 0.0091 |
| Code/repo | TOP3 | f1 | 0.0031 | 0.377 | 0.875 | 0.294 | -0.0036 | 0.0083 |
| Code/deepseek | TOP5 | acc | 0.0000 | nan | 1 | nan | nan | nan |
| Code/deepseek | TOP5 | f1 | 0.0000 | nan | 1 | nan | nan | nan |
| Code/deepseek | TOP3 | acc | 0.0212 | 0.1727 | 0.2188 | 0.468 | -0.0030 | 0.0500 |
| Code/deepseek | TOP3 | f1 | 0.0234 | 0.183 | 0.2324 | 0.456 | -0.0042 | 0.0558 |
| Yelp/repo | TOP5 | acc | 0.0049 | 0.4749 | 0.3281 | 0.236 | -0.0099 | 0.0154 |
| Yelp/repo | TOP5 | f1 | 0.0050 | 0.5133 | 0.375 | 0.215 | -0.0116 | 0.0162 |
| Yelp/repo | TOP3 | acc | 0.0031 | 0.596 | 0.6094 | 0.174 | -0.0093 | 0.0123 |
| Yelp/repo | TOP3 | f1 | 0.0042 | 0.5314 | 0.5566 | 0.206 | -0.0097 | 0.0148 |
| Yelp/deepseek | TOP5 | acc | 0.0093 | 0.3146 | 0.3145 | 0.337 | -0.0068 | 0.0259 |
| Yelp/deepseek | TOP5 | f1 | 0.0139 | 0.2289 | 0.2754 | 0.408 | -0.0050 | 0.0351 |
| Yelp/deepseek | TOP3 | acc | 0.0037 | 0.7705 | 0.8262 | 0.095 | -0.0204 | 0.0259 |
| Yelp/deepseek | TOP3 | f1 | 0.0086 | 0.4671 | 0.4316 | 0.240 | -0.0151 | 0.0277 |