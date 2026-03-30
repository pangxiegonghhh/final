# Stage C Summary: ALL prompts vs TOP-3 prompts

| Domain | Rewriter | Setting | #Prompts | Accuracy (mean±std) | F1 (mean±std) |
|---|---|---|---:|---:|---:|
| Arxiv | deepseek | ALL | 7 | 0.6657 ± 0.0267 | 0.6538 ± 0.0412 |
| Arxiv | deepseek | TOP3 | 3 | 0.6329 ± 0.0494 | 0.6160 ± 0.0603 |
| Arxiv | repo | ALL | 7 | 0.8443 ± 0.0153 | 0.8409 ± 0.0140 |
| Arxiv | repo | TOP3 | 3 | 0.8179 ± 0.0266 | 0.8142 ± 0.0289 |
| Code | deepseek | ALL | 5 | 0.8167 ± 0.0249 | 0.8097 ± 0.0274 |
| Code | deepseek | TOP3 | 3 | 0.8379 ± 0.0296 | 0.8331 ± 0.0294 |
| Code | repo | ALL | 5 | 0.9545 ± 0.0235 | 0.9525 ± 0.0249 |
| Code | repo | TOP3 | 3 | 0.9576 ± 0.0242 | 0.9556 ± 0.0260 |
| Yelp | deepseek | ALL | 7 | 0.7432 ± 0.0390 | 0.7405 ± 0.0396 |
| Yelp | deepseek | TOP3 | 3 | 0.7469 ± 0.0197 | 0.7491 ± 0.0174 |
| Yelp | repo | ALL | 7 | 0.8451 ± 0.0227 | 0.8447 ± 0.0252 |
| Yelp | repo | TOP3 | 3 | 0.8481 ± 0.0173 | 0.8489 ± 0.0184 |