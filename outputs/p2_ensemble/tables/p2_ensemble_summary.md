# P2 Ensemble Results Summary

| Domain | Strategy | Dims | F1 (mean±std) | AUROC (mean±std) | Cost |
|--------|----------|------|---------------|------------------|------|
| ccnews | Best-Single (P4_formal) | 5 | 0.7949±0.0054 | 0.8700±0.0034 | 1 |
| squad | Best-Single (P5_concise) | 5 | 0.7295±0.0166 | 0.7996±0.0109 | 1 |
| ccnews | ensemble_all | 35 | 0.9024±0.0097 | 0.9676±0.0029 | 7 |
| ccnews | ensemble_all_cons | 39 | 0.9252±0.0050 | 0.9790±0.0024 | 7 |
| ccnews | ensemble_top3 | 15 | 0.8510±0.0098 | 0.9243±0.0058 | 3 |
| squad | ensemble_all | 35 | 0.8304±0.0162 | 0.9120±0.0116 | 7 |
| squad | ensemble_all_cons | 39 | 0.9339±0.0077 | 0.9779±0.0037 | 7 |
| squad | ensemble_top3 | 15 | 0.7793±0.0156 | 0.8569±0.0110 | 3 |