import json, random, os
from pathlib import Path

SEED=42
N=50

for domain, gpt_file, human_file in [
    ("Yelp", r"Yelp\rewrite_yelp_GPT_inv.json", r"Yelp\rewrite_yelp_human_inv.json"),
    ("Code", r"Code\rewrite_code_GPT_inv.json", r"Code\rewrite_code_human_inv.json"),
    ("Arxiv", r"Arxiv\rewrite_arxiv_GPT_inv.json", r"Arxiv\rewrite_arxiv_human_inv.json"),
]:
    g = json.load(open(gpt_file, "r", encoding="utf-8"))
    h = json.load(open(human_file, "r", encoding="utf-8"))
    m = min(len(g), len(h))
    random.seed(SEED)
    idx = random.sample(range(m), N)
    out = Path(domain) / f"sample_idx_{N}_seed{SEED}.json"
    out.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(domain, "min_len=", m, "saved", out)