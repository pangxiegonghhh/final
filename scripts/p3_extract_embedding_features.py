# scripts/p3_extract_embedding_features.py
# -*- coding: utf-8 -*-
"""
P3 Embedding: 用 sentence-transformers 提取 384 维 embedding 作为路由特征
输出: {domain}_meta_features.npz (X_meta: 4000×384, y: 4000)
注意: 输出格式与 p3_extract_meta_features.py 完全一致（同名 key），
      因此 p3_train_routing.py 无需修改，直接换目录即可。
"""
import argparse, os, json
import numpy as np
from sentence_transformers import SentenceTransformer

DOMAIN_DIRS = {"ccnews": "CCNews", "squad": "SQuAD"}

def load_inputs(project_root, domain, text_type):
    path = os.path.join(project_root, DOMAIN_DIRS[domain],
                        "rewrites_p1", f"{text_type}_P0_baseline.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["input"] for item in data]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--out_dir", default=r"outputs\p3_routing\embedding_features")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    for domain in ["ccnews", "squad"]:
        print(f"\n=== {domain} ===")
        human_inputs = load_inputs(args.project_root, domain, "human")
        ai_inputs = load_inputs(args.project_root, domain, "ai")
        all_inputs = human_inputs + ai_inputs
        print(f"  Encoding {len(all_inputs)} texts...")

        embeddings = model.encode(all_inputs, batch_size=args.batch_size,
                                  show_progress_bar=True, normalize_embeddings=True)

        y = np.concatenate([np.zeros(len(human_inputs)), np.ones(len(ai_inputs))])

        out_path = os.path.join(args.out_dir, f"{domain}_meta_features.npz")
        np.savez(out_path, X_meta=embeddings, y=y)
        print(f"  shape={embeddings.shape} -> {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()