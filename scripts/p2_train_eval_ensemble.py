# scripts/p2_train_eval_ensemble.py
# -*- coding: utf-8 -*-
"""
P2 Step 3: 对集成特征做 10-seed LR 训练评估
与 P0/P1 完全对齐：80/20 stratified split, LR, 10 seeds (42-51)
"""
import argparse, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_single_seed(X, y, seed, C=1.0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=C, max_iter=1000, random_state=seed,
                             class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    return acc, f1, auroc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="特征 npz 文件路径")
    parser.add_argument("--out_json", required=True, help="输出 JSON 路径")
    parser.add_argument("--tag", default="", help="实验标签")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0, help="LR 正则化参数")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"Loaded: X.shape={X.shape}, y.shape={y.shape}, tag={args.tag}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    acc_list, f1_list, auroc_list = [], [], []

    for seed in seeds:
        acc, f1, auroc = evaluate_single_seed(X, y, seed, C=args.C)
        acc_list.append(acc)
        f1_list.append(f1)
        auroc_list.append(auroc)
        print(f"  seed={seed}: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")

    result = {
        "tag": args.tag,
        "npz": args.npz,
        "dims": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "seeds": seeds,
        "C": args.C,
        "acc_list": acc_list,
        "f1_list": f1_list,
        "auroc_list": auroc_list,
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "auroc_mean": float(np.mean(auroc_list)),
        "auroc_std": float(np.std(auroc_list)),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out_json}")
    print(f"  F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}")
    print(f"  AUROC={result['auroc_mean']:.4f}±{result['auroc_std']:.4f}")

if __name__ == "__main__":
    main()