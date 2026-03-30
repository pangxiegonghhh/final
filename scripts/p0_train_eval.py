# scripts/p0_train_eval.py
# -*- coding: utf-8 -*-
"""
Step 5: 训练 LR / XGBoost / MLP 分类器，评估 AUROC、F1、TPR@FPR。
10-seed 交叉评估，输出 JSON + 汇总表。

运行命令：
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
"""
import argparse, json, os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost")

# ============================================================
# 评估函数（对齐 RAIDAR 原仓库的指标定义）
# ============================================================

def compute_tpr_at_fpr(y_true, y_prob, target_fpr):
    """
    计算 TPR@FPR=target_fpr。
    逻辑：遍历阈值，找到 FPR <= target_fpr 时的最大 TPR。
    这与 RAIDAR 的 TPR@FPR 定义一致。
    """
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    # 找 fpr <= target_fpr 的最大 tpr
    valid = fpr_arr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr_arr[valid].max())

def evaluate(y_true, y_pred, y_prob):
    """返回一组指标的字典"""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    tpr_1 = compute_tpr_at_fpr(y_true, y_prob, 0.01)
    tpr_5 = compute_tpr_at_fpr(y_true, y_prob, 0.05)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "tpr_at_fpr_1pct": tpr_1,
        "tpr_at_fpr_5pct": tpr_5,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# ============================================================
# 分类器工厂
# ============================================================

def make_classifier(name, seed):
    if name == "LR":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
            solver="lbfgs"
        )
    elif name == "XGBoost":
        if not HAS_XGB:
            return None
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    elif name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15
        )
    else:
        raise ValueError(f"unknown classifier: {name}")

# ============================================================
# 单 seed 训练评估
# ============================================================

def train_eval_one_seed(X, y, clf_name, seed, test_ratio=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = make_classifier(clf_name, seed)
    if clf is None:
        return None

    clf.fit(X_tr_s, y_tr)

    y_pred = clf.predict(X_te_s)

    # 获取概率
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te_s)[:, 1]
    else:
        y_prob = clf.decision_function(X_te_s)

    metrics = evaluate(y_te, y_pred, y_prob)
    return metrics

# ============================================================
# 主函数
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="特征 npz 文件")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--dataset_name", required=True, help="squad / ccnews")
    ap.add_argument("--classifiers", default="LR,XGBoost,MLP")
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--num_seeds", type=int, default=10)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    args = ap.parse_args()

    # 加载特征
    npz = np.load(args.npz)
    X_human = npz["X_human"]  # label=0
    X_ai = npz["X_ai"]        # label=1

    X = np.concatenate([X_human, X_ai], axis=0)
    y = np.concatenate([
        np.zeros(X_human.shape[0], dtype=np.int64),
        np.ones(X_ai.shape[0], dtype=np.int64)
    ])

    print(f"Total samples: {len(y)} (human={X_human.shape[0]}, ai={X_ai.shape[0]})")
    print(f"Feature dim: {X.shape[1]}")

    clf_names = [c.strip() for c in args.classifiers.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for clf_name in clf_names:
        print(f"\n=== {clf_name} ===")
        seed_results = []

        for seed in seeds:
            metrics = train_eval_one_seed(X, y, clf_name, seed, args.test_ratio)
            if metrics is None:
                print(f"  [seed={seed}] skipped (classifier unavailable)")
                continue
            seed_results.append(metrics)
            print(f"  [seed={seed}] acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f} "
                  f"tpr@1%={metrics['tpr_at_fpr_1pct']:.4f} "
                  f"tpr@5%={metrics['tpr_at_fpr_5pct']:.4f}")

        if not seed_results:
            continue

        # 汇总
        summary = {
            "classifier": clf_name,
            "dataset": args.dataset_name,
            "num_seeds": len(seed_results),
            "seeds": seeds[:len(seed_results)],
        }

        for metric_key in ["accuracy", "f1", "auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct"]:
            vals = [r[metric_key] for r in seed_results]
            summary[f"{metric_key}_list"] = vals
            summary[f"{metric_key}_mean"] = float(np.mean(vals))
            summary[f"{metric_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        summary["per_seed"] = seed_results

        # 保存单分类器结果
        out_path = os.path.join(args.out_dir, f"{args.dataset_name}_{clf_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  saved: {out_path}")

        all_results[clf_name] = summary

    # 保存总汇总
    summary_path = os.path.join(args.out_dir, f"{args.dataset_name}_all_classifiers.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved summary: {summary_path}")

    # 打印总表
    print(f"\n{'='*80}")
    print(f"{'Classifier':<12} {'Acc':>12} {'F1':>12} {'AUROC':>12} {'TPR@1%':>12} {'TPR@5%':>12}")
    print(f"{'-'*80}")
    for clf_name, s in all_results.items():
        print(f"{clf_name:<12} "
              f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f} "
              f"{s['auroc_mean']:.4f}±{s['auroc_std']:.4f} "
              f"{s['tpr_at_fpr_1pct_mean']:.4f}±{s['tpr_at_fpr_1pct_std']:.4f} "
              f"{s['tpr_at_fpr_5pct_mean']:.4f}±{s['tpr_at_fpr_5pct_std']:.4f}")

if __name__ == "__main__":
    main()
