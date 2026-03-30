# scripts/p3_train_routing.py
# -*- coding: utf-8 -*-
"""
P3 Step 2: 训练 prompt 专家 + 路由器，评估 Routing Top-1/Top-2
核心流程（每个 seed）:
  1. 60/20/20 split -> train/dev/test
  2. 在 train 上训练 7 个专家 LR（每个 prompt 的 5 维特征）
  3. 在 dev 上用专家预测，构造 oracle label（每条样本最佳 prompt）
  4. 在 train+dev 的元特征上训练路由器 LR
  5. 在 test 上评估: Routing Top-1, Top-2, Random, Best-Single
"""
import argparse, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ALL_PROMPTS = [
    "P0_baseline", "P1_fluency", "P2_synonym",
    "P3_restructure", "P4_formal", "P5_concise", "P6_no_length"
]

def load_prompt_features(feat_dir, domain, prompt):
    """加载单 prompt 的 5 维特征"""
    path = os.path.join(feat_dir, f"{domain}_{prompt}.npz")
    data = np.load(path)
    X = np.vstack([data["X_human"], data["X_ai"]])
    y = np.concatenate([np.zeros(len(data["X_human"])), np.ones(len(data["X_ai"]))])
    return X, y

def run_one_seed(all_prompt_X, meta_X, y, seed, domain, C_expert=1.0, C_router=1.0):
    """单个 seed 的完整流水线"""
    n = len(y)
    indices = np.arange(n)

    # 60/20/20 split
    train_dev_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y)
    y_train_dev = y[train_dev_idx]
    train_idx, dev_idx = train_test_split(
        train_dev_idx, test_size=0.25, random_state=seed, stratify=y_train_dev)
    # 此时 train:dev:test ≈ 60:20:20

    # === Step 1: 训练 7 个专家 ===
    experts = {}
    expert_scalers = {}
    for i, prompt in enumerate(ALL_PROMPTS):
        X_p = all_prompt_X[i]  # (4000, 5)
        X_train = X_p[train_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        clf = LogisticRegression(C=C_expert, max_iter=1000, random_state=seed,
                                 class_weight="balanced", solver="lbfgs")
        clf.fit(X_train_s, y_train)

        experts[prompt] = clf
        expert_scalers[prompt] = scaler

    # === Step 2: 在 dev 上构造 oracle label ===
    oracle_labels = []  # 每条 dev 样本的最佳 prompt index
    for idx in dev_idx:
        best_prompt_idx = -1
        best_confidence = -1
        true_label = int(y[idx])

        for pi, prompt in enumerate(ALL_PROMPTS):
            X_sample = all_prompt_X[pi][idx:idx+1]
            X_sample_s = expert_scalers[prompt].transform(X_sample)
            prob = experts[prompt].predict_proba(X_sample_s)[0]
            # 正确类的置信度
            confidence = prob[true_label]
            if confidence > best_confidence:
                best_confidence = confidence
                best_prompt_idx = pi

        oracle_labels.append(best_prompt_idx)

    oracle_labels = np.array(oracle_labels)

    # Oracle 准确率统计
    oracle_distribution = {ALL_PROMPTS[i]: int(np.sum(oracle_labels == i))
                           for i in range(len(ALL_PROMPTS))}

    # === Step 3: 训练路由器 ===
    # 用 train+dev 的元特征训练（dev 提供了 oracle label）
    # 但 oracle label 只存在于 dev，所以路由器只在 dev 上训练
    meta_dev = meta_X[dev_idx]
    meta_scaler = StandardScaler()
    meta_dev_s = meta_scaler.fit_transform(meta_dev)

    router = LogisticRegression(C=C_router, max_iter=1000, random_state=seed,
                                multi_class="multinomial", solver="lbfgs",
                                class_weight="balanced")
    router.fit(meta_dev_s, oracle_labels)

    # 路由器在 dev 上的准确率
    router_dev_acc = accuracy_score(oracle_labels, router.predict(meta_dev_s))

    # === Step 4: 在 test 上评估 ===
    y_test = y[test_idx]
    meta_test = meta_X[test_idx]
    meta_test_s = meta_scaler.transform(meta_test)

    # 路由器预测
    router_probs = router.predict_proba(meta_test_s)  # (n_test, n_classes)
    # 注意: router.classes_ 可能不是 0-6 的完整集合（如果某个 prompt 在 oracle 中从未被选为最佳）
    classes = router.classes_

    # --- Routing Top-1 ---
    top1_indices = np.argmax(router_probs, axis=1)
    top1_prompts = classes[top1_indices]

    # 对每条 test 样本，用路由器选的 prompt 的专家来预测
    top1_preds = []
    top1_probs_list = []
    for j, test_i in enumerate(test_idx):
        pi = top1_prompts[j]
        prompt = ALL_PROMPTS[pi]
        X_sample = all_prompt_X[pi][test_i:test_i+1]
        X_sample_s = expert_scalers[prompt].transform(X_sample)
        pred = experts[prompt].predict(X_sample_s)[0]
        prob = experts[prompt].predict_proba(X_sample_s)[0, 1]
        top1_preds.append(pred)
        top1_probs_list.append(prob)

    top1_acc = accuracy_score(y_test, top1_preds)
    top1_f1 = f1_score(y_test, top1_preds)
    top1_auroc = roc_auc_score(y_test, top1_probs_list)

    # --- Routing Top-2 ---
    top2_preds = []
    top2_probs_list = []
    for j, test_i in enumerate(test_idx):
        sorted_cls = classes[np.argsort(-router_probs[j])][:2]
        # 拼接 top-2 prompt 的特征 (10维)
        feats = []
        for pi in sorted_cls:
            prompt = ALL_PROMPTS[pi]
            X_sample = all_prompt_X[pi][test_i:test_i+1]
            X_sample_s = expert_scalers[prompt].transform(X_sample)
            feats.append(X_sample_s[0])
        feat_concat = np.concatenate(feats).reshape(1, -1)

        # 用简单平均两个专家的概率
        probs = []
        for pi in sorted_cls:
            prompt = ALL_PROMPTS[pi]
            X_sample = all_prompt_X[pi][test_i:test_i+1]
            X_sample_s = expert_scalers[prompt].transform(X_sample)
            probs.append(experts[prompt].predict_proba(X_sample_s)[0, 1])
        avg_prob = np.mean(probs)
        pred = 1 if avg_prob >= 0.5 else 0
        top2_preds.append(pred)
        top2_probs_list.append(avg_prob)

    top2_acc = accuracy_score(y_test, top2_preds)
    top2_f1 = f1_score(y_test, top2_preds)
    top2_auroc = roc_auc_score(y_test, top2_probs_list)

    # --- Random Baseline (Top-1 随机选) ---
    rng = np.random.RandomState(seed)
    random_prompts = rng.randint(0, len(ALL_PROMPTS), size=len(test_idx))
    rand_preds = []
    rand_probs = []
    for j, test_i in enumerate(test_idx):
        pi = random_prompts[j]
        prompt = ALL_PROMPTS[pi]
        X_sample = all_prompt_X[pi][test_i:test_i+1]
        X_sample_s = expert_scalers[prompt].transform(X_sample)
        pred = experts[prompt].predict(X_sample_s)[0]
        prob = experts[prompt].predict_proba(X_sample_s)[0, 1]
        rand_preds.append(pred)
        rand_probs.append(prob)

    rand_acc = accuracy_score(y_test, rand_preds)
    rand_f1 = f1_score(y_test, rand_preds)
    rand_auroc = roc_auc_score(y_test, rand_probs)

    # --- Oracle Top-1 (上界: 如果路由器100%准确) ---
    oracle_test_preds = []
    oracle_test_probs = []
    for j, test_i in enumerate(test_idx):
        best_pi = -1
        best_conf = -1
        true_label = int(y_test[j])
        for pi, prompt in enumerate(ALL_PROMPTS):
            X_sample = all_prompt_X[pi][test_i:test_i+1]
            X_sample_s = expert_scalers[prompt].transform(X_sample)
            prob = experts[prompt].predict_proba(X_sample_s)[0]
            conf = prob[true_label]
            if conf > best_conf:
                best_conf = conf
                best_pi = pi
        prompt = ALL_PROMPTS[best_pi]
        X_sample = all_prompt_X[best_pi][test_i:test_i+1]
        X_sample_s = expert_scalers[prompt].transform(X_sample)
        pred = experts[prompt].predict(X_sample_s)[0]
        prob_ai = experts[prompt].predict_proba(X_sample_s)[0, 1]
        oracle_test_preds.append(pred)
        oracle_test_probs.append(prob_ai)

    oracle_acc = accuracy_score(y_test, oracle_test_preds)
    oracle_f1 = f1_score(y_test, oracle_test_preds)
    oracle_auroc = roc_auc_score(y_test, oracle_test_probs)

    return {
        "seed": seed,
        "domain": domain,
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "router_dev_acc": float(router_dev_acc),
        "oracle_distribution": oracle_distribution,
        "routing_top1": {"acc": top1_acc, "f1": top1_f1, "auroc": top1_auroc},
        "routing_top2": {"acc": top2_acc, "f1": top2_f1, "auroc": top2_auroc},
        "random_top1": {"acc": rand_acc, "f1": rand_f1, "auroc": rand_auroc},
        "oracle_top1": {"acc": oracle_acc, "f1": oracle_f1, "auroc": oracle_auroc},
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", default=r"outputs\p1_sensitivity\features",
                        help="P1 特征目录")
    parser.add_argument("--meta_dir", default=r"outputs\p3_routing\meta_features",
                        help="步骤1 输出的元特征目录")
    parser.add_argument("--out_dir", default=r"outputs\p3_routing\metrics",
                        help="输出目录")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--C_expert", type=float, default=1.0)
    parser.add_argument("--C_router", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    for domain in ["ccnews", "squad"]:
        print(f"\n{'='*50}")
        print(f"  Domain: {domain}")
        print(f"{'='*50}")

        # 加载 7 个 prompt 的特征
        all_prompt_X = []
        y = None
        for prompt in ALL_PROMPTS:
            X_p, y_p = load_prompt_features(args.feat_dir, domain, prompt)
            all_prompt_X.append(X_p)
            if y is None:
                y = y_p
        # all_prompt_X: list of 7 arrays, each (4000, 5)

        # 加载元特征
        meta_data = np.load(os.path.join(args.meta_dir, f"{domain}_meta_features.npz"))
        meta_X = meta_data["X_meta"]  # (4000, 8)

        # 逐 seed 运行
        all_results = []
        top1_f1_list = []
        top2_f1_list = []
        rand_f1_list = []
        oracle_f1_list = []

        for seed in seeds:
            print(f"\n  --- seed={seed} ---")
            result = run_one_seed(all_prompt_X, meta_X, y, seed, domain,
                                 C_expert=args.C_expert, C_router=args.C_router)
            all_results.append(result)

            top1_f1_list.append(result["routing_top1"]["f1"])
            top2_f1_list.append(result["routing_top2"]["f1"])
            rand_f1_list.append(result["random_top1"]["f1"])
            oracle_f1_list.append(result["oracle_top1"]["f1"])

            print(f"    Router dev acc: {result['router_dev_acc']:.4f}")
            print(f"    Routing Top-1 F1: {result['routing_top1']['f1']:.4f}")
            print(f"    Routing Top-2 F1: {result['routing_top2']['f1']:.4f}")
            print(f"    Random  Top-1 F1: {result['random_top1']['f1']:.4f}")
            print(f"    Oracle  Top-1 F1: {result['oracle_top1']['f1']:.4f}")

        # 汇总
        summary = {
            "domain": domain,
            "seeds": seeds,
            "C_expert": args.C_expert,
            "C_router": args.C_router,
            "routing_top1": {
                "f1_list": top1_f1_list,
                "f1_mean": float(np.mean(top1_f1_list)),
                "f1_std": float(np.std(top1_f1_list)),
            },
            "routing_top2": {
                "f1_list": top2_f1_list,
                "f1_mean": float(np.mean(top2_f1_list)),
                "f1_std": float(np.std(top2_f1_list)),
            },
            "random_top1": {
                "f1_list": rand_f1_list,
                "f1_mean": float(np.mean(rand_f1_list)),
                "f1_std": float(np.std(rand_f1_list)),
            },
            "oracle_top1": {
                "f1_list": oracle_f1_list,
                "f1_mean": float(np.mean(oracle_f1_list)),
                "f1_std": float(np.std(oracle_f1_list)),
            },
            "per_seed_details": all_results,
        }

        out_path = os.path.join(args.out_dir, f"{domain}_routing.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Summary for {domain}:")
        print(f"    Routing Top-1 F1: {summary['routing_top1']['f1_mean']:.4f}±{summary['routing_top1']['f1_std']:.4f}")
        print(f"    Routing Top-2 F1: {summary['routing_top2']['f1_mean']:.4f}±{summary['routing_top2']['f1_std']:.4f}")
        print(f"    Random  Top-1 F1: {summary['random_top1']['f1_mean']:.4f}±{summary['random_top1']['f1_std']:.4f}")
        print(f"    Oracle  Top-1 F1: {summary['oracle_top1']['f1_mean']:.4f}±{summary['oracle_top1']['f1_std']:.4f}")
        print(f"  -> {out_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()