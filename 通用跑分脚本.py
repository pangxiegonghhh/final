# detect_inv_generic.py
# -*- coding: utf-8 -*-
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score


def tokenize_and_normalize(sentence: str):
    return [word.lower().strip() for word in sentence.split()]


def extract_ngrams(tokens, n: int):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def common_elements(list1, list2):
    return set(list1) & set(list2)


def calculate_sentence_common(sentence1: str, sentence2: str):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    common_words = common_elements(tokens1, tokens2)
    number_common_hierarchy = [len(common_words)]

    for n in range(2, 5):  # 2-grams..4-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2)
        number_common_hierarchy.append(len(common_ngrams))

    return number_common_hierarchy


def sum_for_list(a, b):
    return [aa + bb for aa, bb in zip(a, b)]


def get_data_stat(data_json, cutoff_start, cutoff_end, max_idx, ngram_num):
    total_len = len(data_json)
    for idxx, each in enumerate(data_json):
        original = each.get("input", "")
        raw = tokenize_and_normalize(original)
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue

        if idxx % 50 == 0:
            print(f"processing idx={idxx} / total={total_len}")

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for _ in range(ngram_num)]
        cnt = 0
        whole_combined = ""

        for pp in list(each.keys()):
            if pp == "common_features":
                continue
            if pp == "fzwz_features":
                continue
            if pp == "avg_common_features":
                continue
            if pp == "common_features_ori_vs_allcombined":
                continue

            # 跳过 input 自己
            if pp == "input":
                continue

            whole_combined += (" " + str(each[pp]))

            res = calculate_sentence_common(original, str(each[pp]))
            statistic_res[pp] = res
            all_statistic_res = sum_for_list(all_statistic_res, res)

            ratio_fzwz[pp] = [
                fuzz.ratio(original, str(each[pp])),
                fuzz.token_set_ratio(original, str(each[pp])),
            ]
            cnt += 1

        each["fzwz_features"] = ratio_fzwz
        each["common_features"] = statistic_res
        each["avg_common_features"] = [a / cnt for a in all_statistic_res] if cnt > 0 else all_statistic_res
        each["common_features_ori_vs_allcombined"] = calculate_sentence_common(original, whole_combined)

        if idxx == max_idx:
            break

    return data_json


def get_feature_vec(input_json, cutoff_start, cutoff_end, max_idx):
    all_list = []
    for idxx, each in enumerate(input_json):
        raw = tokenize_and_normalize(each["input"])
        r_len = float(len(raw))
        if r_len == 0:
            continue
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue

        each_data_fea = []
        each_data_fea.extend([ind_d / r_len for ind_d in each["avg_common_features"]])

        for ek in each["common_features"].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each["common_features"][ek]])

        each_data_fea.extend([ind_d / r_len for ind_d in each["common_features_ori_vs_allcombined"]])

        for ek in each["fzwz_features"].keys():
            each_data_fea.extend(each["fzwz_features"][ek])

        all_list.append(np.array(each_data_fea, dtype=np.float32))

        if idxx == max_idx:
            break

    if not all_list:
        raise RuntimeError("No features extracted. Check json format or cutoff/max_idx.")
    return np.vstack(all_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True, help="path to GPT rewrite json")
    ap.add_argument("--human", required=True, help="path to Human rewrite json")
    ap.add_argument("--out", required=True, help="output results json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_idx", type=int, default=400)
    ap.add_argument("--cutoff_start", type=int, default=0)
    ap.add_argument("--cutoff_end", type=int, default=6_000_000)
    args = ap.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    gpt_path = Path(args.gpt)
    human_path = Path(args.human)
    out_path = Path(args.out)

    with open(gpt_path, "r", encoding="utf-8") as f:
        data_gpt = json.load(f)
    with open(human_path, "r", encoding="utf-8") as f:
        data_human = json.load(f)

    NGRAM_NUM = 4

    gpt_stats = get_data_stat(data_gpt, args.cutoff_start, args.cutoff_end, args.max_idx, NGRAM_NUM)
    human_stats = get_data_stat(data_human, args.cutoff_start, args.cutoff_end, args.max_idx, NGRAM_NUM)

    gpt_all = get_feature_vec(gpt_stats, args.cutoff_start, args.cutoff_end, args.max_idx)
    human_all = get_feature_vec(human_stats, args.cutoff_start, args.cutoff_end, args.max_idx)

    h_train, h_test, yh_train, yh_test = train_test_split(
        human_all, np.zeros(human_all.shape[0]),
        test_size=0.2, random_state=SEED, shuffle=True
    )
    g_train, g_test, yg_train, yg_test = train_test_split(
        gpt_all, np.ones(gpt_all.shape[0]),
        test_size=0.2, random_state=SEED, shuffle=True
    )

    X_train = np.concatenate((g_train, h_train), axis=0)
    y_train = np.concatenate((yg_train, yh_train), axis=0)
    X_test = np.concatenate((g_test, h_test), axis=0)
    y_test = np.concatenate((yg_test, yh_test), axis=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=1000,
        activation="relu",
        solver="adam",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    print("Accuracy:", acc, "F1:", f1)
    print(classification_report(y_test, y_pred))

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": {"gpt": str(gpt_path), "human": str(human_path)},
        "data": {
            "gpt_samples_used": int(gpt_all.shape[0]),
            "human_samples_used": int(human_all.shape[0]),
            "feature_dim": int(gpt_all.shape[1]),
            "max_idx_stop": int(args.max_idx),
            "cutoff_start": int(args.cutoff_start),
            "cutoff_end": int(args.cutoff_end),
        },
        "model": {
            "name": "MLPClassifier",
            "hidden_layer_sizes": [10],
            "max_iter": 1000,
            "activation": "relu",
            "solver": "adam",
        },
        "metrics": {"accuracy": acc, "f1": f1},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()