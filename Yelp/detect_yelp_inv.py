# detect_yelp_inv.py
# -*- coding: utf-8 -*-

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


# =========================
# 0) Reproducibility config
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent  # Yelp/
GPT_REWRITE_FILE = BASE_DIR / "rewrite_yelp_GPT_inv.json"
HUMAN_REWRITE_FILE = BASE_DIR / "rewrite_yelp_human_inv.json"
RESULT_FILE = BASE_DIR / "results_yelp_inv.json"

# feature config
NGRAM_NUM = 4  # common words + 2-gram + 3-gram + 4-gram = 4 numbers
CUTOFF_START = 0
CUTOFF_END = 6_000_000

# how many samples to process at most (the original code stops at idxx==400)
MAX_IDX = 400  # inclusive index stop condition in your original code


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

    for n in range(2, 5):  # 2-grams to 4-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2)
        number_common_hierarchy.append(len(common_ngrams))

    return number_common_hierarchy


def sum_for_list(a, b):
    return [aa + bb for aa, bb in zip(a, b)]


def get_data_stat(data_json):
    total_len = len(data_json)

    for idxx, each in enumerate(data_json):
        original = each["input"]

        raw = tokenize_and_normalize(original)
        if len(raw) < CUTOFF_START or len(raw) > CUTOFF_END:
            continue

        # progress log (avoid printing every single line)
        if idxx % 50 == 0:
            print(f"processing idx={idxx} / total={total_len}")

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for _ in range(NGRAM_NUM)]
        cnt = 0
        whole_combined = ""

        for pp in each.keys():
            if pp != "common_features":
                whole_combined += (" " + str(each[pp]))

                res = calculate_sentence_common(original, str(each[pp]))
                statistic_res[pp] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)

                ratio_fzwz[pp] = [
                    fuzz.ratio(original, str(each[pp])),
                    fuzz.token_set_ratio(original, str(each[pp]))
                ]
                cnt += 1

        each["fzwz_features"] = ratio_fzwz
        each["common_features"] = statistic_res
        each["avg_common_features"] = [a / cnt for a in all_statistic_res] if cnt > 0 else all_statistic_res
        each["common_features_ori_vs_allcombined"] = calculate_sentence_common(original, whole_combined)

        if idxx == MAX_IDX:
            break

    return data_json


def get_feature_vec(input_json):
    all_list = []

    for idxx, each in enumerate(input_json):
        raw = tokenize_and_normalize(each["input"])
        r_len = float(len(raw))

        if r_len == 0:
            continue
        if len(raw) < CUTOFF_START or len(raw) > CUTOFF_END:
            continue

        each_data_fea = []

        # avg common features normalized by length
        each_data_fea.extend([ind_d / r_len for ind_d in each["avg_common_features"]])

        # per rewrite common features normalized by length
        for ek in each["common_features"].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each["common_features"][ek]])

        # original vs all combined
        each_data_fea.extend([ind_d / r_len for ind_d in each["common_features_ori_vs_allcombined"]])

        # fuzzywuzzy features (two scalars per rewrite)
        for ek in each["fzwz_features"].keys():
            each_data_fea.extend(each["fzwz_features"][ek])

        all_list.append(np.array(each_data_fea, dtype=np.float32))

        if idxx == MAX_IDX:
            break

    if len(all_list) == 0:
        raise RuntimeError("No features extracted. Check input json format or cutoff limits.")

    return np.vstack(all_list)


def main():
    # 1) load rewrite json
    with open(GPT_REWRITE_FILE, "r", encoding="utf-8") as f:
        data_gpt = json.load(f)

    with open(HUMAN_REWRITE_FILE, "r", encoding="utf-8") as f:
        data_human = json.load(f)

    # 2) compute stats/features
    gpt_stats = get_data_stat(data_gpt)
    human_stats = get_data_stat(data_human)

    gpt_all = get_feature_vec(gpt_stats)
    human_all = get_feature_vec(human_stats)

    # 3) balanced split (keep your original logic: split each class then concat)
    h_train, h_test, yh_train, yh_test = train_test_split(
        human_all,
        np.zeros(human_all.shape[0]),
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )
    g_train, g_test, yg_train, yg_test = train_test_split(
        gpt_all,
        np.ones(gpt_all.shape[0]),
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    X_train = np.concatenate((g_train, h_train), axis=0)
    y_train = np.concatenate((yg_train, yh_train), axis=0)

    X_test = np.concatenate((g_test, h_test), axis=0)
    y_test = np.concatenate((yg_test, yh_test), axis=0)

    # 4) standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5) train classifier
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

    print("Accuracy:", acc, "F1 score", f1)
    print(classification_report(y_test, y_pred))

    # 6) save results
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": {
            "gpt_rewrite": str(GPT_REWRITE_FILE.name),
            "human_rewrite": str(HUMAN_REWRITE_FILE.name),
        },
        "data": {
            "gpt_samples_used": int(gpt_all.shape[0]),
            "human_samples_used": int(human_all.shape[0]),
            "feature_dim": int(gpt_all.shape[1]),
            "max_idx_stop": MAX_IDX,
            "cutoff_start": CUTOFF_START,
            "cutoff_end": CUTOFF_END,
        },
        "model": {
            "name": "MLPClassifier",
            "hidden_layer_sizes": [10],
            "max_iter": 1000,
            "activation": "relu",
            "solver": "adam",
        },
        "metrics": {
            "accuracy": acc,
            "f1": f1,
        },
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved to {RESULT_FILE.name}")


if __name__ == "__main__":
    main()