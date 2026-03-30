import os, json, math, glob, re
import numpy as np
from scipy import stats

root = r'outputs\10seed测试'
outdir = r'outputs\stats'
os.makedirs(outdir, exist_ok=True)

pairs = {
    'Yelp':  ('yelp_repo',  'yelp_deepseek'),
    'Code':  ('code_repo',  'code_deepseek'),
    'Arxiv': ('arxiv_repo', 'arxiv_deepseek'),
}

def find(prefix: str) -> str:
    fs = sorted(glob.glob(os.path.join(root, prefix + '*.json')))
    if not fs:
        raise FileNotFoundError(f'missing {prefix}*.json in {root}')
    return fs[0]

def load_lists(p: str):
    j = json.load(open(p, 'r', encoding='utf-8'))
    ms = j.get('multi_seed_summary', {})
    return ms['seeds'], np.array(ms['acc_list'], float), np.array(ms['f1_list'], float)

def bh_fdr(pvals):
    p = np.array(pvals, float)
    m = len(p)
    idx = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    # step-down on sorted p (largest to smallest)
    for rank, i in enumerate(idx[::-1], start=1):
        k = m - rank + 1
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q.tolist()

def rank_biserial_from_wilcoxon_stat(T, n):
    S = n * (n + 1) / 2.0
    return 1.0 - (2.0 * T) / S

def analyze(a, b):
    # diff = DeepSeek - Repo
    d = b - a
    n = len(d)

    shapiro_p = float(stats.shapiro(d).pvalue) if n >= 3 else float('nan')

    tt = stats.ttest_rel(b, a, alternative='two-sided')
    ww = stats.wilcoxon(b, a, alternative='two-sided', zero_method='wilcox', method='auto')

    sd = float(np.std(d, ddof=1))
    dz = float(np.mean(d) / sd) if sd > 0 else float('nan')
    J = 1.0 - 3.0 / (4.0 * n - 1.0) if n > 1 else 1.0
    g = float(J * dz) if not math.isnan(dz) else float('nan')

    r_rb = float(rank_biserial_from_wilcoxon_stat(float(ww.statistic), n))

    boot = stats.bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=20000,
                           method='BCa', random_state=0)
    ci_low = float(boot.confidence_interval.low)
    ci_high = float(boot.confidence_interval.high)

    return {
        'n': int(n),
        'diff_mean': float(np.mean(d)),
        'diff_std': float(sd),
        'shapiro_p': shapiro_p,
        'ttest': {'t': float(tt.statistic), 'p': float(tt.pvalue)},
        'wilcoxon': {'T': float(ww.statistic), 'p': float(ww.pvalue)},
        'effect': {'cohen_dz': dz, 'hedges_g': g, 'rank_biserial_r': r_rb},
        'boot_ci95_mean_diff': {'low': ci_low, 'high': ci_high},
    }

res = {
    'meta': {
        'root': root,
        'compare': 'DeepSeek - Repo',
        'tests': ['paired t-test', 'wilcoxon', 'bootstrap CI (BCa)'],
        'n_resamples_bootstrap': 20000,
    },
    'domains': {},
    'p_adjust': {},
}

p_acc = []
p_f1 = []
keys = []

for dom, (pre_r, pre_d) in pairs.items():
    pr = find(pre_r)
    pd = find(pre_d)

    seeds_r, acc_r, f1_r = load_lists(pr)
    seeds_d, acc_d, f1_d = load_lists(pd)

    if seeds_r != seeds_d:
        raise RuntimeError(f'{dom} seeds mismatch: repo={seeds_r} deepseek={seeds_d}')

    res['domains'][dom] = {
        'paths': {'repo': pr, 'deepseek': pd},
        'seeds': seeds_r,
        'acc': analyze(acc_r, acc_d),
        'f1': analyze(f1_r, f1_d),
    }

    p_acc.append(res['domains'][dom]['acc']['ttest']['p'])
    p_f1.append(res['domains'][dom]['f1']['ttest']['p'])
    keys.append(dom)

q_acc = bh_fdr(p_acc)
q_f1 = bh_fdr(p_f1)

res['p_adjust'] = {
    'method': 'BH-FDR on paired t-test p',
    'acc': {k: float(q) for k, q in zip(keys, q_acc)},
    'f1': {k: float(q) for k, q in zip(keys, q_f1)},
}

out_json = os.path.join(outdir, 'paired_stats_repo_vs_deepseek.json')
json.dump(res, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ---- markdown (avoid PowerShell pipe parsing by using chr(124)) ----
pipe = chr(124)
md = []
md.append('# Repo vs DeepSeek 配对统计检验')
md.append('')
md.append('说明：差值定义为 DeepSeek - Repo；t-test 为配对样本；Wilcoxon 为符号秩检验；CI 用 bootstrap(BCa)。')
md.append('')
md.append(f'{pipe} Domain {pipe} Metric {pipe} mean(diff) {pipe} std(diff) {pipe} t p {pipe} wilcoxon p {pipe} dz {pipe} r_rb {pipe} CI95 low {pipe} CI95 high {pipe}')
md.append(f'{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}')

for dom in keys:
    for m in ['acc', 'f1']:
        r = res['domains'][dom][m]
        md.append(
            f'{pipe} {dom} {pipe} {m} {pipe} {r["diff_mean"]:.4f} {pipe} {r["diff_std"]:.4f} {pipe} '
            f'{r["ttest"]["p"]:.4g} {pipe} {r["wilcoxon"]["p"]:.4g} {pipe} '
            f'{r["effect"]["cohen_dz"]:.3f} {pipe} {r["effect"]["rank_biserial_r"]:.3f} {pipe} '
            f'{r["boot_ci95_mean_diff"]["low"]:.4f} {pipe} {r["boot_ci95_mean_diff"]["high"]:.4f} {pipe}'
        )

out_md = os.path.join(outdir, 'paired_stats_repo_vs_deepseek.md')
open(out_md, 'w', encoding='utf-8').write('\n'.join(md))

print('saved', out_json)
print('saved', out_md)
