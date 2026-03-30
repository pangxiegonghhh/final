import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pt = r'outputs\paper_tables'
csv_path = os.path.join(pt, 'all_domains_compare.csv')
df = pd.read_csv(csv_path)

outdir = r'outputs\figures'
os.makedirs(outdir, exist_ok=True)

def parse_ms(s: str):
    a = re.split(r'\s*±\s*', str(s))
    mean = float(a[0])
    std = float(a[1]) if len(a) > 1 else 0.0
    return mean, std

df[['acc_mean','acc_std']] = df['Accuracy(mean±std)'].apply(lambda x: pd.Series(parse_ms(x)))
df[['f1_mean','f1_std']]   = df['F1(mean±std)'].apply(lambda x: pd.Series(parse_ms(x)))

dom = list(dict.fromkeys(df['Domain'].tolist()))
stg = list(dict.fromkeys(df['Setting'].tolist()))

w = 0.35
x = np.arange(len(dom))

def plot(metric: str, ylab: str, fn: str):
    plt.figure()
    for j, s in enumerate(stg):
        sub = df[df['Setting'] == s].set_index('Domain').reindex(dom)
        m = sub[f'{metric}_mean'].values
        e = sub[f'{metric}_std'].values
        xpos = x + (j - (len(stg) - 1) / 2) * w
        plt.bar(xpos, m, width=w, label=s)
        plt.errorbar(xpos, m, yerr=e, fmt='none', capsize=3)

    plt.xticks(x, dom)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fn + '.png'), dpi=300)
    plt.savefig(os.path.join(outdir, fn + '.svg'))
    plt.close()

plot('acc', 'Accuracy', 'fig_acc_repo_vs_deepseek')
plot('f1',  'F1',       'fig_f1_repo_vs_deepseek')

print('saved to', outdir)
