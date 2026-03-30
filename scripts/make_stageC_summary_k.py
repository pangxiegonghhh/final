import os, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.dirname(__file__) + r"\..")
OUT10 = os.path.join(ROOT, "outputs", "10seed测试")
OUT_PAPER = os.path.join(ROOT, "outputs", "paper_tables")
OUT_FIG = os.path.join(ROOT, "outputs", "figures")
OUT_STATS = os.path.join(ROOT, "outputs", "stats")
os.makedirs(OUT_PAPER, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_STATS, exist_ok=True)

def load_ms(path):
    j = json.load(open(path, "r", encoding="utf-8"))
    return j["multi_seed_summary"]

def fmt(m,s): return f"{m:.4f} ± {s:.4f}"

def paired(a, b):
    a=np.array(a,float); b=np.array(b,float); d=b-a
    tt=stats.ttest_rel(b,a,alternative="two-sided")
    ww=stats.wilcoxon(b,a,alternative="two-sided",zero_method="wilcox",method="auto")
    boot=stats.bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=20000, method="BCa", random_state=0)
    sd=float(np.std(d,ddof=1)) if len(d)>1 else 0.0
    dz=float(np.mean(d)/sd) if sd>0 else float("nan")
    return {
      "diff_mean": float(np.mean(d)),
      "ttest_p": float(tt.pvalue),
      "wilcoxon_p": float(ww.pvalue),
      "dz": dz,
      "ci_low": float(boot.confidence_interval.low),
      "ci_high": float(boot.confidence_interval.high),
    }

baseline = {
    ("Arxiv","repo"):     os.path.join(OUT10, "arxiv_repo_350.json"),
    ("Arxiv","deepseek"): os.path.join(OUT10, "arxiv_deepseek_350.json"),
    ("Code","repo"):      os.path.join(OUT10, "code_repo_164.json"),
    ("Code","deepseek"):  os.path.join(OUT10, "code_deepseek_164.json"),
    ("Yelp","repo"):      os.path.join(OUT10, "yelp_repo_401.json"),
    ("Yelp","deepseek"):  os.path.join(OUT10, "yelp_deepseek_401.json"),
}

top3 = {
    ("Arxiv","repo"):     os.path.join(OUT10, "prompt_ensemble", "Arxiv", "repo", "top3.json"),
    ("Arxiv","deepseek"): os.path.join(OUT10, "prompt_ensemble", "Arxiv", "deepseek", "top3.json"),
    ("Code","repo"):      os.path.join(OUT10, "prompt_ensemble", "Code", "repo", "top3.json"),
    ("Code","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Code", "deepseek", "top3.json"),
    ("Yelp","repo"):      os.path.join(OUT10, "prompt_ensemble", "Yelp", "repo", "top3.json"),
    ("Yelp","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Yelp", "deepseek", "top3.json"),
}
top5 = {
    ("Arxiv","repo"):     os.path.join(OUT10, "prompt_ensemble", "Arxiv", "repo", "top5.json"),
    ("Arxiv","deepseek"): os.path.join(OUT10, "prompt_ensemble", "Arxiv", "deepseek", "top5.json"),
    ("Code","repo"):      os.path.join(OUT10, "prompt_ensemble", "Code", "repo", "top5.json"),
    ("Code","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Code", "deepseek", "top5.json"),
    ("Yelp","repo"):      os.path.join(OUT10, "prompt_ensemble", "Yelp", "repo", "top5.json"),
    ("Yelp","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Yelp", "deepseek", "top5.json"),
}

prompt_counts = {"Arxiv":{"ALL":7,"TOP5":5,"TOP3":3}, "Code":{"ALL":5,"TOP5":5,"TOP3":3}, "Yelp":{"ALL":7,"TOP5":5,"TOP3":3}}
settings = [("ALL", baseline), ("TOP5", top5), ("TOP3", top3)]

rows=[]
paired_out={"meta":{"compare":"K - ALL","bootstrap_resamples":20000},"groups":{}}

for dom in ["Arxiv","Code","Yelp"]:
  for rw in ["repo","deepseek"]:
    ms_all=load_ms(baseline[(dom,rw)])
    paired_out["groups"][f"{dom}/{rw}"]={}
    for st,mp in settings:
      ms=load_ms(mp[(dom,rw)])
      rows.append({
        "Domain":dom,"Rewriter":rw,"Setting":st,"#Prompts":prompt_counts[dom][st],
        "Accuracy(mean±std)":fmt(ms["acc_mean"],ms["acc_std"]),
        "F1(mean±std)":fmt(ms["f1_mean"],ms["f1_std"]),
      })
      if st!="ALL":
        paired_out["groups"][f"{dom}/{rw}"][st]={
          "acc": paired(ms_all["acc_list"], ms["acc_list"]),
          "f1":  paired(ms_all["f1_list"],  ms["f1_list"]),
        }

df=pd.DataFrame(rows).sort_values(["Domain","Rewriter","Setting"]).reset_index(drop=True)
out_csv=os.path.join(OUT_PAPER,"stageC_all_vs_top5_vs_top3_summary.csv")
df.to_csv(out_csv,index=False,encoding="utf-8-sig")

# markdown
pipe=chr(124)
out_md=os.path.join(OUT_PAPER,"stageC_all_vs_top5_vs_top3_summary.md")
md=[ "# Stage C Summary: ALL vs TOP5 vs TOP3","",
     f"{pipe} Domain {pipe} Rewriter {pipe} Setting {pipe} #Prompts {pipe} Accuracy (mean±std) {pipe} F1 (mean±std) {pipe}",
     f"{pipe}---{pipe}---{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}"]
for rec in df.to_dict("records"):
  md.append(f"{pipe} {rec['Domain']} {pipe} {rec['Rewriter']} {pipe} {rec['Setting']} {pipe} {int(rec['#Prompts'])} {pipe} {rec['Accuracy(mean±std)']} {pipe} {rec['F1(mean±std)']} {pipe}")
open(out_md,"w",encoding="utf-8").write("\n".join(md))

# paired stats json + md
out_stat_json=os.path.join(OUT_STATS,"stageC_paired_k_minus_all.json")
json.dump(paired_out, open(out_stat_json,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

out_stat_md=os.path.join(OUT_PAPER,"stageC_paired_k_minus_all.md")
md2=["# Stage C Paired Tests: TOP5/TOP3 minus ALL (10 seeds paired)","",
     "说明：差值定义为 K - ALL；paired t-test + Wilcoxon；CI 为 bootstrap(BCa) 95%。","",
     f"{pipe} Group {pipe} K {pipe} Metric {pipe} mean(diff) {pipe} t p {pipe} wilcoxon p {pipe} dz {pipe} CI95 low {pipe} CI95 high {pipe}",
     f"{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}"]
for g,gv in paired_out["groups"].items():
  for K in ["TOP5","TOP3"]:
    if K not in gv: 
      continue
    for m in ["acc","f1"]:
      s=gv[K][m]
      md2.append(f"{pipe} {g} {pipe} {K} {pipe} {m} {pipe} {s['diff_mean']:.4f} {pipe} {s['ttest_p']:.4g} {pipe} {s['wilcoxon_p']:.4g} {pipe} {s['dz']:.3f} {pipe} {s['ci_low']:.4f} {pipe} {s['ci_high']:.4f} {pipe}")
open(out_stat_md,"w",encoding="utf-8").write("\n".join(md2))

# plots
def parse_ms(s):
  a=str(s).split("±")
  return float(a[0].strip()), float(a[1].strip())

plot_df=df.copy()
plot_df[["acc_mean","acc_std"]]=plot_df["Accuracy(mean±std)"].apply(lambda x: pd.Series(parse_ms(x)))
plot_df[["f1_mean","f1_std"]]=plot_df["F1(mean±std)"].apply(lambda x: pd.Series(parse_ms(x)))

def plot(metric,ylab,fn):
  plt.figure()
  domains=["Arxiv","Code","Yelp"]
  rewriters=["repo","deepseek"]
  settings=["ALL","TOP5","TOP3"]
  x=np.arange(len(domains))
  w=0.12
  idx=0
  for rw in rewriters:
    for st in settings:
      sub=plot_df[(plot_df["Rewriter"]==rw)&(plot_df["Setting"]==st)].set_index("Domain").reindex(domains)
      m=sub[f"{metric}_mean"].values
      e=sub[f"{metric}_std"].values
      xpos=x+(idx-(len(rewriters)*len(settings)-1)/2)*w
      plt.bar(xpos,m,width=w,label=f"{rw}-{st}")
      plt.errorbar(xpos,m,yerr=e,fmt="none",capsize=3)
      idx+=1
  plt.xticks(x,domains)
  plt.ylabel(ylab)
  plt.legend(fontsize=7)
  plt.tight_layout()
  plt.savefig(os.path.join(OUT_FIG, fn+".png"), dpi=300)
  plt.savefig(os.path.join(OUT_FIG, fn+".svg"))
  plt.close()

plot("f1","F1","fig_stageC_f1_all_vs_top5_vs_top3")
plot("acc","Accuracy","fig_stageC_acc_all_vs_top5_vs_top3")

print("saved:", out_csv)
print("saved:", out_md)
print("saved:", out_stat_json)
print("saved:", out_stat_md)
print("saved figures to:", OUT_FIG)
