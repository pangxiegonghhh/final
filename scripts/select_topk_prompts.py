import argparse, pandas as pd, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.sens_csv).sort_values("f1_mean", ascending=False).head(args.topk)
    slugs = df["prompt_file"].tolist()  # 这里是 prompt_{slug}.json 的文件名
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"topk": args.topk, "prompt_files": slugs}, open(args.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved:", args.out_json)
    print("topk:", slugs)

if __name__ == "__main__":
    main()