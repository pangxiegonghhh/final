# scripts/make_prompt_subset_json.py
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--keep_key", required=True, help="prompt key to keep (exact key string)")
    ap.add_argument("--keep_idx", action="store_true", help="keep _idx if exists")
    args = ap.parse_args()

    data = json.load(open(args.in_json, "r", encoding="utf-8"))
    out = []
    for it in data:
        o = {"input": it.get("input", "")}
        if args.keep_idx and "_idx" in it:
            o["_idx"] = it["_idx"]
        if args.keep_key in it:
            o[args.keep_key] = it[args.keep_key]
        else:
            # 任一条缺失会导致维度不一致；直接写空串，后续统一处理
            o[args.keep_key] = ""
        out.append(o)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.out_json)

if __name__ == "__main__":
    main()