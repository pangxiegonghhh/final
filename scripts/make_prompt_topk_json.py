import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--keep_keys_json", required=True, help="json file with {'keys':[...]} or {'prompt_keys':[...]} or list")
    args = ap.parse_args()

    keep = json.load(open(args.keep_keys_json, "r", encoding="utf-8"))
    if isinstance(keep, list):
        keys = keep
    else:
        keys = keep.get("keys") or keep.get("prompt_keys") or keep.get("keep_keys") or []
    assert keys, "keep keys empty"

    data = json.load(open(args.in_json, "r", encoding="utf-8"))
    out = []
    for it in data:
        o = {"input": it.get("input", "")}
        for k in keys:
            o[k] = it.get(k, "")
        out.append(o)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.out_json)

if __name__ == "__main__":
    main()