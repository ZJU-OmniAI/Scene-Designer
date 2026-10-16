"""Assemble the final continued-training set from new-scene data.

Combines:
  * core n2/n3/n4 ranking shards (build_dataset; gemini reasoning),
  * severity multitarget/multikind + pairwise records with image-grounded
    reasoning (gpt-5.2).

All n4 records get label-permuted (anti-ABCD-shortcut) and the reasoning
bare-letter sequences fixed (fix_perm_letter_sequences), because training
uses the reasoning text. n2 records are already balanced and kept as-is.

Usage:
  python -m src_critic.assemble_train_set \
      --core-glob 'datasets/critic_new_train/sft_train_w*.jsonl' \
      --ig datasets/critic_new_train/sft_train_severity_pairwise_ig.jsonl \
      --out datasets/critic_new_train/sft_train_new.jsonl
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, tempfile
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _fp(r):
    scene = r["id"].split("__")[0]
    plb = r["ground_truth"].get("perturbations_by_label", {})
    items = [(p.get("kind"), p.get("object_id")) for _, ps in plb.items() for p in ps]
    return (scene, r["n"], r["ground_truth"].get("ranking_mode"), frozenset(items))


def _permute_and_fix(n4_records, out_dir, seed_xor):
    if not n4_records:
        return []
    raw = out_dir / "_tr_n4_raw.jsonl"
    perm = out_dir / "_tr_n4_perm.jsonl"
    fixed = out_dir / "_tr_n4_fixed.jsonl"
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in n4_records) + "\n")
    subprocess.run([sys.executable, "-m", "src_critic.permute_n4_labels",
                    "--inp", str(raw), "--out", str(perm), "--seed-xor", str(seed_xor)],
                   check=True)
    subprocess.run([sys.executable, "-m", "src_critic.fix_perm_letter_sequences",
                    "--inp", str(perm), "--out", str(fixed)], check=True)
    res = load(fixed)
    for t in (raw, perm, fixed):
        t.unlink(missing_ok=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-glob", required=True)
    ap.add_argument("--ig", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-xor", type=int, default=20260608)
    args = ap.parse_args()
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge core (dedup by id, then by fingerprint)
    by_id = {}
    for f in sorted(glob.glob(args.core_glob)):
        for r in load(f):
            by_id[r["id"]] = r
    seen_fp, core = set(), []
    for r in by_id.values():
        k = _fp(r)
        if k in seen_fp:
            continue
        seen_fp.add(k); core.append(r)
    ig = load(args.ig)

    pool = core + ig
    n2 = [r for r in pool if r.get("n") == 2]
    n4 = [r for r in pool if r.get("n") == 4]
    n3 = [r for r in pool if r.get("n") == 3]
    # n3 dropped: permute_n4_labels supports only n=2/4, and the held-out
    # test has no n=3 — keeping un-permuted n3 would reintroduce the
    # label-order shortcut, so exclude for a clean, consistent train set.
    print(f"[train-assemble] core={len(core)} ig={len(ig)} → n2={len(n2)} n4={len(n4)} (dropped n3={len(n3)})")

    n4_perm = _permute_and_fix(n4, out_dir, args.seed_xor)
    final = n2 + n4_perm
    # shuffle deterministically by id hash for training mix
    final.sort(key=lambda r: r["id"])
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final) + "\n")

    from collections import Counter
    c = Counter((r["n"], r["ground_truth"].get("ranking_mode")) for r in final)
    print(f"[train-assemble] wrote {len(final)} → {args.out}")
    for k in sorted(c, key=lambda x: str(x)):
        print(f"   {k}: {c[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
