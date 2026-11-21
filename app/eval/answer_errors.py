import argparse
from collections import Counter
from pathlib import Path

from app.eval.metrics import relaxed_match
from app.eval.score import load

CATS = (
    "gold retrieved, correct",
    "gold retrieved, WRONG",
    "gold missing, correct (closed-book also correct)",
    "gold missing, correct (closed-book wrong)",
    "gold missing, wrong",
)


def category(r, closed_ok):
    full = set(r["gold_titles"]) <= set(r["retrieved_titles"])
    ok = relaxed_match(r["prediction"], r["gold_answer"]) == 1.0
    if full:
        return CATS[0] if ok else CATS[1]
    if ok:
        return CATS[2] if closed_ok else CATS[3]
    return CATS[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("closed", help="closed-book run, e.g. data/runs/baseline_k0_test.jsonl")
    ap.add_argument("runs", nargs="+", help="retrieval runs to break down")
    ap.add_argument("--show", action="store_true", help="list the 'gold retrieved, WRONG' cases")
    a = ap.parse_args()

    closed = load(a.closed)
    runs = {Path(p).stem: load(p) for p in a.runs}
    common = sorted(set(closed).intersection(*(set(r) for r in runs.values())))
    print(f"{len(common)} common questions | closed-book run: {Path(a.closed).stem}\n")

    for typ in ("all", "bridge", "comparison"):
        ids = [i for i in common if typ == "all" or closed[i]["type"] == typ]
        if not ids:
            continue
        print(f"{typ} (n={len(ids)})")
        print(f"  {'category':<52}" + "".join(f"{n:>20}" for n in runs))
        counts = {
            n: Counter(category(r[i], relaxed_match(closed[i]["prediction"], closed[i]["gold_answer"]) == 1.0) for i in ids)
            for n, r in runs.items()
        }
        for c in CATS:
            print(f"  {c:<52}" + "".join(f"{counts[n][c]:>20}" for n in runs))
        print()

    if a.show:
        for n, r in runs.items():
            print(f"--- {n}: gold retrieved but answer WRONG ---")
            for i in common:
                row = r[i]
                if category(row, False) == CATS[1]:
                    print(f"  {i} [{row['type']}] gold={row['gold_answer']!r} pred={row['prediction']!r}\n     Q: {row['question']}")


if __name__ == "__main__":
    main()