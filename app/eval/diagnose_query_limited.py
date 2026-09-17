import argparse
import json
from pathlib import Path

from app.eval.analyze_traces import classify
from app.retrieval import retrieve


def deep_rank(query, gold_title, k):
    for i, h in enumerate(retrieve(query, k=k), start=1):
        if h["title"] == gold_title:
            return i
    return None


def all_tried_queries(row):
    qs = [row["question"]]
    for s in row["trace"]:
        if s["step"] == "rewrite" and "candidates" in s:
            qs += [c["query"] for c in s["candidates"]]
    seen, out = set(), []
    for q in qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def main(path, k, show):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    ql = [r for r in rows if classify(r) == "query-limited"]
    print(f"{Path(path).stem}: {len(ql)} query-limited questions | deep-checking at k={k} (local, no API cost)\n")
    if not ql:
        return

    buckets = {"found <=10": [], "found 11-50": [], f"not in top-{k}": []}
    for r in ql:
        gold = set(r["gold_titles"])
        missing = gold - set(r["seen_titles"])  
        queries = all_tried_queries(r)
        best = {}
        for title in missing:
            ranks = [rk for q in queries if (rk := deep_rank(q, title, k)) is not None]
            best[title] = min(ranks) if ranks else None
        worst_of_the_missing = max(best.values(), key=lambda x: (x is None, x))  
        if worst_of_the_missing is None:
            bucket = f"not in top-{k}"
        elif worst_of_the_missing <= 10:
            bucket = "found <=10"
        else:
            bucket = "found 11-50"
        buckets[bucket].append((r, missing, best))

    print(f"{'bucket':<16}{'n':>4}   meaning")
    print(f"{'found <=10':<16}{len(buckets['found <=10']):>4}   more depth per query (bigger k) would likely have found it")
    print(f"{'found 11-50':<16}{len(buckets['found 11-50']):>4}   borderline - reranking / better candidates might help")
    print(f"{f'not in top-{k}':<16}{len(buckets[f'not in top-{k}']):>4}   genuine embedding/vocabulary gap - needs a different retrieval strategy")

    if show:
        for name, items in buckets.items():
            for r, missing, best in items[:show]:
                print(f"\n[{name}] {r['id']}  {r['question'][:90]}")
                for title in missing:
                    rk = best[title]
                    print(f"    missing paragraph {title!r}: best rank found = {rk if rk else f'not in top-{k}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    main(a.run, a.k, a.show)