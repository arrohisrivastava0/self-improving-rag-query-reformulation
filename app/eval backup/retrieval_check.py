"""Sanity check + your first thesis number.

Uses the RAW question as the query (exactly what naive RAG does) and measures how
many of the gold supporting paragraphs land in the top-k. Multi-hop 'bridge'
questions should score noticeably worse - that gap is the motivation for query
reformulation, and the per-question misses are good candidates for your demo.

Run:  python -m app.eval.retrieval_check
"""
import json
from collections import defaultdict

from tqdm import tqdm

from app.config import QUESTIONS_PATH, RETRIEVAL_CHECK_PATH
from app.retrieval import retrieve

KS = (2, 5, 10)


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    per_q = []
    agg = defaultdict(lambda: {k: {"recall": [], "all": []} for k in KS})

    for q in tqdm(questions):
        hits = retrieve(q["question"], k=max(KS))
        titles = [h["title"] for h in hits]
        gold = set(q["supporting_titles"])
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "gold": sorted(gold)}
        for k in KS:
            found = gold & set(titles[:k])
            r, full = len(found) / len(gold), len(found) == len(gold)
            for grp in ("all", q["type"]):
                agg[grp][k]["recall"].append(r)
                agg[grp][k]["all"].append(full)
            if k == 5:
                row["missing_at_5"] = sorted(gold - found)
        per_q.append(row)

    print("\nSupporting-paragraph retrieval with the RAW question as query")
    print(f"{'group':<12}{'n':>5}" + "".join(f"  R@{k:<2} all@{k:<2}" for k in KS))
    summary = {}
    for grp, d in agg.items():
        n = len(d[KS[0]]["recall"])
        line = f"{grp:<12}{n:>5}"
        summary[grp] = {"n": n}
        for k in KS:
            rec = sum(d[k]["recall"]) / n
            allf = sum(d[k]["all"]) / n
            line += f"  {rec:5.2f}  {allf:5.2f} "
            summary[grp][f"recall@{k}"] = rec
            summary[grp][f"all_gold@{k}"] = allf
        print(line)

    RETRIEVAL_CHECK_PATH.write_text(
        json.dumps({"summary": summary, "per_question": per_q}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved -> {RETRIEVAL_CHECK_PATH}")
    print("Tip: questions with a non-empty 'missing_at_5' are your best demo candidates.")


if __name__ == "__main__":
    main()
