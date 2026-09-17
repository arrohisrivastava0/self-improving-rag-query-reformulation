import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from app.eval.metrics import relaxed_match


def initial_titles(row):
    for s in row["trace"]:
        if s["step"] == "retrieve":
            return set(s["titles"])
    return set()


def classify(row, max_cands=None):
    gold = set(row["gold_titles"])
    if gold <= set(row["retrieved_titles"]):
        return "recovered"
    if gold <= set(row["seen_titles"]):
        return "cap-limited"
    if row["stop_reason"] == "sufficient":
        return "gate fooled"
    seen = initial_titles(row)
    for s in row["trace"]:
        if s["step"] != "rewrite" or "candidates" not in s:
            continue
        if any(gold <= (seen | set(c["titles"])) for c in s["candidates"][:max_cands]):
            return "selector-limited"         
        seen |= set(s["candidates"][s["chosen"]]["titles"])
    return "query-limited"


def main(path, show):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    if not rows:
        sys.exit("empty run")
    fail = [r for r in rows if not set(r["gold_titles"]) <= initial_titles(r)]
    ok = [r for r in rows if set(r["gold_titles"]) <= initial_titles(r)]
    print(f"{Path(path).stem}: {len(rows)} questions | naive retrieval missed gold on {len(fail)}, got it right on {len(ok)}\n")

    cats = defaultdict(list)
    for r in fail:
        cats[classify(r)].append(r)
    notes = {
        "recovered": "final evidence has every gold paragraph",
        "cap-limited": "saw every gold paragraph, the 5-passage cut dropped one",
        "gate fooled": "checklist said 'sufficient' though gold was missing",
        "selector-limited": "a better candidate existed -> headroom for the Phase-3 scorer",
        "query-limited": "no candidate query retrieved the missing paragraph",
    }
    has_pred = all(r["prediction"] != "" for r in rows)

    def acc(rs):
        return f"Acc {sum(relaxed_match(r['prediction'], r['gold_answer']) for r in rs) / len(rs):.2f}" if (has_pred and rs) else "        "

    print(f"FAILURES ({len(fail)})" + ("   [Acc = did the final ANSWER come out right anyway?]" if has_pred else ""))
    for name in ("recovered", "cap-limited", "gate fooled", "selector-limited", "query-limited"):
        n = len(cats[name])
        pct = f"{n / len(fail):>4.0%}" if fail else "  - "
        print(f"  {name:<17}{n:>3}  {pct}  {acc(cats[name])}  {notes[name]}")
    if fail:
        rec, sel, cap = len(cats["recovered"]), len(cats["selector-limited"]), len(cats["cap-limited"])
        print(f"\n  recovery now: {rec}/{len(fail)} ({rec / len(fail):.0%})")
        print(f"  ceiling with a PERFECT candidate selector (single-step estimate): {rec + sel}/{len(fail)} ({(rec + sel) / len(fail):.0%})")
        widest = max((len(st["candidates"]) for r in fail for st in r["trace"] if st["step"] == "rewrite" and "candidates" in st), default=0)
        for k in sorted({3, 5} & set(range(1, widest))):
            sel_k = sum(1 for r in fail if not set(r["gold_titles"]) <= set(r["retrieved_titles"]) and classify(r, k) == "selector-limited")
            print(f"    ...if the scorer could only choose among the first {k} candidates: {rec + sel_k}/{len(fail)} ({(rec + sel_k) / len(fail):.0%})")
        print(f"  ceiling if the final cut also kept everything seen: {rec + sel + cap}/{len(fail)} ({(rec + sel + cap) / len(fail):.0%})")
        print(f"  stop reasons among failures: {dict(Counter(r['stop_reason'] for r in fail))}")

    loops = [r for r in ok if r["iterations"] > 0]
    lost = [r for r in ok if not set(r["gold_titles"]) <= set(r["retrieved_titles"])]
    if has_pred:
        print(f"\nOverall answer Acc of this run: {sum(relaxed_match(r['prediction'], r['gold_answer']) for r in rows) / len(rows):.3f}")
    print(f"\nSUCCESSES ({len(ok)}): the loop was entered anyway on {len(loops)} ({len(loops) / max(len(ok), 1):.0%}); gold lost in {len(lost)}")
    if ok:
        print(f"  mean agent tokens: {sum(r['agent_tokens'] for r in ok) / len(ok):.0f} "
              f"(questions with no loop: {sum(r['agent_tokens'] for r in ok if r['iterations'] == 0) / max(len(ok) - len(loops), 1):.0f}, "
              f"with a loop: {sum(r['agent_tokens'] for r in loops) / max(len(loops), 1):.0f})")
        print(f"  stop reasons among successes: {dict(Counter(r['stop_reason'] for r in ok))}")

    if show:
        print("\nEXAMPLES (question ids, for inspection with --ids):")
        for name in ("gate fooled", "selector-limited", "query-limited", "cap-limited"):
            for r in cats[name][:show]:
                print(f"  [{name}] {r['id']}  {r['question'][:90]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    main(a.run, a.show)
