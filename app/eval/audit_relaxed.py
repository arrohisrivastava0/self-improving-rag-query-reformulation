import argparse
from collections import Counter
from pathlib import Path

from app.eval.metrics import _relaxed_tokens, exact_match, f1_score, relaxed_match
from app.eval.score import load

SUBSET = "prediction is a subset of gold"


def kind(pred, gold):
    """Why relaxed_match credited a prediction that exact match did not."""
    ps, gs = set(_relaxed_tokens(pred)), set(_relaxed_tokens(gold))
    if ps == gs:
        return "same words (order/punctuation/article)"
    if ps < gs:
        return SUBSET                                   # e.g. "Smith" for "John Smith"
    return "gold is a subset of prediction (<=3 extra words)"   # e.g. "the city of Paris" for "Paris"


def strict_match(pred, gold):
    """Relaxed accuracy WITHOUT credit for predictions that are only a fragment of the gold answer."""
    if relaxed_match(pred, gold) == 0.0:
        return 0.0
    if exact_match(pred, gold) == 1.0:
        return 1.0
    return float(kind(pred, gold) != SUBSET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", action="store_true", help="list every relaxed-only match for manual review")
    a = ap.parse_args()

    print(f"{'run':<22}{'n':>5}{'EM':>7}{'F1':>7}{'Acc':>7}{'AccStrict':>10}{'relaxed-only':>14}")
    detail = {}
    for p in a.runs:
        name, rows = Path(p).stem, list(load(p).values())
        n = len(rows)
        em = sum(exact_match(r["prediction"], r["gold_answer"]) for r in rows) / n
        f1 = sum(f1_score(r["prediction"], r["gold_answer"]) for r in rows) / n
        acc = sum(relaxed_match(r["prediction"], r["gold_answer"]) for r in rows) / n
        strict = sum(strict_match(r["prediction"], r["gold_answer"]) for r in rows) / n
        extra = [r for r in rows if relaxed_match(r["prediction"], r["gold_answer"]) == 1.0
                and exact_match(r["prediction"], r["gold_answer"]) == 0.0]
        detail[name] = extra
        print(f"{name:<22}{n:>5}{em:>7.3f}{f1:>7.3f}{acc:>7.3f}{strict:>10.3f}{len(extra):>14}")

    for name, extra in detail.items():
        kinds = Counter(kind(r["prediction"], r["gold_answer"]) for r in extra)
        print(f"\n{name}: relaxed-only matches by kind -> {dict(kinds)}")
        if a.show:
            for r in extra:
                print(f"  [{kind(r['prediction'], r['gold_answer'])}] gold={r['gold_answer']!r} pred={r['prediction']!r}\n     Q: {r['question']}")


if __name__ == "__main__":
    main()