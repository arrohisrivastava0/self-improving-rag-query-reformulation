import json
import sys
from math import comb
from pathlib import Path

from app.eval.metrics import exact_match, f1_score, relaxed_match


def load(path):
    return {r["id"]: r for r in map(json.loads, Path(path).open(encoding="utf-8"))}


def summarize(rows):
    n = len(rows)
    em = sum(exact_match(r["prediction"], r["gold_answer"]) for r in rows) / n
    f1 = sum(f1_score(r["prediction"], r["gold_answer"]) for r in rows) / n
    acc = sum(relaxed_match(r["prediction"], r["gold_answer"]) for r in rows) / n
    has_retrieval = any(r["retrieved_titles"] for r in rows)
    gold = sum(set(r["gold_titles"]) <= set(r["retrieved_titles"]) for r in rows) / n if has_retrieval else None
    tok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n
    return n, em, f1, acc, gold, tok


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def _g(x):
    return f"{x:>9.3f}" if x is not None else f"{'-':>9}"


def main(paths):
    runs = {Path(p).stem: load(p) for p in paths}
    common = set.intersection(*(set(r) for r in runs.values()))
    if len(common) != max(len(r) for r in runs.values()):
        print(f"NOTE: scoring only the {len(common)} questions present in ALL runs.\n")
    if not common:
        sys.exit("No questions in common.")
    first = next(iter(runs.values()))

    print(f"{'run':<24}{'n':>5}{'EM':>7}{'F1':>7}{'Acc':>7}{'allGold':>9}{'tok/q':>8}")
    for name, r in runs.items():
        n, em, f1, acc, gold, tok = summarize([r[i] for i in common])
        print(f"{name:<24}{n:>5}{em:>7.3f}{f1:>7.3f}{acc:>7.3f}{_g(gold)}{tok:>8.0f}")

    for typ in ("bridge", "comparison"):
        ids = [i for i in common if first[i]["type"] == typ]
        if not ids:
            continue
        print(f"\n  {typ} (n={len(ids)})")
        for name, r in runs.items():
            n, em, f1, acc, gold, _ = summarize([r[i] for i in ids])
            print(f"  {name:<22}{'':>5}{em:>7.3f}{f1:>7.3f}{acc:>7.3f}{_g(gold)}")

    names = list(runs)
    for other in names[1:]:
        print(f"\n{other} vs {names[0]} (paired, {len(common)} questions)")
        for label, fn in (("EM ", exact_match), ("Acc", relaxed_match)):
            w = l = 0
            for i in common:
                a = fn(runs[names[0]][i]["prediction"], runs[names[0]][i]["gold_answer"])
                b = fn(runs[other][i]["prediction"], runs[other][i]["gold_answer"])
                w += b > a
                l += b < a
            print(f"  {label}: wins {w}, losses {l}, ties {len(common) - w - l}, sign-test p = {sign_p(w, l):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.eval.score RUN.jsonl [RUN2.jsonl ...]")
    main(sys.argv[1:])
