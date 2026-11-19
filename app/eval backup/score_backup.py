"""Score one or more run files and compare them on the SAME questions.

Run:  python -m app.eval.score data/runs/baseline_k5.jsonl [data/runs/pipeline.jsonl ...]
Columns: EM / F1 (HotpotQA official), allGold = share of questions whose retrieved set contains
every gold paragraph, tok/q = mean prompt+completion tokens per question (your cost metric).
With 2+ files, also prints per-question wins/losses vs the first file.
"""
import json
import sys
from pathlib import Path

from app.eval.metrics import exact_match, f1_score


def load(path):
    return {r["id"]: r for r in map(json.loads, Path(path).open(encoding="utf-8"))}


def summarize(rows):
    n = len(rows)
    em = sum(exact_match(r["prediction"], r["gold_answer"]) for r in rows) / n
    f1 = sum(f1_score(r["prediction"], r["gold_answer"]) for r in rows) / n
    gold = sum(set(r["gold_titles"]) <= set(r["retrieved_titles"]) for r in rows) / n
    tok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n
    return n, em, f1, gold, tok


def main(paths):
    runs = {Path(p).stem: load(p) for p in paths}
    common = set.intersection(*(set(r) for r in runs.values()))
    if len(common) != max(len(r) for r in runs.values()):
        print(f"NOTE: scoring only the {len(common)} questions present in ALL runs.\n")
    if not common:
        sys.exit("No questions in common.")

    print(f"{'run':<24}{'n':>5}{'EM':>7}{'F1':>7}{'allGold':>9}{'tok/q':>8}")
    for name, r in runs.items():
        n, em, f1, gold, tok = summarize([r[i] for i in common])
        print(f"{name:<24}{n:>5}{em:>7.3f}{f1:>7.3f}{gold:>9.3f}{tok:>8.0f}")

    for typ in ("bridge", "comparison"):
        ids = [i for i in common if next(iter(runs.values()))[i]["type"] == typ]
        if not ids:
            continue
        print(f"\n  {typ} (n={len(ids)})")
        for name, r in runs.items():
            n, em, f1, gold, _ = summarize([r[i] for i in ids])
            print(f"  {name:<22}{'':>5}{em:>7.3f}{f1:>7.3f}{gold:>9.3f}")

    names = list(runs)
    for other in names[1:]:
        w = l = 0
        for i in common:
            a = exact_match(runs[names[0]][i]["prediction"], runs[names[0]][i]["gold_answer"])
            b = exact_match(runs[other][i]["prediction"], runs[other][i]["gold_answer"])
            w += b > a
            l += b < a
        print(f"\n{other} vs {names[0]}: wins {w}, losses {l}, ties {len(common) - w - l} (by exact match)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.eval.score RUN.jsonl [RUN2.jsonl ...]")
    main(sys.argv[1:])
