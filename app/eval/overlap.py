import json
import sys

from app.eval.metrics import relaxed_match


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def full(r):
    return set(r["gold_titles"]) <= set(r["retrieved_titles"])


def table(title, a_name, b_name, a, b, universe):
    both = len(a & b)
    only_a, only_b = len(a - b), len(b - a)
    neither = len(universe) - len(a | b)
    print(f"{title}")
    print(f"  both {both:>3} | {a_name} only {only_a:>3} | {b_name} only {only_b:>3} | neither {neither:>3} | either {len(a | b):>3} of {len(universe)}")


def main(base_p, k10_p, pipe_p):
    base, k10, pipe = load(base_p), load(k10_p), load(pipe_p)
    ids = sorted(set(base) & set(k10) & set(pipe))
    fail = {i for i in ids if not full(base[i])}
    table(f"Baseline-k5 retrieval FAILURES that each method fixes (all gold in evidence)  [n={len(fail)}]",
          "k10", "pipeline", {i for i in fail if full(k10[i])}, {i for i in fail if full(pipe[i])}, fail)
    ok = lambda r: relaxed_match(r["prediction"], r["gold_answer"]) == 1.0
    if all(k10[i]["prediction"] and pipe[i]["prediction"] for i in ids):
        table(f"\nQuestions ANSWERED correctly (relaxed Acc)  [n={len(ids)}]",
              "k10", "pipeline", {i for i in ids if ok(k10[i])}, {i for i in ids if ok(pipe[i])}, set(ids))
        table("\n...restricted to the baseline-k5 failures",
              "k10", "pipeline", {i for i in fail if ok(k10[i])}, {i for i in fail if ok(pipe[i])}, fail)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python -m app.eval.overlap BASE_K5.jsonl K10.jsonl PIPELINE.jsonl")
    main(*sys.argv[1:4])
