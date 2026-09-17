import json
import sys
from pathlib import Path

from app.eval.metrics import exact_match, relaxed_match


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).open(encoding="utf-8"))}


def full(r):
    return set(r["gold_titles"]) <= set(r["retrieved_titles"])


def tokens(r):
    return r["prompt_tokens"] + r["completion_tokens"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main(base_path, pipe_path):
    base, pipe = load(base_path), load(pipe_path)
    ids = sorted(set(base) & set(pipe))
    if not ids:
        sys.exit("No questions in common.")
    has_pred = all(pipe[i]["prediction"] != "" for i in ids)
    fail = [i for i in ids if not full(base[i])]
    ok = [i for i in ids if full(base[i])]
    print(f"{len(ids)} common questions | baseline: {Path(base_path).stem} | pipeline: {Path(pipe_path).stem}\n")

    rec = sum(full(pipe[i]) for i in fail)
    harm = sum(not full(pipe[i]) for i in ok)
    print("RETRIEVAL (all gold paragraphs in the evidence the reader sees)")
    print(f"  baseline-failure questions: {len(fail):>3}   pipeline recovered {rec:>3}   ({rec / len(fail):.0%})" if fail else "  no baseline failures in this sample")
    print(f"  baseline-success questions: {len(ok):>3}   pipeline lost      {harm:>3}   ({harm / len(ok):.0%})" if ok else "  no baseline successes in this sample")
    all_b = sum(full(base[i]) for i in ids) / len(ids)
    all_p = sum(full(pipe[i]) for i in ids) / len(ids)
    print(f"  overall all-gold: baseline {all_b:.3f} -> pipeline {all_p:.3f}")

    if has_pred:
        print("\nANSWERS            EM base  EM pipe   Acc base  Acc pipe")
        for name, sub in (("baseline-failure", fail), ("baseline-success", ok), ("all", ids)):
            if not sub:
                continue
            print(
                f"  {name:<16} {mean(exact_match(base[i]['prediction'], base[i]['gold_answer']) for i in sub):>8.3f}"
                f" {mean(exact_match(pipe[i]['prediction'], pipe[i]['gold_answer']) for i in sub):>8.3f}"
                f" {mean(relaxed_match(base[i]['prediction'], base[i]['gold_answer']) for i in sub):>10.3f}"
                f" {mean(relaxed_match(pipe[i]['prediction'], pipe[i]['gold_answer']) for i in sub):>9.3f}"
            )
    else:
        print("\n(pipeline run has no answers - retrieval-only run)")

    print("\nCOST (tokens per question)")
    for name, sub in (("baseline-failure", fail), ("baseline-success", ok), ("all", ids)):
        if sub:
            print(f"  {name:<16} baseline {mean(tokens(base[i]) for i in sub):>7.0f}   pipeline {mean(tokens(pipe[i]) for i in sub):>7.0f}")
    routes = [pipe[i].get("route") for i in ids]
    stops = {}
    for i in ids:
        stops[pipe[i].get("stop_reason")] = stops.get(pipe[i].get("stop_reason"), 0) + 1
    print(f"\nrouted simple: {routes.count('simple')} / complex: {routes.count('complex')} | stop reasons: {stops}")
    print(f"mean rewrites per question: {mean(pipe[i]['iterations'] for i in ids):.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m app.eval.compare_recovery BASELINE.jsonl PIPELINE.jsonl")
    main(sys.argv[1], sys.argv[2])
