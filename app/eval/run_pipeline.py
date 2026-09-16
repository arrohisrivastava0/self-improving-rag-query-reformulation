import argparse
import json
import sys

from tqdm import tqdm

from app.config import QUESTIONS_PATH, RETRIEVAL_CHECK_PATH, RUNS_DIR
from app.llm import DailyQuotaExceeded, stats_summary
from app.eval.trace_format import format_trace
from app.pipeline import PipelineConfig, build_graph, run_question


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--failures", type=int)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pool", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--evidence-chars", type=int, default=450)
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument("--prompts", choices=["v3", "v4"], default="v3")
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--selector", default="first", choices=["first", "novelty"])
    ap.add_argument("--router", action="store_true")
    ap.add_argument("--no-name-guard", action="store_true")
    ap.add_argument("--max-rewrites", type=int, default=2)
    ap.add_argument("--name", default="pipeline")
    ap.add_argument("--quiet", action="store_true", help="don't print traces")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}
    split = min(100, len(questions) // 2)
    pool_name = a.pool
    pool = {"dev": questions[:split], "test": questions[split:], "all": questions}[pool_name]
    if a.ids:
        chosen = [by_id[i] for i in a.ids]
    elif a.failures:
        rc = json.loads(RETRIEVAL_CHECK_PATH.read_text(encoding="utf-8"))["per_question"]
        pool_ids = {q["id"] for q in pool}
        fail_ids = [r["id"] for r in rc if r["missing_at_5"] and r["id"] in pool_ids]
        chosen = [by_id[i] for i in fail_ids[: a.failures]]
    else:
        chosen = pool[: a.n]
    print(f"pool: {pool_name} ({len(pool)} questions) | selected {len(chosen)}")
    if pool_name == "test":
        print("NOTE: TEST pool = held-out questions. Run it once with frozen prompts; never tune on it.")

    cfg = PipelineConfig(
        selector=a.selector, use_router=a.router, name_guard=not a.no_name_guard, max_rewrites=a.max_rewrites, answer=not a.no_answer,
        evidence_chars=a.evidence_chars, prompt_version=a.prompts, n_candidates=a.n_candidates
    )
    graph = build_graph(cfg)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{a.name}.jsonl"
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [q for q in chosen if q["id"] not in done]
    print(f"{out.name}: {len(done)} already done, {len(todo)} to run")

    try:
        with out.open("a", encoding="utf-8") as f:
            for q in tqdm(todo, disable=not a.quiet):
                st = run_question(graph, q["question"])
                u = st["usage"]
                ev = st["evidence"]
                row = {
                    "id": q["id"],
                    "type": q["type"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "prediction": st["answer"],
                    "retrieved_titles": [ev[i]["title"] for i in st["final_ids"]],
                    "gold_titles": q["supporting_titles"],
                    "prompt_tokens": u["agent_p"] + u["reader_p"],
                    "completion_tokens": u["agent_c"] + u["reader_c"],
                    "agent_tokens": u["agent_p"] + u["agent_c"],
                    "reader_tokens": u["reader_p"] + u["reader_c"],
                    "llm_calls": u["calls"],
                    "iterations": st["rewrites"],
                    "route": st["route"],
                    "stop_reason": st["stop_reason"],
                    "seen_titles": [ev[i]["title"] for i in st["order"]],
                    "trace": st["trace"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if not a.quiet:
                    print(format_trace(row))
    except DailyQuotaExceeded as e:
        print(f"\nStopped: {e}\nRe-run the same command later to resume.")
    if a.failures and out.exists():
        rows = [r for r in map(json.loads, out.open(encoding="utf-8")) if r["id"] in {q["id"] for q in chosen}]
        if rows:
            rec = sum(set(r["gold_titles"]) <= set(r["retrieved_titles"]) for r in rows)
            tok = sum(r["agent_tokens"] + r["reader_tokens"] for r in rows) / len(rows)
            print(f"\nRECOVERY: {rec}/{len(rows)} of these baseline-failure questions now have every gold paragraph "
                  f"in the final evidence | mean tokens/question {tok:.0f}")
    print("\n" + stats_summary())
    print(f"Compare with baseline:  python -m app.eval.compare_recovery data/runs/baseline_k5.jsonl {out}")


if __name__ == "__main__":
    main()
