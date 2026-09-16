import argparse
import json

from tqdm import tqdm

from app.config import QUESTIONS_PATH, RUNS_DIR, TOP_K
from app.llm import DailyQuotaExceeded, stats_summary
from app.reader import answer_question
from app.retrieval import retrieve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="how many questions (from the start of the eval set)")
    ap.add_argument("--k", type=int, default=TOP_K, help="passages given to the reader")
    ap.add_argument("--pool", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    allq = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    split = min(100, len(allq) // 2)
    questions = {"dev": allq[:split], "test": allq[split:], "all": allq}[a.pool][: a.n]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.pool == "dev" else f"_{a.pool}"
    out = RUNS_DIR / f"{a.name or f'baseline_k{a.k}{suffix}'}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open(encoding="utf-8")}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{out.name}: {len(done)} already done, {len(todo)} to run")

    try:
        with out.open("a", encoding="utf-8") as f:
            for q in tqdm(todo):
                hits = retrieve(q["question"], k=a.k) if a.k > 0 else []
                pred, resp = answer_question(q["question"], hits)
                row = {
                    "id": q["id"],
                    "type": q["type"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "prediction": pred,
                    "retrieved_titles": [h["title"] for h in hits],
                    "gold_titles": q["supporting_titles"],
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "iterations": 0,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    except DailyQuotaExceeded as e:
        print(f"\nStopped: {e}\nRe-run the same command later to resume.")
    print(stats_summary())
    print(f"Score it:  python -m app.eval.score {out}")


if __name__ == "__main__":
    main()
