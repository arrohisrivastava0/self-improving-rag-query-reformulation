import csv
from collections import Counter

import matplotlib

matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np

from app.config import BASE_DIR, RUNS_DIR
from app.eval.analyze_traces import classify, initial_titles
from app.eval.answer_errors import CATS, category
from app.eval.audit_relaxed import strict_match
from app.eval.metrics import exact_match, f1_score, relaxed_match
from app.eval.score import load
from app.eval.stats import boot_ci, reader_tokens, tokens

OUT = BASE_DIR / "report_assets"
CLOSED = "baseline_k0_test"
RUNS = {
    "Baseline k=5": "baseline_k5_test",
    "Baseline k=10": "baseline_k10_test",
    "Adaptive (2 rewrites)": "pipeline_v3_test",
    "Adaptive (1 rewrite)": "pipeline_v3_test_rw1",
}
MAIN = ("Baseline k=5", "Baseline k=10", "Adaptive (2 rewrites)")
METRICS = (("EM", exact_match), ("F1", f1_score), ("Acc", relaxed_match), ("AccStrict", strict_match))
FAIL_ORDER = ("recovered", "cap-limited", "gate fooled", "selector-limited", "query-limited")


def scores(rows, fn):
    return np.array([fn(r["prediction"], r["gold_answer"]) for r in rows])


def fmt(xs):
    lo, hi = boot_ci(xs)
    return f"{np.mean(xs):.3f} [{lo:.3f}, {hi:.3f}]"


def write_table(stem, header, rows):
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    data = {label: load(RUNS_DIR / f"{stem}.jsonl") for label, stem in RUNS.items()}
    common = sorted(set.intersection(*(set(r) for r in data.values())))
    rows = {label: [r[i] for i in common] for label, r in data.items()}
    print(f"{len(common)} questions common to all runs -> {OUT}")

    table = []
    for label, rs in rows.items():
        t = np.mean([tokens(r) for r in rs])
        rd = np.mean([reader_tokens(r) for r in rs])
        gold = np.mean([set(r["gold_titles"]) <= set(r["retrieved_titles"]) for r in rs])
        table.append([label, *[fmt(scores(rs, fn)) for _, fn in METRICS], f"{gold:.3f}", f"{t:.0f} ({t - rd:.0f} agent + {rd:.0f} reader)"])
    write_table(OUT / "results_main", ["System", *[m for m, _ in METRICS], "All-gold", "Tokens/q"], table)

    types = ("bridge", "comparison")
    first = next(iter(rows.values()))
    ns = {t: sum(r["type"] == t for r in first) for t in types}
    header = ["System"] + [f"{t} {m} (n={ns[t]})" for t in types for m in ("Acc", "AccStrict")]
    table = []
    for label, rs in rows.items():
        line = [label]
        for t in types:
            sub = [r for r in rs if r["type"] == t]
            line += [f"{scores(sub, relaxed_match).mean():.3f}", f"{scores(sub, strict_match).mean():.3f}"]
        table.append(line)
    write_table(OUT / "results_by_type", header, table)

    # 3. accuracy vs cost
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (ylab, fn) in zip(axes, (("Relaxed Accuracy", relaxed_match), ("AccStrict", strict_match))):
        for label, rs in rows.items():
            x, s = np.mean([tokens(r) for r in rs]), scores(rs, fn)
            lo, hi = boot_ci(s)
            ax.errorbar(x, s.mean(), yerr=[[s.mean() - lo], [hi - s.mean()]], fmt="o", capsize=4)
            left = label == "Adaptive (1 rewrite)"
            ax.annotate(label, (x, s.mean()), textcoords="offset points", xytext=(-6 if left else 6, -12), ha="right" if left else "left", fontsize=8)        ax.set_xlim(0, 3800)
        ax.set_xlabel("tokens per question (agent + reader)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs. cost, held-out test set (n=100, 95% bootstrap CIs)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_accuracy.png", dpi=200)
    plt.close(fig)

    pipe = list(data["Adaptive (2 rewrites)"].values())
    fail = [r for r in pipe if not set(r["gold_titles"]) <= initial_titles(r)]
    cats = Counter(classify(r) for r in fail)
    counts = [cats[c] for c in FAIL_ORDER]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(FAIL_ORDER[::-1], counts[::-1])
    for y, c in enumerate(counts[::-1]):
        ax.text(c + 0.2, y, f"{c} ({c / len(fail):.0%})", va="center")
    ax.set_xlim(0, max(counts) + 4)
    ax.set_xlabel("questions")
    ax.set_title(f"Initial k=5 retrieval failures ({len(fail)}): outcome with the adaptive pipeline")
    fig.tight_layout()
    fig.savefig(OUT / "fig_failure_taxonomy.png", dpi=200)
    plt.close(fig)

    closed = load(RUNS_DIR / f"{CLOSED}.jsonl")

    def cat_of(r):
        c = closed[r["id"]]
        return category(r, relaxed_match(c["prediction"], c["gold_answer"]) == 1.0)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = np.zeros(len(MAIN))
    per = {label: Counter(cat_of(r) for r in rows[label]) for label in MAIN}
    for c in CATS:
        vals = np.array([per[label][c] for label in MAIN])
        ax.barh(MAIN, vals, left=left, label=c)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("questions (n=100)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Answer outcome by evidence coverage")
    fig.tight_layout()
    fig.savefig(OUT / "fig_answer_categories.png", dpi=200)
    plt.close(fig)

    print("wrote: results_main.md/.csv, results_by_type.md/.csv, fig_cost_accuracy.png, fig_failure_taxonomy.png, fig_answer_categories.png")


if __name__ == "__main__":
    main()