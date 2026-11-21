import itertools
import sys
from pathlib import Path

import numpy as np

from app.config import SEED
from app.eval.metrics import exact_match, f1_score, relaxed_match
from app.eval.score import load, sign_p

N_BOOT = 10000
METRICS = (("EM", exact_match), ("F1", f1_score), ("Acc", relaxed_match))


def boot_ci(xs, n_boot=N_BOOT):
    """95% percentile bootstrap CI for the mean of xs (per-question scores or paired differences)."""
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(SEED)
    means = xs[rng.integers(0, len(xs), size=(n_boot, len(xs)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def tokens(r):
    return r["prompt_tokens"] + r["completion_tokens"]


def reader_tokens(r):
    return r.get("reader_tokens", tokens(r))   # baseline rows have no agent/reader split


def _cell(xs):
    lo, hi = boot_ci(xs)
    return f"{np.mean(xs):.3f} [{lo:.3f},{hi:.3f}]"


def main(paths):
    runs = {Path(p).stem: load(p) for p in paths}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not common:
        sys.exit("No questions in common.")
    names = list(runs)
    per = {
        n: {
            lab: np.array([fn(runs[n][i]["prediction"], runs[n][i]["gold_answer"]) for i in common])
            for lab, fn in METRICS
        }
        for n in names
    }

    print(f"{len(common)} questions common to all runs | {N_BOOT} bootstrap resamples, 95% percentile CIs\n")
    print(f"{'run':<22}" + "".join(f"{lab:>22}" for lab, _ in METRICS))
    for n in names:
        print(f"{n:<22}" + "".join(f"{_cell(per[n][lab]):>22}" for lab, _ in METRICS))

    for a, b in itertools.combinations(names, 2):
        print(f"\n{b} vs {a} (paired)")
        for lab in ("EM", "Acc"):
            d = per[b][lab] - per[a][lab]
            w, l = int((d > 0).sum()), int((d < 0).sum())
            lo, hi = boot_ci(d)
            print(f"  {lab:<4} diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] | wins {w}, losses {l}, ties {len(d) - w - l}, sign-test p = {sign_p(w, l):.3f}")

    print("\nCOST (tokens per question)")
    print(f"{'run':<22}{'total':>8}{'agent':>8}{'reader':>8}{'Acc per 1k tok':>16}")
    tot = {}
    for n in names:
        rows = [runs[n][i] for i in common]
        t = float(np.mean([tokens(r) for r in rows]))
        rd = float(np.mean([reader_tokens(r) for r in rows]))
        tot[n] = t
        print(f"{n:<22}{t:>8.0f}{t - rd:>8.0f}{rd:>8.0f}{per[n]['Acc'].mean() / (t / 1000):>16.3f}")
    print()
    for a, b in itertools.combinations(names, 2):
        extra_tok = (tot[b] - tot[a]) * len(common)
        extra_ok = per[b]["Acc"].sum() - per[a]["Acc"].sum()
        if extra_ok <= 0:
            msg = "no accuracy gain"
        elif extra_tok <= 0:
            msg = "more accurate and not more expensive"
        else:
            msg = f"{extra_tok / extra_ok:,.0f} extra tokens per extra correct answer"
        print(f"  {b} vs {a}: {int(extra_ok)} extra correct answers | {msg}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m app.eval.stats RUN_A.jsonl RUN_B.jsonl [RUN_C.jsonl ...]")
    main(sys.argv[1:])