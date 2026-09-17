import argparse
import json
import sys

from app.eval.analyze_traces import classify, initial_titles
from app.eval.trace_format import format_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ids", nargs="*", help="question ids or id prefixes")
    ap.add_argument("--category", choices=["recovered", "cap-limited", "gate fooled", "selector-limited", "query-limited"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8")]
    if a.category:
        sel = [r for r in rows if not set(r["gold_titles"]) <= initial_titles(r) and classify(r) == a.category]
    elif a.ids:
        sel = [r for r in rows if any(r["id"].startswith(i) for i in a.ids)]
    else:
        sys.exit("give question ids or --category")
    print(f"{len(sel)} trace(s)")
    for r in sel:
        print(format_trace(r))


if __name__ == "__main__":
    main()
