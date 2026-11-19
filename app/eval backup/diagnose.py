import json
import sys

from app.eval.metrics import exact_match

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
for r in rows:
    r["full"] = set(r["gold_titles"]) <= set(r["retrieved_titles"])
    r["em"] = exact_match(r["prediction"], r["gold_answer"])

def mean_em(rs):
    return sum(r["em"] for r in rs) / max(len(rs), 1)

full = [r for r in rows if r["full"]]
miss = [r for r in rows if not r["full"]]
print(f"EM when all gold retrieved      : {mean_em(full):.3f} (n={len(full)})")
print(f"EM when a gold paragraph missing: {mean_em(miss):.3f} (n={len(miss)})")

print("\nEM=0 even though all gold was retrieved (reader / answer-format errors):")
for r in [r for r in full if r["em"] == 0][:25]:
    print(f"  gold={r['gold_answer']!r:38} pred={r['prediction']!r}")