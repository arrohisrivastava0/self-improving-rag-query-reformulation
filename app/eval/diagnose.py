import json
import sys

from app.eval.metrics import exact_match, relaxed_match

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
for r in rows:
    r["full"] = set(r["gold_titles"]) <= set(r["retrieved_titles"])
    r["em"] = exact_match(r["prediction"], r["gold_answer"])
    r["acc"] = relaxed_match(r["prediction"], r["gold_answer"])


def mean(rs, key):
    return sum(r[key] for r in rs) / max(len(rs), 1)


full = [r for r in rows if r["full"]]
miss = [r for r in rows if not r["full"]]
print(f"{'':34}{'n':>4}{'EM':>7}{'Acc':>7}")
print(f"{'all gold paragraphs retrieved':<34}{len(full):>4}{mean(full, 'em'):>7.3f}{mean(full, 'acc'):>7.3f}")
print(f"{'a gold paragraph missing':<34}{len(miss):>4}{mean(miss, 'em'):>7.3f}{mean(miss, 'acc'):>7.3f}")

print("\nGENUINE errors with all gold retrieved (reader/reasoning problems):")
for r in [r for r in full if r["acc"] == 0]:
    print(f"  gold={r['gold_answer']!r:34} pred={r['prediction']!r}")

print("\nGENUINE errors with a gold paragraph missing (retrieval problems - the loop's target):")
for r in [r for r in miss if r["acc"] == 0][:20]:
    print(f"  gold={r['gold_answer']!r:34} pred={r['prediction']!r}")
