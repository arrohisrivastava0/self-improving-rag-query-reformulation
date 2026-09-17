import json
from app.config import RETRIEVAL_CHECK_PATH

d = json.loads(RETRIEVAL_CHECK_PATH.read_text(encoding="utf-8"))
rows = [r for r in d["per_question"] if len(r["missing_at_5"]) == 1]
rows.sort(key=lambda r: len(r["question"]))
print(f"{len(rows)} questions with exactly one gold paragraph missing at top-5 (shortest first)\n")
for r in rows[:25]:
    print(r["id"], "|", r["type"], "|", r["question"])
    print("    missing:", r["missing_at_5"], "\n")