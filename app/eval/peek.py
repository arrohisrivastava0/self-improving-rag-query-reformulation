import json
import sys

from app.config import CORPUS_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if len(sys.argv) < 2:
    sys.exit('usage: python -m app.eval.peek "TITLE" ["PHRASE"]')
title, phrase = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
found = False
for line in CORPUS_PATH.open(encoding="utf-8"):
    d = json.loads(line)
    if d["title"].lower() == title.lower():
        found = True
        print(f"[{d['title']}]  ({len(d['text'])} chars)\n{d['text']}\n")
        if phrase:
            pos = d["text"].lower().find(phrase.lower())
            if pos < 0:
                print(f"'{phrase}' is not in this paragraph")
            else:
                note = "  <-- BEYOND the 450-char cap: the checklist never sees it" if pos > 450 else "  (inside the 450-char cap)"
                print(f"'{phrase}' starts at char {pos}{note}")
if not found:
    print(f"no paragraph titled {title!r}")
