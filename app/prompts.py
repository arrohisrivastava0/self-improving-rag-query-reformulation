import re
import unicodedata

from app.llm import parse_json

ROUTER_SYSTEM = "You classify questions for a document-retrieval system."
CHECKLIST_SYSTEM = "You are a strict evidence auditor. You only trust the passages you are shown."
REWRITE_SYSTEM = "You write search queries for a document-retrieval system."

def router_prompt(question):
    return (
        "Decide whether a document search system can answer this question from ONE document "
        "or needs facts from MULTIPLE documents.\n\n"
        "SIMPLE = a single fact about ONE directly named thing, stated in one document.\n"
        "COMPLEX = answering needs at least two lookups. Typical signs:\n"
        '- the question refers to something INDIRECTLY (\"the actress who starred in X\", \"the facility where Y '
        'worked\", \"the author of the book that ...\"), so you must first find out WHAT it is and then look up '
        "a fact about it;\n"
        "- it compares two or more things;\n"
        '- it combines two conditions (\"which X is also a Y\").\n\n'
        "Examples:\n"
        "Q: What year was the Eiffel Tower completed? -> SIMPLE\n"
        "Q: Who directed the film that won Best Picture in 1998? -> COMPLEX\n"
        "Q: Which is older, the Colosseum or the Pantheon? -> COMPLEX\n"
        "Q: What is the birthplace of the author of the novel Dracula? -> COMPLEX\n"
        "Q: What is the capital of Australia? -> SIMPLE\n\n"
        f"Q: {question}\n"
        "If in doubt, answer COMPLEX.\n"
        "Reply with exactly one word: SIMPLE or COMPLEX."
    )


def parse_route(text):
    t = text.lower()
    return "simple" if ("simple" in t and "complex" not in t) else "complex"  # when unsure, be thorough

def checklist_prompt(question, passages, previous=None, version="v3"):
    plist = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    prev = ""
    if previous:
        lines = []
        for it in previous:
            line = f"- {it['need']} [{it['status']}]"
            if it.get("finding"):
                line += f" - {it['finding']}"
            lines.append(line)
        prev = (
            "Previous checklist (keep these items; re-check each one against the passages below):\n"
            + "\n".join(lines)
            + "\n\n"
        )
    v4 = version != "v3"
    lit = (
        "0. Read the question literally. A phrase like \"the actress and writer\" or \"the singer and songwriter\" "
        "usually describes ONE person with two roles - do not split it into two different people unless the "
        "question clearly refers to two.\n"
        if v4
        else ""
    )
    exc = (
        " EXCEPTION: if a passage lists several candidates that could answer the question (a cast, band "
        "members, any list of names), name ALL of them in the finding."
        if v4
        else ""
    )
    return (
        "You audit whether the passages below contain the information needed to answer a question.\n\n"
        f"Question: {question}\n\n"
        + prev
        + f"Passages:\n{plist}\n\n"
        "Task:\n"
        + lit
        + "1. List the minimal facts needed to answer the question (2-4 items). Each \"need\" is a SHORT phrase "
        "(under 12 words) naming the fact to find - it must not contain the answer itself.\n"
        '2. Mark an item "confirmed" ONLY if a passage explicitly states it; otherwise "missing". '
        "Do not use outside knowledge.\n"
        "3. For confirmed items give the finding in under 15 words, including the specific names/values "
        "found, and the numbers of the supporting passages."
        + exc
        + "\n\n"
        "Return ONLY a JSON object, no other text:\n"
        '{"checklist": [{"need": "...", "status": "confirmed", "finding": "...", "passages": [1]}, '
        '{"need": "...", "status": "missing", "finding": "", "passages": []}]}'
    )


def parse_checklist(text, n_passages):
    """Validate the model's JSON. A 'confirmed' item must cite a real passage number, otherwise it
    is demoted to 'missing' (guards against hallucinated confirmations)."""
    obj = parse_json(text)
    items = obj.get("checklist")
    if not isinstance(items, list) or not items:
        raise ValueError("no checklist list")
    out = []
    for it in items[:6]:
        if not isinstance(it, dict):
            continue
        need = str(it.get("need", "")).strip()
        if not need:
            continue
        raw = it.get("passages", [])
        raw = raw if isinstance(raw, list) else [raw]
        nums = []
        for x in raw:
            try:
                n = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= n_passages and n not in nums:
                nums.append(n)
        confirmed = str(it.get("status", "")).lower().startswith("confirm") and bool(nums)
        out.append(
            {
                "need": need[:200],
                "status": "confirmed" if confirmed else "missing",
                "finding": str(it.get("finding", "")).strip()[:200] if confirmed else "",
                "passages": nums if confirmed else [],
            }
        )
    if not out:
        raise ValueError("empty checklist")
    return out

def rewrite_prompt(question, found, missing, tried, n, stalled=False, version="v3"):
    def bullets(xs):
        return "\n".join(f"- {x}" for x in xs) if xs else "- (nothing yet)"

    return (
        "A search over a document collection has not yet found everything needed to answer a question.\n\n"
        f"Question: {question}\n\n"
        f"Already found:\n{bullets(found)}\n\n"
        f"Still missing:\n{bullets(missing)}\n\n"
        f"Queries already tried (do not repeat or rephrase them):\n{bullets(tried)}\n\n"
        "First write ONE sentence: what the passages found so far establish, and what link is still missing.\n"
        f"Then write {n} different search queries that would retrieve a passage containing the MISSING information.\n"
        "Rules:\n"
        "- Use ONLY names that appear in the question or under \"Already found\". NEVER introduce a name from "
        "your own memory.\n"
        "- If \"Already found\" lists several candidates and the missing fact is a property only one of them "
        "has, combine that property with the candidate names (e.g. \"Which of A, B, C appeared in <property>?\" "
        "or \"<A> <property>\").\n"
        "- Target the missing information directly. Keep each query under 20 words.\n"
        "- At most one query may be a close paraphrase of the original question. Vary the style: a natural "
        "question, a keyword-style query, and a statement that resembles how the target passage would be written.\n\n"
        + (
            "IMPORTANT: your previous queries did not confirm any new fact. Do not continue the same line of "
            "search. Re-read the question: it may mean something different from what you assumed (for example "
            "one person with two roles rather than two people) or concern a different entity. Use other names "
            "from \"Already found\" - for example probe each candidate in a list.\n\n"
            if (stalled and version != "v3")
            else ""
        )
        + 'Return ONLY a JSON object: {"reflection": "...", "queries": ["...", "...", "..."]}'
    )


def clean_query(q):
    """gpt-oss likes typographic characters (narrow no-break spaces, non-breaking hyphens, curly quotes)
    that silently change embeddings. Normalise them away."""
    q = unicodedata.normalize("NFKC", str(q))
    q = re.sub(r"[\u2010-\u2015\u2212]", "-", q)
    q = q.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(q.split())


def parse_rewrite(text):
    obj = parse_json(text)
    qs = obj.get("queries")
    if not isinstance(qs, list):
        raise ValueError("no queries list")
    queries, seen = [], set()
    for q in qs:
        q = clean_query(q)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    if not queries:
        raise ValueError("empty queries")
    return str(obj.get("reflection", "")).strip()[:300], queries
