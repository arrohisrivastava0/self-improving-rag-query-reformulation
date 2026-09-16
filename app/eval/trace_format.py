def _fmt_titles(titles, gold):
    return ", ".join(("*" if t in gold else "") + t for t in titles)


def format_trace(row):
    gold = set(row["gold_titles"])
    lines = [f"\nQ [{row['type']}]: {row['question']}", f"   gold answer: {row['gold_answer']!r} | gold paragraphs (*): {sorted(gold)}"]
    for s in row["trace"]:
        if s["step"] == "route":
            lines.append(f"   route: {s['route']}")
        elif s["step"] == "retrieve":
            lines.append(f"   retrieve  {s['query']!r}\n      -> {_fmt_titles(s['titles'], gold)}")
        elif s["step"] == "checklist":
            if "error" in s:
                lines.append(f"   checklist: PARSE FAILED ({s['error']})")
                continue
            lines.append("   checklist:" + ("   (no new fact confirmed - STALLED)" if s.get("stalled") else ""))
            for it in s["items"]:
                extra = f" - {it['finding']}" if it["finding"] else ""
                lines.append(f"      [{it['status']:9}] {it['need']}{extra}")
        elif s["step"] == "rewrite":
            if "reflection" in s:
                lines.append(f"   reflect: {s['reflection']}")
            for i, c in enumerate(s.get("candidates", [])):
                mark = "  <== chosen" if i == s["chosen"] else ""
                if c.get("new_names"):
                    mark = f"  [invented names: {', '.join(c['new_names'])}]" + mark
                lines.append(f"   candidate {i + 1}: {c['query']!r}  (new={c['novel']}){mark}\n      -> {_fmt_titles(c['titles'], gold)}")
            if "error" in s:
                lines.append(f"   rewrite: PARSE FAILED ({s['error']})")
    has_all = gold <= set(row["retrieved_titles"])
    lines.append(
        f"   stop: {row['stop_reason']} | rewrites: {row['iterations']} | final evidence has all gold: {'YES' if has_all else 'no'}"
    )
    if row["prediction"] != "":
        lines.append(f"   answer: {row['prediction']!r}  (gold {row['gold_answer']!r})")
    lines.append(f"   tokens: agent {row['agent_tokens']}, reader {row['reader_tokens']}, LLM calls {row['llm_calls']}")
    return "\n".join(lines)
