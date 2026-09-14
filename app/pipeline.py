import re
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app import prompts
from app.config import AGENT_MODEL, TOP_K
from app.llm import chat
from app.reader import answer_question
from app.retrieval import retrieve

ROUTER_MODEL = AGENT_MODEL  


@dataclass
class PipelineConfig:
    k: int = TOP_K               
    final_k: int = 5             
    max_rewrites: int = 2        
    n_candidates: int = 3        
    selector: str = "first"      
    use_router: bool = True
    prompt_version: str = "v3"   
    name_guard: bool = True      
    show_max: int = 8            
    evidence_chars: int = 450    
    answer: bool = True          


class State(TypedDict, total=False):
    question: str
    route: str
    queries: List[str]
    evidence: Dict[str, Any]     
    order: List[str]             
    latest_ids: List[str]        
    useful: List[str]            
    checklist: List[dict]
    stalled: bool                
    rewrites: int
    stop_reason: str
    final_ids: List[str]
    answer: str
    trace: List[dict]
    usage: Dict[str, int]


def _bump(usage, resp, kind):
    u = dict(usage)
    u[f"{kind}_p"] += resp.prompt_tokens
    u[f"{kind}_c"] += resp.completion_tokens
    u["calls"] += 1
    return u


def _titles(evidence, ids):
    return [evidence[i]["title"] for i in ids]


_STOP = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how", "is", "are", "was", "were",
         "does", "did", "do", "in", "the", "a", "an", "of", "on", "and", "or", "for", "to"}


def _near_dup(q, tried, thresh=0.7):
    """True if q is (almost) the same bag of words as an already-tried query."""
    a = set(re.findall(r"\w+", q.lower()))
    for t in tried:
        b = set(re.findall(r"\w+", t.lower()))
        if a and b and len(a & b) / len(a | b) >= thresh:
            return True
    return False


def _new_names(query, allowed_text, possessive=False):
    """Capitalised words in `query` that appear nowhere in the question / findings / earlier queries.
    A cheap guard against the LLM inventing an entity from its own memory."""
    def base(w):
        w = w.lower()
        return re.sub(r"['\u2019]s$", "", w) if possessive else w   # v4: "Houten's" == "Houten"

    allowed = {base(w) for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", allowed_text)}
    out = []
    for w in re.findall(r"\b[A-Z][a-z][A-Za-z'\-]*\b", query):
        if base(w) not in allowed and w.lower() not in _STOP and w not in out:
            out.append(w)
    return out


def build_graph(cfg=None):
    cfg = cfg or PipelineConfig()

    def route_query(state):
        if not cfg.use_router:
            step = {"step": "route", "route": "complex", "note": "router disabled"}
            return {"route": "complex", "trace": state["trace"] + [step]}
        resp = chat(
            prompts.router_prompt(state["question"]),
            system=prompts.ROUTER_SYSTEM,
            model=ROUTER_MODEL,
            max_tokens=256,
        )
        route = prompts.parse_route(resp.text)
        return {
            "route": route,
            "usage": _bump(state["usage"], resp, "agent"),
            "trace": state["trace"] + [{"step": "route", "route": route}],
        }

    def initial_retrieval(state):
        q = state["question"]
        hits = retrieve(q, k=cfg.k)
        ids = [h["doc_id"] for h in hits]
        step = {"step": "retrieve", "query": q, "titles": [h["title"] for h in hits]}
        return {
            "queries": [q],
            "evidence": {h["doc_id"]: h for h in hits},
            "order": ids,
            "latest_ids": ids,
            "trace": state["trace"] + [step],
        }

    def checklist_gate(state):
        ev = state["evidence"]
        shown = (state["useful"] + [i for i in state["latest_ids"] if i not in state["useful"]])[: cfg.show_max]
        passages = [ev[i]["text"][: cfg.evidence_chars] for i in shown]
        usage, items, err = state["usage"], None, ""
        for attempt in range(2):
            prompt = prompts.checklist_prompt(
                state["question"], passages, state["checklist"] or None, version=cfg.prompt_version
            )
            if attempt:
                prompt += "\n\nYour previous reply was not valid. Return ONLY the JSON object."
            resp = chat(
                prompt,
                system=prompts.CHECKLIST_SYSTEM,
                model=AGENT_MODEL,
                max_tokens=1024,
                salt="retry" if attempt else "",
            )
            usage = _bump(usage, resp, "agent")
            try:
                items = prompts.parse_checklist(resp.text, len(passages))
                break
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                err = f"{type(e).__name__}: {e}"
        step = {"step": "checklist", "shown": _titles(ev, shown)}
        if items is None:  
            step["error"] = err
            return {"usage": usage, "stop_reason": "checklist_parse_failed", "trace": state["trace"] + [step]}

        useful = list(state["useful"])
        for it in items:
            if it["status"] == "confirmed":
                for n in it["passages"]:
                    if shown[n - 1] not in useful:
                        useful.append(shown[n - 1])
        sufficient = all(it["status"] == "confirmed" for it in items)
        prev_conf = sum(it["status"] == "confirmed" for it in state["checklist"])
        stalled = state["rewrites"] > 0 and sum(it["status"] == "confirmed" for it in items) <= prev_conf
        step.update(items=items, sufficient=sufficient, stalled=stalled)
        out = {
            "checklist": items,
            "useful": useful,
            "stalled": stalled,
            "usage": usage,
            "trace": state["trace"] + [step],
        }
        if sufficient:
            out["stop_reason"] = "sufficient"
        elif state["rewrites"] >= cfg.max_rewrites:
            out["stop_reason"] = "max_rewrites"
        return out

    def rewrite_query(state):
        found = [it["finding"] or it["need"] for it in state["checklist"] if it["status"] == "confirmed"]
        missing = [it["need"] for it in state["checklist"] if it["status"] == "missing"]
        tried = state["queries"]
        usage, parsed, err = state["usage"], None, ""
        for attempt in range(2):
            prompt = prompts.rewrite_prompt(
                state["question"], found, missing, tried, cfg.n_candidates,
                stalled=state.get("stalled", False), version=cfg.prompt_version
            )
            if attempt:
                prompt += "\n\nYour previous reply was not valid. Return ONLY the JSON object."
            resp = chat(
                prompt,
                system=prompts.REWRITE_SYSTEM,
                model=AGENT_MODEL,
                max_tokens=1024,
                salt="retry" if attempt else "",
            )
            usage = _bump(usage, resp, "agent")
            try:
                parsed = prompts.parse_rewrite(resp.text)
                break
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                err = f"{type(e).__name__}: {e}"
        step = {"step": "rewrite", "found": found, "missing": missing}
        if parsed is None:
            step["error"] = err
            return {"usage": usage, "stop_reason": "rewrite_parse_failed", "trace": state["trace"] + [step]}

        reflection, queries = parsed
        queries = [q for q in queries if not _near_dup(q, tried)][: cfg.n_candidates]
        step["reflection"] = reflection
        if not queries:
            return {"usage": usage, "stop_reason": "no_new_query", "trace": state["trace"] + [step]}

        allowed = " ".join([state["question"]] + found + missing + tried)
        ev = dict(state["evidence"])
        cands = []
        for q in queries:
            hits = retrieve(q, k=cfg.k)
            novel = sum(h["doc_id"] not in ev for h in hits)
            cands.append({"query": q, "hits": hits, "novel": novel, "new_names": _new_names(q, allowed, possessive=cfg.prompt_version != "v3")})
        pool = [i for i, c in enumerate(cands) if not c["new_names"]] if cfg.name_guard else []
        pool = pool or list(range(len(cands)))
        chosen = max(pool, key=lambda i: (cands[i]["novel"], -i)) if cfg.selector == "novelty" else pool[0]
        step["candidates"] = [
            {
                "query": c["query"],
                "titles": [h["title"] for h in c["hits"]],
                "novel": c["novel"],
                "new_names": c["new_names"],
            }
            for c in cands
        ]
        step["chosen"] = chosen

        order = list(state["order"])
        for h in cands[chosen]["hits"]:
            if h["doc_id"] not in ev:
                ev[h["doc_id"]] = h
                order.append(h["doc_id"])
            elif h["score"] > ev[h["doc_id"]]["score"]:
                ev[h["doc_id"]] = h
        out = {
            "evidence": ev,
            "order": order,
            "latest_ids": [h["doc_id"] for h in cands[chosen]["hits"]],
            "queries": tried + [cands[chosen]["query"]],
            "rewrites": state["rewrites"] + 1,
            "usage": usage,
            "trace": state["trace"] + [step],
        }
        if cands[chosen]["novel"] == 0:
            out["stop_reason"] = "no_new_evidence"
        return out

    def generate(state):
        ev, order = state["evidence"], state["order"]
        if state["rewrites"] == 0:
            final_ids = order[: cfg.final_k]  
        else:
            useful = sorted(state["useful"], key=lambda i: -ev[i]["score"])[: cfg.final_k]
            rest = sorted((i for i in order if i not in useful), key=lambda i: -ev[i]["score"])
            final_ids = (useful + rest)[: cfg.final_k]
        answer, usage = "", state["usage"]
        if cfg.answer:
            answer, resp = answer_question(state["question"], [ev[i] for i in final_ids])
            usage = _bump(usage, resp, "reader")
        stop = state.get("stop_reason") or ("simple_route" if state.get("route") == "simple" else "done")
        step = {"step": "generate", "final_titles": _titles(ev, final_ids), "answer": answer}
        return {
            "final_ids": final_ids,
            "answer": answer,
            "usage": usage,
            "stop_reason": stop,
            "trace": state["trace"] + [step],
        }

    def after_retrieval(state):
        return "generate" if state["route"] == "simple" else "checklist_gate"

    def after_checklist(state):
        return "generate" if state.get("stop_reason") else "rewrite_query"

    def after_rewrite(state):
        return "generate" if state.get("stop_reason") else "checklist_gate"

    g = StateGraph(State)
    g.add_node("route_query", route_query)
    g.add_node("initial_retrieval", initial_retrieval)
    g.add_node("checklist_gate", checklist_gate)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_edge(START, "route_query")
    g.add_edge("route_query", "initial_retrieval")
    g.add_conditional_edges("initial_retrieval", after_retrieval, ["generate", "checklist_gate"])
    g.add_conditional_edges("checklist_gate", after_checklist, ["generate", "rewrite_query"])
    g.add_conditional_edges("rewrite_query", after_rewrite, ["generate", "checklist_gate"])
    g.add_edge("generate", END)
    return g.compile()


def run_question(graph, question):
    """Run one question through the compiled graph; returns the final state dict."""
    init = {
        "question": question,
        "usage": {"agent_p": 0, "agent_c": 0, "reader_p": 0, "reader_c": 0, "calls": 0},
        "trace": [],
        "stop_reason": "",
        "queries": [],
        "evidence": {},
        "order": [],
        "latest_ids": [],
        "useful": [],
        "checklist": [],
        "stalled": False,
        "rewrites": 0,
    }
    return graph.invoke(init, config={"recursion_limit": 50})
