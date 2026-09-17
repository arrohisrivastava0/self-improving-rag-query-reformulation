import argparse
import json

from app import prompts
from app.config import QUESTIONS_PATH
from app.llm import stats_summary
from app.pipeline import ROUTER_MODEL
from app.llm import chat

SIMPLE_QS = [
    "What year did World War II end?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the capital of Canada?",
    "Who painted the Starry Night?",
    "What is the chemical symbol for sodium?",
    "Who was the first person to walk on the Moon?",
    "What is the largest planet in the solar system?",
    "In what year did the Berlin Wall fall?",
    "Who discovered penicillin?",
    "What is the longest river in Africa?",
    "How many players are on the field for one soccer team?",
    "What is the boiling point of water in Celsius?",
]


def route(q):
    r = chat(prompts.router_prompt(q), system=prompts.ROUTER_SYSTEM, model=ROUTER_MODEL, max_tokens=256)
    return prompts.parse_route(r.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    hotpot = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[: a.n]

    hp = [(q, route(q["question"])) for q in hotpot]
    sm = [(q, route(q)) for q in SIMPLE_QS]
    ok_c = sum(r == "complex" for _, r in hp)
    ok_s = sum(r == "simple" for _, r in sm)
    print(f"multi-hop (HotpotQA) routed COMPLEX: {ok_c}/{len(hp)}   <- want ~100%")
    print(f"single-fact routed SIMPLE          : {ok_s}/{len(sm)}   <- want ~100%")
    for q, r in hp:
        if r != "complex":
            print(f"  misrouted multi-hop: [{q['type']}] {q['question']}")
    for q, r in sm:
        if r != "simple":
            print(f"  misrouted single-fact: {q}")
    print(stats_summary())


if __name__ == "__main__":
    main()
