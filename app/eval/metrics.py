import re
import string
import unicodedata
from collections import Counter


def _canon(s):
    """Unicode canonicalisation: gpt-oss emits narrow no-break spaces, non-breaking hyphens, curly quotes."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    return s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def normalize_answer(s):
    s = _canon(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    p, g = normalize_answer(prediction), normalize_answer(ground_truth)
    special = ("yes", "no", "noanswer")
    if (p in special or g in special) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def _relaxed_tokens(s):
    s = re.sub(r"[^\w\s]", " ", _canon(s).lower())          
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def relaxed_match(prediction, ground_truth):
    """1.0 if one answer's tokens are contained in the other's (with a length guard).

    yes/no answers must match exactly. A prediction that merely CONTAINS the gold answer only
    counts if it adds at most 3 extra words, so long rambling answers are not rewarded.
    """
    p, g = _relaxed_tokens(prediction), _relaxed_tokens(ground_truth)
    if not p or not g:
        return 0.0
    special = {"yes", "no", "noanswer"}
    if (len(p) == 1 and p[0] in special) or (len(g) == 1 and g[0] in special):
        return float(p == g)
    ps, gs = set(p), set(g)
    if ps <= gs:
        return 1.0
    if gs <= ps and len(p) <= len(g) + 3:
        return 1.0
    return 0.0
