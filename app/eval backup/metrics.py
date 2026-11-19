"""HotpotQA-official Exact Match and F1 (same normalisation as the official eval script)."""
import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()
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
