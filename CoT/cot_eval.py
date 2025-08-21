# -*- coding: utf-8 -*-
"""
Lightweight evaluation utilities for CoT outputs.

Covers:
- Answer correctness: exact/normalized/numeric/regex/MCQ
- Heuristics for CoT steps: count/length/redundancy
- Self-consistency aggregation: vote share & entropy
- Calibration metrics (optional): ECE, Brier (needs confidences)

No external deps. Keep it simple & fast.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Tuple
import math
import re
import statistics

# -------------------------
# Normalization helpers
# -------------------------
WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("\u3000", " ")  # 全角スペース
    s = WHITESPACE_RE.sub(" ", s)
    # 末尾句点のゆらぎ吸収（日本語タスク向け）
    s = s.rstrip("。.")
    return s

def only_digits_and_sign(s: str) -> str:
    return re.sub(r"[^0-9\.\-\+eE]", "", s or "")

# -------------------------
# Answer extraction helpers
# -------------------------
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def extract_first_number(s: str) -> Optional[float]:
    if not s:
        return None
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

# -------------------------
# Core correctness checks
# -------------------------
def exact_match(pred: str, gold: str) -> bool:
    return (pred or "") == (gold or "")

def normalized_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def regex_match(pred: str, pattern: str) -> bool:
    """pattern: Python regex that should match pred."""
    try:
        return re.search(pattern, pred or "") is not None
    except re.error:
        return False

def numeric_close(
    pred: str, gold: str,
    atol: float = 0.0, rtol: float = 0.0
) -> Optional[bool]:
    """
    Compare first numbers in pred & gold with tolerances.
    Returns True/False if both numbers exist; otherwise None.
    """
    p = extract_first_number(pred)
    g = extract_first_number(gold)
    if p is None or g is None:
        return None
    if atol > 0 and abs(p - g) <= atol:
        return True
    if rtol > 0 and g != 0 and abs(p - g) <= rtol * abs(g):
        return True
    return p == g  # exact float string equality fallback

def mcq_match(pred: str, correct_option: str) -> bool:
    """
    多肢選択の判定。
    例: correct_option='C'、predが 'Answer: C' / 'C) ...' / '選択肢C' 等でも拾いたい。
    """
    if not pred:
        return False
    p = normalize_text(pred).upper()
    c = normalize_text(correct_option).upper()
    # 代表的表現の吸収
    patterns = [
        rf"\b{re.escape(c)}\b",
        rf"ANSWER\s*[:：]\s*{re.escape(c)}",
        rf"選択肢\s*{re.escape(c)}",
        rf"\b\({re.escape(c)}\)\b",
        rf"^{re.escape(c)}\)",
    ]
    return any(re.search(pat, p) for pat in patterns)

# -------------------------
# Heuristic step quality
# -------------------------
@dataclass
class StepStats:
    n_steps: int
    mean_len: float
    max_len: int
    redundancy_ratio: float  # 重複行の割合（0〜1）
    empty_ratio: float       # 空/短すぎる行の割合（0〜1）

def compute_step_stats(steps: List[str]) -> StepStats:
    if not steps:
        return StepStats(0, 0.0, 0, 0.0, 1.0)
    lens = [len(s.strip()) for s in steps]
    n = len(steps)
    mean_len = statistics.mean(lens)
    max_len = max(lens)
    # 重複率
    uniq = set(s.strip() for s in steps if s.strip())
    redundancy = 1.0 - (len(uniq) / n) if n > 0 else 0.0
    # 短すぎる（<=3文字）を空扱い
    empty = sum(1 for s in steps if len(s.strip()) <= 3) / n
    return StepStats(n_steps=n, mean_len=mean_len, max_len=max_len,
                     redundancy_ratio=redundancy, empty_ratio=empty)

# -------------------------
# Keyword recall (optional)
# -------------------------
def keyword_recall(pred_text: str, required_keywords: Iterable[str]) -> float:
    """
    CoT中に含んでほしい語（根拠キーワード）の再現率をざっくり測る。
    """
    if not required_keywords:
        return 1.0
    p = normalize_text(pred_text)
    total = 0
    hit = 0
    for kw in required_keywords:
        kw = kw.strip()
        if not kw:
            continue
        total += 1
        if kw in p:
            hit += 1
    return hit / total if total else 1.0

# -------------------------
# Self-consistency aggregation
# -------------------------
def vote_share(answers: List[Optional[str]]) -> Dict[str, float]:
    """
    それぞれの答えの票割合を返す。None は除外。
    """
    counts: Dict[str, int] = {}
    for a in answers:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}

def entropy_from_share(share: Dict[str, float]) -> float:
    """自己一貫性サンプルの不確実性（エントロピー, bits）。"""
    if not share:
        return 0.0
    return -sum(p * math.log(p, 2) for p in share.values())

# -------------------------
# Calibration metrics (optional)
# -------------------------
def brier_score(prob_correct: float, is_correct: bool) -> float:
    """
    2値Brier。予測確信度（0..1）と正誤で評価。
    """
    y = 1.0 if is_correct else 0.0
    return (prob_correct - y) ** 2

def expected_calibration_error(
    confidences: List[float], corrects: List[bool], n_bins: int = 10
) -> float:
    """
    ECE（ナイーブ実装）。confidences: [0..1]
    """
    assert len(confidences) == len(corrects)
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, corrects):
        c = min(max(c, 0.0), 1.0)
        # 1.0 は最後のビンへ
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, 1.0 if y else 0.0))
    ece = 0.0
    n = len(confidences)
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        weight = len(bucket) / n
        ece += weight * abs(avg_conf - acc)
    return ece

# -------------------------
# High-level evaluator
# -------------------------
@dataclass
class EvalItem:
    problem: str
    gold: Optional[str] = None
    gold_regex: Optional[str] = None
    gold_numeric: bool = False
    mcq_correct: Optional[str] = None     # e.g., "C"
    required_keywords: Optional[List[str]] = None

@dataclass
class EvalResult:
    correct: Optional[bool]
    reason: str
    step_stats: StepStats
    metrics: Dict[str, Any]

def evaluate_single(
    pred_answer: Optional[str],
    steps: List[str],
    item: EvalItem,
    numeric_atol: float = 0.0,
    numeric_rtol: float = 0.0,
) -> EvalResult:
    """
    汎用の単一サンプル評価。
    ゴールドの種類（MCQ/正規表現/数値/文字列）を自動で優先判定。
    """
    # Step stats
    sstats = compute_step_stats(steps)

    # 判定順序：MCQ > regex > numeric > normalized string
    if item.mcq_correct:
        ok = mcq_match(pred_answer or "", item.mcq_correct)
        return EvalResult(correct=ok, reason="mcq", step_stats=sstats,
                          metrics={"match": ok})

    if item.gold_regex:
        ok = regex_match(pred_answer or "", item.gold_regex)
        return EvalResult(correct=ok, reason="regex", step_stats=sstats,
                          metrics={"match": ok})

    if item.gold_numeric:
        comp = numeric_close(pred_answer or "", item.gold or "",
                             atol=numeric_atol, rtol=numeric_rtol)
        return EvalResult(correct=comp, reason="numeric", step_stats=sstats,
                          metrics={"close": comp, "atol": numeric_atol, "rtol": numeric_rtol})

    if item.gold is not None:
        ok = normalized_match(pred_answer or "", item.gold or "")
        return EvalResult(correct=ok, reason="normalized", step_stats=sstats,
                          metrics={"match": ok})

    # ゴールドなし：キーワード再現だけ見る
    if item.required_keywords:
        rec = keyword_recall("\n".join(steps) + "\n" + (pred_answer or ""),
                             item.required_keywords)
        return EvalResult(correct=None, reason="keywords_only", step_stats=sstats,
                          metrics={"keyword_recall": rec})

    # 何も比較できない
    return EvalResult(correct=None, reason="no_gold", step_stats=sstats, metrics={})

def evaluate_self_consistency(
    answers: List[Optional[str]],
    gold: Optional[str] = None,
) -> Dict[str, Any]:
    """
    SCサンプル群のまとめ。goldがあれば正解率も返す。
    """
    share = vote_share(answers)
    ent = entropy_from_share(share)
    top_answer = max(share.items(), key=lambda x: x[1])[0] if share else None
    acc = None
    if gold is not None and answers:
        acc = sum(1 for a in answers if normalize_text(a) == normalize_text(gold)) / len(answers)
    return {"vote_share": share, "entropy": ent, "top_answer": top_answer, "sample_acc": acc}

# -------------------------
# Batch utilities
# -------------------------
def evaluate_batch(
    preds: List[Tuple[Optional[str], List[str]]],  # (answer, steps)
    items: List[EvalItem],
    numeric_atol: float = 0.0,
    numeric_rtol: float = 0.0,
) -> Dict[str, Any]:
    """
    一括評価。戻り値は集計と各サンプル詳細。
    """
    assert len(preds) == len(items)
    results: List[EvalResult] = []
    for (ans, steps), it in zip(preds, items):
        res = evaluate_single(ans, steps, it, numeric_atol, numeric_rtol)
        results.append(res)

    # 集計（correct が True/False のものだけ）
    judged = [r for r in results if isinstance(r.correct, bool)]
    acc = sum(1 for r in judged if r.correct) / len(judged) if judged else None

    # ステップ統計の要約
    n_steps = [r.step_stats.n_steps for r in results]
    mean_steps = statistics.mean(n_steps) if n_steps else 0.0

    return {
        "accuracy": acc,
        "mean_steps": mean_steps,
        "n_items": len(items),
        "n_judged": len(judged),
        "details": results,
    }
