# -*- coding: utf-8 -*-
"""
Lightweight evaluation utilities for CoT outputs (clean ver.).

Covers:
- Answer correctness: exact/normalized/numeric/regex/MCQ
- Heuristics for CoT steps: count/length/redundancy
- Self-consistency aggregation: vote share & entropy
- Calibration metrics: ECE, Brier (needs confidences)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Tuple
from collections import Counter
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
    s = s.strip().replace("\u3000", " ")
    s = WHITESPACE_RE.sub(" ", s)
    return s.rstrip("。.")

def _nz(s: Optional[str]) -> str:
    """None → '' の薄い補助。"""
    return s or ""

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
    return _nz(pred) == _nz(gold)

def normalized_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def regex_match(pred: str, pattern: str) -> bool:
    """pattern: Python regex that should match pred."""
    try:
        return re.search(pattern, _nz(pred)) is not None
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
    return p == g  # fallback

def mcq_match(pred: str, correct_option: str) -> bool:
    """
    多肢選択の判定。
    例: correct_option='C'、predが 'Answer: C' / 'C) ...' / '選択肢C' 等でも拾う。
    """
    if not pred:
        return False
    p = normalize_text(pred).upper()
    c = normalize_text(correct_option).upper()
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
    cleaned = [s.strip() for s in steps]
    lens = [len(s) for s in cleaned]
    n = len(cleaned)
    uniq = set(s for s in cleaned if s)
    redundancy = 1.0 - (len(uniq) / n)
    empty = sum(1 for s in cleaned if len(s) <= 3) / n
    return StepStats(
        n_steps=n,
        mean_len=statistics.mean(lens),
        max_len=max(lens),
        redundancy_ratio=redundancy,
        empty_ratio=empty,
    )

# -------------------------
# Keyword recall (optional)
# -------------------------
def keyword_recall(pred_text: str, required_keywords: Iterable[str]) -> float:
    """
    CoT中に含んでほしい語（根拠キーワード）の再現率をざっくり測る。
    """
    kws = [kw.strip() for kw in (required_keywords or []) if kw and kw.strip()]
    if not kws:
        return 1.0
    p = normalize_text(pred_text)
    hit = sum(1 for kw in kws if kw in p)
    return hit / len(kws)

# -------------------------
# Self-consistency aggregation
# -------------------------
def vote_share(answers: List[Optional[str]]) -> Dict[str, float]:
    """
    それぞれの答えの票割合を返す。None は除外。
    """
    cnt = Counter(a for a in answers if a is not None)
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()} if total else {}

def entropy_from_share(share: Dict[str, float]) -> float:
    """自己一貫性サンプルの不確実性（エントロピー, bits）。"""
    if not share:
        return 0.0
    return -sum(p * math.log2(p) for p in share.values())

# -------------------------
# Calibration metrics (optional)
# -------------------------
def brier_score(prob_correct: float, is_correct: bool) -> float:
    """2値Brier。予測確信度（0..1）と正誤で評価。"""
    y = 1.0 if is_correct else 0.0
    return (prob_correct - y) ** 2

def expected_calibration_error(
    confidences: List[float], corrects: List[bool], n_bins: int = 10
) -> float:
    """ECE（ナイーブ実装）。confidences: [0..1]"""
    assert len(confidences) == len(corrects)
    if not confidences:
        return 0.0

    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, corrects):
        c = min(max(c, 0.0), 1.0)
        idx = min(int(c * n_bins), n_bins - 1)  # c==1.0 は最後のビンへ
        bins[idx].append((c, 1.0 if y else 0.0))

    ece = 0.0
    n = len(confidences)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - acc)
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
    sstats = compute_step_stats(steps)
    pa = _nz(pred_answer)

    # 判定順序：MCQ > regex > numeric > normalized string
    if item.mcq_correct:
        ok = mcq_match(pa, item.mcq_correct)
        return EvalResult(ok, "mcq", sstats, {"match": ok})

    if item.gold_regex:
        ok = regex_match(pa, item.gold_regex)
        return EvalResult(ok, "regex", sstats, {"match": ok})

    if item.gold_numeric:
        comp = numeric_close(pa, _nz(item.gold), atol=numeric_atol, rtol=numeric_rtol)
        return EvalResult(comp, "numeric", sstats, {"close": comp, "atol": numeric_atol, "rtol": numeric_rtol})

    if item.gold is not None:
        ok = normalized_match(pa, _nz(item.gold))
        return EvalResult(ok, "normalized", sstats, {"match": ok})

    # ゴールドなし：キーワード再現だけ見る
    if item.required_keywords:
        rec = keyword_recall("\n".join(steps) + "\n" + pa, item.required_keywords)
        return EvalResult(None, "keywords_only", sstats, {"keyword_recall": rec})

    return EvalResult(None, "no_gold", sstats, {})

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
        g = normalize_text(gold)
        acc = sum(1 for a in answers if normalize_text(a) == g) / len(answers)
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
    results: List[EvalResult] = [
        evaluate_single(ans, steps, it, numeric_atol, numeric_rtol)
        for (ans, steps), it in zip(preds, items)
    ]

    judged = [r for r in results if isinstance(r.correct, bool)]
    acc = (sum(1 for r in judged if r.correct) / len(judged)) if judged else None
    mean_steps = statistics.mean([r.step_stats.n_steps for r in results]) if results else 0.0

    return {
        "accuracy": acc,
        "mean_steps": mean_steps,
        "n_items": len(items),
        "n_judged": len(judged),
        "details": results,
    }


# --- Generic extractors ------------------------------------------------------
from typing import Protocol, Tuple, Any

class EvalExtractor(Protocol):
    def __call__(self, pred: Any) -> Tuple[Optional[str], List[str]]: ...

def default_extractor(pred: Any) -> Tuple[Optional[str], List[str]]:
    """dict / dataclass / 任意オブジェクトから (answer, steps) を素直に取り出す。"""
    # dict っぽい
    if isinstance(pred, dict):
        ans = pred.get("answer")
        steps = pred.get("steps") or []
        # もし steps が [[...], ...] なら最初だけ拾う（SCの steps_list などを想定）
        if steps and isinstance(steps[0], list):
            steps = steps[0]
        return ans, steps
    # 属性アクセス（dataclass想定）
    ans = getattr(pred, "answer", None)
    steps = getattr(pred, "steps", []) or []
    return ans, steps

def evaluate_generic_batch(
    preds: List[Any],
    items: List[EvalItem],
    extractor: EvalExtractor = default_extractor,
    numeric_atol: float = 0.0,
    numeric_rtol: float = 0.0,
) -> Dict[str, Any]:
    """どんなモジュールの出力でも、extractor で (answer, steps) に正規化して評価。"""
    assert len(preds) == len(items)
    normed = [extractor(p) for p in preds]
    return evaluate_batch(normed, items, numeric_atol=numeric_atol, numeric_rtol=numeric_rtol)

# --- Self-consistency (SC) の汎用集計（answers を持っていれば評価） ----------
def extract_sc_answers(pred: Any) -> List[Optional[str]]:
    """pred が {'answers': [...]} や .answers を持っていれば取り出す。無ければ []。"""
    if isinstance(pred, dict):
        a = pred.get("answers")
        return a if isinstance(a, list) else []
    return getattr(pred, "answers", []) or []

def evaluate_sc_generic(pred: Any, gold: Optional[str] = None) -> Dict[str, Any]:
    answers = extract_sc_answers(pred)
    return evaluate_self_consistency(answers, gold=gold)

# --- Confidence signals & combiner ------------------------------------------
from dataclasses import dataclass

@dataclass
class ConfidenceSignals:
    sc_vote_share_max: Optional[float] = None   # SCの最大得票率 [0..1]
    ap_conf: Optional[float] = None             # UQ(AP) 由来 [0..1]
    se_conf: Optional[float] = None             # UQ(SE) 由来 [0..1]

@dataclass
class ConfidencePolicy:
    w_sc: float = 0.55
    w_se: float = 0.25
    w_ap: float = 0.20
    clamp_min: float = 0.0
    clamp_max: float = 1.0

def confidence_from_vote_share(share: Dict[str, float]) -> float:
    return max(share.values()) if share else 0.0

def combine_confidence(sig: ConfidenceSignals, pol: ConfidencePolicy = ConfidencePolicy()) -> float:
    val = 0.0
    if sig.sc_vote_share_max is not None: val += pol.w_sc * sig.sc_vote_share_max
    if sig.se_conf is not None:           val += pol.w_se * sig.se_conf
    if sig.ap_conf is not None:           val += pol.w_ap * sig.ap_conf
    return max(pol.clamp_min, min(pol.clamp_max, val))

