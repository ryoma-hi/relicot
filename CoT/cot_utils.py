# -*- coding: utf-8 -*-
"""
Utility helpers for CoT (clean ver.)
- JSON復旧/抽出
- テキスト整形
- 再現性（乱数固定）
- 合議/集計
- トークン安全トリミング
- セクション検出
"""
from __future__ import annotations
import json
import random
import re
from typing import Any, Iterable, Optional, List, Tuple

import numpy as np

# -------- Constants & precompiled regex --------
_ZWSP = "\u200b"
_BOM  = "\ufeff"
_CODE_FENCE_HEAD_RE = re.compile(r"^```[a-zA-Z]*\n")   # e.g., ```json\n
_CODE_FENCE_TAIL_RE = re.compile(r"\n```$")
_WS_MULTI_RE        = re.compile(r"[ \t]+")
_JSON_OPEN_RE       = re.compile(r"[\{\[]")
_ANSWER_HEAD_RE     = re.compile(r"(?mi)^(Step\s*\d+|Answer)\s*:")

# -------- JSON helpers --------
def clean_json_string(s: str) -> str:
    """Remove zero-width chars/BOM and strip."""
    return (s or "").replace(_ZWSP, "").replace(_BOM, "").strip()

def strip_markdown_code_fences(s: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = _CODE_FENCE_HEAD_RE.sub("", s)
        s = _CODE_FENCE_TAIL_RE.sub("", s)
    return s.strip()

def safe_json_loads(s: str) -> Optional[Any]:
    """Best-effort JSON loading with cleanup."""
    try:
        return json.loads(strip_markdown_code_fences(clean_json_string(s)))
    except Exception:
        return None

def extract_first_json_block(text: str) -> Optional[str]:
    """
    Extract first JSON-like block using a bracket matcher.
    Returns raw substring or None.
    """
    if not text:
        return None
    starts = [m.start() for m in _JSON_OPEN_RE.finditer(text)]
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            # outside string
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    if safe_json_loads(candidate) is not None:
                        return candidate
                    break  # 該当開始点は不成立。次の開始点へ
    return None

# -------- Text helpers --------
def normalize_whitespace(s: str) -> str:
    return _WS_MULTI_RE.sub(" ", (s or "")).strip()

def dedent_trim(s: str) -> str:
    import textwrap
    return textwrap.dedent(s or "").strip()

def ensure_suffix(text: str, suffix: str) -> str:
    return text if (text or "").endswith(suffix) else (text or "") + suffix

# -------- Reproducibility --------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 安定化（挙動は同じ、性能だけ影響し得る）
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # torch 未導入/環境差分でも静かに無視
        pass

# -------- Aggregation / voting --------
def majority_vote(items: Iterable[str]) -> Optional[str]:
    """Return the most common item (ties -> None)."""
    from collections import Counter
    lst = [x for x in items]
    if not lst:
        return None
    counts = Counter(lst).most_common()
    if len(counts) == 1:
        return counts[0][0]
    return None if counts[0][1] == counts[1][1] else counts[0][0]

# -------- Token-safe truncation --------
def truncate_tokens(text: str, tokenizer=None, max_tokens: int = 2048) -> str:
    """
    Token-safe truncation if tokenizer provided; else rough char fallback.
    """
    if tokenizer is None:
        # ~4 chars/token heuristics
        approx = max_tokens * 4
        return (text or "")[:approx]
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) <= max_tokens:
        return text
    cut = ids[:max_tokens]
    return tokenizer.decode(cut, skip_special_tokens=True)

# -------- Section locator --------
def find_sections(text: str, heading_regex: str = r"(?mi)^(Step\s*\d+|Answer)\s*:") -> List[Tuple[str, int, int]]:
    """
    Find (label, start, end) for sections like 'Step 1:' ... 'Answer:'.
    """
    spans: List[Tuple[str, int, int]] = []
    for m in re.finditer(heading_regex, text or ""):
        spans.append((m.group(1), m.start(), m.end()))
    return spans
