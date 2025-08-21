# -*- coding: utf-8 -*-
"""
JSONまわりの強化

clean_json_string()：BOM/ゼロ幅スペース除去で壊れJSONを救済

strip_markdown_code_fences()：json … の柵を剥がす

safe_json_loads()：上記クレンジング込みの安全ロード

extract_first_json_block()：カッコ整合で最初のJSONブロックを抽出

→ CoTをJSON出力モードで使うときの安定化（崩れたJSONへの耐性）

テキスト整形

normalize_whitespace() / dedent_trim() / ensure_suffix()

→ モデル出力の余分な空白・字下げ・末尾不足の正規化

再現性

seed_everything()：random/numpy/torch の乱数固定

→ 自己一貫性サンプリングや比較実験での再現性確保

集計・合議

majority_vote()：最頻値（同票ならNone）

→ Self-Consistencyの最終決定を簡単に

トークン安全なトリミング

truncate_tokens(text, tokenizer, max_tokens)：Tokenizerあり/なし両対応

→ 長すぎるプロンプトを安全に切る（語途中切断の副作用を抑える）

構造把握の補助

find_sections()：Step n:/Answer:の見出し位置を取得

→ パーサの拡張や可視化で便利
"""
from __future__ import annotations
import json
import math
import random
import re
from typing import Any, Iterable, Optional

import numpy as np

_ZWSP = "\u200b"
_BOM = "\ufeff"

def clean_json_string(s: str) -> str:
    """Remove zero-width chars/BOM and strip."""
    return s.replace(_ZWSP, "").replace(_BOM, "").strip()

def strip_markdown_code_fences(s: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
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
    start_idxs = [m.start() for m in re.finditer(r"[\{\[]", text)]
    for start in start_idxs:
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
            else:
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
                        break
        # if not closed, try next start
    return None

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def dedent_trim(s: str) -> str:
    import textwrap
    return textwrap.dedent(s).strip()

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def majority_vote(items: Iterable[str]) -> Optional[str]:
    """Return the most common item (ties -> None)."""
    items = list(items)
    if not items:
        return None
    from collections import Counter
    c = Counter(items).most_common()
    if len(c) == 1:
        return c[0][0]
    if len(c) >= 2 and c[0][1] == c[1][1]:
        return None
    return c[0][0]

def truncate_tokens(text: str, tokenizer=None, max_tokens: int = 2048) -> str:
    """
    Token-safe truncation if tokenizer provided; else rough char fallback.
    """
    if tokenizer is None:
        # ~4 chars/token heuristics
        approx = max_tokens * 4
        return text[:approx]
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) <= max_tokens:
        return text
    cut = ids[:max_tokens]
    return tokenizer.decode(cut, skip_special_tokens=True)

def ensure_suffix(text: str, suffix: str) -> str:
    return text if text.endswith(suffix) else text + suffix

def find_sections(text: str, heading_regex: str = r"(?mi)^(Step\s*\d+|Answer)\s*:") -> list[tuple[str, int, int]]:
    """
    Find (label, start, end) for sections like 'Step 1:' ... 'Answer:'.
    """
    spans = []
    for m in re.finditer(heading_regex, text):
        spans.append((m.group(1), m.start(), m.end()))
    return spans
