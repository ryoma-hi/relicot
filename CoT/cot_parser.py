# -*- coding: utf-8 -*-
"""
多様な書式のCoTを安定して構造化

ルール：

Step検出:Step n: / 1. / - の3系統に対応(正規表現)

Answer検出:Answer: / A: / 結論: / 最終解: に対応

フォールバック：見出しが無い場合の行スキャン（上限あり）

付加機能：

normalize_whitespace() と連携して整形

CoTParseResult で型付き返却

想定用途:自動評価・ログ収集・後工程(UQ/集計)に流す前処理

- parse_steps(text) -> list[str]
- extract_answer(text) -> str|None
- parse_all(text) -> {"steps": [...], "answer": "...", "raw": text}
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .cot_utils import normalize_whitespace

STEP_PATTERNS = [
    r"(?mi)^\s*Step\s*\d+\s*:\s*(.+)$",
    r"(?mi)^\s*\d+\.\s*(.+)$",
    r"(?mi)^\s*-\s*(.+)$",
]

ANSWER_PATTERNS = [
    r"(?mi)^\s*Answer\s*:\s*(.+)$",
    r"(?mi)^\s*A\s*:\s*(.+)$",
    r"(?mi)^\s*結論\s*[:：]\s*(.+)$",
    r"(?mi)^\s*最終解\s*[:：]\s*(.+)$",
]

@dataclass
class CoTParseResult:
    steps: List[str]
    answer: Optional[str]
    raw: str

def _match_first(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return normalize_whitespace(m.group(1))
    return None

def parse_steps(text: str) -> List[str]:
    steps: List[str] = []
    for pat in STEP_PATTERNS:
        ms = re.findall(pat, text)
        if ms:
            steps = [normalize_whitespace(x) for x in ms if normalize_whitespace(x)]
            break
    # Fallback: split by lines if no explicit step markers
    if not steps:
        lines = [normalize_whitespace(l) for l in text.splitlines()]
        lines = [l for l in lines if l]
        # Heuristic: keep lines until we hit 'Answer:'
        acc = []
        for l in lines:
            if re.match(r"(?mi)^\s*(Answer|A|結論|最終解)\s*[:：]", l):
                break
            acc.append(l)
        # Avoid dumping all if it's too long
        steps = acc[:10]  # cap to 10 lines
    return steps

def extract_answer(text: str) -> Optional[str]:
    return _match_first(text, ANSWER_PATTERNS)

def parse_all(text: str) -> CoTParseResult:
    return CoTParseResult(
        steps=parse_steps(text),
        answer=extract_answer(text),
        raw=text,
    )
