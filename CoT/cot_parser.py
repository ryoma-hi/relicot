# -*- coding: utf-8 -*-
"""
多様な書式のCoTを安定して構造化（clean ver.）

- parse_steps(text) -> List[str]
- extract_answer(text) -> Optional[str]
- parse_all(text) -> CoTParseResult
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re

from .cot_utils import normalize_whitespace

# 既存の正規表現パターン（順序で優先度が決まる）

STEP_PATTERNS = [
    r"(?mi)^\s*Step\s*\d+\s*[:：]\s*(.+)$",
    r"(?mi)^\s*ステップ\s*\d+\s*[:：]\s*(.+)$",  # ← 日本語も許容
    r"(?mi)^\s*\d+\.\s*(.+)$",
    r"(?mi)^\s*[-・]\s*(.+)$",
]

ANSWER_PATTERNS = [
    r"(?mi)^\s*Answer\s*[:：]\s*(.+)$",
    r"(?mi)^\s*Final\s*Answer\s*[:：]\s*(.+)$",     # ← 追加
    r"(?mi)^\s*A\s*[:：]\s*(.+)$",
    r"(?mi)^\s*結論\s*[:：]\s*(.+)$",
    r"(?mi)^\s*最終(?:解|解答|回答)\s*[:：]\s*(.+)$",  # ← 追加
]

# フォールバック時に「ここで打ち切る」ための見出し検出
ANSWER_HEAD_RE = re.compile(r"(?mi)^\s*(Answer|A|結論|最終解)\s*[:：]")
FALLBACK_MAX_LINES = 10


@dataclass
class CoTParseResult:
    steps: List[str]
    answer: Optional[str]
    raw: str


# ---- helpers ---------------------------------------------------------------
def _match_first_group(text: str, patterns: List[str]) -> Optional[str]:
    """パターン群のどれかに最初にマッチしたグループ1を返す。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return normalize_whitespace(m.group(1))
    return None


def _find_steps_by_patterns(text: str) -> List[str]:
    """STEP_PATTERNSのいずれかで抽出（最初にヒットした規則だけ採用）。"""
    for pat in STEP_PATTERNS:
        matches = re.findall(pat, text)
        if matches:
            # 正規化して空要素を除去
            steps = [normalize_whitespace(s) for s in matches]
            return [s for s in steps if s]
    return []


def _fallback_lines_until_answer(text: str) -> List[str]:
    """見出しが無い場合、Answer見出しまで（最大FALLBACK_MAX_LINES）を行で返す。"""
    steps: List[str] = []
    for line in text.splitlines():
        line_norm = normalize_whitespace(line)
        if not line_norm:
            continue
        if ANSWER_HEAD_RE.match(line_norm):
            break
        steps.append(line_norm)
        if len(steps) >= FALLBACK_MAX_LINES:
            break
    return steps


# ---- public APIs -----------------------------------------------------------
def parse_steps(text: str) -> List[str]:
    steps = _find_steps_by_patterns(text)
    return steps if steps else _fallback_lines_until_answer(text)


def extract_answer(text: str) -> Optional[str]:
    return _match_first_group(text, ANSWER_PATTERNS)


def parse_all(text: str) -> CoTParseResult:
    return CoTParseResult(
        steps=parse_steps(text),
        answer=extract_answer(text),
        raw=text,
    )
