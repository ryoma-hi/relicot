# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import lru_cache
from importlib import resources
from typing import Optional, Mapping

class _DefaultDict(dict):
    """format_map用のデフォルト空文字（strict=Falseのときに使う）"""
    def __missing__(self, key): return ""

class PromptStore:
    """
    - 各パッケージ内の `prompts/` ディレクトリから <name>.txt を読み込む
    - {var} 形式でフォーマット
    """
    def __init__(self, package: str, subdir: str = "prompts"):
        self.package = package   # 例: "relicot.CoT"
        self.subdir  = subdir

    @lru_cache(maxsize=None)
    def _load_text(self, name: str) -> str:
        # prompts/<name>.txt を読む（importlib.resourcesでパッケージ同梱OK）
        path = f"{self.subdir}/{name}.txt"
        data = resources.files(self.package).joinpath(path).read_text(encoding="utf-8")
        return data

    def render(self, name: str, vars: Optional[Mapping] = None, *, strict: bool = True) -> str:
        tmpl = self._load_text(name)
        vars = dict(vars or {})
        if strict:
            return tmpl.format(**vars)          # キー不足なら例外（早期に気づける）
        return tmpl.format_map(_DefaultDict(vars))  # 不足キーは空文字
