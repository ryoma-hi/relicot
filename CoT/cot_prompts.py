# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Mapping
from ..common.prompt_store import PromptStore

# このモジュールの prompts ディレクトリをストアに紐づけ
_store = PromptStore(package=__package__)

def render(template: str, *, problem: str = "", instruction: str = "", vars: Optional[Mapping] = None) -> str:
    """
    互換I/F維持: render("cot_default_ja", problem=..., instruction=...)
    追加: 任意の追加変数を vars= で渡せる
    """
    v = {"problem": problem, "instruction": instruction}
    if vars: v.update(vars)
    return _store.render(template, v, strict=True)
