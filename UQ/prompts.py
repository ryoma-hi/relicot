# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Mapping
from ..common.prompt_store import PromptStore  # 既に作った共通PromptStoreを再利用

_store = PromptStore(package=__package__.rsplit(".", 1)[0])  # "relicot.UQ"

def render(name: str, *, vars: Optional[Mapping] = None, strict: bool = True) -> str:
    return _store.render(name, vars or {}, strict=strict)
