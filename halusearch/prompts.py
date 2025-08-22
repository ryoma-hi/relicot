# halusearch/prompts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Mapping, List
from importlib import resources

# ---- 1) 共通PromptStoreがあれば使う（無ければローカル実装へ） -----------------
_PS = None
try:
    # 例: relicot/common/prompt_store.py
    from ..common.prompt_store import PromptStore as _PS  # type: ignore
except Exception:
    try:
        # 例: relicot/common/prompts.py （ファイル名違いにも対応）
        from ..common.prompts import PromptStore as _PS  # type: ignore
    except Exception:
        _PS = None  # フォールバックへ

# ---- 2) ローカルStore（最小実装） -------------------------------------------
if _PS is None:
    class _LocalStore:
        """
        prompts/<name>.txt を importlib.resources で読み込み、str.format で埋め込むだけの軽実装。
        - strict=True: 不足キーがあれば KeyError
        - strict=False: 不足キーは空文字に
        - サブディレクトリは 'prompts' → 無ければ 'prompt' を探す
        """
        def __init__(self, package: str, subdir: str = "prompts", alt_subdir: str = "prompt"):
            self.package = package
            self.subdir = subdir
            self.alt_subdir = alt_subdir

        def _load_text(self, name: str) -> str:
            last_err = None
            for d in (self.subdir, self.alt_subdir):
                try:
                    return resources.files(self.package).joinpath(f"{d}/{name}.txt").read_text(encoding="utf-8")
                except Exception as e:
                    last_err = e
            raise FileNotFoundError(f"template '{name}.txt' not found in {self.package}/prompts or /prompt") from last_err

        def render(self, name: str, vars: Mapping | None = None, *, strict: bool = True) -> str:
            txt = self._load_text(name)
            data = dict(vars or {})
            if strict:
                return txt.format(**data)
            class _DDict(dict):
                def __missing__(self, key): return ""
            return txt.format_map(_DDict(data))

        def available(self) -> List[str]:
            names: set[str] = set()
            for d in ("prompts", "prompt"):
                try:
                    for p in resources.files(self.package).joinpath(d).iterdir():
                        if str(p).endswith(".txt"):
                            names.add(p.name[:-4])
                except Exception:
                    pass
            return sorted(names)

    _store = _LocalStore(package=__package__)
else:
    # 共通PromptStoreがある場合（推奨パス）
    _store = _PS(package=__package__, subdir="prompts")  # subdir='prompts' を明示

# ---- 3) 公開API --------------------------------------------------------------
def render(name: str, *, vars: Optional[Mapping] = None, strict: bool = True) -> str:
    """
    halusearch/prompts/<name>.txt を読み込み、vars を埋め込む。
    strict=True だと不足キーで KeyError を出して早期に気づけます。
    """
    return _store.render(name, vars or {}, strict=strict)

def available_templates() -> List[str]:
    """利用可能なテンプレート名の一覧（拡張子 .txt を除く）"""
    if hasattr(_store, "available"):
        return _store.available()
    # ここに来ることはほぼ無い想定だが、一応安全側に
    return []
