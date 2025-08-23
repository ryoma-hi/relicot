# -*- coding: utf-8 -*-
"""
Core CoT module (clean ver.) with progress logs.
- 既存の関数名/引数/返り値(dict)は維持（※ __init__ に verbose/log_fn を追加）
- verbose=True のとき、printで進捗を出力（log_fnを渡せばprint以外にも出せる）
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Any, List, Optional
from time import perf_counter

from .cot_prompts import render as render_prompt_default
from .cot_parser import parse_all
from .cot_utils import safe_json_loads, extract_first_json_block, majority_vote, truncate_tokens

# --------- Type aliases ----------
GenerateFn = Callable[..., str]   # (prompt: str, **decode_kwargs) -> str
RenderFn   = Callable[..., str]   # (template: str, **vars) -> str

# --------- Datamodels ----------
@dataclass
class CoTOutput:
    steps: List[str]
    answer: Optional[str]
    raw: str
    prompt: str
    decode_kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SelfConsistencyOutput:
    answer_majority: Optional[str]
    answers: List[Optional[str]]
    steps_list: List[List[str]]
    raw_list: List[str]
    prompt: str
    decode_kwargs: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# --------- Defaults ----------
_DEFAULT_DECODE      = {"max_new_tokens": 256, "temperature": 0.7}
_SC_DECODE           = {"max_new_tokens": 192, "temperature": 0.9, "top_p": 0.95}
_JSON_MODE_DECODE    = {"max_new_tokens": 256, "temperature": 0.3}

def _merge_decode(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**base, **(override or {})}

def _dk_brief(d: Dict[str, Any]) -> str:
    keys = ["max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"]
    return ", ".join(f"{k}={d[k]}" for k in keys if k in d)

class CoTGenerator:
    def __init__(self,
                 generate_fn: GenerateFn,
                 *,
                 max_input_tokens: int = 4096,
                 render_fn: RenderFn = render_prompt_default,
                 verbose: bool = False,
                 log_fn: Optional[Callable[[str], None]] = None):
        self.generate_fn = generate_fn
        self.max_input_tokens = max_input_tokens
        self.render_fn = render_fn
        self.verbose = verbose
        self._log_fn = log_fn or print

    # ---- logging helper ---------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            self._log_fn(msg)

    # ---- helpers ----------------------------------------------------------
    def _build_prompt(self, template: str, problem: str, instruction: str = "",
                      prompt_override: Optional[str] = None,
                      render_vars: Optional[Dict[str, Any]] = None) -> str:
        if prompt_override:
            self._log(f"[prompt] override used (template={template})")
            return prompt_override
        v = {"problem": problem, "instruction": instruction}
        if render_vars: v.update(render_vars)
        self._log(f"[prompt] render template='{template}' "
                  f"(instr={'yes' if instruction else 'no'}, vars={list((render_vars or {}).keys())})")
        return self.render_fn(template, **v)

    def _generate(self, prompt: str, decode_kwargs: Dict[str, Any], tokenizer=None) -> str:
        # 長さの記録（文字/トークン）
        char_len_before = len(prompt)
        tok_len_before = None
        if tokenizer is not None:
            try:
                tok_len_before = len(tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0])
            except Exception:
                tok_len_before = None

        # トリミング
        t0 = perf_counter()
        prompt_trim = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        char_len_after = len(prompt_trim)
        tok_len_after = None
        if tokenizer is not None:
            try:
                tok_len_after = len(tokenizer(prompt_trim, return_tensors="pt", add_special_tokens=False).input_ids[0])
            except Exception:
                tok_len_after = None

        if (char_len_before != char_len_after) or (tok_len_before and tok_len_after and tok_len_before != tok_len_after):
            self._log(f"[gen] prompt truncated "
                      f"(chars {char_len_before}->{char_len_after}"
                      + (f", toks {tok_len_before}->{tok_len_after}" if tok_len_before is not None else "")
                      + f")")

        self._log(f"[gen] decode_kwargs: {_dk_brief(decode_kwargs)}")
        # 生成
        t1 = perf_counter()
        text = self.generate_fn(prompt_trim, **decode_kwargs)
        t2 = perf_counter()
        self._log(f"[gen] generated {len(text)} chars in {t2 - t1:.2f}s (total {t2 - t0:.2f}s)")
        # 先頭だけプレビュー（行頭1行）
        head = text.splitlines()[0].strip() if text else ""
        if head:
            self._log(f"[gen] preview: {head[:120] + ('…' if len(head) > 120 else '')}")
        return text

    # ---- public APIs (互換維持) ------------------------------------------
    def generate_raw(self, prompt: str,
                     decode_kwargs: Optional[Dict[str, Any]] = None,
                     tokenizer=None) -> str:
        dk = _merge_decode(_DEFAULT_DECODE, decode_kwargs)
        self._log("[generate_raw] start")
        text = self._generate(prompt, dk, tokenizer=tokenizer)
        self._log("[generate_raw] done")
        return text

    def run(self, problem: str, template: str = "cot_default_ja", instruction: str = "",
            decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
            prompt_override: Optional[str] = None,
            render_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._log(f"[run] problem='{problem[:50] + ('…' if len(problem) > 50 else '')}' template='{template}'")
        dk = _merge_decode(_DEFAULT_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, instruction, prompt_override, render_vars)
        t0 = perf_counter()
        text = self._generate(prompt, dk, tokenizer=tokenizer)
        parsed = parse_all(text)
        t1 = perf_counter()
        self._log(f"[run] parsed: steps={len(parsed.steps)} answer={'yes' if parsed.answer else 'no'} "
                  f"in {t1 - t0:.2f}s")
        out = CoTOutput(steps=parsed.steps, answer=parsed.answer, raw=parsed.raw,
                        prompt=prompt, decode_kwargs=dk)
        self._log("[run] done")
        return out.to_dict()

    def run_self_consistency(self, problem: str, template: str = "cot_sc_ja", k: int = 8,
                             instruction: str = "", decode_kwargs: Optional[Dict[str, Any]] = None,
                             tokenizer=None, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        self._log(f"[SC] start: k={k} template='{template}' problem='{problem[:40] + ('…' if len(problem) > 40 else '')}'")
        dk = _merge_decode(_SC_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, instruction, prompt_override)
        raws, answers, steps_list = [], [], []
        t0 = perf_counter()
        for i in range(k):
            text = self._generate(prompt, dk, tokenizer=tokenizer)
            pr = parse_all(text)
            raws.append(pr.raw); answers.append(pr.answer); steps_list.append(pr.steps)
            self._log(f"[SC] sample {i+1}/{k}: steps={len(pr.steps)} answer={'yes' if pr.answer else 'no'}")
        voted = majority_vote([a for a in answers if a is not None])
        t1 = perf_counter()
        self._log(f"[SC] majority='{voted}' in {t1 - t0:.2f}s")
        out = SelfConsistencyOutput(
            answer_majority=voted, answers=answers, steps_list=steps_list,
            raw_list=raws, prompt=prompt, decode_kwargs=dk
        )
        self._log("[SC] done")
        return out.to_dict()

    def run_json(self, problem: str, template: str = "cot_json_ja",
                 decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
                 strict: bool = False, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        self._log(f"[JSON] start template='{template}' problem='{problem[:50] + ('…' if len(problem) > 50 else '')}'")
        dk = _merge_decode(_JSON_MODE_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, "", prompt_override)
        text = self._generate(prompt, dk, tokenizer=tokenizer)

        block = extract_first_json_block(text)
        if block is None:
            self._log("[JSON] no JSON block found → fallback parse_all")
        else:
            self._log(f"[JSON] JSON block found (len={len(block)}) → loading")

        data = safe_json_loads(block or "")
        if data is None:
            if strict:
                self._log("[JSON] invalid JSON and strict=True → return {}")
                return {}
            parsed = parse_all(text)
            out = CoTOutput(
                steps=parsed.steps, answer=parsed.answer, raw=parsed.raw,
                prompt=prompt, decode_kwargs=dk,
                meta={"note": "fallback from invalid JSON"}
            )
            self._log(f"[JSON] fallback parsed: steps={len(parsed.steps)} answer={'yes' if parsed.answer else 'no'}")
            return out.to_dict()

        out = CoTOutput(
            steps=list(data.get("steps", [])),
            answer=data.get("answer"),
            raw=text,
            prompt=prompt,
            decode_kwargs=dk
        )
        self._log(f"[JSON] loaded: steps={len(out.steps)} answer={'yes' if out.answer else 'no'}")
        self._log("[JSON] done")
        return out.to_dict()
